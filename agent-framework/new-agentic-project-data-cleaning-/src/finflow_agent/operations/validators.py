from typing import List, Any
import pandas as pd
from finflow_agent.operations.errors import OperationValidationError

def validate_columns_exist(df: pd.DataFrame, columns: Any) -> None:
    """
    Validates that the specified columns exist in the DataFrame.
    """
    if isinstance(columns, str) and columns in {"__all_string_columns__", "__all_numeric_columns__"}:
        return

    if isinstance(columns, str):
        columns = [columns]

    if not isinstance(columns, list):
        return

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise OperationValidationError(f"Missing required columns in dataset: {missing}")

def get_string_columns(df: pd.DataFrame, requested_cols: Any) -> List[str]:
    """
    Resolves __all_string_columns__ macro or filters to string columns.
    """
    if requested_cols == "__all_string_columns__":
        return list(df.select_dtypes(include=['object', 'string']).columns)
    
    if isinstance(requested_cols, str):
        requested_cols = [requested_cols]
        
    return [col for col in requested_cols if col in df.columns and pd.api.types.is_string_dtype(df[col])]

def required_columns_for_operation(op) -> List[str]:
    required = []

    if hasattr(op, "column") and getattr(op, "column"):
        if op.column not in {"__all_string_columns__", "__all_numeric_columns__"}:
            required.append(op.column)

    if hasattr(op, "columns"):
        cols = op.columns
        if cols not in {"__all_string_columns__", "__all_numeric_columns__"}:
            if isinstance(cols, str):
                required.append(cols)
            elif isinstance(cols, list):
                required.extend(cols)

    if hasattr(op, "group_by") and op.group_by:
        required.extend(op.group_by)

    if hasattr(op, "secondary_column") and op.secondary_column:
        required.append(op.secondary_column)

    if hasattr(op, "sort_by") and op.sort_by:
        required.append(op.sort_by)

    if hasattr(op, "partition_by") and op.partition_by:
        required.extend(op.partition_by)

    return list(dict.fromkeys(required))

import hashlib
import json
from pydantic import BaseModel

def hash_operation_params(op) -> str:
    """
    Stably hashes operation parameters.
    """
    if isinstance(op, BaseModel):
        data = op.model_dump()
    elif isinstance(op, dict):
        data = op
    else:
        data = getattr(op, "__dict__", {})
        
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
