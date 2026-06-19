import pandas as pd
import numpy as np
from typing import Dict, Any
from finflow_agent.operations.schemas import CalculationOperation
from finflow_agent.operations.errors import OperationExecutionError

def _check_numeric(df: pd.DataFrame, col: str):
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise OperationExecutionError(f"Column {col} must be numeric for this calculation.")

def _round_if_currency(val: float, col_name: str) -> float:
    currency_keywords = ["revenue", "price", "amount", "cost", "sales", "profit", "total", "sum", "balance", "value", "metric"]
    col_lower = col_name.lower()
    if any(kw in col_lower for kw in currency_keywords):
        return round(val, 2)
    return val

def _round_series_if_currency(series: pd.Series, col_name: str) -> pd.Series:
    currency_keywords = ["revenue", "price", "amount", "cost", "sales", "profit", "total", "sum", "balance", "value", "metric"]
    col_lower = col_name.lower()
    if any(kw in col_lower for kw in currency_keywords):
        return series.round(2)
    return series

def calc_sum(df: pd.DataFrame, op: CalculationOperation) -> Dict[str, Any]:
    _check_numeric(df, op.column)
    out_col = op.output_column or f"sum_{op.column}"
    val = float(df[op.column].sum())
    val = _round_if_currency(val, out_col)
    return {"metrics": {out_col: val}}

def calc_mean(df: pd.DataFrame, op: CalculationOperation) -> Dict[str, Any]:
    _check_numeric(df, op.column)
    out_col = op.output_column or f"mean_{op.column}"
    val = float(df[op.column].mean())
    val = _round_if_currency(val, out_col)
    return {"metrics": {out_col: val}}

def calc_median(df: pd.DataFrame, op: CalculationOperation) -> Dict[str, Any]:
    _check_numeric(df, op.column)
    out_col = op.output_column or f"median_{op.column}"
    val = float(df[op.column].median())
    val = _round_if_currency(val, out_col)
    return {"metrics": {out_col: val}}

def calc_min(df: pd.DataFrame, op: CalculationOperation) -> Dict[str, Any]:
    _check_numeric(df, op.column)
    out_col = op.output_column or f"min_{op.column}"
    val = float(df[op.column].min())
    val = _round_if_currency(val, out_col)
    return {"metrics": {out_col: val}}

def calc_max(df: pd.DataFrame, op: CalculationOperation) -> Dict[str, Any]:
    _check_numeric(df, op.column)
    out_col = op.output_column or f"max_{op.column}"
    val = float(df[op.column].max())
    val = _round_if_currency(val, out_col)
    return {"metrics": {out_col: val}}

def calc_count(df: pd.DataFrame, op: CalculationOperation) -> Dict[str, Any]:
    out_col = op.output_column or f"count_{op.column}"
    val = int(df[op.column].count())
    return {"metrics": {out_col: val}}

def calc_count_distinct(df: pd.DataFrame, op: CalculationOperation) -> Dict[str, Any]:
    out_col = op.output_column or f"count_distinct_{op.column}"
    val = int(df[op.column].nunique())
    return {"metrics": {out_col: val}}

def calc_variance(df: pd.DataFrame, op: CalculationOperation) -> Dict[str, Any]:
    _check_numeric(df, op.column)
    out_col = op.output_column or f"variance_{op.column}"
    val = float(df[op.column].var())
    val = _round_if_currency(val, out_col)
    return {"metrics": {out_col: val}}

def calc_standard_deviation(df: pd.DataFrame, op: CalculationOperation) -> Dict[str, Any]:
    _check_numeric(df, op.column)
    out_col = op.output_column or f"standard_deviation_{op.column}"
    val = float(df[op.column].std())
    val = _round_if_currency(val, out_col)
    return {"metrics": {out_col: val}}

def calc_group_sum(df: pd.DataFrame, op: CalculationOperation) -> Dict[str, Any]:
    _check_numeric(df, op.column)
    out_col = op.output_column or f"sum_{op.column}"
    grouped = df.groupby(op.group_by, as_index=False)[op.column].sum()
    grouped.rename(columns={op.column: out_col}, inplace=True)
    grouped[out_col] = _round_series_if_currency(grouped[out_col], out_col)
    return {"df": grouped}

def calc_group_mean(df: pd.DataFrame, op: CalculationOperation) -> Dict[str, Any]:
    _check_numeric(df, op.column)
    out_col = op.output_column or f"mean_{op.column}"
    grouped = df.groupby(op.group_by, as_index=False)[op.column].mean()
    grouped.rename(columns={op.column: out_col}, inplace=True)
    grouped[out_col] = _round_series_if_currency(grouped[out_col], out_col)
    return {"df": grouped}

def calc_group_count(df: pd.DataFrame, op: CalculationOperation) -> Dict[str, Any]:
    out_col = op.output_column or f"count_{op.column}"
    grouped = df.groupby(op.group_by, as_index=False).size()
    grouped.rename(columns={'size': out_col}, inplace=True)
    return {"df": grouped}

def calc_running_total(df: pd.DataFrame, op: CalculationOperation) -> Dict[str, Any]:
    _check_numeric(df, op.column)
    if not op.sort_by:
        raise OperationExecutionError("running_total requires sort_by.")
    
    # Sort by sort_by
    df = df.sort_values(by=op.sort_by)
    out_col = op.output_column or f"running_total_{op.column}"
    warnings = []
    
    if not op.partition_by:
        # Detect multiple likely entity/account/category columns to warn
        likely_entity_cols = []
        for col in df.columns:
            if col != op.sort_by and col != op.column:
                if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                    if df[col].nunique() > 1 and df[col].nunique() < len(df) * 0.5:
                        likely_entity_cols.append(col)
        if likely_entity_cols:
            warnings.append(f"running_total has no partition_by but dataset contains potential entity/category columns: {likely_entity_cols}")
            
    if op.partition_by:
        df[out_col] = df.groupby(op.partition_by)[op.column].transform(lambda x: x.cumsum())
    else:
        df[out_col] = df[op.column].cumsum()
        
    df[out_col] = _round_series_if_currency(df[out_col], out_col)
    return {"df": df, "warnings": warnings}

def calc_percentage_change(df: pd.DataFrame, op: CalculationOperation) -> Dict[str, Any]:
    _check_numeric(df, op.column)
    if not op.sort_by:
        raise OperationExecutionError("percentage_change requires sort_by.")
        
    df = df.sort_values(by=op.sort_by)
    out_col = op.output_column or f"pct_change_{op.column}"
    warnings = []
    
    if not op.partition_by:
        likely_entity_cols = []
        for col in df.columns:
            if col != op.sort_by and col != op.column:
                if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                    if df[col].nunique() > 1 and df[col].nunique() < len(df) * 0.5:
                        likely_entity_cols.append(col)
        if likely_entity_cols:
            warnings.append(f"percentage_change has no partition_by but dataset contains potential entity/category columns: {likely_entity_cols}")
            
    if op.partition_by:
        df[out_col] = df.groupby(op.partition_by)[op.column].transform(lambda x: x.pct_change())
    else:
        df[out_col] = df[op.column].pct_change()
        
    return {"df": df, "warnings": warnings}

def calc_difference(df: pd.DataFrame, op: CalculationOperation) -> Dict[str, Any]:
    _check_numeric(df, op.column)
    if not op.secondary_column:
        raise OperationExecutionError("Difference requires secondary_column.")
    _check_numeric(df, op.secondary_column)
    out_col = op.output_column or f"diff_{op.column}_{op.secondary_column}"
    df[out_col] = df[op.column] - df[op.secondary_column]
    df[out_col] = _round_series_if_currency(df[out_col], out_col)
    return {"df": df}

def calc_ratio(df: pd.DataFrame, op: CalculationOperation) -> Dict[str, Any]:
    _check_numeric(df, op.column)
    if not op.secondary_column:
        raise OperationExecutionError("Ratio requires secondary_column.")
    _check_numeric(df, op.secondary_column)
    out_col = op.output_column or f"ratio_{op.column}_{op.secondary_column}"
    
    zeros_count = int((df[op.secondary_column] == 0).sum())
    warnings = []
    if zeros_count > 0:
        warnings.append(f"Ratio calculation encountered {zeros_count} rows with zero denominator in column '{op.secondary_column}'. These were set to NaN.")
        
    df[out_col] = np.where(df[op.secondary_column] == 0, np.nan, df[op.column] / df[op.secondary_column])
    df[out_col] = _round_series_if_currency(df[out_col], out_col)
    return {"df": df, "warnings": warnings}

def calc_absolute_value(df: pd.DataFrame, op: CalculationOperation) -> Dict[str, Any]:
    if op.column == "__all_numeric_columns__":
        numeric_columns = list(df.select_dtypes(include=["number"]).columns)
        for column in numeric_columns:
            df[column] = df[column].abs()
        return {
            "df": df,
            "warnings": [] if numeric_columns else ["absolute_value found no numeric columns to transform."],
        }

    _check_numeric(df, op.column)
    out_col = op.output_column or op.column
    df[out_col] = df[op.column].abs()
    return {"df": df}

CALCULATION_HANDLERS = {
    "sum": calc_sum,
    "mean": calc_mean,
    "median": calc_median,
    "min": calc_min,
    "max": calc_max,
    "count": calc_count,
    "count_distinct": calc_count_distinct,
    "variance": calc_variance,
    "standard_deviation": calc_standard_deviation,
    "group_sum": calc_group_sum,
    "group_mean": calc_group_mean,
    "group_count": calc_group_count,
    "running_total": calc_running_total,
    "percentage_change": calc_percentage_change,
    "difference": calc_difference,
    "ratio": calc_ratio,
    "absolute_value": calc_absolute_value,
}
