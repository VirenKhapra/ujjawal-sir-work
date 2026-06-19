from typing import Annotated, Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, model_validator

# -------------------------------------------------------------------
# BASE VALIDATORS & HELPERS
# -------------------------------------------------------------------
class BaseOperation(BaseModel):
    pass

# -------------------------------------------------------------------
# CLEANING SCHEMAS
# -------------------------------------------------------------------
class TrimWhitespaceOperation(BaseOperation):
    type: Literal["trim_whitespace"] = "trim_whitespace"
    columns: Union[List[str], Literal["__all_string_columns__"]]

class NormalizeColumnNamesOperation(BaseOperation):
    type: Literal["normalize_column_names"] = "normalize_column_names"
    style: Literal["lowercase", "snake_case", "camel_case", "pascal_case"] = "snake_case"

class DropDuplicatesOperation(BaseOperation):
    type: Literal["drop_duplicates"] = "drop_duplicates"
    subset: Optional[List[str]] = None
    keep: Literal["first", "last", False] = "first"

class FillNullsOperation(BaseOperation):
    type: Literal["fill_nulls"] = "fill_nulls"
    columns: List[str]
    strategy: Literal["zero", "empty_string", "mean", "median", "mode", "constant"]
    value: Optional[Any] = None

class DropNullsOperation(BaseOperation):
    type: Literal["drop_nulls"] = "drop_nulls"
    columns: Optional[List[str]] = None
    how: Literal["any", "all"] = "any"

class NormalizeDateOperation(BaseOperation):
    type: Literal["normalize_date"] = "normalize_date"
    column: str
    target_format: str = "%Y-%m-%d"
    dayfirst: bool = False
    errors: Literal["raise", "coerce", "ignore"] = "coerce"

class NormalizeCurrencyOperation(BaseOperation):
    type: Literal["normalize_currency"] = "normalize_currency"
    column: str

class NormalizeNumberOperation(BaseOperation):
    type: Literal["normalize_number"] = "normalize_number"
    column: str

class NormalizeTextCaseOperation(BaseOperation):
    type: Literal["normalize_text_case"] = "normalize_text_case"
    columns: Union[List[str], Literal["__all_string_columns__"]]
    case: Literal["lower", "upper", "title", "capitalize"]

class ReplaceValuesOperation(BaseOperation):
    type: Literal["replace_values"] = "replace_values"
    column: str
    mapping: Dict[Any, Any]

class StripCurrencySymbolsOperation(BaseOperation):
    type: Literal["strip_currency_symbols"] = "strip_currency_symbols"
    column: str

class RemoveCommasFromNumbersOperation(BaseOperation):
    type: Literal["remove_commas_from_numbers"] = "remove_commas_from_numbers"
    column: str

class CoerceColumnTypeOperation(BaseOperation):
    type: Literal["coerce_column_type"] = "coerce_column_type"
    column: str
    target_type: Literal["string", "integer", "float", "decimal", "boolean", "date"]
    errors: Literal["raise", "coerce", "ignore"] = "coerce"

class RemoveEmptyRowsOperation(BaseOperation):
    type: Literal["remove_empty_rows"] = "remove_empty_rows"

class RemoveEmptyColumnsOperation(BaseOperation):
    type: Literal["remove_empty_columns"] = "remove_empty_columns"

class RenameColumnsOperation(BaseOperation):
    type: Literal["rename_columns"] = "rename_columns"
    mapping: Dict[str, str]

class ReorderColumnsOperation(BaseOperation):
    type: Literal["reorder_columns"] = "reorder_columns"
    columns: List[str]

CleaningOperationType = Annotated[
    Union[
        TrimWhitespaceOperation,
        NormalizeColumnNamesOperation,
        DropDuplicatesOperation,
        FillNullsOperation,
        DropNullsOperation,
        NormalizeDateOperation,
        NormalizeCurrencyOperation,
        NormalizeNumberOperation,
        NormalizeTextCaseOperation,
        ReplaceValuesOperation,
        StripCurrencySymbolsOperation,
        RemoveCommasFromNumbersOperation,
        CoerceColumnTypeOperation,
        RemoveEmptyRowsOperation,
        RemoveEmptyColumnsOperation,
        RenameColumnsOperation,
        ReorderColumnsOperation,
    ],
    Field(discriminator="type"),
]

class CleaningOperationPlan(BaseModel):
    operations: List[CleaningOperationType]

# -------------------------------------------------------------------
# FILTER SCHEMAS
# -------------------------------------------------------------------
class FilterCondition(BaseModel):
    column: str
    operator: Literal["eq", "neq", "gt", "gte", "lt", "lte", "contains", "not_contains", 
                     "starts_with", "ends_with", "between", "in", "not_in", "is_null", "is_not_null"]
    value: Optional[Any] = None
    value_to: Optional[Any] = None # For 'between'
    case_sensitive: bool = False

    @model_validator(mode="before")
    @classmethod
    def validate_values(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        op = data.get("operator")
        val = data.get("value")
        val_to = data.get("value_to")
        
        if op in ["is_null", "is_not_null"]:
            pass # Value not needed
        elif op in ["in", "not_in"]:
            if not isinstance(val, list):
                raise ValueError(f"Operator {op} requires a list value.")
        elif op == "between":
            if val is None or val_to is None:
                raise ValueError("Operator 'between' requires both value and value_to.")
        else:
            if val is None:
                raise ValueError(f"Operator {op} requires a value.")
        return data

class FilterOperationPlan(BaseModel):
    conditions: List[FilterCondition] = Field(default_factory=list)
    logic: Literal["and", "or"] = "and"
    select_columns: Optional[List[str]] = None
    limit: Optional[int] = None

# -------------------------------------------------------------------
# CALCULATION SCHEMAS
# -------------------------------------------------------------------
class CalculationOperation(BaseModel):
    type: Literal["sum", "mean", "median", "min", "max", "count", "count_distinct", 
                  "variance", "standard_deviation", "group_sum", "group_mean", "group_count", 
                  "running_total", "percentage_change", "difference", "ratio", "absolute_value"]
    column: str
    output_column: Optional[str] = None
    group_by: Optional[List[str]] = None
    secondary_column: Optional[str] = None # For ratio, difference, percentage_change
    sort_by: Optional[str] = None
    partition_by: Optional[List[str]] = None

    @model_validator(mode="before")
    @classmethod
    def validate_calc(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        t = data.get("type")
        if t in ["group_sum", "group_mean", "group_count"]:
            gb = data.get("group_by")
            if not gb or not isinstance(gb, list) or len(gb) == 0:
                raise ValueError(f"Operator '{t}' requires non-empty group_by columns.")
        elif t in ["running_total", "percentage_change"]:
            if not data.get("sort_by"):
                raise ValueError(f"Operator '{t}' requires a sort_by column.")
        elif t in ["ratio", "difference"]:
            if not data.get("secondary_column"):
                raise ValueError(f"Operator '{t}' requires secondary_column.")
        return data

class CalculationOperationPlan(BaseModel):
    operations: List[CalculationOperation]

# -------------------------------------------------------------------
# VISUALIZATION SCHEMAS
# -------------------------------------------------------------------
class ChartSpec(BaseModel):
    type: Literal["bar", "line", "pie", "scatter", "area", "stacked_bar"]
    x: str
    y: Union[str, List[str]]
    title: str

class VisualizationOperationPlan(BaseModel):
    charts: List[ChartSpec]

# -------------------------------------------------------------------
# REPORTING SCHEMAS
# -------------------------------------------------------------------
class ReportingOperationPlan(BaseModel):
    output_format: Literal["xlsx", "csv", "json", "txt"]
    sheet_name: Optional[str] = None
    title: Optional[str] = None
