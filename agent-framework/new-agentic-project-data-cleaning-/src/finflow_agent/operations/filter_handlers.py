import pandas as pd
import re
from typing import Dict, Any
from finflow_agent.operations.schemas import FilterCondition
from finflow_agent.operations.errors import OperationExecutionError

def _check_numeric(s: pd.Series, operator: str):
    if not pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_datetime64_any_dtype(s):
        raise OperationExecutionError(f"Operator {operator} requires numeric or datetime column, got {s.dtype}")

def filter_eq(s: pd.Series, cond: FilterCondition) -> pd.Series:
    if _is_text_series(s) and isinstance(cond.value, str) and not cond.case_sensitive:
        return s.astype(str).str.lower().str.strip() == cond.value.lower().strip()
    return s == cond.value

def filter_neq(s: pd.Series, cond: FilterCondition) -> pd.Series:
    if _is_text_series(s) and isinstance(cond.value, str) and not cond.case_sensitive:
        return s.astype(str).str.lower().str.strip() != cond.value.lower().strip()
    return s != cond.value

def filter_gt(s: pd.Series, cond: FilterCondition) -> pd.Series:
    _check_numeric(s, "gt")
    return s > cond.value

def filter_gte(s: pd.Series, cond: FilterCondition) -> pd.Series:
    _check_numeric(s, "gte")
    return s >= cond.value

def filter_lt(s: pd.Series, cond: FilterCondition) -> pd.Series:
    _check_numeric(s, "lt")
    return s < cond.value

def filter_lte(s: pd.Series, cond: FilterCondition) -> pd.Series:
    _check_numeric(s, "lte")
    return s <= cond.value

def filter_contains(s: pd.Series, cond: FilterCondition) -> pd.Series:
    return s.astype(str).str.contains(str(cond.value), case=cond.case_sensitive, regex=False, na=False)

def filter_not_contains(s: pd.Series, cond: FilterCondition) -> pd.Series:
    return ~s.astype(str).str.contains(str(cond.value), case=cond.case_sensitive, regex=False, na=False)

def filter_starts_with(s: pd.Series, cond: FilterCondition) -> pd.Series:
    if not cond.case_sensitive:
        return s.astype(str).str.lower().str.startswith(str(cond.value).lower(), na=False)
    return s.astype(str).str.startswith(str(cond.value), na=False)

def filter_ends_with(s: pd.Series, cond: FilterCondition) -> pd.Series:
    if not cond.case_sensitive:
        return s.astype(str).str.lower().str.endswith(str(cond.value).lower(), na=False)
    return s.astype(str).str.endswith(str(cond.value), na=False)

def filter_between(s: pd.Series, cond: FilterCondition) -> pd.Series:
    _check_numeric(s, "between")
    return s.between(cond.value, cond.value_to)

def _is_text_series(s: pd.Series) -> bool:
    return pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s)

def _normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def _compact_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())

def _text_mask(
    s: pd.Series,
    values: list[Any],
    *,
    mode: str,
    case_sensitive: bool = False,
) -> pd.Series:
    series = s.astype(str)
    if case_sensitive:
        normalized = series
        compact = series.str.replace(r"[^A-Za-z0-9]+", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
        compact = series.str.replace(r"[^A-Za-z0-9]+", "", regex=True)
    else:
        normalized = series.str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
        compact = series.str.lower().str.replace(r"[^a-z0-9]+", "", regex=True)

    mask = pd.Series(False, index=s.index)
    for value in values:
        request_raw = str(value or "").strip()
        if not request_raw:
            continue
        if case_sensitive:
            request_norm = request_raw
            request_compact = re.sub(r"[^A-Za-z0-9]+", "", request_raw)
        else:
            request_norm = _normalize_text(request_raw)
            request_compact = _compact_text(request_raw)

        if mode == "eq":
            mask |= normalized.eq(request_norm) | compact.eq(request_compact)
        elif mode == "contains":
            mask |= normalized.str.contains(re.escape(request_norm), case=True, regex=True, na=False)
            mask |= compact.str.contains(re.escape(request_compact), case=True, regex=True, na=False)
        elif mode == "starts_with":
            mask |= normalized.str.startswith(request_norm, na=False)
            mask |= compact.str.startswith(request_compact, na=False)
        elif mode == "ends_with":
            mask |= normalized.str.endswith(request_norm, na=False)
            mask |= compact.str.endswith(request_compact, na=False)
        else:
            raise ValueError(f"Unsupported text match mode: {mode}")
    return mask

def filter_in(s: pd.Series, cond: FilterCondition) -> pd.Series:
    if _is_text_series(s):
        return _text_mask(s, list(cond.value), mode="eq", case_sensitive=cond.case_sensitive)
    return s.isin(cond.value)

def filter_not_in(s: pd.Series, cond: FilterCondition) -> pd.Series:
    if _is_text_series(s):
        return ~_text_mask(s, list(cond.value), mode="eq", case_sensitive=cond.case_sensitive)
    return ~s.isin(cond.value)

def filter_is_null(s: pd.Series, cond: FilterCondition) -> pd.Series:
    return s.isnull()

def filter_is_not_null(s: pd.Series, cond: FilterCondition) -> pd.Series:
    return s.notnull()

FILTER_HANDLERS = {
    "eq": filter_eq,
    "neq": filter_neq,
    "gt": filter_gt,
    "gte": filter_gte,
    "lt": filter_lt,
    "lte": filter_lte,
    "contains": filter_contains,
    "not_contains": filter_not_contains,
    "starts_with": filter_starts_with,
    "ends_with": filter_ends_with,
    "between": filter_between,
    "in": filter_in,
    "not_in": filter_not_in,
    "is_null": filter_is_null,
    "is_not_null": filter_is_not_null,
}

# Import-time coverage check: ensures every CanonicalOperator has a handler
# and no unknown operators are registered. Raises ImportError on violation.
from finflow_agent.contract_registry import check_operator_handler_coverage
check_operator_handler_coverage(FILTER_HANDLERS)
