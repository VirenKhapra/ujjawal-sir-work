"""Deterministic normalization for raw LLM PlanIntent payloads.

Normalization repairs syntax and known representational aliases. It does not
infer business meaning, invent thresholds, guess columns, inspect dataframes,
or call the LLM.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


PLAN_SCHEMA_VERSION = "1.0"
NORMALIZER_VERSION = "1.0"
COMPILER_VERSION = "1.0"


@dataclass(frozen=True)
class NormalizationEvent:
    path: str
    original_value: Any
    normalized_value: Any
    rule: str


@dataclass(frozen=True)
class NormalizationResult:
    payload: dict[str, Any]
    events: list[NormalizationEvent]


FILTER_OPERATOR_ALIASES: dict[str, str] = {
    "equals": "eq",
    "equal": "eq",
    "equal_to": "eq",
    "is_equal": "eq",
    "==": "eq",
    "not_equals": "neq",
    "not_equal": "neq",
    "not_equal_to": "neq",
    "!=": "neq",
    "greater_than": "gt",
    "more_than": "gt",
    ">": "gt",
    "greater_than_or_equal": "gte",
    "greater_than_or_equal_to": "gte",
    ">=": "gte",
    "less_than": "lt",
    "below": "lt",
    "<": "lt",
    "less_than_or_equal": "lte",
    "less_than_or_equal_to": "lte",
    "<=": "lte",
    "contains_any": "in",
    "not_in_list": "not_in",
}

LOGIC_ALIASES: dict[str, str] = {
    "and": "and",
    "or": "or",
}

CLEANING_OPERATION_ALIASES: dict[str, str] = {
    "remove_duplicates": "drop_duplicates",
    "deduplicate": "drop_duplicates",
    "trim_spaces": "trim_whitespace",
    "strip_whitespace": "trim_whitespace",
    "drop_blank_rows": "remove_empty_rows",
    "drop_blank_columns": "remove_empty_columns",
}

CALCULATION_OPERATION_ALIASES: dict[str, str] = {
    "abs": "absolute_value",
    "absolute": "absolute_value",
}

OUTPUT_FORMAT_ALIASES: dict[str, str] = {
    "xlsx": "xlsx",
    "csv": "csv",
    "json": "json",
    "txt": "txt",
}

CHART_TYPE_ALIASES: dict[str, str] = {
    "bar": "bar",
    "line": "line",
    "pie": "pie",
    "scatter": "scatter",
    "area": "area",
    "stacked_bar": "stacked_bar",
}


def normalize_plan_intent_payload(payload: dict[str, Any]) -> NormalizationResult:
    normalized = deepcopy(payload)
    events: list[NormalizationEvent] = []

    if not isinstance(normalized, dict):
        return NormalizationResult(payload=normalized, events=events)

    _normalize_root(normalized, events)
    _normalize_cleaning_plan(normalized.get("cleaning_plan"), events)
    _normalize_filter_plan(normalized.get("filter_plan"), events)
    _normalize_calculation_plan(normalized.get("calculation_plan"), events)
    _normalize_visualization_plan(normalized.get("visualization_plan"), events)

    return NormalizationResult(payload=normalized, events=events)


def _record(
    events: list[NormalizationEvent],
    path: str,
    original_value: Any,
    normalized_value: Any,
    rule: str,
) -> None:
    if original_value != normalized_value:
        events.append(
            NormalizationEvent(
                path=path,
                original_value=original_value,
                normalized_value=normalized_value,
                rule=rule,
            )
        )


def _normalize_root(payload: dict[str, Any], events: list[NormalizationEvent]) -> None:
    if isinstance(payload.get("output_format"), str):
        original = payload["output_format"]
        canonical = OUTPUT_FORMAT_ALIASES.get(original.strip().lower(), original)
        payload["output_format"] = canonical
        _record(events, "output_format", original, canonical, "output_format_case")


def _normalize_cleaning_plan(plan: Any, events: list[NormalizationEvent]) -> None:
    if not isinstance(plan, dict):
        return
    operations = plan.get("operations")
    if not isinstance(operations, list):
        return

    for index, operation in enumerate(operations):
        if not isinstance(operation, dict):
            continue
        path = f"cleaning_plan.operations.{index}"
        _normalize_operation_type(
            operation, f"{path}.type", CLEANING_OPERATION_ALIASES, events
        )

        if operation.get("type") == "drop_duplicates" and operation.get("subset") == "__all_columns__":
            original = operation["subset"]
            operation["subset"] = None
            _record(
                events,
                f"{path}.subset",
                original,
                None,
                "drop_duplicates_all_columns_sentinel",
            )

        _coerce_string_to_list(operation, "columns", f"{path}.columns", events)


def _normalize_filter_plan(plan: Any, events: list[NormalizationEvent]) -> None:
    if not isinstance(plan, dict):
        return

    if isinstance(plan.get("logic"), str):
        original = plan["logic"]
        canonical = LOGIC_ALIASES.get(original.strip().lower(), original)
        plan["logic"] = canonical
        _record(events, "filter_plan.logic", original, canonical, "filter_logic_case")

    _coerce_string_to_list(plan, "select_columns", "filter_plan.select_columns", events)

    conditions = plan.get("conditions")
    if not isinstance(conditions, list):
        return

    for index, condition in enumerate(conditions):
        if not isinstance(condition, dict):
            continue
        path = f"filter_plan.conditions.{index}"
        operator = condition.get("operator")
        if isinstance(operator, str):
            canonical = FILTER_OPERATOR_ALIASES.get(operator.strip().lower(), operator)
            condition["operator"] = canonical
            _record(
                events,
                f"{path}.operator",
                operator,
                canonical,
                "filter_operator_alias",
            )
        if condition.get("operator") in {"in", "not_in"}:
            _coerce_string_to_list(condition, "value", f"{path}.value", events)


def _normalize_calculation_plan(plan: Any, events: list[NormalizationEvent]) -> None:
    if not isinstance(plan, dict):
        return
    operations = plan.get("operations")
    if not isinstance(operations, list):
        return

    for index, operation in enumerate(operations):
        if not isinstance(operation, dict):
            continue
        path = f"calculation_plan.operations.{index}"
        _normalize_operation_type(
            operation, f"{path}.type", CALCULATION_OPERATION_ALIASES, events
        )
        _coerce_string_to_list(operation, "group_by", f"{path}.group_by", events)
        _coerce_string_to_list(
            operation, "partition_by", f"{path}.partition_by", events
        )
        if operation.get("type") == "absolute_value" and "column" not in operation:
            columns = operation.get("columns")
            if columns == "__all_numeric_columns__":
                operation["column"] = "__all_numeric_columns__"
                operation.pop("columns", None)
                _record(
                    events,
                    f"{path}.columns",
                    columns,
                    {"column": "__all_numeric_columns__"},
                    "absolute_value_numeric_columns_sentinel",
                )
            elif isinstance(columns, list) and len(columns) == 1:
                operation["column"] = columns[0]
                operation.pop("columns", None)
                _record(
                    events,
                    f"{path}.columns",
                    columns,
                    {"column": columns[0]},
                    "absolute_value_single_column_list",
                )


def _normalize_visualization_plan(plan: Any, events: list[NormalizationEvent]) -> None:
    if not isinstance(plan, dict):
        return
    charts = plan.get("charts")
    if not isinstance(charts, list):
        return

    for index, chart in enumerate(charts):
        if not isinstance(chart, dict):
            continue
        chart_type = chart.get("type")
        if isinstance(chart_type, str):
            canonical = CHART_TYPE_ALIASES.get(chart_type.strip().lower(), chart_type)
            chart["type"] = canonical
            _record(
                events,
                f"visualization_plan.charts.{index}.type",
                chart_type,
                canonical,
                "chart_type_case",
            )


def _normalize_operation_type(
    operation: dict[str, Any],
    path: str,
    aliases: dict[str, str],
    events: list[NormalizationEvent],
) -> None:
    op_type = operation.get("type")
    if not isinstance(op_type, str):
        return
    canonical = aliases.get(op_type.strip().lower(), op_type)
    operation["type"] = canonical
    _record(events, path, op_type, canonical, "operation_type_alias")


def _coerce_string_to_list(
    container: dict[str, Any],
    key: str,
    path: str,
    events: list[NormalizationEvent],
) -> None:
    value = container.get(key)
    if isinstance(value, str) and not value.startswith("__all_"):
        normalized = [value]
        container[key] = normalized
        _record(events, path, value, normalized, "known_list_field_scalar")
