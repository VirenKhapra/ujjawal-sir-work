"""Schema-aware repair helpers for PlanIntent validation failures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from finflow_agent.planning.normalizer import (
    CALCULATION_OPERATION_ALIASES,
    CLEANING_OPERATION_ALIASES,
    FILTER_OPERATOR_ALIASES,
)


SUPPORTED_FILTER_OPERATORS = {
    "eq",
    "neq",
    "gt",
    "gte",
    "lt",
    "lte",
    "contains",
    "not_contains",
    "starts_with",
    "ends_with",
    "between",
    "in",
    "not_in",
    "is_null",
    "is_not_null",
}

SUPPORTED_FILTER_LOGIC = {"and", "or"}

SUPPORTED_CALCULATION_TYPES = {
    "sum",
    "mean",
    "median",
    "min",
    "max",
    "count",
    "count_distinct",
    "variance",
    "standard_deviation",
    "group_sum",
    "group_mean",
    "group_count",
    "running_total",
    "percentage_change",
    "difference",
    "ratio",
    "absolute_value",
}

SUPPORTED_CLEANING_TYPES = {
    "trim_whitespace",
    "normalize_column_names",
    "drop_duplicates",
    "fill_nulls",
    "drop_nulls",
    "normalize_date",
    "normalize_currency",
    "normalize_number",
    "normalize_text_case",
    "replace_values",
    "strip_currency_symbols",
    "remove_commas_from_numbers",
    "coerce_column_type",
    "remove_empty_rows",
    "remove_empty_columns",
    "rename_columns",
    "reorder_columns",
}


@dataclass(frozen=True)
class PlanValidationIssue:
    path: str
    message: str
    received_value: Any
    allowed_values: list[Any] | None
    repairable: bool
    category: str


def classify_plan_validation_error(error: ValidationError) -> list[PlanValidationIssue]:
    return [_classify_error(item) for item in error.errors()]


def issues_are_repairable(issues: list[PlanValidationIssue]) -> bool:
    return bool(issues) and all(issue.repairable for issue in issues)


def build_repair_messages(
    *,
    original_instruction: str,
    invalid_payload: dict[str, Any],
    issues: list[PlanValidationIssue],
) -> list[dict[str, Any]]:
    system = (
        "Repair only the invalid fields in the PlanIntent JSON. Preserve fields "
        "that already satisfy the schema. Do not invent thresholds, columns, "
        "operations, agents, or business meaning. Do not return PlanStep or "
        "ExecutionPlan. Return one JSON object matching PlanIntent."
    )
    user = {
        "original_instruction": original_instruction,
        "invalid_payload": invalid_payload,
        "validation_issues": [
            {
                "path": issue.path,
                "message": issue.message,
                "received_value": issue.received_value,
                "allowed_values": issue.allowed_values,
                "repairable": issue.repairable,
                "category": issue.category,
            }
            for issue in issues
        ],
        "allowed_schema": {
            "filter_operators": sorted(SUPPORTED_FILTER_OPERATORS),
            "filter_logic": sorted(SUPPORTED_FILTER_LOGIC),
            "cleaning_operations": sorted(SUPPORTED_CLEANING_TYPES),
            "calculation_operations": sorted(SUPPORTED_CALCULATION_TYPES),
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _classify_error(item: dict[str, Any]) -> PlanValidationIssue:
    path = _format_path(item.get("loc", ()))
    message = str(item.get("msg", "Validation error"))
    received = item.get("input")
    error_type = str(item.get("type", ""))

    if _is_aliasable_literal(path, received):
        return PlanValidationIssue(
            path=path,
            message=message,
            received_value=received,
            allowed_values=_allowed_values_for_path(path),
            repairable=True,
            category="known_alias_or_enum",
        )

    if error_type in {"list_type"} and _is_known_list_path(path):
        return PlanValidationIssue(
            path=path,
            message=message,
            received_value=received,
            allowed_values=None,
            repairable=True,
            category="scalar_for_known_list",
        )

    if error_type in {"missing"} and _is_safely_defaultable_missing_path(path):
        return PlanValidationIssue(
            path=path,
            message=message,
            received_value=received,
            allowed_values=None,
            repairable=True,
            category="safe_missing_default",
        )

    return PlanValidationIssue(
        path=path,
        message=message,
        received_value=received,
        allowed_values=_allowed_values_for_path(path),
        repairable=False,
        category="requires_business_meaning_or_unsupported_capability",
    )


def _format_path(loc: Any) -> str:
    if isinstance(loc, (list, tuple)):
        return ".".join(str(part) for part in loc)
    return str(loc)


def _is_aliasable_literal(path: str, value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower()
    if path.endswith(".operator"):
        return normalized in FILTER_OPERATOR_ALIASES or normalized in SUPPORTED_FILTER_OPERATORS
    if path.endswith(".logic"):
        return normalized in SUPPORTED_FILTER_LOGIC
    if path.endswith(".type") and "cleaning_plan.operations" in path:
        return normalized in CLEANING_OPERATION_ALIASES or normalized in SUPPORTED_CLEANING_TYPES
    if path.endswith(".type") and "calculation_plan.operations" in path:
        return normalized in CALCULATION_OPERATION_ALIASES or normalized in SUPPORTED_CALCULATION_TYPES
    return False


def _is_known_list_path(path: str) -> bool:
    if path in {
        "filter_plan.conditions",
        "cleaning_plan.operations",
        "calculation_plan.operations",
        "visualization_plan.charts",
    }:
        return True
    return path.endswith(
        (
            ".columns",
            ".group_by",
            ".partition_by",
            ".select_columns",
            ".value",
        )
    )


def _is_safely_defaultable_missing_path(path: str) -> bool:
    return path.endswith(".keep") or path.endswith(".style")


def _allowed_values_for_path(path: str) -> list[Any] | None:
    if path.endswith(".operator"):
        return sorted(SUPPORTED_FILTER_OPERATORS)
    if path.endswith(".logic"):
        return sorted(SUPPORTED_FILTER_LOGIC)
    if path.endswith(".type") and "cleaning_plan.operations" in path:
        return sorted(SUPPORTED_CLEANING_TYPES)
    if path.endswith(".type") and "calculation_plan.operations" in path:
        return sorted(SUPPORTED_CALCULATION_TYPES)
    return None
