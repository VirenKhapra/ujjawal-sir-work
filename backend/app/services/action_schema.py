from __future__ import annotations

from typing import Any

from app.services.canonical_intent import canonical_intent_to_legacy_action_schema


def action_schema_to_constraints(action_schema: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(action_schema, dict):
        return []

    if any(isinstance(action, dict) and "kind" in action for action in action_schema.get("actions", [])):
        action_schema = canonical_intent_to_legacy_action_schema(action_schema)

    constraints: list[dict[str, Any]] = []
    for action in action_schema.get("actions", []):
        if not isinstance(action, dict):
            continue

        action_type = str(action.get("action", "")).strip()
        if action_type == "drop_columns":
            for role in action.get("roles", []):
                role_name = str(role or "").strip()
                if not role_name:
                    continue
                constraints.append(
                    {
                        "column": role_name,
                        "rule": "drop_column",
                        "severity": "warning",
                        "reason": f"Instruction removes the {role_name} field.",
                        "source": "action_schema",
                    }
                )
            continue

        if action_type in {"keep_rows_where", "drop_rows_where"}:
            rule = "allowed_values" if action_type == "keep_rows_where" else "not_allowed_values"
            for leaf in _iter_leaf_conditions(action.get("condition_tree")):
                role_name = str(leaf.get("role", "")).strip()
                op = str(leaf.get("op", "")).strip().lower()
                value = str(leaf.get("value", "")).strip()
                if not role_name or not value:
                    continue
                if op == "contains":
                    constraints.append(
                        {
                            "column": role_name,
                            "rule": "contains" if action_type == "keep_rows_where" else "forbidden_substring",
                            "severity": "warning",
                            "reason": f"Instruction filters on {role_name}.",
                            "value": value,
                            "source": "action_schema",
                        }
                    )
                elif op in {"eq", "equals"}:
                    constraints.append(
                        {
                            "column": role_name,
                            "rule": rule,
                            "severity": "warning",
                            "reason": f"Instruction filters on {role_name}.",
                            "allowed_values": [value] if rule == "allowed_values" else [],
                            "not_allowed_values": [value] if rule == "not_allowed_values" else [],
                            "source": "action_schema",
                        }
                    )
                elif op in {"in", "one_of"}:
                    values = _stringify_values(leaf.get("value"))
                    if not values:
                        continue
                    constraints.append(
                        {
                            "column": role_name,
                            "rule": "allowed_values" if action_type == "keep_rows_where" else "not_allowed_values",
                            "severity": "warning",
                            "reason": f"Instruction filters on {role_name}.",
                            "allowed_values": values if action_type == "keep_rows_where" else [],
                            "not_allowed_values": values if action_type == "drop_rows_where" else [],
                            "source": "action_schema",
                        }
                    )
    return constraints


def _iter_leaf_conditions(tree: Any) -> list[dict[str, Any]]:
    if not isinstance(tree, dict):
        return []
    conditions = tree.get("conditions", [])
    leaves: list[dict[str, Any]] = []
    for item in conditions if isinstance(conditions, list) else []:
        if isinstance(item, dict) and "conditions" in item:
            leaves.extend(_iter_leaf_conditions(item))
        elif isinstance(item, dict):
            leaves.append(item)
    return leaves


def _stringify_values(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []
