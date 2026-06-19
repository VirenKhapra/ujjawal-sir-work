from copy import deepcopy


def test_normalizer_repairs_filter_operators_and_logic():
    from finflow_agent.planning.normalizer import normalize_plan_intent_payload

    payload = {
        "filter_plan": {
            "logic": "AND",
            "conditions": [
                {"column": "Age", "operator": "greater_than", "value": 45},
                {"column": "Status", "operator": "not_equals", "value": "Closed"},
            ],
        }
    }

    result = normalize_plan_intent_payload(payload)

    assert result.payload["filter_plan"]["logic"] == "and"
    assert result.payload["filter_plan"]["conditions"][0]["operator"] == "gt"
    assert result.payload["filter_plan"]["conditions"][1]["operator"] == "neq"
    assert {event.path for event in result.events} == {
        "filter_plan.logic",
        "filter_plan.conditions.0.operator",
        "filter_plan.conditions.1.operator",
    }


def test_normalizer_repairs_all_required_filter_operator_aliases():
    from finflow_agent.planning.normalizer import normalize_plan_intent_payload

    aliases = {
        "equals": "eq",
        "not_equals": "neq",
        "greater_than": "gt",
        "greater_than_or_equal": "gte",
        "less_than": "lt",
        "less_than_or_equal": "lte",
    }

    for alias, canonical in aliases.items():
        result = normalize_plan_intent_payload(
            {
                "filter_plan": {
                    "conditions": [
                        {"column": "Age", "operator": alias, "value": 45}
                    ]
                }
            }
        )
        assert result.payload["filter_plan"]["conditions"][0]["operator"] == canonical


def test_normalizer_logic_casing():
    from finflow_agent.planning.normalizer import normalize_plan_intent_payload

    for value, expected in {"AND": "and", "And": "and", "OR": "or", "Or": "or"}.items():
        result = normalize_plan_intent_payload({"filter_plan": {"logic": value}})
        assert result.payload["filter_plan"]["logic"] == expected


def test_normalizer_coerces_only_known_list_fields():
    from finflow_agent.planning.normalizer import normalize_plan_intent_payload

    result = normalize_plan_intent_payload(
        {
            "filter_plan": {
                "select_columns": "Age",
                "conditions": [{"column": "Status", "operator": "in", "value": "Paid"}],
            },
            "reporting_title": "Age",
        }
    )

    assert result.payload["filter_plan"]["select_columns"] == ["Age"]
    assert result.payload["filter_plan"]["conditions"][0]["value"] == ["Paid"]
    assert result.payload["reporting_title"] == "Age"


def test_normalizer_repairs_duplicate_and_numeric_sentinels():
    from finflow_agent.planning.normalizer import normalize_plan_intent_payload

    result = normalize_plan_intent_payload(
        {
            "cleaning_plan": {
                "operations": [
                    {"type": "deduplicate", "subset": "__all_columns__"}
                ]
            },
            "calculation_plan": {
                "operations": [
                    {"type": "absolute_value", "columns": "__all_numeric_columns__"}
                ]
            },
        }
    )

    assert result.payload["cleaning_plan"]["operations"][0]["type"] == "drop_duplicates"
    assert result.payload["cleaning_plan"]["operations"][0]["subset"] is None
    assert result.payload["calculation_plan"]["operations"][0] == {
        "type": "absolute_value",
        "column": "__all_numeric_columns__",
    }


def test_normalizer_records_events_and_does_not_mutate_input():
    from finflow_agent.planning.normalizer import normalize_plan_intent_payload

    payload = {"output_format": "XLSX"}
    original = deepcopy(payload)

    result = normalize_plan_intent_payload(payload)

    assert payload == original
    assert result.payload["output_format"] == "xlsx"
    assert result.events[0].path == "output_format"
    assert result.events[0].original_value == "XLSX"
    assert result.events[0].normalized_value == "xlsx"
    assert result.events[0].rule == "output_format_case"


def test_normalizer_does_not_map_unknown_operations_or_invent_meaning():
    from finflow_agent.planning.normalizer import normalize_plan_intent_payload

    result = normalize_plan_intent_payload(
        {
            "calculation_plan": {
                "operations": [{"type": "high_value", "column": "Amount"}]
            },
            "filter_plan": {
                "conditions": [
                    {"column": "Amount", "operator": "contains", "value": "high value"}
                ]
            },
        }
    )

    assert result.payload["calculation_plan"]["operations"][0]["type"] == "high_value"
    assert result.payload["filter_plan"]["conditions"][0]["value"] == "high value"
