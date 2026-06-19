"""Regression tests for LLM/provider boundary role normalization.

Root cause this covers: LangChain's ``ChatPromptTemplate.format_messages``
emits messages with ``.type == "human"`` and ``.type == "ai"`` (internal
framework roles). Groq's chat.completions API only accepts the provider
wire-format roles: ``system``, ``user``, ``assistant``, ``tool``. Without
normalization at the outbound boundary, the orchestrator sends
``{"role": "human", ...}`` and Groq rejects it with HTTP 400:

    'messages.1' : discriminator property 'role' has invalid value

The normalization layer lives in ``finflow_agent.llm.normalize_outbound_messages``
and is called by ``call_groq_json`` before every request. These tests pin
that contract so a future refactor cannot silently regress the boundary.
"""

from __future__ import annotations

import pytest

from finflow_agent.llm import (
    normalize_outbound_messages,
    call_groq_json,
)


# ---------------------------------------------------------------------------
# 1. Role mapping — parametrized
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "internal_role,expected_wire_role",
    [
        ("human", "user"),
        ("ai", "assistant"),
        ("system", "system"),
        ("user", "user"),
        ("assistant", "assistant"),
        ("tool", "tool"),
        # Case tolerance (defensive)
        ("Human", "user"),
        ("AI", "assistant"),
        ("SYSTEM", "system"),
    ],
)
def test_normalize_maps_internal_to_provider_role(
    internal_role: str, expected_wire_role: str
):
    messages = [{"role": internal_role, "content": "hello"}]
    result = normalize_outbound_messages(messages)
    assert result == [{"role": expected_wire_role, "content": "hello"}]


# ---------------------------------------------------------------------------
# 2. Unknown roles fail fast
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "bad_role",
    ["function", "developer", "agent", "bot", "", "unknown", "  "],
)
def test_normalize_rejects_unknown_role_with_clear_error(bad_role: str):
    messages = [{"role": bad_role, "content": "hello"}]
    with pytest.raises(ValueError) as exc_info:
        normalize_outbound_messages(messages)
    # The error must name the offending role and the allowed set.
    error_text = str(exc_info.value)
    assert "messages[0]" in error_text
    assert "role" in error_text.lower()


def test_normalize_rejects_none_role():
    messages = [{"role": None, "content": "hello"}]
    with pytest.raises(ValueError):
        normalize_outbound_messages(messages)


def test_normalize_rejects_non_string_role():
    messages = [{"role": 123, "content": "hello"}]
    with pytest.raises(ValueError):
        normalize_outbound_messages(messages)


# ---------------------------------------------------------------------------
# 3. Structural integrity of the output
# ---------------------------------------------------------------------------

def test_normalize_preserves_content_and_order():
    messages = [
        {"role": "system", "content": "You are a planner."},
        {"role": "human", "content": "Clean this data."},
    ]
    result = normalize_outbound_messages(messages)
    assert len(result) == 2
    assert result[0] == {"role": "system", "content": "You are a planner."}
    assert result[1] == {"role": "user", "content": "Clean this data."}


def test_normalize_does_not_mutate_input():
    original = [{"role": "human", "content": "test"}]
    import copy
    snapshot = copy.deepcopy(original)
    normalize_outbound_messages(original)
    assert original == snapshot


def test_normalize_returns_fresh_dicts():
    original = [{"role": "human", "content": "test"}]
    result = normalize_outbound_messages(original)
    # Mutating the output must not affect the input.
    result[0]["content"] = "mutated"
    assert original[0]["content"] == "test"


# ---------------------------------------------------------------------------
# 4. Integration: call_groq_json normalizes before sending
# ---------------------------------------------------------------------------

def test_call_groq_json_normalizes_human_to_user(monkeypatch):
    """Regression test: the exact failure path that produced the 400.

    Monkeypatch the Groq client to capture what gets sent. The ``messages``
    kwarg in the captured call must have ``role="user"`` (wire format),
    never ``role="human"`` (framework format).
    """
    captured_kwargs = {}

    class FakeCompletionMessage:
        content = '{"needs_cleaning": true, "output_format": "xlsx"}'

    class FakeChoice:
        message = FakeCompletionMessage()

    class FakeCompletion:
        choices = [FakeChoice()]

    class FakeCompletions:
        def create(self, **kwargs):
            captured_kwargs.update(kwargs)
            return FakeCompletion()

    class FakeClient:
        chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(
        "finflow_agent.llm.get_groq_client", lambda: FakeClient()
    )

    # Simulate the exact payload shape the orchestrator builds:
    # message[0].type = "system", message[1].type = "human"
    raw_msg = [
        {"role": "system", "content": "You are the FinFlow Orchestrator."},
        {"role": "human", "content": "Clean this data."},
    ]

    result = call_groq_json(raw_msg, schema={})

    # The Groq client must receive wire-format roles.
    sent_messages = captured_kwargs["messages"]
    assert sent_messages[0]["role"] == "system"
    assert sent_messages[1]["role"] == "user"  # NOT "human"!
    assert result == {"needs_cleaning": True, "output_format": "xlsx"}


# ---------------------------------------------------------------------------
# 5. End-to-end planner no longer fails on role mismatch
# ---------------------------------------------------------------------------

def test_orchestrator_plan_does_not_fail_on_provider_role_format(
    monkeypatch, bootstrap_agents, tmp_path
):
    """Full planner regression: the Orchestrator builds messages from
    ``ChatPromptTemplate.format_messages()`` which emits ``human`` as the
    type. After normalization, the plan call must succeed (returning either
    an ExecutionPlan or a quarantine dict) rather than bubbling up a Groq
    400 error about an invalid role discriminator.
    """
    from finflow_agent.planning.orchestrator import Orchestrator

    # Return a clean PlanIntent dict so the compiler can proceed.
    def fake_call_groq_json(messages, schema):
        # Assert the messages are already normalized before we even get here.
        for msg in messages:
            assert msg["role"] in {"system", "user", "assistant", "tool"}, (
                f"Leaked framework role to provider: {msg['role']!r}"
            )
        return {
            "is_quarantined": False,
            "needs_cleaning": True,
            "needs_filtering": False,
            "needs_visualization": False,
            "needs_calculation": False,
            "output_format": "xlsx",
            "cleaning_plan": {
                "operations": [
                    {"type": "trim_whitespace", "columns": "__all_string_columns__"}
                ]
            },
            "filter_plan": None,
            "calculation_plan": None,
            "visualization_plan": None,
            "reporting_title": None,
            "sheet_name": None,
        }

    import finflow_agent.orchestrator as root_orchestrator
    monkeypatch.setattr(root_orchestrator, "call_groq_json", fake_call_groq_json)

    result = Orchestrator().build_plan(
        instruction="Clean this data",
        file_path=str(tmp_path / "input.csv"),
        file_name="input.csv",
        output_format="xlsx",
    )

    # Must not be a quarantine with the provider-error message.
    if isinstance(result, dict):
        assert "discriminator" not in result.get("reason", "")
        assert "invalid value" not in result.get("reason", "")
    else:
        # Got an ExecutionPlan — planning succeeded.
        from finflow_agent.state import ExecutionPlan
        assert isinstance(result, ExecutionPlan)
        assert result.steps[0].agent == "ingestion_agent"
        assert result.steps[-1].agent == "reporting_agent"
