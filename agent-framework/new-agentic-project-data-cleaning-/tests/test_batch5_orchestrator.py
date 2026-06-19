import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from finflow_agent.orchestrator import Orchestrator
from finflow_agent.state import ExecutionPlan, PlanStep
from finflow_agent.registry import registry

# Register dummy agents for testing if not already present
import finflow_agent.agents.ingestion_agent
import finflow_agent.agents.cleaning_agent
import finflow_agent.agents.reporting_agent
import json

@patch("finflow_agent.orchestrator.call_groq_json")
def test_orchestrator_successful_plan(mock_call_groq):
    orchestrator = Orchestrator()
    
    mock_call_groq.return_value = {
        "is_quarantined": False,
        "needs_cleaning": True,
        "cleaning_plan": {"operations": []},
        "needs_filtering": False,
        "needs_calculation": False,
        "needs_visualization": False,
        "output_format": "xlsx",
        "cleaning_plan": {
            "operations": [{"type": "normalize_column_names", "style": "snake_case"}]
        }
    }
    
    res = orchestrator.build_plan("clean and report CSV", "test.csv", "test.csv", "xlsx")
    assert isinstance(res, ExecutionPlan)
    assert len(res.steps) == 3
    assert res.steps[0].agent == "ingestion_agent"
    assert res.steps[1].agent == "cleaning_agent"
    assert res.steps[2].agent == "reporting_agent"
    assert res.steps[2].params["plan"]["output_format"] == "xlsx"

@patch("finflow_agent.orchestrator.call_groq_json")
def test_orchestrator_quarantined_request(mock_call_groq):
    orchestrator = Orchestrator()
    
    mock_call_groq.return_value = {
        "is_quarantined": True,
        "quarantine_reason": "Instruction refers to stock trading, which is an unsupported capability."
    }
    
    res = orchestrator.build_plan("analyze options trading", "test.csv", "test.csv", "xlsx")
    assert isinstance(res, dict)
    assert res["status"] == "quarantined"
    assert "stock trading" in res["reason"]

@patch("finflow_agent.orchestrator.call_groq_json")
def test_orchestrator_quarantines_initial_llm_error_without_blind_retry(mock_call_groq):
    orchestrator = Orchestrator()
    
    mock_call_groq.side_effect = ValueError("Invalid JSON returned from LLM")
    
    res = orchestrator.build_plan("run pipeline", "test.csv", "test.csv", "xlsx")
    assert isinstance(res, dict)
    assert res["status"] == "quarantined"
    assert "Initial planning call failed" in res["reason"]
    assert mock_call_groq.call_count == 1

@patch("finflow_agent.orchestrator.call_groq_json")
def test_orchestrator_rejects_legacy_steps_response(mock_call_groq):
    orchestrator = Orchestrator()
    
    # Return a legacy response containing steps
    mock_call_groq.return_value = {
        "is_quarantined": False,
        "steps": [
            {
                "step_id": "step_1",
                "agent": "ingestion_agent",
                "params": {},
                "depends_on": []
            }
        ]
    }
    
    res = orchestrator.build_plan("run legacy pipeline", "test.csv", "test.csv", "xlsx")
    assert isinstance(res, dict)
    assert res["status"] == "quarantined"
    assert "Only PlanIntent is allowed" in res["reason"]
    assert mock_call_groq.call_count == 1


def test_orchestrator_initializes_groq_client_without_proxy_kwargs(monkeypatch):
    import finflow_agent.llm as llm_module

    captured_kwargs = {}
    captured_completion_kwargs = {}
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")

    class FakeGroq:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)
            assert "proxies" not in kwargs
            assert kwargs["http_client"].trust_env is False
            assert kwargs["api_key"] == "test-groq-key"
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create_completion)
            )

        def _create_completion(self, **kwargs):
            captured_completion_kwargs.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(
                                {
                                    "is_quarantined": False,
                                    "needs_cleaning": False,
                                    "needs_filtering": False,
                                    "needs_calculation": False,
                                    "needs_visualization": False,
                                    "output_format": "xlsx",
                                }
                            )
                        )
                    )
                ]
            )

    monkeypatch.setattr(llm_module, "Groq", FakeGroq)

    result = Orchestrator().build_plan("make a report", "test.csv", "test.csv", "xlsx")

    assert isinstance(result, ExecutionPlan)
    assert captured_kwargs["api_key"] == "test-groq-key"
    assert "http_client" in captured_kwargs
    assert captured_completion_kwargs.get("model") == "llama-3.3-70b-versatile"
    assert len(result.steps) == 2
