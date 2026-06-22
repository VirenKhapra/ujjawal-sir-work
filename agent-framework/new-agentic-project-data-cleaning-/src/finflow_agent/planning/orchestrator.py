from __future__ import annotations

from finflow_agent.llm_telemetry import get_runtime_context, log_runtime_event


class Orchestrator:
    """Deprecated prompt-planning entrypoint.

    The canonical-intent pipeline no longer routes through prompt-driven
    planning. This stub remains only so old imports fail with a clear error
    instead of silently invoking a legacy fallback.
    """

    def build_plan(self, *args, **kwargs):
        runtime = get_runtime_context()
        log_runtime_event(
            "legacy_planner_entered",
            service="agent-service",
            trigger=str(runtime.get("trigger", "worker")),
            instruction_present=bool(runtime.get("instruction_present")),
            canonical_intent_present=bool(runtime.get("canonical_intent_present")),
            legacy_schema_state_present=bool(runtime.get("legacy_schema_state_present")),
            planner="planning.Orchestrator.build_plan",
        )
        if runtime.get("canonical_intent_present"):
            log_runtime_event(
                "architecture_violation_canonical_job_entered_legacy_planner",
                service="agent-service",
                trigger=str(runtime.get("trigger", "worker")),
                instruction_present=bool(runtime.get("instruction_present")),
                canonical_intent_present=True,
                legacy_schema_state_present=bool(runtime.get("legacy_schema_state_present")),
                planner="planning.Orchestrator.build_plan",
            )
        raise RuntimeError(
            "canonical_intent_required: use canonical intents and typed plans."
        )


__all__ = ["Orchestrator"]
