"""Bridge module: connects the backend to the new SemanticPipeline extractor.

Provides `run_new_semantic_pipeline_sync()` which:
1. Imports `SemanticExtractor` and supporting models from `finflow_agent`
2. Uses a Groq-backed SemanticResolver to run extraction
3. Performs lightweight column grounding via the new CandidateGenerator + ColumnGrounder
4. Converts the resulting SemanticIntentDraft to the legacy canonical intent dict

If the new pipeline is not importable, GROQ_API_KEY is not set, or any error
occurs, the function returns None so the backend falls through to the legacy path.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path setup: make finflow_agent importable
# ---------------------------------------------------------------------------

# In Docker: the agent-framework source is mounted at /app/finflow_agent_src
# Locally: resolve relative to this file's location in the repo
_DOCKER_SRC = "/app/finflow_agent_src"
_LOCAL_SRC = str(
    Path(__file__).resolve().parents[3]
    / "agent-framework"
    / "new-agentic-project-data-cleaning-"
    / "src"
)

# Try Docker path first, then local
for _src_path in (_DOCKER_SRC, _LOCAL_SRC):
    if Path(_src_path).is_dir() and _src_path not in sys.path:
        sys.path.insert(0, _src_path)
        break


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_new_semantic_pipeline_sync(
    instruction: str,
    source_columns: list[str],
    *,
    column_types: dict[str, str] | None = None,
    output_format: str = "xlsx",
    submission_id: str = "",
    trigger: str = "upload",
) -> dict[str, Any] | None:
    """Run the new semantic extraction + grounding pipeline synchronously.

    Returns a legacy canonical intent dict on success, or None to signal
    that the caller should fall back to the existing pipeline.
    """
    # Guard: GROQ_BRIDGE_API_KEY must be set (separate org to avoid competing
    # with agent-service's rate limit on the same GROQ_API_KEY)
    api_key = os.environ.get("GROQ_BRIDGE_API_KEY", "")
    if not api_key:
        # Fall back to GROQ_API_KEY only if bridge key isn't available
        api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return None

    if not instruction or not instruction.strip():
        return None

    if not source_columns:
        return None

    try:
        return _run_pipeline(
            instruction,
            source_columns,
            column_types,
            output_format,
            api_key,
            submission_id=submission_id,
            trigger=trigger,
        )
    except Exception:
        logger.exception("new_pipeline_bridge: unhandled error — falling back to legacy")
        return None


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------


def _run_pipeline(
    instruction: str,
    source_columns: list[str],
    column_types: dict[str, str] | None,
    output_format: str,
    api_key: str,
    *,
    submission_id: str,
    trigger: str,
) -> dict[str, Any] | None:
    """Core implementation — may raise on import/runtime failure."""
    # Lazy import to avoid crashing the backend if finflow_agent is unavailable
    try:
        from finflow_agent.grounding.semantic_extractor import (
            SchemaContext,
            SemanticExtractor,
        )
        from finflow_agent.grounding.llm_adapter import (
            DEFAULT_CONSTRAINTS,
            LLMCallSite,
            LLMConstraint,
            LLMProviderError,
            LLMResponse,
            SemanticResolver,
        )
        from finflow_agent.models.draft import (
            DraftAction,
            DropAction,
            FilterAction,
            ProjectAction,
            ReferenceKind,
            RenameAction,
            SemanticColumnReference,
            SemanticIntentDraft,
            SortAction,
        )
    except ImportError:
        logger.warning("new_pipeline_bridge: finflow_agent not importable — skipping")
        return None

    # Build resolver and extractor
    resolver = _GroqResolver(api_key, submission_id=submission_id, trigger=trigger)
    extractor = SemanticExtractor(resolver)

    schema_context = SchemaContext(
        column_names=source_columns,
        column_types=column_types or {},
    )

    # Run the async extraction synchronously
    draft = _run_async(extractor.extract(instruction, schema_context))
    if draft is None:
        return None

    # Lightweight grounding: resolve column references against source_columns
    _ground_draft_references(draft, source_columns)

    # Check if everything resolved
    has_unresolved = _has_unresolved_references(draft)

    # Convert draft to legacy canonical intent dict
    return _draft_to_legacy_dict(
        draft,
        instruction=instruction,
        source_columns=source_columns,
        output_format=output_format,
        has_unresolved=has_unresolved,
    )


# ---------------------------------------------------------------------------
# Groq Resolver (SemanticResolver protocol implementation)
# ---------------------------------------------------------------------------


class _GroqResolver:
    """Minimal Groq LLM adapter satisfying the SemanticResolver protocol."""

    def __init__(self, api_key: str, *, submission_id: str = "", trigger: str = "upload") -> None:
        self._api_key = api_key
        self._base_url = "https://api.groq.com/openai/v1/chat/completions"
        self._model = os.environ.get("GROQ_BRIDGE_MODEL", "llama-3.3-70b-versatile")
        self._submission_id = submission_id
        self._trigger = trigger

    async def call(
        self,
        messages: list[dict[str, str]],
        *,
        call_site: Any,
        constraint: Any,
        timeout: float = 30.0,
    ) -> Any:
        """Call Groq chat completions API using httpx."""
        import httpx
        from finflow_agent.grounding.llm_adapter import LLMProviderError, LLMResponse

        # --- Telemetry: log call start ---
        try:
            from app.services.llm_telemetry import log_llm_started, log_llm_completed, log_llm_failed, log_runtime_event
            log_runtime_event(
                "canonical_extractor_entered",
                service="backend",
                operation="new_pipeline_extraction",
                trigger=self._trigger,
                submission_id=self._submission_id,
                http_method="POST" if self._trigger == "upload" else "",
                instruction_present=True,
                canonical_intent_present=False,
                legacy_schema_state_present=False,
                messages=messages,
                model=self._model,
                api_key=self._api_key,
                api_key_source="GROQ_BRIDGE_API_KEY",
            )
            _telemetry_ctx = log_llm_started(
                service="backend",
                operation="new_pipeline_extraction",
                caller_file="new_pipeline_bridge.py",
                caller_function="_GroqResolver.call",
                model=self._model,
                api_key_source="GROQ_BRIDGE_API_KEY",
                api_key=self._api_key,
                attempt=1,
                trigger=str(call_site),
                messages=messages,
                submission_id=self._submission_id,
                http_method="POST" if self._trigger == "upload" else "",
            )
        except Exception:
            _telemetry_ctx = None
        # --- End telemetry start ---

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        max_tokens = 4096
        temperature = 0.0
        if hasattr(constraint, "max_tokens") and constraint.max_tokens:
            max_tokens = constraint.max_tokens
        if hasattr(constraint, "temperature"):
            temperature = constraint.temperature

        payload = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(self._base_url, json=payload, headers=headers)
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            # --- Telemetry: log failure ---
            if _telemetry_ctx:
                try:
                    log_llm_failed(
                        _telemetry_ctx,
                        status_code=0,
                        error_type="timeout",
                        error_message=str(exc),
                    )
                except Exception:
                    pass
            # --- End telemetry failure ---
            raise LLMProviderError(
                f"Groq API call failed: {exc}",
                error_type="timeout",
                call_site=str(call_site),
            ) from exc

        latency_ms = (time.perf_counter() - start) * 1000

        if resp.status_code != 200:
            error_type = "rate_limit" if resp.status_code == 429 else "server_error"
            # --- Telemetry: log failure ---
            if _telemetry_ctx:
                try:
                    log_llm_failed(
                        _telemetry_ctx,
                        status_code=resp.status_code,
                        error_type=error_type,
                        error_message=resp.text[:200],
                        headers=dict(resp.headers),
                    )
                except Exception:
                    pass
            # --- End telemetry failure ---
            raise LLMProviderError(
                f"Groq API returned {resp.status_code}: {resp.text[:200]}",
                error_type=error_type,
                call_site=str(call_site),
            )

        data = resp.json()
        content = ""
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            # --- Telemetry: log failure ---
            if _telemetry_ctx:
                try:
                    log_llm_failed(
                        _telemetry_ctx,
                        status_code=resp.status_code,
                        error_type="parse_error",
                        error_message="Groq API response has unexpected structure",
                    )
                except Exception:
                    pass
            # --- End telemetry failure ---
            raise LLMProviderError(
                "Groq API response has unexpected structure",
                error_type="server_error",
                call_site=str(call_site),
            )

        # --- Telemetry: log success ---
        if _telemetry_ctx:
            try:
                usage = data.get("usage", {})
                log_llm_completed(
                    _telemetry_ctx,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                    finish_reason=data.get("choices", [{}])[0].get("finish_reason", ""),
                )
            except Exception:
                pass
        # --- End telemetry success ---

        # Attempt to parse JSON from content
        parsed = None
        try:
            text = content.strip()
            # Strip code fences
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            parsed = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            pass

        return LLMResponse(
            content=content,
            parsed=parsed,
            call_site=call_site,
            latency_ms=latency_ms,
            retries_used=0,
        )


# ---------------------------------------------------------------------------
# Lightweight grounding (no DataFrameProfile required)
# ---------------------------------------------------------------------------


def _ground_draft_references(
    draft: Any,  # SemanticIntentDraft
    source_columns: list[str],
) -> None:
    """Resolve column references in the draft using simple name matching.

    This does direct name/fuzzy matching against source_columns without needing
    a full DataFrameProfile. Modifies the draft in place.
    """
    import difflib

    col_lower_map: dict[str, str] = {c.lower(): c for c in source_columns}
    # Normalized map: underscore/space/dash collapsed
    col_norm_map: dict[str, str] = {}
    for c in source_columns:
        norm = c.lower().replace("_", " ").replace("-", " ").strip()
        col_norm_map[norm] = c

    def _resolve_ref(ref: Any) -> None:
        """Try to resolve a SemanticColumnReference."""
        if ref.resolved_column is not None:
            return

        text = ref.reference_text.strip().lower()

        # Skip generic references — leave unresolved (the new pipeline handles them)
        from finflow_agent.models.draft import ReferenceKind
        if ref.reference_kind == ReferenceKind.GENERIC_REFERENCE:
            return

        # Exact case-insensitive match
        if text in col_lower_map:
            ref.resolved_column = col_lower_map[text]
            ref.confidence = 1.0
            return

        # Normalized match
        norm_text = text.replace("_", " ").replace("-", " ").strip()
        if norm_text in col_norm_map:
            ref.resolved_column = col_norm_map[norm_text]
            ref.confidence = 0.95
            return

        # Fuzzy match
        matches = difflib.get_close_matches(text, col_lower_map.keys(), n=1, cutoff=0.75)
        if matches:
            ref.resolved_column = col_lower_map[matches[0]]
            ref.confidence = 0.85
            return

    from finflow_agent.models.draft import (
        FilterAction, ProjectAction, DropAction, SortAction, RenameAction,
    )

    for action in draft.actions:
        if isinstance(action, FilterAction):
            for group in action.logical_groups:
                for pred in group.predicates:
                    _resolve_ref(pred.field_ref)
        elif isinstance(action, ProjectAction):
            for col_ref in action.columns:
                _resolve_ref(col_ref)
        elif isinstance(action, DropAction):
            for col_ref in action.columns:
                _resolve_ref(col_ref)
        elif isinstance(action, SortAction):
            for col_ref in action.keys:
                _resolve_ref(col_ref)
        elif isinstance(action, RenameAction):
            for col_ref, _ in action.mappings:
                _resolve_ref(col_ref)


def _has_unresolved_references(draft: Any) -> bool:
    """Check if the draft has any unresolved column references."""
    from finflow_agent.models.draft import (
        FilterAction, ProjectAction, DropAction, SortAction, RenameAction,
        ReferenceKind,
    )

    def _is_unresolved(ref: Any) -> bool:
        # Generic references that are unresolved count as unresolved
        return ref.resolved_column is None

    for action in draft.actions:
        if isinstance(action, FilterAction):
            for group in action.logical_groups:
                for pred in group.predicates:
                    if _is_unresolved(pred.field_ref):
                        return True
        elif isinstance(action, ProjectAction):
            for col_ref in action.columns:
                if _is_unresolved(col_ref):
                    return True
        elif isinstance(action, DropAction):
            for col_ref in action.columns:
                if _is_unresolved(col_ref):
                    return True
        elif isinstance(action, SortAction):
            for col_ref in action.keys:
                if _is_unresolved(col_ref):
                    return True
        elif isinstance(action, RenameAction):
            for col_ref, _ in action.mappings:
                if _is_unresolved(col_ref):
                    return True

    return False


# ---------------------------------------------------------------------------
# Conversion: SemanticIntentDraft → legacy canonical intent dict
# ---------------------------------------------------------------------------


def _draft_to_legacy_dict(
    draft: Any,
    *,
    instruction: str,
    source_columns: list[str],
    output_format: str,
    has_unresolved: bool,
) -> dict[str, Any] | None:
    """Convert a SemanticIntentDraft to the backend's legacy canonical intent dict."""
    from finflow_agent.models.draft import (
        FilterAction, ProjectAction, DropAction, SortAction, RenameAction,
    )

    actions: list[dict[str, Any]] = []

    for action in draft.actions:
        converted = _convert_action(action, source_columns)
        if converted is not None:
            actions.append(converted)

    if not actions:
        # No usable actions extracted
        return None

    if has_unresolved:
        resolution_status = "needs_clarification"
    else:
        resolution_status = "resolved"

    # Build the legacy dict structure matching backend CanonicalIntent
    import uuid
    from datetime import datetime, timezone

    canonical = {
        "schema_version": "2.0",
        "intent_id": str(uuid.uuid4()),
        "intent_revision": 1,
        "intent_hash": "",
        "parent_intent_id": None,
        "original_prompt": instruction,
        "normalized_prompt": instruction.strip().lower(),
        "resolution_status": resolution_status,
        "decision": _build_decision(actions),
        "evidence": [f"new_pipeline_extraction: {draft.schema_version}"],
        "alternatives_considered": [],
        "actions": actions,
        "output_format": output_format if output_format in ("xlsx", "csv", "json", "txt") else "xlsx",
        "assumptions": [],
        "repair_notes": [],
        "dataframe_profile": {"columns": source_columns},
        "capability_version": "backend.capability.1",
        "capability_snapshot": {},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "grounded_at": datetime.now(timezone.utc).isoformat(),
    }

    # Compute intent_hash
    canonical["intent_hash"] = _compute_hash(canonical)

    return canonical


def _convert_action(action: Any, source_columns: list[str]) -> dict[str, Any] | None:
    """Convert a single DraftAction to a legacy action dict."""
    from finflow_agent.models.draft import (
        FilterAction, ProjectAction, DropAction, SortAction, RenameAction,
    )

    if isinstance(action, ProjectAction):
        return _convert_project(action)
    elif isinstance(action, DropAction):
        return _convert_drop(action)
    elif isinstance(action, FilterAction):
        return _convert_filter(action)
    elif isinstance(action, SortAction):
        return _convert_sort(action)
    elif isinstance(action, RenameAction):
        return _convert_rename(action)
    return None


def _convert_project(action: Any) -> dict[str, Any]:
    """Convert ProjectAction → legacy project_columns dict."""
    fields = []
    for ref in action.columns:
        fields.append(_ref_to_legacy(ref))
    return {"kind": "project_columns", "requested_fields": fields}


def _convert_drop(action: Any) -> dict[str, Any]:
    """Convert DropAction → legacy drop_columns dict."""
    fields = []
    for ref in action.columns:
        fields.append(_ref_to_legacy(ref))
    return {"kind": "drop_columns", "requested_fields": fields}


def _convert_filter(action: Any) -> dict[str, Any]:
    """Convert FilterAction → legacy filter_rows dict."""
    conditions = []
    logic: str | None = None

    for group in action.logical_groups:
        group_logic = str(group.operator).strip().lower() or "and"
        for pred in group.predicates:
            field_ref = _ref_to_legacy(pred.field_ref)
            operator = _map_operator_to_legacy(pred.operator)
            conditions.append(
                {
                    "field": field_ref,
                    "operator": operator,
                    "value": pred.value,
                }
            )
            if logic is None:
                logic = group_logic

    return {
        "kind": "filter_rows",
        "mode": "keep",
        "conditions": conditions,
        "logic": logic or "and",
    }


def _convert_sort(action: Any) -> dict[str, Any]:
    """Convert SortAction → legacy sort_rows dict."""
    sort_keys = []
    for i, ref in enumerate(action.keys):
        direction = action.directions[i] if i < len(action.directions) else "asc"
        sort_keys.append({
            "column": _ref_to_legacy(ref),
            "direction": direction,
        })
    return {"kind": "sort_rows", "sort_keys": sort_keys}


def _convert_rename(action: Any) -> dict[str, Any]:
    """Convert RenameAction → legacy rename_columns dict."""
    mapping = []
    for ref, new_name in action.mappings:
        mapping.append({
            "source": _ref_to_legacy(ref),
            "target_name": new_name,
        })
    return {"kind": "rename_columns", "mapping": mapping}


def _ref_to_legacy(ref: Any) -> dict[str, Any]:
    """Convert a SemanticColumnReference to a legacy UnresolvedColumnReference dict."""
    result: dict[str, Any] = {
        "raw_reference": ref.reference_text,
        "resolved_column": ref.resolved_column,
        "resolution_method": ref.reference_kind.value if ref.reference_kind else None,
        "candidate_columns": [],
        "evidence": [],
    }
    if ref.resolved_column:
        result["resolved_columns"] = [ref.resolved_column]
    else:
        result["resolved_columns"] = []
    return result


def _map_operator_to_legacy(operator: str) -> str:
    """Map new pipeline operator strings to legacy operator enum values."""
    _MAP = {
        "eq": "eq",
        "==": "eq",
        "ne": "neq",
        "!=": "neq",
        "not_equal": "neq",
        "gt": "gt",
        ">": "gt",
        "gte": "gte",
        ">=": "gte",
        "lt": "lt",
        "<": "lt",
        "lte": "lte",
        "<=": "lte",
        "in": "in",
        "contains": "contains",
        "not_in": "not_in",
        "is_null": "eq",
        "is_not_null": "neq",
    }
    return _MAP.get(operator.lower().strip(), "contains")


def _build_decision(actions: list[dict[str, Any]]) -> str:
    """Build a human-readable decision summary from converted actions."""
    parts = []
    for a in actions:
        kind = a.get("kind", "unknown")
        if kind == "project_columns":
            n = len(a.get("requested_fields", []))
            parts.append(f"select {n} column(s)")
        elif kind == "drop_columns":
            n = len(a.get("requested_fields", []))
            parts.append(f"drop {n} column(s)")
        elif kind == "filter_rows":
            n = len(a.get("conditions", []))
            parts.append(f"filter rows ({n} condition(s))")
        elif kind == "sort_rows":
            n = len(a.get("sort_keys", []))
            parts.append(f"sort by {n} key(s)")
        elif kind == "rename_columns":
            n = len(a.get("mapping", []))
            parts.append(f"rename {n} column(s)")
    return "; ".join(parts) if parts else "unknown operation"


def _compute_hash(canonical: dict[str, Any]) -> str:
    """Compute a stable hash for the canonical intent."""
    import hashlib
    content = json.dumps(
        {k: v for k, v in canonical.items() if k not in ("intent_hash", "intent_id", "created_at", "grounded_at")},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Async helper
# ---------------------------------------------------------------------------


def _run_async(coro: Any) -> Any:
    """Run an async coroutine synchronously, handling existing event loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside an existing event loop (e.g., FastAPI)
        # Use a new thread to avoid blocking
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result(timeout=60)
    else:
        return asyncio.run(coro)


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Return True when an exception chain indicates an upstream 429."""
    current: BaseException | None = exc
    while current is not None:
        error_type = getattr(current, "error_type", None)
        if error_type == "rate_limit":
            return True
        message = str(current)
        if "429" in message or "rate limit" in message.lower():
            return True
        current = current.__cause__ or current.__context__
    return False
