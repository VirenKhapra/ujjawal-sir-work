from __future__ import annotations

from dataclasses import dataclass, field as dc_field
import hashlib
import difflib
import json
import re
import uuid
from datetime import UTC, datetime
from typing import Any, Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from app.services.llm_telemetry import log_runtime_event
from app.services.semantic_schema import canonical_target_for_column, infer_column_roles, normalize_semantic_name


CANONICAL_INTENT_SCHEMA_VERSION = "2.0"
SUPPORTED_CANONICAL_INTENT_SCHEMA_VERSIONS = {"1.0", "2.0"}
INTENT_EXTRACTOR_VERSION = "backend.canonical_intent.1"
INTENT_NORMALIZER_VERSION = "backend.canonical_intent.1"
INTENT_GROUNDING_VERSION = "backend.canonical_intent.1"
CANONICAL_INTENT_CAPABILITY_VERSION = "backend.capability.1"

RESOLUTION_STATUS = Literal[
    "resolved",
    "repaired",
    "ambiguous",
    "needs_clarification",
    "unsupported",
    "rejected",
]


class UnresolvedColumnReference(BaseModel):
    model_config = ConfigDict(extra="ignore")

    raw_reference: str
    resolved_column: str | None = None
    grounded_value: str | None = None
    resolution_method: str | None = None
    selection_mode: Literal["single", "semantic_family", "ambiguous"] | None = None
    resolved_columns: list[str] = Field(default_factory=list)
    candidate_columns: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class FilterCondition(BaseModel):
    model_config = ConfigDict(extra="ignore")

    field: UnresolvedColumnReference
    operator: Literal["eq", "neq", "gt", "lt", "gte", "lte", "contains", "in", "not_in"]
    value: Any


@dataclass(slots=True)
class _FilterFieldGroundingCandidate:
    column: str
    exact_value_matches: int = 0
    requested_value_count: int = 0
    value_coverage: float = 0.0
    observed_value_matches: list[str] = dc_field(default_factory=list)
    semantic_role_score: float = 0.0
    column_name_similarity: float = 0.0
    type_compatibility_score: float = 0.0
    final_score: float = 0.0
    resolution_reason: str = "insufficient_evidence"


class ProjectColumnsIntent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: Literal["project_columns"]
    requested_fields: list[UnresolvedColumnReference]


class DropColumnsIntent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: Literal["drop_columns"]
    requested_fields: list[UnresolvedColumnReference]


class FilterRowsIntent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: Literal["filter_rows"]
    mode: Literal["keep", "drop"] = "keep"
    conditions: list[FilterCondition]
    logic: Literal["and", "or"] = "and"


class RenameMapping(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source: UnresolvedColumnReference
    target_name: str


class RenameColumnsIntent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: Literal["rename_columns"]
    mapping: list[RenameMapping]


class SortKey(BaseModel):
    model_config = ConfigDict(extra="ignore")

    column: UnresolvedColumnReference
    direction: Literal["asc", "desc"] = "asc"


class SortRowsIntent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: Literal["sort_rows"]
    sort_keys: list[SortKey]


class LimitRowsIntent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: Literal["limit_rows"]
    limit: int


class CalculateIntent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: Literal["calculate"]
    operations: list[Any] = Field(default_factory=list)  # Accepts both str and dict operations


class VisualizeIntent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: Literal["visualize"]
    chart_type: str | None = None
    fields: list[UnresolvedColumnReference] = Field(default_factory=list)


class ReportIntent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: Literal["report"]
    sections: list[str] = Field(default_factory=list)


class CleaningIntentOperation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class CleanIntent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: Literal["clean"]
    mode: Literal["safe_default", "explicit"] = "safe_default"
    operations: list[CleaningIntentOperation] = Field(default_factory=list)


IntentAction = Annotated[
    ProjectColumnsIntent
    | DropColumnsIntent
    | FilterRowsIntent
    | RenameColumnsIntent
    | SortRowsIntent
    | LimitRowsIntent
    | CalculateIntent
    | VisualizeIntent
    | ReportIntent
    | CleanIntent,
    Field(discriminator="kind"),
]


class CanonicalIntent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schema_version: str = CANONICAL_INTENT_SCHEMA_VERSION
    intent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    intent_revision: int = 1
    intent_hash: str = ""
    parent_intent_id: str | None = None
    original_prompt: str = ""
    normalized_prompt: str = ""
    resolution_status: RESOLUTION_STATUS = "resolved"
    decision: str = ""
    evidence: list[str] = Field(default_factory=list)
    alternatives_considered: list[str] = Field(default_factory=list)
    actions: list[IntentAction] = Field(default_factory=list)
    output_format: Literal["xlsx", "csv", "json", "txt"] = "xlsx"
    assumptions: list[str] = Field(default_factory=list)
    repair_notes: list[str] = Field(default_factory=list)
    dataframe_profile: dict[str, Any] = Field(default_factory=dict)
    capability_version: str = CANONICAL_INTENT_CAPABILITY_VERSION
    capability_snapshot: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    grounded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CapabilitySnapshot(BaseModel):
    model_config = ConfigDict(extra="ignore")

    capability_version: str = CANONICAL_INTENT_CAPABILITY_VERSION
    available_action_kinds: list[str] = Field(default_factory=list)
    available_operators: list[str] = Field(default_factory=list)
    available_output_formats: list[str] = Field(default_factory=list)
    registered_agent_versions: dict[str, str] = Field(default_factory=dict)
    operation_schema_versions: dict[str, str] = Field(default_factory=dict)


_ROLE_ALIASES: dict[str, set[str]] = {
    "merchant": {"merchant", "vendor", "provider", "payment method", "payment_method", "payment type", "gateway"},
    "status": {"status", "state", "payment status", "payment_status", "loan status", "loan_status"},
    "gender": {"gender", "sex", "male", "female", "man", "woman", "men", "women"},
    "marital_status": {"marital status", "marital_status", "relationship status", "relationship_status"},
    "education": {"education", "education level", "education_level", "degree", "qualification"},
    "transaction_id": {"transaction id", "transaction_id", "txn id", "txn_id", "invoice id", "invoice_id", "id", "identifier"},
    "payment_value": {"amount", "payment", "payment value", "payment_value", "price", "cost", "value", "total", "subtotal"},
    "quantity": {"quantity", "qty", "units", "count"},
    "date": {"date", "transaction date", "invoice date", "application date", "voucher date"},
}

_DROP_COLUMN_VERBS = (
    "remove",
    "drop",
    "delete",
    "omit",
    "exclude",
    "without",
    "get rid of",
    "strip",
)
_OUTPUT_RESTRICTION_RE = re.compile(
    r"\b(?:keep only|only keep|show only|only show|only return|return only|"
    r"give me only|only give me|just give me|just give|only need|output only|"
    r"extract only|and nothing else|nothing else)\b|\bonly\b|\bjust\b",
    re.IGNORECASE,
)
_CLEAN_RE = re.compile(
    r"\b(?:clean(?:up)?|clean up|normalize|normalise|standardize|standardise|"
    r"deduplicate|de-duplicate|trim whitespace|remove duplicates)\b",
    re.IGNORECASE,
)
_ROW_HINT_RE = re.compile(
    r"\b(?:row|rows|record|records|where|which|that|with|for)\b",
    re.IGNORECASE,
)
_SORT_RE = re.compile(r"\b(?:sort|order)\s+by\s+(.+)$", re.IGNORECASE)
_LIMIT_RE = re.compile(r"\b(?:top|first|limit|only first)\s+(\d+)\b", re.IGNORECASE)
_OUTPUT_FORMAT_RE = re.compile(r"\b(xlsx|csv|json|txt)\b", re.IGNORECASE)
_FILTER_INTENT_PREFIX = re.compile(
    r"^\s*(?:clean(?:\s+(?:the|this)\s+data)?\s+and\s+)?"
    r"(?:only allow|keep only|only keep|show only|only show|only return|filter|"
    r"extract\s+rows?|return\s+rows?|pull(?:\s+out)?\s+rows?|remove|drop|exclude|"
    r"do not allow|don't allow|dont allow|wipe out|get rid of|delete)\s*",
    re.IGNORECASE,
)

_NULL_ROW_CLEANUP_VERB_RE = re.compile(
    r"\b(?:remove(?:s)?|drop(?:s)?|delete(?:s)?|omit(?:s)?|exclude(?:s)?|wipe out|get rid of)\s+(?:the\s+)?rows?\b",
    re.IGNORECASE,
)
_NULL_ROW_CLEANUP_HINT_RE = re.compile(
    r"\b(?:null|empty|blank|missing)\b",
    re.IGNORECASE,
)

_SUPPORTED_ACTION_KINDS = {
    "clean",
    "project_columns",
    "drop_columns",
    "rename_columns",
    "filter_rows",
    "sort_rows",
    "limit_rows",
    "calculate",
    "visualize",
    "report",
}
_SUPPORTED_OPERATORS = {"eq", "neq", "gt", "lt", "gte", "lte", "contains", "in", "not_in"}
_SUPPORTED_OUTPUT_FORMATS = {"xlsx", "csv", "json", "txt"}


def _stable_json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _canonical_intent_payload(intent: CanonicalIntent | dict[str, Any]) -> dict[str, Any]:
    if isinstance(intent, CanonicalIntent):
        payload = intent.model_dump(mode="json")
    else:
        payload = dict(intent)
    for key in (
        "intent_id",
        "intent_revision",
        "intent_hash",
        "parent_intent_id",
        "created_at",
        "grounded_at",
    ):
        payload.pop(key, None)
    return payload


def compute_intent_hash(intent: CanonicalIntent | dict[str, Any]) -> str:
    return hashlib.sha256(_stable_json_dumps(_canonical_intent_payload(intent)).encode("utf-8")).hexdigest()


def build_capability_snapshot() -> CapabilitySnapshot:
    return CapabilitySnapshot(
        available_action_kinds=sorted(_SUPPORTED_ACTION_KINDS),
        available_operators=sorted(_SUPPORTED_OPERATORS),
        available_output_formats=sorted(_SUPPORTED_OUTPUT_FORMATS),
        registered_agent_versions={
            "cleaning": "1.0",
            "filtering": "1.0",
            "calculation": "1.0",
            "visualization": "1.0",
            "reporting": "1.0",
        },
        operation_schema_versions={
            "cleaning": "1.0",
            "filtering": "1.0",
            "calculation": "1.0",
            "visualization": "1.0",
            "reporting": "1.0",
        },
    )


def _try_semantic_extraction(
    instruction: str,
    source_columns: list[str],
    *,
    preview_rows: list[dict[str, Any]] | None = None,
    data_profile: dict[str, Any] | None = None,
    output_format: str = "xlsx",
    detected_types: dict[str, str] | None = None,
    submission_id: str = "",
    trigger: str = "upload",
) -> dict[str, Any] | None:
    """Try the hybrid semantic extraction pipeline.

    Returns a canonical intent dict if semantic extraction succeeds,
    or None to fall back to deterministic regex extraction.
    """
    import os
    import logging

    logger = logging.getLogger(__name__)

    # Only use semantic pipeline if GROQ_API_KEY or GROQ_BRIDGE_API_KEY is configured
    if not os.environ.get("GROQ_API_KEY", "") and not os.environ.get("GROQ_BRIDGE_API_KEY", ""):
        return None

    # Don't use semantic pipeline for empty instructions
    if not instruction or not instruction.strip():
        return None

    log_runtime_event(
        "canonical_extractor_entered",
        service="backend",
        trigger=trigger,
        submission_id=submission_id,
        http_method="POST" if trigger == "upload" else "",
        instruction_present=True,
        canonical_intent_present=False,
        legacy_schema_state_present=False,
        prompt_text=instruction,
        source_column_count=len(source_columns),
        preview_row_count=len(preview_rows or []),
        output_format=output_format,
        bridge_enabled=bool(os.environ.get("GROQ_BRIDGE_API_KEY", "") or os.environ.get("GROQ_API_KEY", "")),
    )

    try:
        # ---- Try the NEW semantic pipeline (finflow_agent) first ----
        from app.services.new_pipeline_bridge import run_new_semantic_pipeline_sync

        new_result = run_new_semantic_pipeline_sync(
            instruction,
            source_columns,
            column_types=detected_types,
            output_format=output_format,
            submission_id=submission_id,
            trigger=trigger,
        )
        if new_result is not None:
            dataframe_profile = _build_dataframe_profile(
                source_columns,
                preview_rows or [],
                detected_types or {},
                data_profile=data_profile,
            )
            role_columns = infer_column_roles(source_columns)
            new_result = _repair_select_all_projection(new_result, instruction=instruction, source_columns=source_columns)
            new_result = _repair_profile_grounded_references(new_result, dataframe_profile, submission_id=submission_id)
            new_result = _repair_null_row_cleanup(new_result, instruction=instruction)
            new_result = _repair_missing_clean_action(new_result, instruction=instruction)
            new_result = _repair_missing_clean_operations(new_result, instruction=instruction)
            new_result = _repair_filter_mode(new_result, instruction=instruction)
            new_result = _repair_missing_calculate_action(
                new_result, instruction=instruction,
                source_columns=source_columns, dataframe_profile=dataframe_profile,
            )
            # Enrich with visualization intent if trigger language detected
            try:
                from finflow_agent.planning.intent_enricher import enrich_intent_with_visualization
                new_result = enrich_intent_with_visualization(new_result)
            except ImportError:
                pass  # Visualization module not available
            if isinstance(new_result, dict):
                new_result["intent_hash"] = compute_intent_hash(new_result)
            logger.info("New pipeline extraction succeeded for: %s", instruction[:80])
            return new_result

        # ---- Old semantic pipeline DISABLED to save LLM rate limit ----
        # The new pipeline handles extraction; if it fails (rate limit, etc.)
        # fall directly to deterministic regex instead of burning more tokens.
        logger.info("New pipeline returned None; skipping old semantic pipeline to save rate limit")
        return None

    except Exception as e:
        logger.warning("Semantic extraction error: %s", e)
        return None


def _semantic_actions_to_typed(
    canonical_actions: list[dict[str, Any]],
    source_columns: list[str],
) -> list[IntentAction]:
    """Convert compiled semantic actions (dicts) into typed IntentAction models."""
    typed: list[IntentAction] = []
    for action in canonical_actions:
        if not isinstance(action, dict):
            continue
        kind = str(action.get("kind", "")).strip()
        try:
            if kind == "clean":
                operations = []
                for op in action.get("operations", []):
                    if isinstance(op, dict):
                        operations.append(CleaningIntentOperation(
                            name=str(op.get("name", "")),
                            parameters=op.get("parameters", {}),
                        ))
                typed.append(CleanIntent(
                    kind="clean",
                    mode=action.get("mode", "safe_default"),
                    operations=operations,
                ))
            elif kind == "drop_columns":
                fields = _dict_fields_to_unresolved(action.get("requested_fields", []), source_columns)
                if fields:
                    typed.append(DropColumnsIntent(kind="drop_columns", requested_fields=fields))
            elif kind == "project_columns":
                fields = _dict_fields_to_unresolved(action.get("requested_fields", []), source_columns)
                if fields:
                    typed.append(ProjectColumnsIntent(kind="project_columns", requested_fields=fields))
            elif kind == "filter_rows":
                conditions = _dict_conditions_to_typed(action.get("conditions", []), source_columns)
                if conditions:
                    typed.append(FilterRowsIntent(
                        kind="filter_rows",
                        mode=action.get("mode", "keep"),
                        conditions=conditions,
                        logic=action.get("logic", "and"),
                    ))
            elif kind == "sort_rows":
                sort_keys = []
                for sk in action.get("sort_keys", []):
                    if isinstance(sk, dict):
                        col_ref = sk.get("column", {})
                        ref = _single_dict_to_unresolved(col_ref, source_columns)
                        if ref:
                            sort_keys.append(SortKey(column=ref, direction=sk.get("direction", "asc")))
                if sort_keys:
                    typed.append(SortRowsIntent(kind="sort_rows", sort_keys=sort_keys))
            elif kind == "limit_rows":
                try:
                    limit = int(action.get("limit", 0))
                    typed.append(LimitRowsIntent(kind="limit_rows", limit=max(0, limit)))
                except (TypeError, ValueError):
                    pass
            elif kind == "calculate":
                typed.append(CalculateIntent(kind="calculate", operations=action.get("operations", [])))
            elif kind == "visualize":
                fields = _dict_fields_to_unresolved(action.get("fields", []), source_columns)
                typed.append(VisualizeIntent(
                    kind="visualize",
                    chart_type=action.get("chart_type"),
                    fields=fields,
                ))
            elif kind == "report":
                typed.append(ReportIntent(kind="report", sections=action.get("sections", [])))
        except Exception:
            continue
    return typed


def _dict_fields_to_unresolved(
    fields: list[dict[str, Any] | Any],
    source_columns: list[str],
) -> list[UnresolvedColumnReference]:
    """Convert dict-based field references to UnresolvedColumnReference objects."""
    result: list[UnresolvedColumnReference] = []
    for field in fields:
        ref = _single_dict_to_unresolved(field, source_columns)
        if ref:
            result.append(ref)
    return result


def _single_dict_to_unresolved(
    field: dict[str, Any] | Any,
    source_columns: list[str],
) -> UnresolvedColumnReference | None:
    """Convert a single dict field reference to an UnresolvedColumnReference."""
    if not isinstance(field, dict):
        return None
    raw_ref = str(field.get("raw_reference", "")).strip()
    resolved = str(field.get("resolved_column") or "").strip() or None
    method = str(field.get("resolution_method") or "").strip() or None
    if not raw_ref and not resolved:
        return None
    return UnresolvedColumnReference(
        raw_reference=raw_ref or (resolved or ""),
        resolved_column=resolved,
        grounded_value=str(field.get("grounded_value") or "").strip() or None,
        resolution_method=method,
        selection_mode=str(field.get("selection_mode") or "").strip() or None,
        resolved_columns=[
            str(column).strip()
            for column in (field.get("resolved_columns") or [])
            if str(column).strip()
        ],
        candidate_columns=[
            str(column).strip()
            for column in (field.get("candidate_columns") or [])
            if str(column).strip()
        ],
        evidence=[
            str(item).strip()
            for item in (field.get("evidence") or [])
            if str(item).strip()
        ],
    )


def _dict_conditions_to_typed(
    conditions: list[dict[str, Any]],
    source_columns: list[str],
) -> list[FilterCondition]:
    """Convert dict-based filter conditions to typed FilterCondition objects."""
    result: list[FilterCondition] = []
    for cond in conditions:
        if not isinstance(cond, dict):
            continue
        field_dict = cond.get("field", {})
        field_ref = _single_dict_to_unresolved(field_dict, source_columns)
        if not field_ref:
            continue
        operator = str(cond.get("operator", "eq")).strip()
        if operator not in {"eq", "neq", "gt", "lt", "gte", "lte", "contains", "in", "not_in"}:
            operator = "eq"
        result.append(FilterCondition(
            field=field_ref,
            operator=operator,
            value=cond.get("value"),
        ))
    return result


def build_canonical_intent(
    source_columns: list[str],
    preview_rows: list[dict[str, Any]],
    instruction: str,
    *,
    output_format: str = "xlsx",
    detected_types: dict[str, str] | None = None,
    data_profile: dict[str, Any] | None = None,
    capability_snapshot: CapabilitySnapshot | dict[str, Any] | None = None,
    submission_id: str = "",
    trigger: str = "upload",
) -> dict[str, Any]:
    # ---------------------------------------------------------------
    # HYBRID SEMANTIC PIPELINE: Try semantic extraction first.
    # Falls back to deterministic regex if semantic extraction fails
    # or if GROQ_API_KEY is not configured.
    # ---------------------------------------------------------------
    source_columns = [str(column) for column in source_columns if str(column).strip()]
    log_runtime_event(
        "canonical_compiler_entered",
        service="backend",
        trigger=trigger,
        submission_id=submission_id,
        http_method="POST" if trigger == "upload" else "",
        instruction_present=bool((instruction or "").strip()),
        canonical_intent_present=False,
        legacy_schema_state_present=False,
        prompt_text=instruction or "",
        source_column_count=len(source_columns),
        preview_row_count=len(preview_rows or []),
        detected_type_count=len(detected_types or {}),
    )
    semantic_result = _try_semantic_extraction(
        instruction,
        source_columns,
        preview_rows=preview_rows,
        data_profile=data_profile,
        output_format=output_format,
        detected_types=detected_types,
        submission_id=submission_id,
        trigger=trigger,
    )
    if semantic_result is not None:
        return semantic_result

    # ---------------------------------------------------------------
    # FALLBACK: Deterministic regex-based extraction (original path)
    # ---------------------------------------------------------------
    normalized_prompt = _normalize_text(instruction)
    source_columns = [str(column) for column in source_columns if str(column).strip()]
    dataframe_profile = _build_dataframe_profile(
        source_columns,
        preview_rows,
        detected_types or {},
        data_profile=data_profile,
    )
    role_columns = infer_column_roles(source_columns)
    capability_snapshot_model = (
        capability_snapshot
        if isinstance(capability_snapshot, CapabilitySnapshot)
        else CapabilitySnapshot.model_validate(capability_snapshot or build_capability_snapshot().model_dump(mode="json"))
    )

    actions: list[IntentAction] = []
    evidence: list[str] = []
    assumptions: list[str] = []
    repair_notes: list[str] = []
    alternatives_considered: list[str] = []

    clean_action = _extract_clean_action(normalized_prompt)
    if clean_action is not None:
        actions.append(clean_action)
        evidence.append("The instruction contains an explicit data-cleaning request.")

    drop_columns_action, drop_evidence, drop_assumptions, drop_notes = _extract_drop_columns_action(
        normalized_prompt,
        source_columns,
        role_columns,
        dataframe_profile,
    )
    if drop_columns_action is not None:
        actions.append(drop_columns_action)
        evidence.extend(drop_evidence)
        assumptions.extend(drop_assumptions)
        repair_notes.extend(drop_notes)

    filter_action, filter_evidence, filter_assumptions, filter_notes = _extract_filter_rows_action(
        normalized_prompt,
        source_columns,
        role_columns,
        dataframe_profile,
    )
    if filter_action is not None:
        actions.append(filter_action)
        evidence.extend(filter_evidence)
        assumptions.extend(filter_assumptions)
        repair_notes.extend(filter_notes)

    projection_action, projection_evidence, projection_assumptions, projection_notes = _extract_projection_action(
        normalized_prompt,
        source_columns,
        role_columns,
        dataframe_profile,
    )
    if projection_action is not None:
        actions.append(projection_action)
        evidence.extend(projection_evidence)
        assumptions.extend(projection_assumptions)
        repair_notes.extend(projection_notes)

    sort_action = _extract_sort_action(normalized_prompt, source_columns, role_columns, dataframe_profile)
    if sort_action is not None:
        actions.append(sort_action)
        evidence.append("The instruction requests a specific ordering.")

    limit_action = _extract_limit_action(normalized_prompt)
    if limit_action is not None:
        actions.append(limit_action)
        evidence.append("The instruction constrains the number of rows to return.")

    calculate_action = _extract_calculate_action(
        normalized_prompt,
        source_columns=source_columns,
        role_columns=role_columns,
        dataframe_profile=dataframe_profile,
    )
    if calculate_action is not None:
        actions.append(calculate_action)
        evidence.append("The instruction requests an aggregate calculation.")

    visualize_action = _extract_visualize_action(normalized_prompt, source_columns, role_columns, dataframe_profile)
    if visualize_action is not None:
        actions.append(visualize_action)
        evidence.append("The instruction requests a visualization.")

    report_action = _extract_report_action(normalized_prompt)
    if report_action is not None:
        actions.append(report_action)
        evidence.append("The instruction requests a report-style summary.")

    if _looks_like_output_format_request(normalized_prompt):
        inferred_output = _extract_output_format(normalized_prompt)
        if inferred_output:
            output_format = inferred_output
            repair_notes.append(f"Normalized output format request to '{output_format}'.")

    if not actions and normalized_prompt:
        alternatives_considered.extend(["project_columns", "filter_rows", "drop_columns"])
        evidence.append("No supported intent could be grounded with the current schema profile.")

    grounded_references = _iter_grounded_references(actions)
    has_unresolved_references = any(
        not item.get("resolved_column") and not item.get("resolved_columns")
        for item in grounded_references
    )
    has_ambiguous_references = any(str(item.get("selection_mode", "")).strip() == "ambiguous" for item in grounded_references)

    if not actions:
        resolution_status: RESOLUTION_STATUS = "needs_clarification" if normalized_prompt else "unsupported"
    elif has_unresolved_references or has_ambiguous_references:
        resolution_status = "needs_clarification"
        evidence.append("One or more requested fields could not be grounded decisively to available columns.")
    elif repair_notes:
        resolution_status = "repaired"
    elif any(item.get("resolution_method") not in {None, "exact_name"} for item in grounded_references):
        resolution_status = "repaired"
    else:
        resolution_status = "resolved"

    if not source_columns and normalized_prompt:
        resolution_status = "needs_clarification"
        evidence.append("No dataframe columns were available to ground the request.")

    canonical = CanonicalIntent(
        schema_version="2.0",
        intent_hash="",
        original_prompt=str(instruction or ""),
        normalized_prompt=normalized_prompt,
        resolution_status=resolution_status,
        decision=_build_decision_summary(actions),
        evidence=_dedupe_preserve_order(evidence),
        alternatives_considered=_dedupe_preserve_order(alternatives_considered),
        actions=actions,
        output_format=output_format if output_format in {"xlsx", "csv", "json", "txt"} else "xlsx",
        assumptions=_dedupe_preserve_order(assumptions),
        repair_notes=_dedupe_preserve_order(repair_notes),
        dataframe_profile=dataframe_profile,
        capability_version=capability_snapshot_model.capability_version,
        capability_snapshot=capability_snapshot_model.model_dump(mode="json"),
        grounded_at=datetime.now(UTC),
    )
    canonical_dict = _repair_null_row_cleanup(
        canonical.model_dump(mode="json"),
        instruction=instruction,
    )
    canonical_dict = _remove_spurious_analysis_filter_actions(
        canonical_dict,
        instruction=instruction,
    )
    canonical_dict = _repair_profile_grounded_references(
        canonical_dict,
        dataframe_profile,
        submission_id=submission_id,
    )
    # Enrich with visualization intent if trigger language detected
    try:
        from finflow_agent.planning.intent_enricher import enrich_intent_with_visualization
        canonical_dict = enrich_intent_with_visualization(canonical_dict)
    except ImportError:
        pass
    if isinstance(canonical_dict, dict):
        canonical_dict["intent_hash"] = compute_intent_hash(canonical_dict)
    return canonical_dict


def canonical_intent_to_legacy_action_schema(canonical_intent: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(canonical_intent, dict):
        return {"actions": [], "required_capabilities": [], "optional_hints": {}, "source": "deferred_to_agent_parser"}

    actions: list[dict[str, Any]] = []
    required_capabilities: set[str] = set()
    for action in canonical_intent.get("actions", []):
        if not isinstance(action, dict):
            continue
        kind = str(action.get("kind", "")).strip()
        if kind == "clean":
            actions.append(
                {
                    "action": "clean",
                    "mode": str(action.get("mode", "safe_default")),
                    "operations": action.get("operations", []),
                }
            )
            required_capabilities.add("cleaning")
            continue
        if kind == "project_columns":
            requested_fields = _serialize_requested_fields(action.get("requested_fields", []))
            if requested_fields:
                actions.append({"action": "keep_columns", "roles": requested_fields})
                required_capabilities.add("column_keep")
            continue
        if kind == "drop_columns":
            requested_fields = _serialize_requested_fields(action.get("requested_fields", []))
            if requested_fields:
                actions.append({"action": "drop_columns", "roles": requested_fields})
                required_capabilities.add("column_drop")
            continue
        if kind == "filter_rows":
            legacy_conditions = []
            for condition in action.get("conditions", []):
                if not isinstance(condition, dict):
                    continue
                role = _legacy_condition_role(condition)
                if not role:
                    continue
                legacy_conditions.append(
                    {
                        "role": role,
                        "op": str(condition.get("operator", "eq")),
                        "value": condition.get("value"),
                    }
                )
            legacy_logic = "or" if len(legacy_conditions) <= 1 else str(action.get("logic", "and"))
            condition_tree = {
                "logic": legacy_logic,
                "conditions": legacy_conditions,
            }
            legacy_action = "drop_rows_where" if str(action.get("mode", "keep")) == "drop" else "keep_rows_where"
            if condition_tree["conditions"]:
                actions.append({"action": legacy_action, "condition_tree": condition_tree})
                required_capabilities.add("row_filter")
            continue
        if kind == "rename_columns":
            mapping: dict[str, str] = {}
            for item in action.get("mapping", []):
                if not isinstance(item, dict):
                    continue
                source = _condition_role(item.get("source"))
                target = str(item.get("target_name", "")).strip()
                if source and target:
                    mapping[source] = target
            if mapping:
                actions.append({"action": "rename_columns", "mapping": mapping})
                required_capabilities.add("column_rename")
            continue
        if kind == "sort_rows":
            sort_keys = []
            for item in action.get("sort_keys", []):
                if not isinstance(item, dict):
                    continue
                column = _condition_role(item.get("column"))
                if column:
                    sort_keys.append({"column": column, "direction": str(item.get("direction", "asc"))})
            if sort_keys:
                actions.append({"action": "sort_rows", "sort_keys": sort_keys})
                required_capabilities.add("row_sort")
            continue
        if kind == "limit_rows":
            try:
                limit = int(action.get("limit"))
            except (TypeError, ValueError):
                continue
            actions.append({"action": "limit_rows", "limit": limit})
            required_capabilities.add("row_limit")
            continue
        if kind == "calculate":
            actions.append({"action": "calculate", "operations": action.get("operations", [])})
            required_capabilities.add("calculation")
            continue
        if kind == "visualize":
            actions.append(
                {
                    "action": "visualize",
                    "chart_type": action.get("chart_type"),
                    "fields": _serialize_requested_fields(action.get("fields", [])),
                }
            )
            required_capabilities.add("visualization")
            continue
        if kind == "report":
            actions.append({"action": "report", "sections": action.get("sections", [])})
            required_capabilities.add("reporting")

    return {
        "actions": actions,
        "required_capabilities": sorted(required_capabilities),
        "optional_hints": {"source": "deferred_to_agent_parser"},
        "source": "deferred_to_agent_parser",
    }


def build_action_schema_from_canonical_intent(
    source_columns: list[str],
    preview_rows: list[dict[str, Any]],
    instruction: str,
    *,
    output_format: str = "xlsx",
    detected_types: dict[str, str] | None = None,
) -> dict[str, Any]:
    canonical_intent = build_canonical_intent(
        source_columns,
        preview_rows,
        instruction,
        output_format=output_format,
        detected_types=detected_types,
    )
    return canonical_intent_to_legacy_action_schema(canonical_intent)


def _extract_clean_action(normalized_prompt: str) -> CleanIntent | None:
    if not _CLEAN_RE.search(normalized_prompt):
        if not _looks_like_null_row_cleanup(normalized_prompt):
            # Check for specific cleaning keywords even without "clean" mentioned
            if not re.search(
                r"\b(?:remove\s+(?:negative|duplicat|empty|null|missing|blank)|"
                r"fill\s+(?:missing|null|empty|blank)|"
                r"strip|convert|normalize|standardize)\b",
                normalized_prompt, re.IGNORECASE
            ):
                return None
    operations = []

    # Deduplication
    if re.search(r"\b(?:deduplicate|de-duplicate|remove\s+duplicate|drop\s+duplicate)\b", normalized_prompt, re.IGNORECASE):
        operations.append(CleaningIntentOperation(name="drop_duplicates"))

    # Trim whitespace
    if re.search(r"\b(?:trim|whitespace|strip\s+spaces?)\b", normalized_prompt, re.IGNORECASE):
        operations.append(CleaningIntentOperation(name="trim_whitespace"))

    # Normalize/standardize
    if re.search(r"\b(?:normalize|normalise|standardize|standardise)\s+(?:column\s+)?names?\b", normalized_prompt, re.IGNORECASE):
        operations.append(CleaningIntentOperation(name="normalize_column_names"))

    # Drop nulls/missing
    if _looks_like_null_row_cleanup(normalized_prompt):
        operations.append(
            CleaningIntentOperation(
                name="drop_nulls",
                parameters={"columns": None, "how": "any"},
            )
        )

    # Fill nulls/missing
    if re.search(r"\b(?:fill|impute|replace)\s+(?:missing|null|empty|blank|nan)\b", normalized_prompt, re.IGNORECASE):
        operations.append(CleaningIntentOperation(name="fill_nulls", parameters={"value": 0}))

    # Remove negative signs / absolute value
    if re.search(r"\b(?:remove|strip|delete|drop)\s+(?:the\s+)?(?:negative\s+(?:sign|value|number)|minus\s+sign)\b", normalized_prompt, re.IGNORECASE):
        operations.append(CleaningIntentOperation(name="absolute_value", parameters={"columns": "__all_numeric_columns__"}))
    elif re.search(r"\b(?:make\s+(?:all\s+)?(?:values?\s+)?positive|absolute\s+value|remove\s+negatives?)\b", normalized_prompt, re.IGNORECASE):
        operations.append(CleaningIntentOperation(name="absolute_value", parameters={"columns": "__all_numeric_columns__"}))

    # Strip currency symbols
    if re.search(r"\b(?:strip|remove)\s+(?:currency|dollar|rupee)\s*(?:symbol|sign)?\b", normalized_prompt, re.IGNORECASE):
        operations.append(CleaningIntentOperation(name="strip_currency_symbols"))

    # Remove commas from numbers
    if re.search(r"\b(?:remove|strip)\s+commas?\s+(?:from\s+)?(?:number|numeric|amount)\b", normalized_prompt, re.IGNORECASE):
        operations.append(CleaningIntentOperation(name="remove_commas_from_numbers"))

    # Normalize dates
    if re.search(r"\b(?:normalize|standardize|fix|convert)\s+(?:the\s+)?date\b", normalized_prompt, re.IGNORECASE):
        operations.append(CleaningIntentOperation(name="normalize_date"))

    # Normalize text case
    if re.search(r"\b(?:lowercase|uppercase|title\s*case|normalize\s+(?:text\s+)?case)\b", normalized_prompt, re.IGNORECASE):
        operations.append(CleaningIntentOperation(name="normalize_text_case"))

    # Remove empty rows/columns
    if re.search(r"\b(?:remove|drop|delete)\s+(?:empty|blank)\s+(?:row|column)\b", normalized_prompt, re.IGNORECASE):
        operations.append(CleaningIntentOperation(name="remove_empty_rows"))

    return CleanIntent(kind="clean", mode="explicit" if operations else "safe_default", operations=operations)


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _extract_projection_action(
    normalized_prompt: str,
    source_columns: list[str],
    role_columns: dict[str, list[str]],
    dataframe_profile: dict[str, Any],
) -> tuple[ProjectColumnsIntent | None, list[str], list[str], list[str]]:
    if not _looks_like_projection_request(normalized_prompt):
        return None, [], [], []
    if _looks_like_filter_request(normalized_prompt):
        return None, [], [], []
    if any(re.search(pattern, normalized_prompt) for pattern in _SELECT_ALL_PATTERNS):
        return (
            ProjectColumnsIntent(
                kind="project_columns",
                requested_fields=[
                    UnresolvedColumnReference(
                        raw_reference="all columns",
                        resolved_columns=[str(column) for column in source_columns if str(column).strip()],
                        candidate_columns=[str(column) for column in source_columns if str(column).strip()],
                        selection_mode="semantic_family",
                        resolution_method="all_columns",
                    )
                ],
            ),
            ["The instruction explicitly requests that every column be preserved in the output."],
            [],
            [],
        )

    references = _extract_requested_columns(normalized_prompt, source_columns, role_columns, dataframe_profile)
    if not references:
        return None, [], [], []

    evidence = ["The instruction restricts the output to specific columns."]
    assumptions = []
    notes = []
    if any(ref.resolution_method not in {None, "exact_name"} for ref in references):
        notes.append("Resolved one or more requested fields through schema grounding.")
    if any(ref.resolved_column is None for ref in references):
        notes.append("Some requested fields remain unresolved after grounding.")

    return (
        ProjectColumnsIntent(kind="project_columns", requested_fields=references),
        evidence,
        assumptions,
        notes,
    )


def _extract_drop_columns_action(
    normalized_prompt: str,
    source_columns: list[str],
    role_columns: dict[str, list[str]],
    dataframe_profile: dict[str, Any],
) -> tuple[DropColumnsIntent | None, list[str], list[str], list[str]]:
    if _looks_like_null_row_cleanup(normalized_prompt):
        return None, [], [], []
    if not any(verb in normalized_prompt for verb in _DROP_COLUMN_VERBS):
        return None, [], [], []

    references = _extract_requested_columns(normalized_prompt, source_columns, role_columns, dataframe_profile)
    if not references:
        return None, [], [], []
    if not any(_reference_appears_in_prompt(ref.raw_reference, normalized_prompt) for ref in references):
        return None, [], [], []

    evidence = ["The instruction removes one or more columns from the output."]
    notes = []
    if any(ref.resolution_method not in {None, "exact_name"} for ref in references):
        notes.append("Resolved one or more removed columns through schema grounding.")
    return (
        DropColumnsIntent(kind="drop_columns", requested_fields=references),
        evidence,
        [],
        notes,
    )


def _extract_filter_rows_action(
    normalized_prompt: str,
    source_columns: list[str],
    role_columns: dict[str, list[str]],
    dataframe_profile: dict[str, Any],
) -> tuple[FilterRowsIntent | None, list[str], list[str], list[str]]:
    if not _looks_like_filter_request(normalized_prompt):
        return None, [], [], []

    membership_filter, membership_evidence, membership_assumptions, membership_notes = _extract_membership_filter_action(
        normalized_prompt,
        source_columns,
        role_columns,
        dataframe_profile,
    )
    if membership_filter is not None:
        return membership_filter, membership_evidence, membership_assumptions, membership_notes

    filter_prompt = _strip_filter_prefix(normalized_prompt)
    if _looks_like_null_row_cleanup(filter_prompt):
        cleanup_match = _NULL_ROW_CLEANUP_VERB_RE.search(filter_prompt)
        if cleanup_match:
            filter_prompt = filter_prompt[: cleanup_match.start()].strip(" ,.;")

    clauses, connectors = _split_filter_clauses(filter_prompt)
    if not clauses:
        return None, [], [], []

    conditions: list[FilterCondition] = []
    notes: list[str] = []
    assumptions: list[str] = []
    for clause in clauses:
        parsed = _parse_filter_clause(clause, source_columns, role_columns, dataframe_profile)
        if parsed is not None:
            conditions.append(parsed)
            if parsed.field.resolution_method not in {None, "exact_name"}:
                notes.append("Resolved one or more filter fields through schema grounding.")
        else:
            assumptions.append(f"Could not fully ground filter clause: {clause}")

    if not conditions:
        return None, [], [], []

    logic = "or" if connectors and all(connector == "or" for connector in connectors) else "and"
    mode = "drop" if _looks_like_drop_row_request(normalized_prompt) else "keep"
    evidence = ["The instruction constrains row selection with explicit filter logic."]
    return (
        FilterRowsIntent(kind="filter_rows", mode=mode, conditions=conditions, logic=logic),
        evidence,
        assumptions,
        notes,
    )


def _extract_membership_filter_action(
    normalized_prompt: str,
    source_columns: list[str],
    role_columns: dict[str, list[str]],
    dataframe_profile: dict[str, Any],
) -> tuple[FilterRowsIntent | None, list[str], list[str], list[str]]:
    prompt = _strip_noise_tokens(_strip_filter_prefix(normalized_prompt))
    if not prompt:
        return None, [], [], []

    patterns = (
        r"(?P<op>contains?|with)\s+(?P<values>.+?)\s+as\s+(?:a\s+)?(?P<field>.+)",
        r"(?P<field>.+?)\s+(?:is|equals?|equal to|=|:)\s*(?P<values>.+)",
    )

    for pattern in patterns:
        match = re.search(pattern, prompt, flags=re.IGNORECASE)
        if not match:
            continue
        field_text = match.groupdict().get("field", "")
        values_text = match.groupdict().get("values", "")
        if not field_text or not values_text:
            continue
        value = _coerce_filter_value(values_text)
        if not isinstance(value, list) or len(value) <= 1:
            continue
        field_ref = _ground_column_reference(field_text, source_columns, role_columns)
        if field_ref is None:
            continue
        condition = FilterCondition(field=field_ref, operator="in", value=value)
        evidence = ["The instruction constrains rows with a membership-style value list."]
        notes = []
        assumptions = []
        if condition.field.resolution_method not in {None, "exact_name"}:
            notes.append("Resolved one or more filter fields through schema grounding.")
        return (
            FilterRowsIntent(kind="filter_rows", mode="drop" if _looks_like_drop_row_request(normalized_prompt) else "keep", conditions=[condition], logic="and"),
            evidence,
            assumptions,
            notes,
        )

    return None, [], [], []


def _extract_sort_action(
    normalized_prompt: str,
    source_columns: list[str],
    role_columns: dict[str, list[str]],
    dataframe_profile: dict[str, Any],
) -> SortRowsIntent | None:
    match = _SORT_RE.search(normalized_prompt)
    if not match:
        return None
    sort_fields = _extract_requested_columns(match.group(1), source_columns, role_columns, dataframe_profile)
    if not sort_fields:
        return None
    keys = [SortKey(column=field) for field in sort_fields]
    return SortRowsIntent(kind="sort_rows", sort_keys=keys)


def _extract_limit_action(normalized_prompt: str) -> LimitRowsIntent | None:
    match = _LIMIT_RE.search(normalized_prompt)
    if not match:
        return None
    try:
        limit = int(match.group(1))
    except (TypeError, ValueError):
        return None
    return LimitRowsIntent(kind="limit_rows", limit=max(0, limit))


def _extract_calculate_action(
    normalized_prompt: str,
    source_columns: list[str] | None = None,
    role_columns: dict[str, list[str]] | None = None,
    dataframe_profile: dict[str, Any] | None = None,
) -> CalculateIntent | None:
    """Extract structured calculation operations from user prompt.

    Handles:
    - Basic aggregates: sum, average, mean, count, median, min, max
    - Percentage/conditional percentage queries
    - Quarterly revenue/aggregation queries
    - Ratio and difference queries

    Returns structured operations (dicts) that the compiler can process,
    rather than raw prompt strings.
    """
    source_columns = source_columns or []
    source_lower = {c.lower(): c for c in source_columns}

    operations: list[dict] = []

    # --- Percentage-based demographic calculations ---
    pct_match = re.search(
        r"\bpercentage\s+of\s+(?P<denom_group>\w+)?\s*(?:employees?|workers?|staff|people|records?|rows?)?\s*"
        r"(?:who\s+are|that\s+are|which\s+are|having|with)\s+(?P<value>\w+)",
        normalized_prompt,
        re.IGNORECASE,
    )
    if pct_match:
        denom_group = (pct_match.group("denom_group") or "").strip().lower()
        target_value = pct_match.group("value").strip().lower()

        # Determine filter and denominator columns from source data
        filter_column = _find_column_for_value(target_value, source_columns, dataframe_profile)
        denom_filter_column = None
        denom_filter_value = None

        # "percentage of female employees who are single" → denominator = female
        if denom_group and denom_group not in {"total", "all", "overall", "entire"}:
            denom_filter_column = _find_column_for_value(denom_group, source_columns, dataframe_profile)
            denom_filter_value = denom_group

        op = {
            "type": "conditional_percentage",
            "column": filter_column or (source_columns[0] if source_columns else "id"),
            "filter_column": filter_column or "status",
            "filter_value": target_value,
        }
        if denom_filter_column:
            op["denominator_filter_column"] = denom_filter_column
            op["denominator_filter_value"] = denom_filter_value
        operations.append(op)

    # --- Quarterly revenue/aggregation ---
    quarterly_match = re.search(
        r"\b(?:quarterly|quarter[- ]?wise|by\s+quarter|per\s+quarter)\s*"
        r"(?:revenue|sales|income|amount|total|sum|count|average|mean)?",
        normalized_prompt,
        re.IGNORECASE,
    )
    if not quarterly_match:
        quarterly_match = re.search(
            r"\b(?:revenue|sales|income|amount|total)\s*(?:by|per|for\s+each)\s*quarter",
            normalized_prompt,
            re.IGNORECASE,
        )
    if quarterly_match:
        # Find date and numeric columns
        date_col = _find_date_column(source_columns, dataframe_profile)
        numeric_col = _find_revenue_column(normalized_prompt, source_columns, dataframe_profile)

        if date_col and numeric_col:
            op_type = "quarterly_sum"
            if re.search(r"\b(?:average|avg|mean)\b", normalized_prompt, re.IGNORECASE):
                op_type = "quarterly_mean"
            elif re.search(r"\bcount\b", normalized_prompt, re.IGNORECASE):
                op_type = "quarterly_count"
            operations.append({
                "type": op_type,
                "column": numeric_col,
                "date_column": date_col,
            })

    # --- Basic aggregates (existing but now structured) ---
    if not operations:
        basic_agg = re.search(
            r"\b(?P<op>sum|total|average|avg|mean|count|median|min|max|minimum|maximum)\b"
            r"(?:\s+of)?\s*(?P<col>\w+)?",
            normalized_prompt,
            re.IGNORECASE,
        )
        if basic_agg:
            op_name = basic_agg.group("op").lower()
            col_name = (basic_agg.group("col") or "").strip().lower()

            op_map = {
                "sum": "sum", "total": "sum",
                "average": "mean", "avg": "mean", "mean": "mean",
                "count": "count",
                "median": "median",
                "min": "min", "minimum": "min",
                "max": "max", "maximum": "max",
            }
            calc_type = op_map.get(op_name, "sum")

            # Resolve column name
            resolved_col = source_lower.get(col_name, col_name) if col_name else None
            if not resolved_col and source_columns:
                # Try to find a numeric column
                resolved_col = _find_revenue_column(normalized_prompt, source_columns, dataframe_profile)

            if resolved_col:
                operations.append({"type": calc_type, "column": resolved_col})

    if not operations:
        # Fallback: detect any calculation keywords and return raw
        if re.search(r"\b(?:sum|total|average|avg|mean|count|percentage|percent|quarterly|ratio)\b",
                     normalized_prompt, re.IGNORECASE):
            return CalculateIntent(kind="calculate", operations=[normalized_prompt])
        return None

    return CalculateIntent(kind="calculate", operations=operations)


def _find_column_for_value(value: str, source_columns: list[str], profile: dict | None) -> str | None:
    """Find which column likely contains the given value based on the data profile."""
    if not profile:
        return None
    columns_info = profile.get("columns", [])
    if isinstance(columns_info, list):
        for col_info in columns_info:
            if not isinstance(col_info, dict):
                continue
            col_name = col_info.get("name", "")
            samples = col_info.get("sample_values", [])
            if any(value.lower() in str(s).lower() for s in samples):
                return col_name
    # Fallback: try column name matching
    for col in source_columns:
        if value.lower() in col.lower():
            return col
    return None


def _find_date_column(source_columns: list[str], profile: dict | None) -> str | None:
    """Find the most likely date column."""
    date_keywords = {"date", "time", "timestamp", "created", "period", "month", "year", "day"}
    for col in source_columns:
        if any(kw in col.lower() for kw in date_keywords):
            return col
    if profile:
        columns_info = profile.get("columns", [])
        if isinstance(columns_info, list):
            for col_info in columns_info:
                if isinstance(col_info, dict):
                    if col_info.get("detected_type") in {"date", "datetime"}:
                        return col_info.get("name")
                    if col_info.get("semantic_type_hint") in {"transaction_date", "date"}:
                        return col_info.get("name")
    return None


def _find_revenue_column(prompt: str, source_columns: list[str], profile: dict | None) -> str | None:
    """Find the most likely revenue/amount numeric column."""
    revenue_keywords = {"revenue", "sales", "income", "amount", "price", "total", "profit", "cost"}
    # Check if prompt mentions a specific column
    for col in source_columns:
        if col.lower() in prompt.lower():
            return col
    # Match by keyword
    for col in source_columns:
        if any(kw in col.lower() for kw in revenue_keywords):
            return col
    # Fallback: first numeric column from profile
    if profile:
        columns_info = profile.get("columns", [])
        if isinstance(columns_info, list):
            for col_info in columns_info:
                if isinstance(col_info, dict) and col_info.get("detected_type") == "number":
                    return col_info.get("name")
    return None


def _extract_visualize_action(
    normalized_prompt: str,
    source_columns: list[str],
    role_columns: dict[str, list[str]],
    dataframe_profile: dict[str, Any],
) -> VisualizeIntent | None:
    if not re.search(r"\b(?:chart|plot|graph|visuali[sz]e|bar chart|line chart|pie chart)\b", normalized_prompt):
        return None
    fields = _extract_requested_columns(normalized_prompt, source_columns, role_columns, dataframe_profile)
    chart_type = None
    chart_type_aliases = {
        "bar chart": "bar",
        "line chart": "line",
        "pie chart": "pie",
        "scatter plot": "scatter",
        "chart": "chart",
        "plot": "plot",
        "graph": "graph",
    }
    for candidate in ("bar chart", "line chart", "pie chart", "scatter plot", "chart", "plot", "graph"):
        if candidate in normalized_prompt:
            chart_type = chart_type_aliases.get(candidate, candidate)
            break
    return VisualizeIntent(kind="visualize", chart_type=chart_type, fields=fields)


def _suggest_calculation_output_column(aggregation_type: str, group_by: UnresolvedColumnReference) -> str | None:
    base = group_by.resolved_column or group_by.raw_reference
    base = normalize_semantic_name(base).replace(" ", "_")
    if not base:
        return None
    suffix = {
        "group_count": "count",
        "group_sum": "sum",
        "group_mean": "mean",
    }.get(aggregation_type)
    if not suffix:
        return base
    return f"{base}_{suffix}"


def _extract_report_action(normalized_prompt: str) -> ReportIntent | None:
    if not re.search(r"\b(?:report|summary|summarize|summarise|narrative)\b", normalized_prompt):
        return None
    return ReportIntent(kind="report", sections=[normalized_prompt])


def _looks_like_projection_request(normalized_prompt: str) -> bool:
    if any(re.search(pattern, normalized_prompt) for pattern in _SELECT_ALL_PATTERNS):
        return True
    if not _OUTPUT_RESTRICTION_RE.search(normalized_prompt):
        return False
    return not _looks_like_filter_request(normalized_prompt)


def _looks_like_filter_request(normalized_prompt: str) -> bool:
    return bool(
        re.search(r"\b(?:where|equals?|equal to|is\s+\d|greater than|less than|at least|at most|contains?|matches?)\b", normalized_prompt)
        or re.search(r"\b(?:rows?|records?)\s+(?:for|where|with)\b", normalized_prompt)
        or re.search(r"\bas\s+(?:a\s+)?[a-z0-9_ ]+\b", normalized_prompt)
        or any(verb in normalized_prompt for verb in _DROP_COLUMN_VERBS)
    )


def _looks_like_drop_row_request(normalized_prompt: str) -> bool:
    return bool(re.search(r"\b(?:remove(?:s)?|drop(?:s)?|delete(?:s)?|omit(?:s)?|exclude(?:s)?|wipe out|get rid of)\s+(?:rows?|records?)\b", normalized_prompt))


def _looks_like_null_row_cleanup(normalized_prompt: str) -> bool:
    if not _ROW_HINT_RE.search(normalized_prompt):
        return False
    if not _NULL_ROW_CLEANUP_HINT_RE.search(normalized_prompt):
        return False
    return bool(_NULL_ROW_CLEANUP_VERB_RE.search(normalized_prompt))


def _looks_like_output_format_request(normalized_prompt: str) -> bool:
    return bool(_OUTPUT_FORMAT_RE.search(normalized_prompt))


def _extract_output_format(normalized_prompt: str) -> str | None:
    match = _OUTPUT_FORMAT_RE.search(normalized_prompt)
    return match.group(1).lower() if match else None


def _remove_spurious_analysis_filter_actions(
    canonical_intent: dict[str, Any],
    *,
    instruction: str,
) -> dict[str, Any]:
    if not isinstance(canonical_intent, dict):
        return canonical_intent

    normalized_prompt = _normalize_text(instruction)
    if not re.search(r"\b(?:chart|plot|graph|visuali[sz]e|ratio|share|distribution|breakdown|percentage|percent)\b", normalized_prompt):
        return canonical_intent

    actions = canonical_intent.get("actions")
    if not isinstance(actions, list):
        return canonical_intent

    retained_actions: list[Any] = []
    removed_any = False
    for action in actions:
        if not isinstance(action, dict) or str(action.get("kind", "")).strip() != "filter_rows":
            retained_actions.append(action)
            continue
        if _filter_action_looks_spurious_for_analysis(action):
            removed_any = True
            continue
        retained_actions.append(action)

    if not removed_any:
        return canonical_intent

    canonical_intent["actions"] = retained_actions
    canonical_intent["decision"] = _build_decision_summary(retained_actions)

    evidence = canonical_intent.get("evidence")
    if not isinstance(evidence, list):
        evidence = []
    evidence.append("Removed a spurious filter interpretation from visualization or ratio language.")
    canonical_intent["evidence"] = _dedupe_preserve_order(evidence)

    repair_notes = canonical_intent.get("repair_notes")
    if not isinstance(repair_notes, list):
        repair_notes = []
    repair_notes.append("Discarded a filter-like parse artifact from a chart or ratio request.")
    canonical_intent["repair_notes"] = _dedupe_preserve_order(repair_notes)

    if str(canonical_intent.get("resolution_status", "")).strip() in {"needs_clarification", "unsupported"}:
        canonical_intent["resolution_status"] = "repaired"

    return canonical_intent


def _filter_action_looks_spurious_for_analysis(action: dict[str, Any]) -> bool:
    if str(action.get("kind", "")).strip() != "filter_rows":
        return False
    conditions = action.get("conditions", [])
    if not isinstance(conditions, list):
        return False
    for condition in conditions:
        if not isinstance(condition, dict):
            continue
        field = condition.get("field")
        field_raw = ""
        if isinstance(field, dict):
            field_raw = _normalize_text(str(field.get("raw_reference", "")))
        value_text = _normalize_text(str(condition.get("value", "")))
        if field_raw in {"chart", "plot", "graph", "pie chart", "bar chart", "line chart", "scatter plot"}:
            return True
        if re.search(r"\b(?:ratio|share|distribution|breakdown|percentage|percent)\b", value_text):
            return True
        if re.search(r"\b(?:chart|plot|graph)\b", value_text):
            return True
    return False


def _build_dataframe_profile(
    source_columns: list[str],
    preview_rows: list[dict[str, Any]],
    detected_types: dict[str, str],
    *,
    data_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    semantic_roles = infer_column_roles(source_columns)
    preview_samples: dict[str, list[Any]] = {}
    for column in source_columns:
        values: list[Any] = []
        for row in preview_rows[:5]:
            if isinstance(row, dict) and column in row and row[column] not in {None, ""}:
                values.append(row[column])
            if len(values) >= 3:
                break
        preview_samples[column] = values

    profile_columns: dict[str, dict[str, Any]] = {}
    if isinstance(data_profile, dict):
        for column in data_profile.get("columns", []):
            if not isinstance(column, dict):
                continue
            name = str(column.get("name", "")).strip()
            if name:
                profile_columns[name] = column
        if not preview_rows and isinstance(data_profile.get("preview_rows"), list):
            preview_rows = [row for row in data_profile.get("preview_rows", []) if isinstance(row, dict)]
            for column in source_columns:
                values: list[Any] = []
                for row in preview_rows[:5]:
                    if isinstance(row, dict) and column in row and row[column] not in {None, ""}:
                        values.append(row[column])
                    if len(values) >= 3:
                        break
                preview_samples[column] = values

    merged_columns: list[dict[str, Any]] = []
    for column in source_columns:
        column_profile = dict(profile_columns.get(column, {}))
        column_profile.setdefault("name", column)
        column_profile.setdefault("normalized_name", normalize_semantic_name(column))
        column_profile.setdefault("semantic_type_hint", canonical_target_for_column(column))
        column_profile.setdefault("sample_values", preview_samples.get(column, []))
        column_profile.setdefault("distinct_count", len(preview_samples.get(column, [])))
        merged_columns.append(column_profile)

    merged_profile = {
        "source_columns": source_columns,
        "normalized_columns": {column: normalize_semantic_name(column) for column in source_columns},
        "detected_types": {str(key): str(value) for key, value in detected_types.items()},
        "semantic_roles": semantic_roles,
        "preview_values": preview_samples,
        "columns": merged_columns,
    }
    if isinstance(data_profile, dict):
        for key in ("file_fingerprint", "profiler_version", "profile_status", "row_count", "preview_row_count"):
            if key in data_profile:
                merged_profile[key] = data_profile[key]
    return merged_profile


_SELECT_ALL_PATTERNS = (
    r"\breturn all columns\b",
    r"\bkeep all columns\b",
    r"\binclude every column\b",
    r"\bpreserve all columns\b",
    r"\bshow all fields\b",
    r"\bretain all fields\b",
    r"\bkeep every field\b",
)

_RETURN_ROWS_PATTERNS = (
    r"\breturn\s+(?:me\s+)?(?:the\s+)?(?:only\s+)?(?:the\s+)?rows\b",
    r"\bgive\s+(?:me\s+)?(?:the\s+)?rows\b",
    r"\bshow\s+(?:me\s+)?(?:the\s+)?rows\b",
    r"\bget\s+(?:me\s+)?(?:the\s+)?rows\b",
    r"\breturn\s+(?:me\s+)?(?:the\s+)?data\b",
    r"\breturn\s+(?:me\s+)?(?:the\s+)?records\b",
    r"\bshow\s+(?:it\s+)?as\s+a\s+(?:bar|pie|line|scatter)\s*(?:chart|graph)?\b",
    r"\b(?:bar|pie|pi|line|scatter)\s+(?:chart|graph)\b",
    r"\b(?:generate|create)\s+(?:a\s+|the\s+)?(?:bar|pie|pi|line|scatter)\s+(?:chart|graph)\b",
    r"\bvisuali[sz]e\b",
)


def _repair_select_all_projection(
    canonical_intent: dict[str, Any],
    *,
    instruction: str,
    source_columns: list[str],
) -> dict[str, Any]:
    if not isinstance(canonical_intent, dict):
        return canonical_intent
    normalized_instruction = _normalize_text(instruction)

    explicit_all = any(re.search(pattern, normalized_instruction) for pattern in _SELECT_ALL_PATTERNS)
    implicit_all = any(re.search(pattern, normalized_instruction, re.IGNORECASE) for pattern in _RETURN_ROWS_PATTERNS)

    if not explicit_all and not implicit_all:
        return canonical_intent

    actions = canonical_intent.get("actions")
    if not isinstance(actions, list):
        return canonical_intent

    repaired = False
    actions_to_remove: list[int] = []
    for idx, action in enumerate(actions):
        if not isinstance(action, dict) or str(action.get("kind", "")).strip() != "project_columns":
            continue
        fields = action.get("requested_fields")
        if not isinstance(fields, list):
            continue
        for field in fields:
            if not isinstance(field, dict):
                continue
            raw_reference = _normalize_text(str(field.get("raw_reference", "")))
            if raw_reference not in {"all", "all columns", "every column", "all fields", "every field", "everything"}:
                continue

            if implicit_all and not explicit_all:
                actions_to_remove.append(idx)
                repaired = True
            else:
                field["raw_reference"] = "all columns"
                field["resolution_method"] = "all_columns"
                field["selection_mode"] = "semantic_family"
                field["resolved_column"] = None
                field["resolved_columns"] = [str(column) for column in source_columns if str(column).strip()]
                field["candidate_columns"] = [str(column) for column in source_columns if str(column).strip()]
                repaired = True

    if actions_to_remove:
        canonical_intent["actions"] = [a for i, a in enumerate(actions) if i not in actions_to_remove]

    if not repaired:
        return canonical_intent

    canonical_intent["resolution_status"] = "resolved"
    evidence = canonical_intent.get("evidence")
    if not isinstance(evidence, list):
        evidence = []
    evidence.append("Recognized an explicit select-all projection request.")
    canonical_intent["evidence"] = _dedupe_preserve_order(evidence)

    repair_notes = canonical_intent.get("repair_notes")
    if not isinstance(repair_notes, list):
        repair_notes = []
    repair_notes.append("Expanded explicit select-all wording to a deterministic universal projection.")
    canonical_intent["repair_notes"] = _dedupe_preserve_order(repair_notes)
    return canonical_intent


_GENERIC_FIELD_REFERENCES = {
    "column",
    "columns",
    "field",
    "fields",
    "value",
    "values",
    "entry",
    "entries",
    "row",
    "rows",
    "record",
    "records",
    "which",
    "that",
    "education",
    "status",
    "state",
    "type",
    "category",
    "level",
    "kind",
}

_PAYMENT_VALUE_HINTS = {
    "paypal",
    "pay",
    "cash",
    "card",
    "credit",
    "debit",
    "upi",
    "wallet",
    "bank",
    "transfer",
    "visa",
    "mastercard",
}

_STATUS_VALUE_HINTS = {
    "pending",
    "completed",
    "complete",
    "failed",
    "approved",
    "rejected",
    "declined",
    "processing",
    "open",
    "closed",
    "cancelled",
    "canceled",
}

def _repair_profile_grounded_references(
    canonical_intent: dict[str, Any],
    dataframe_profile: dict[str, Any],
    *,
    submission_id: str = "",
) -> dict[str, Any]:
    if not isinstance(canonical_intent, dict):
        return canonical_intent

    actions = canonical_intent.get("actions")
    if not isinstance(actions, list) or not isinstance(dataframe_profile, dict):
        return canonical_intent

    repaired_any = False
    unresolved_before = _count_unresolved_action_references(actions)

    for action in actions:
        if not isinstance(action, dict) or str(action.get("kind", "")).strip() != "filter_rows":
            continue
        conditions = action.get("conditions")
        if not isinstance(conditions, list):
            continue
        for condition in conditions:
            if not isinstance(condition, dict):
                continue
            field = condition.get("field")
            if not isinstance(field, dict):
                continue
            if field.get("resolved_column") or field.get("resolved_columns"):
                continue
            grounded = _ground_filter_field_from_profile(
                field=field,
                operator=str(condition.get("operator", "eq")).strip(),
                value=condition.get("value"),
                dataframe_profile=dataframe_profile,
                submission_id=submission_id,
            )
            if grounded is None:
                continue
            condition["field"] = grounded
            repaired_any = True

    if not repaired_any:
        return canonical_intent

    unresolved_after = _count_unresolved_action_references(actions)
    ambiguous_after = _count_ambiguous_action_references(actions)
    repair_notes = canonical_intent.get("repair_notes")
    if not isinstance(repair_notes, list):
        repair_notes = []
        canonical_intent["repair_notes"] = repair_notes
    repair_notes.append("Resolved one or more generic filter references using schema and preview-value evidence.")
    canonical_intent["repair_notes"] = _dedupe_preserve_order(repair_notes)

    if unresolved_after < unresolved_before:
        evidence = canonical_intent.get("evidence")
        if not isinstance(evidence, list):
            evidence = []
            canonical_intent["evidence"] = evidence
        evidence.append("Profile-aware grounding resolved previously generic filter references.")
        canonical_intent["evidence"] = _dedupe_preserve_order(evidence)

    if (
        unresolved_after == 0
        and ambiguous_after == 0
        and str(canonical_intent.get("resolution_status", "")).strip() == "needs_clarification"
    ):
        canonical_intent["resolution_status"] = "repaired"

    return canonical_intent


def _repair_filter_mode(
    canonical_intent: dict[str, Any],
    *,
    instruction: str,
) -> dict[str, Any]:
    """Fix filter mode when user says 'remove/drop/delete rows' but LLM produced mode='keep'.

    Also removes spurious drop_columns actions when the user clearly says
    'remove rows' (the LLM sometimes confuses 'remove rows having X' with
    'drop column X').
    """
    if not isinstance(canonical_intent, dict):
        return canonical_intent

    normalized = _normalize_text(instruction)

    # Detect "remove/drop/delete rows/records/entries WHERE condition" language
    is_row_removal = bool(re.search(
        r"\b(?:remove|drop|delete|exclude|discard|filter\s+out)\s+"
        r"(?:the\s+)?(?:rows?|records?|entries?|data|values?)?\s*"
        r"(?:which|that|where|having|with|containing|contains)",
        normalized,
        re.IGNORECASE,
    ))

    if not is_row_removal:
        return canonical_intent

    actions = canonical_intent.get("actions")
    if not isinstance(actions, list):
        return canonical_intent

    # Fix filter mode
    for action in actions:
        if not isinstance(action, dict) or action.get("kind") != "filter_rows":
            continue
        if action.get("mode") == "keep":
            # Check if operators are already inverted (neq, not_in) — if so,
            # the LLM already expressed removal semantics, just flip mode without
            # creating double negation
            conditions = action.get("conditions", [])
            has_negated_ops = any(
                isinstance(c, dict) and c.get("operator") in ("neq", "not_in", "not_contains")
                for c in conditions
            )
            if has_negated_ops:
                # LLM used negated operators — flip them to positive + set mode=drop
                for c in conditions:
                    if isinstance(c, dict):
                        op = c.get("operator")
                        if op == "neq":
                            c["operator"] = "eq"
                        elif op == "not_in":
                            c["operator"] = "in"
                        elif op == "not_contains":
                            c["operator"] = "contains"
            action["mode"] = "drop"

    # Remove spurious drop_columns when user said "remove rows"
    # (LLM confused "remove rows having X as education" with "drop education column")
    canonical_intent["actions"] = [
        a for a in actions
        if not (isinstance(a, dict) and a.get("kind") == "drop_columns")
    ]

    return canonical_intent


def _repair_missing_clean_action(
    canonical_intent: dict[str, Any],
    *,
    instruction: str,
) -> dict[str, Any]:
    """Inject a safe_default clean action if the user explicitly requests cleaning
    but the semantic pipeline didn't extract one."""
    if not isinstance(canonical_intent, dict):
        return canonical_intent

    normalized = _normalize_text(instruction)
    if not re.search(
        r"\b(?:clean(?:\s+(?:the|my|this))?\s+(?:data|file|sheet|table|dataset)|"
        r"clean\s+up|cleanup|cleanse|sanitize|standardize|normalise|normalize)\b",
        normalized,
        re.IGNORECASE,
    ):
        return canonical_intent

    actions = canonical_intent.get("actions")
    if not isinstance(actions, list):
        return canonical_intent

    for action in actions:
        if isinstance(action, dict) and str(action.get("kind", "")).strip() == "clean":
            return canonical_intent

    clean_action = {"kind": "clean", "mode": "safe_default", "operations": []}
    actions.insert(0, clean_action)
    canonical_intent["actions"] = actions

    evidence = canonical_intent.get("evidence")
    if not isinstance(evidence, list):
        evidence = []
    evidence.append("Injected clean action: user explicitly requested data cleaning.")
    canonical_intent["evidence"] = evidence

    return canonical_intent


def _repair_missing_clean_operations(
    canonical_intent: dict[str, Any],
    *,
    instruction: str,
) -> dict[str, Any]:
    """Inject specific cleaning operations that were mentioned in the instruction
    but not captured in the existing clean action's operations list."""
    if not isinstance(canonical_intent, dict):
        return canonical_intent

    normalized = _normalize_text(instruction)
    actions = canonical_intent.get("actions")
    if not isinstance(actions, list):
        return canonical_intent

    # Find existing clean action
    clean_action = None
    for action in actions:
        if isinstance(action, dict) and action.get("kind") == "clean":
            clean_action = action
            break

    if clean_action is None:
        return canonical_intent

    operations = clean_action.get("operations")
    if not isinstance(operations, list):
        operations = []
        clean_action["operations"] = operations

    existing_names = {op.get("name") for op in operations if isinstance(op, dict)}

    # Check for "remove negative signs" / "absolute value"
    if "absolute_value" not in existing_names:
        if re.search(
            r"\b(?:remove|strip|delete|drop)\s+(?:the\s+)?(?:negative\s+(?:sign|value|number)|minus\s+sign)\b",
            normalized, re.IGNORECASE,
        ) or re.search(
            r"\b(?:make\s+(?:all\s+)?(?:values?\s+)?positive|absolute\s+value|remove\s+negatives?)\b",
            normalized, re.IGNORECASE,
        ):
            operations.append({"name": "absolute_value", "parameters": {"columns": "__all_numeric_columns__"}})

    # Check for "remove duplicates"
    if "drop_duplicates" not in existing_names:
        if re.search(r"\b(?:remove|drop|delete)\s+(?:the\s+)?duplicat", normalized, re.IGNORECASE):
            operations.append({"name": "drop_duplicates", "parameters": {}})

    # Check for "strip currency symbols"
    if "strip_currency_symbols" not in existing_names:
        if re.search(r"\b(?:strip|remove)\s+(?:the\s+)?(?:currency|dollar|rupee)\s*(?:symbol|sign)?\b", normalized, re.IGNORECASE):
            operations.append({"name": "strip_currency_symbols", "parameters": {}})

    # Check for "normalize values" / "normalize inconsistent values"
    if "normalize_categorical_values" not in existing_names:
        if re.search(r"\b(?:normalize|standardize)\s+(?:inconsistent\s+)?(?:values?|categories|categorical)\b", normalized, re.IGNORECASE):
            operations.append({"name": "normalize_categorical_values", "parameters": {"columns": "__all_string_columns__"}})

    if operations:
        clean_action["mode"] = "explicit"

    return canonical_intent


def _repair_missing_calculate_action(
    canonical_intent: dict[str, Any],
    *,
    instruction: str,
    source_columns: list[str] | None = None,
    dataframe_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Inject a calculate action if the user explicitly requests a calculation
    but the LLM didn't produce one.

    Detects patterns like:
    - "calculate average X by Y"
    - "calculate sum of X by Y"
    - "average X by Y"
    """
    if not isinstance(canonical_intent, dict):
        return canonical_intent

    actions = canonical_intent.get("actions")
    if not isinstance(actions, list):
        return canonical_intent

    # Check if calculate action already exists
    for action in actions:
        if isinstance(action, dict) and str(action.get("kind", "")).strip() == "calculate":
            return canonical_intent

    source_columns = source_columns or []
    col_lower_map = {c.lower(): c for c in source_columns}

    # Detect "calculate/compute average/sum/mean X by Y" patterns
    calc_match = re.search(
        r"\b(?:calculate|compute|find|get|show)\s+(?:the\s+)?(?P<agg>average|avg|mean|sum|total|count|median|min|max)"
        r"(?:\s+(?:of\s+)?)?(?P<col>[\w\s]+?)\s+(?:by|per|grouped?\s+by|for\s+each)\s+(?P<group>[\w\s]+?)(?:\s*[,.]|\s+and\b|$)",
        instruction,
        re.IGNORECASE,
    )
    if not calc_match:
        # Also try "average X by Y" without explicit "calculate"
        calc_match = re.search(
            r"\b(?P<agg>average|avg|mean|sum|total|count|median)\s+(?P<col>[\w\s]+?)\s+(?:by|per|grouped?\s+by|for\s+each)\s+(?P<group>[\w\s]+?)(?:\s*[,.]|\s+and\b|$)",
            instruction,
            re.IGNORECASE,
        )

    if not calc_match:
        return canonical_intent

    agg_name = calc_match.group("agg").strip().lower()
    col_text = calc_match.group("col").strip().lower()
    group_text = calc_match.group("group").strip().lower()

    # Map aggregation names
    agg_map = {"average": "mean", "avg": "mean", "mean": "mean", "sum": "sum", "total": "sum", "count": "count", "median": "median", "min": "min", "max": "max"}
    op_type = f"group_{agg_map.get(agg_name, 'mean')}"

    # Resolve column names against source_columns
    def _resolve_col(text: str) -> str | None:
        text = text.strip().replace(" ", "_")
        if text in col_lower_map:
            return col_lower_map[text]
        # Try partial match
        for col_lower, col_real in col_lower_map.items():
            if text in col_lower or col_lower in text:
                return col_real
        # Try word-by-word
        words = text.split("_")
        for col_lower, col_real in col_lower_map.items():
            if all(w in col_lower for w in words):
                return col_real
        return None

    resolved_col = _resolve_col(col_text)
    resolved_group = _resolve_col(group_text)

    if not resolved_col or not resolved_group:
        return canonical_intent

    # Build the calculate action with structured operations
    calc_action = {
        "kind": "calculate",
        "operations": [{
            "type": op_type,
            "column": resolved_col,
            "group_by": [resolved_group],
        }],
    }
    # Insert before visualize action if present, otherwise append
    insert_idx = len(actions)
    for i, action in enumerate(actions):
        if isinstance(action, dict) and action.get("kind") == "visualize":
            insert_idx = i
            break
    actions.insert(insert_idx, calc_action)

    evidence = canonical_intent.get("evidence")
    if not isinstance(evidence, list):
        evidence = []
    evidence.append(f"Injected calculate action: {op_type} of {resolved_col} by {resolved_group}.")
    canonical_intent["evidence"] = evidence

    return canonical_intent


def _extract_null_drop_columns(instruction: str, canonical_intent: dict) -> list[str] | None:
    """Extract specific column names for null dropping from the instruction.

    Detects patterns like:
    - "remove rows with missing values for education or credit score column"
    - "drop nulls in age and income columns"
    - "remove rows having null in gender column"

    Returns a list of column names if specific columns are mentioned,
    or None to drop nulls across all columns.
    """
    # Pattern: "missing/null values for/in <col1> or/and <col2> column(s)"
    col_match = re.search(
        r"\b(?:missing|null|empty|blank)\s+(?:values?\s+)?(?:for|in|from|of)\s+"
        r"(?:any\s+(?:these|those|the)\s+)?"
        r"(?P<cols>.+?)\s*(?:column|field|col)s?\b",
        instruction,
        re.IGNORECASE,
    )
    if not col_match:
        # Pattern: "drop nulls in <col1>, <col2>"
        col_match = re.search(
            r"\b(?:drop|remove)\s+(?:rows?\s+(?:with|having)\s+)?(?:null|missing|blank)s?\s+"
            r"(?:in|from|for)\s+(?P<cols>.+?)\s*(?:column|field|$)",
            instruction,
            re.IGNORECASE,
        )

    if not col_match:
        return None

    cols_text = col_match.group("cols").strip()
    # Split on "or", "and", ","
    raw_cols = re.split(r"\s+(?:or|and)\s+|,\s*", cols_text)
    raw_cols = [c.strip().lower().replace(" ", "_") for c in raw_cols if c.strip()]

    if not raw_cols:
        return None

    # Try to resolve against source columns in the dataframe profile
    source_columns = []
    profile = canonical_intent.get("dataframe_profile")
    if isinstance(profile, dict):
        src = profile.get("source_columns") or profile.get("columns")
        if isinstance(src, list):
            source_columns = [str(c) for c in src]
        elif isinstance(src, list) and src and isinstance(src[0], dict):
            source_columns = [str(c.get("name", "")) for c in src if isinstance(c, dict)]

    if not source_columns:
        # Return raw extracted column names
        return raw_cols if raw_cols else None

    col_lower_map = {c.lower().replace(" ", "_"): c for c in source_columns}
    resolved = []
    for raw in raw_cols:
        # Exact match
        if raw in col_lower_map:
            resolved.append(col_lower_map[raw])
            continue
        # Partial match
        for key, real_name in col_lower_map.items():
            if raw in key or key in raw:
                resolved.append(real_name)
                break

    return resolved if resolved else None


def _repair_null_row_cleanup(
    canonical_intent: dict[str, Any],
    *,
    instruction: str,
) -> dict[str, Any]:
    if not isinstance(canonical_intent, dict):
        return canonical_intent

    normalized_instruction = _normalize_text(instruction)
    if not _looks_like_null_row_cleanup(normalized_instruction):
        return canonical_intent

    actions = canonical_intent.get("actions")
    if not isinstance(actions, list):
        return canonical_intent

    repaired = False
    retained_actions: list[Any] = []
    clean_action: dict[str, Any] | None = None

    for action in actions:
        if not isinstance(action, dict):
            retained_actions.append(action)
            continue

        kind = str(action.get("kind", "")).strip()
        if kind == "drop_columns":
            repaired = True
            continue
        if kind == "clean" and clean_action is None:
            clean_action = action
        retained_actions.append(action)

    if clean_action is None:
        clean_action = {
            "kind": "clean",
            "mode": "explicit",
            "operations": [],
        }
        retained_actions.insert(0, clean_action)
        repaired = True

    operations = clean_action.get("operations")
    if not isinstance(operations, list):
        operations = []
        clean_action["operations"] = operations

    if not any(isinstance(op, dict) and str(op.get("name", "")).strip() == "drop_nulls" for op in operations):
        # Try to extract specific columns for null checking from the instruction
        null_columns = _extract_null_drop_columns(normalized_instruction, canonical_intent)
        operations.append(
            {
                "name": "drop_nulls",
                "parameters": {"columns": null_columns, "how": "any"},
            }
        )
        repaired = True

    clean_action["mode"] = "explicit"
    canonical_intent["actions"] = retained_actions
    canonical_intent["decision"] = _build_decision_summary(retained_actions)

    if repaired:
        canonical_intent["resolution_status"] = "repaired"
        evidence = canonical_intent.get("evidence")
        if not isinstance(evidence, list):
            evidence = []
        evidence.append("Interpreted null/empty row cleanup as a drop_nulls cleaning request.")
        canonical_intent["evidence"] = _dedupe_preserve_order(evidence)

        repair_notes = canonical_intent.get("repair_notes")
        if not isinstance(repair_notes, list):
            repair_notes = []
        repair_notes.append("Repaired row-level null cleanup into an explicit clean(drop_nulls) action.")
        canonical_intent["repair_notes"] = _dedupe_preserve_order(repair_notes)

    return canonical_intent


def _count_unresolved_action_references(actions: list[Any]) -> int:
    unresolved = 0
    for action in actions:
        if not isinstance(action, dict):
            continue
        kind = str(action.get("kind", "")).strip()
        if kind in {"project_columns", "drop_columns"}:
            for field in action.get("requested_fields", []):
                if isinstance(field, dict) and not field.get("resolved_column") and not field.get("resolved_columns"):
                    unresolved += 1
        elif kind == "filter_rows":
            for condition in action.get("conditions", []):
                if not isinstance(condition, dict):
                    continue
                field = condition.get("field")
                if isinstance(field, dict) and not field.get("resolved_column") and not field.get("resolved_columns"):
                    unresolved += 1
        elif kind == "rename_columns":
            for mapping in action.get("mapping", []):
                if not isinstance(mapping, dict):
                    continue
                source = mapping.get("source")
                if isinstance(source, dict) and not source.get("resolved_column") and not source.get("resolved_columns"):
                    unresolved += 1
        elif kind == "sort_rows":
            for item in action.get("sort_keys", []):
                if not isinstance(item, dict):
                    continue
                column = item.get("column")
                if isinstance(column, dict) and not column.get("resolved_column") and not column.get("resolved_columns"):
                    unresolved += 1
    return unresolved


def _count_ambiguous_action_references(actions: list[Any]) -> int:
    ambiguous = 0
    for action in actions:
        if not isinstance(action, dict):
            continue
        kind = str(action.get("kind", "")).strip()
        if kind in {"project_columns", "drop_columns"}:
            for field in action.get("requested_fields", []):
                if isinstance(field, dict) and str(field.get("selection_mode", "")).strip() == "ambiguous":
                    ambiguous += 1
        elif kind == "filter_rows":
            for condition in action.get("conditions", []):
                if not isinstance(condition, dict):
                    continue
                field = condition.get("field")
                if isinstance(field, dict) and str(field.get("selection_mode", "")).strip() == "ambiguous":
                    ambiguous += 1
        elif kind == "rename_columns":
            for mapping in action.get("mapping", []):
                if not isinstance(mapping, dict):
                    continue
                source = mapping.get("source")
                if isinstance(source, dict) and str(source.get("selection_mode", "")).strip() == "ambiguous":
                    ambiguous += 1
        elif kind == "sort_rows":
            for item in action.get("sort_keys", []):
                if not isinstance(item, dict):
                    continue
                column = item.get("column")
                if isinstance(column, dict) and str(column.get("selection_mode", "")).strip() == "ambiguous":
                    ambiguous += 1
    return ambiguous


def _ground_filter_field_from_profile(
    *,
    field: dict[str, Any],
    operator: str,
    value: Any,
    dataframe_profile: dict[str, Any],
    submission_id: str = "",
) -> dict[str, Any] | None:
    raw_reference = str(field.get("raw_reference", "")).strip()
    source_columns = [
        str(column)
        for column in dataframe_profile.get("source_columns", [])
        if str(column).strip()
    ]
    if not raw_reference or not source_columns:
        return None

    role_columns = dataframe_profile.get("semantic_roles", {})
    detected_types = {
        str(key): str(val).strip().lower()
        for key, val in (dataframe_profile.get("detected_types") or {}).items()
    }

    requested_tokens = set(_normalize_reference(raw_reference).split())
    generic_reference = not requested_tokens or requested_tokens <= _GENERIC_FIELD_REFERENCES
    generic_field_tokens = requested_tokens & _GENERIC_FIELD_REFERENCES
    value_strings = _flatten_filter_values(value)
    value_tokens = _value_tokens(value_strings)
    compact_values = {_compact_token(value_string) for value_string in value_strings if _compact_token(value_string)}
    value_concepts = _value_concepts(value_tokens, value_strings)

    scored: list[_FilterFieldGroundingCandidate] = []
    for column in source_columns:
        column_profile = _profile_column_metadata(dataframe_profile, column)
        observed_values = _profile_observed_values(dataframe_profile, column)
        observed_normalized = {_normalize_reference(item) for item in observed_values if _normalize_reference(item)}
        observed_compact = {_compact_token(item) for item in observed_values if _compact_token(item)}
        observed_tokens = _value_tokens(observed_values)
        column_normalized = normalize_semantic_name(column)
        column_tokens = {token for token in _normalize_reference(column).split() if token}

        exact_value_matches = 0
        matched_values: list[str] = []
        for requested in value_strings:
            requested_normalized = _normalize_reference(requested)
            requested_compact = _compact_token(requested)
            matched_observed_value = None
            if requested_normalized:
                for observed in observed_values:
                    if _normalize_reference(observed) == requested_normalized:
                        matched_observed_value = observed
                        break
            if matched_observed_value is None and requested_compact:
                for observed in observed_values:
                    if _compact_token(observed) == requested_compact:
                        matched_observed_value = observed
                        break
            if matched_observed_value is not None:
                exact_value_matches += 1
                matched_values.append(matched_observed_value)

        requested_value_count = len(value_strings)
        value_coverage = (exact_value_matches / requested_value_count) if requested_value_count else 0.0

        semantic_role_score = 0.0
        column_roles = {
            role
            for role, values in role_columns.items()
            if isinstance(values, list) and column in values
        }
        if "payment" in value_concepts:
            if {"merchant", "payment_value"} & column_roles:
                semantic_role_score += 0.35
            if "payment" in column_normalized or "method" in column_normalized or "gateway" in column_normalized:
                semantic_role_score += 0.20
            if column_profile.get("semantic_type_hint") in {"payment_method", "merchant", "payment_value"}:
                semantic_role_score += 0.25
        if "status" in value_concepts:
            if "status" in column_roles or "status" in column_normalized:
                semantic_role_score += 0.35
            if column_profile.get("semantic_type_hint") in {"status", "payment_status"}:
                semantic_role_score += 0.20
        if "date" in value_concepts:
            if "date" in column_roles or detected_types.get(column) == "date":
                semantic_role_score += 0.35
            if column_profile.get("semantic_type_hint") == "date":
                semantic_role_score += 0.15
        if "numeric" in value_concepts and detected_types.get(column) == "number":
            semantic_role_score += 0.20
        if observed_tokens & value_tokens:
            semantic_role_score += 0.10
        if generic_field_tokens and any(token in column_normalized for token in generic_field_tokens):
            semantic_role_score += 0.35
        if generic_field_tokens and column_profile.get("semantic_type_hint") in generic_field_tokens:
            semantic_role_score += 0.20

        column_name_similarity = 0.0
        if not generic_reference:
            overlap = requested_tokens & set(column_normalized.replace("_", " ").split())
            if overlap:
                column_name_similarity += 0.30
            semantic_aliases = {
                _normalize_reference(alias)
                for alias in _semantic_role_aliases(column_normalized)
            }
            alias_overlap = requested_tokens & {token for alias in semantic_aliases for token in alias.split()}
            if alias_overlap:
                column_name_similarity += 0.20
        elif value_coverage > 0:
            # Generic field references should only auto-resolve on value evidence
            # or strong semantic role evidence, not on fuzzy name matching alone.
            column_name_similarity += 0.05 if column_tokens else 0.0

        type_compatibility_score = 0.0
        detected_type = detected_types.get(column, "")
        if requested_value_count:
            if detected_type in {"string", "object", "category", ""}:
                type_compatibility_score += 0.05
            if "payment" in value_concepts and detected_type in {"string", "object", "category", ""}:
                type_compatibility_score += 0.10
            if "status" in value_concepts and detected_type in {"string", "object", "category", ""}:
                type_compatibility_score += 0.10
            if "date" in value_concepts and detected_type in {"date", "datetime", "string", "object", ""}:
                type_compatibility_score += 0.10
            if "numeric" in value_concepts and detected_type == "number":
                type_compatibility_score += 0.10

        final_score = 0.0
        if requested_value_count:
            final_score += min(0.70, value_coverage * 0.70)
        final_score += semantic_role_score
        final_score += column_name_similarity
        final_score += type_compatibility_score
        if operator == "contains" and detected_type in {"string", "object", "category", ""}:
            final_score += 0.05
        if operator in {"in", "not_in"}:
            final_score += 0.05

        resolution_reason = "insufficient_evidence"
        if exact_value_matches and value_coverage >= 1.0:
            resolution_reason = "observed_value_unique_match"
        elif exact_value_matches:
            resolution_reason = "observed_value_match"
        elif semantic_role_score >= 0.35:
            resolution_reason = "semantic_role_match"
        elif column_name_similarity >= 0.30:
            resolution_reason = "column_name_match"

        scored.append(
            _FilterFieldGroundingCandidate(
                column=column,
                exact_value_matches=exact_value_matches,
                requested_value_count=requested_value_count,
                value_coverage=value_coverage,
                observed_value_matches=matched_values,
                semantic_role_score=semantic_role_score,
                column_name_similarity=column_name_similarity,
                type_compatibility_score=type_compatibility_score,
                final_score=final_score,
                resolution_reason=resolution_reason,
            )
        )

    scored.sort(key=lambda item: (item.final_score, item.exact_value_matches, item.value_coverage), reverse=True)
    if not scored:
        return None

    def _supports_value_resolution(candidate: _FilterFieldGroundingCandidate) -> bool:
        if requested_value_count <= 0:
            return False
        if candidate.exact_value_matches <= 0:
            return False
        if requested_value_count == 1:
            value_match = candidate.exact_value_matches >= 1
        else:
            value_match = candidate.exact_value_matches >= requested_value_count and candidate.value_coverage >= 1.0
        semantic_match = candidate.semantic_role_score >= 0.35 or candidate.column_name_similarity >= 0.30
        type_match = candidate.type_compatibility_score > 0
        return value_match and semantic_match and type_match

    supported_candidates = [candidate for candidate in scored if _supports_value_resolution(candidate)]

    if requested_value_count > 0:
        if len(supported_candidates) == 1:
            best = supported_candidates[0]
            try:
                log_runtime_event(
                    "profile_grounding_decision",
                    service="backend",
                    submission_id=submission_id,
                    operator=operator,
                    candidate_count=len(scored),
                    requested_value_count=best.requested_value_count,
                    exact_value_matches=best.exact_value_matches,
                    value_coverage=round(best.value_coverage, 3),
                    best_score=round(best.final_score, 3),
                    second_score=0.0,
                    resolution_reason=best.resolution_reason,
                    generic_reference=generic_reference,
                )
            except Exception:
                pass
            grounded_value = best.observed_value_matches[0] if best.observed_value_matches else (value_strings[0] if value_strings else None)
            return {
                **field,
                "resolved_column": best.column,
                "resolved_columns": [best.column],
                "grounded_value": grounded_value,
                "candidate_columns": [candidate.column for candidate in supported_candidates],
                "selection_mode": "single",
                "resolution_method": "profile_value_evidence",
                "evidence": _dedupe_preserve_order(
                    list(field.get("evidence", []))
                    + [
                        f"profile_match={best.resolution_reason}",
                        f"candidate_count={len(supported_candidates)}",
                    ]
                ),
            }

        if len(supported_candidates) > 1:
            try:
                log_runtime_event(
                    "profile_grounding_decision",
                    service="backend",
                    submission_id=submission_id,
                    operator=operator,
                    candidate_count=len(scored),
                    requested_value_count=supported_candidates[0].requested_value_count,
                    exact_value_matches=supported_candidates[0].exact_value_matches,
                    value_coverage=round(supported_candidates[0].value_coverage, 3),
                    best_score=round(supported_candidates[0].final_score, 3),
                    second_score=round(supported_candidates[1].final_score, 3),
                    resolution_reason="ambiguous",
                    generic_reference=generic_reference,
                )
            except Exception:
                pass
            return {
                **field,
                "resolved_column": None,
                "resolved_columns": [],
                "grounded_value": None,
                "candidate_columns": [candidate.column for candidate in supported_candidates],
                "selection_mode": "ambiguous",
                "resolution_method": "profile_value_ambiguous",
                "candidates": [
                    {
                        "column": candidate.column,
                        "matched_value": candidate.observed_value_matches[0] if candidate.observed_value_matches else None,
                    }
                    for candidate in supported_candidates
                ],
                "evidence": _dedupe_preserve_order(
                    list(field.get("evidence", []))
                    + [
                        "Multiple candidate columns matched the requested value evidence.",
                        f"candidate_count={len(supported_candidates)}",
                    ]
                ),
            }

    best = scored[0]
    second = scored[1] if len(scored) > 1 else None

    if requested_value_count > 0 and best.exact_value_matches > 0 and best.final_score >= 0.45:
        try:
            log_runtime_event(
                "profile_grounding_decision",
                service="backend",
                submission_id=submission_id,
                operator=operator,
                candidate_count=len(scored),
                requested_value_count=best.requested_value_count,
                exact_value_matches=best.exact_value_matches,
                value_coverage=round(best.value_coverage, 3),
                best_score=round(best.final_score, 3),
                second_score=round(second.final_score, 3) if second else 0.0,
                resolution_reason=best.resolution_reason,
                generic_reference=generic_reference,
            )
        except Exception:
            pass
        return {
            **field,
            "resolved_column": best.column,
            "resolved_columns": [best.column],
            "grounded_value": best.observed_value_matches[0] if best.observed_value_matches else (value_strings[0] if value_strings else None),
            "candidate_columns": [candidate.column for candidate in scored[:3]],
            "selection_mode": "single",
            "resolution_method": "profile_value_evidence",
            "evidence": _dedupe_preserve_order(
                list(field.get("evidence", []))
                + [
                    f"profile_match={best.resolution_reason}",
                    f"candidate_count={len(scored)}",
                ]
            ),
        }

    if requested_value_count > 0 and generic_reference and best.exact_value_matches == 0:
        return None

    decisive_semantic_match = (
        best.final_score >= 0.60
        and (second is None or (best.final_score - second.final_score) >= 0.15)
    )

    if decisive_semantic_match:
        try:
            log_runtime_event(
                "profile_grounding_decision",
                service="backend",
                submission_id=submission_id,
                operator=operator,
                candidate_count=len(scored),
                requested_value_count=best.requested_value_count,
                exact_value_matches=best.exact_value_matches,
                value_coverage=round(best.value_coverage, 3),
                best_score=round(best.final_score, 3),
                second_score=round(second.final_score, 3) if second else 0.0,
                resolution_reason=best.resolution_reason,
                generic_reference=generic_reference,
            )
        except Exception:
            pass
        return {
            **field,
            "resolved_column": best.column,
            "resolved_columns": [best.column],
            "grounded_value": best.observed_value_matches[0] if best.observed_value_matches else None,
            "candidate_columns": [candidate.column for candidate in scored[:3]],
            "selection_mode": "single",
            "resolution_method": "profile_semantic_match",
            "evidence": _dedupe_preserve_order(
                list(field.get("evidence", []))
                + [
                    f"profile_match={best.resolution_reason}",
                    f"candidate_count={len(scored)}",
                ]
            ),
        }

    try:
        log_runtime_event(
            "profile_grounding_decision",
            service="backend",
            submission_id=submission_id,
            operator=operator,
            candidate_count=len(scored),
            requested_value_count=best.requested_value_count,
            exact_value_matches=best.exact_value_matches,
            value_coverage=round(best.value_coverage, 3),
            best_score=round(best.final_score, 3),
            second_score=round(second.final_score, 3) if second else 0.0,
            resolution_reason="ambiguous",
            generic_reference=generic_reference,
        )
    except Exception:
        pass

    if best.final_score < 0.45:
        return None

    return {
        **field,
        "resolved_column": best.column,
        "resolved_columns": [best.column],
        "grounded_value": best.observed_value_matches[0] if best.observed_value_matches else None,
        "candidate_columns": [candidate.column for candidate in scored[:3]],
        "selection_mode": "single",
        "resolution_method": "profile_semantic_match",
        "evidence": _dedupe_preserve_order(
            list(field.get("evidence", []))
            + [f"profile_match={best.resolution_reason}", f"candidate_count={len(scored)}"]
        ),
    }


def _profile_column_metadata(dataframe_profile: dict[str, Any], column: str) -> dict[str, Any]:
    for item in dataframe_profile.get("columns", []):
        if not isinstance(item, dict):
            continue
        if str(item.get("name", "")).strip() == column:
            return item
    return {}


def _profile_observed_values(dataframe_profile: dict[str, Any], column: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()

    def _append(raw: Any) -> None:
        text = str(raw).strip()
        if not text:
            return
        marker = _compact_token(text) or text.lower()
        if marker in seen:
            return
        seen.add(marker)
        values.append(text)

    preview_values = dataframe_profile.get("preview_values", {})
    if isinstance(preview_values, dict):
        for item in preview_values.get(column, []):
            _append(item)

    column_profile = _profile_column_metadata(dataframe_profile, column)
    sample_values = column_profile.get("sample_values", [])
    if isinstance(sample_values, list):
        for item in sample_values:
            _append(item)

    return values


def _flatten_filter_values(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _compact_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _value_tokens(values: list[str]) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        normalized = _normalize_reference(value)
        if normalized:
            tokens.update(token for token in normalized.split() if token)
        compact = _compact_token(value)
        if compact:
            tokens.add(compact)
    return tokens


def _value_concepts(value_tokens: set[str], value_strings: list[str]) -> set[str]:
    concepts: set[str] = set()
    if value_tokens & _PAYMENT_VALUE_HINTS:
        concepts.add("payment")
    if value_tokens & _STATUS_VALUE_HINTS:
        concepts.add("status")
    if any(re.fullmatch(r"-?\d+(?:\.\d+)?", item) for item in value_strings):
        concepts.add("numeric")
    if any(re.fullmatch(r"\d{4}-\d{2}-\d{2}", item) for item in value_strings):
        concepts.add("date")
    if value_strings:
        concepts.add("text")
    return concepts


def _extract_requested_columns(
    normalized_prompt: str,
    source_columns: list[str],
    role_columns: dict[str, list[str]],
    dataframe_profile: dict[str, Any],
) -> list[UnresolvedColumnReference]:
    references = list(_iter_column_reference_candidates(normalized_prompt, source_columns, role_columns))
    if references:
        referenced_columns = {ref.resolved_column for ref in references if ref.resolved_column}
        if _OUTPUT_RESTRICTION_RE.search(normalized_prompt):
            stripped = _OUTPUT_RESTRICTION_RE.sub("", normalized_prompt)
            stripped = _strip_noise_tokens(stripped)
            for fragment in re.split(r"\s*,\s*|\s+and\s+", stripped):
                fragment = _strip_noise_tokens(fragment)
                if not fragment:
                    continue
                grounded = _ground_column_reference(fragment, source_columns, role_columns)
                if grounded is None:
                    continue
                if grounded.resolved_column and grounded.resolved_column in referenced_columns:
                    continue
                references.append(grounded)
                if grounded.resolved_column:
                    referenced_columns.add(grounded.resolved_column)
        return _dedupe_references(references)

    if _OUTPUT_RESTRICTION_RE.search(normalized_prompt):
        # Fallback for prompts such as "customer id only" where the text is
        # short and the groundable field is the whole prompt fragment.
        stripped = normalized_prompt
        stripped = _OUTPUT_RESTRICTION_RE.sub("", stripped)
        stripped = _strip_noise_tokens(stripped)
        grounded = _ground_column_reference(stripped, source_columns, role_columns)
        if grounded is not None:
            return [grounded]
    return []


def _iter_column_reference_candidates(
    normalized_prompt: str,
    source_columns: list[str],
    role_columns: dict[str, list[str]],
) -> list[UnresolvedColumnReference]:
    seen: set[str] = set()
    candidates: list[UnresolvedColumnReference] = []
    aliases = _build_column_alias_index(source_columns)
    for alias, columns in aliases.items():
        if not _phrase_in_prompt(alias, normalized_prompt):
            continue
        for column in columns:
            if column in seen:
                continue
            seen.add(column)
            candidates.append(
                UnresolvedColumnReference(
                    raw_reference=alias,
                    resolved_column=column,
                    resolution_method="exact_name" if normalize_semantic_name(alias) == normalize_semantic_name(column) else "alias_match",
                )
            )

    # Try semantic roles for phrases like "merchant" or "payment method".
    for role, values in role_columns.items():
        for alias in _ROLE_ALIASES.get(role, {role}):
            if not _phrase_in_prompt(alias, normalized_prompt):
                continue
            if len(values) == 1:
                column = values[0]
                if column in seen:
                    continue
                seen.add(column)
                candidates.append(
                    UnresolvedColumnReference(
                        raw_reference=alias,
                        resolved_column=column,
                        resolution_method="semantic_role",
                        selection_mode="single",
                    )
                )
            elif len(values) > 1:
                candidate_columns = [column for column in values if column not in seen]
                if not candidate_columns:
                    continue
                seen.update(candidate_columns)
                candidates.append(
                    UnresolvedColumnReference(
                        raw_reference=alias,
                        resolution_method="semantic_role",
                        selection_mode="ambiguous",
                        resolved_columns=candidate_columns,
                        candidate_columns=candidate_columns,
                        evidence=[f"Semantic role {role!r} matches multiple columns."],
                    )
                )

    return _dedupe_references(candidates)


def _parse_filter_clause(
    clause: str,
    source_columns: list[str],
    role_columns: dict[str, list[str]],
    dataframe_profile: dict[str, Any],
) -> FilterCondition | None:
    clause = _strip_noise_tokens(clause)
    if not clause:
        return None

    contains_value_first = re.match(
        r"^(?P<op>contains?|with)\s+(?P<value>.+?)\s+as\s+(?:a\s+)?(?P<field>.+)$",
        clause,
        flags=re.IGNORECASE,
    )
    if contains_value_first:
        field_ref = _ground_column_reference(contains_value_first.group("field"), source_columns, role_columns)
        value = _coerce_filter_value(contains_value_first.group("value"))
        operator = "in" if isinstance(value, list) and len(value) > 1 else "eq"
        if field_ref is None:
            field_ref = UnresolvedColumnReference(
                raw_reference=_strip_noise_tokens(contains_value_first.group("field")),
                resolution_method="unresolved",
                selection_mode="ambiguous" if _normalize_reference(contains_value_first.group("field")) else None,
            )
        return FilterCondition(field=field_ref, operator=operator, value=value)

    value_first = re.match(r"^(?P<value>.+?)\s+as\s+(?:a\s+)?(?P<field>.+)$", clause, flags=re.IGNORECASE)
    if value_first:
        field_ref = _ground_column_reference(value_first.group("field"), source_columns, role_columns)
        value = _coerce_filter_value(value_first.group("value"))
        operator = "in" if isinstance(value, list) and len(value) > 1 else "eq"
        if field_ref is None:
            field_ref = UnresolvedColumnReference(
                raw_reference=_strip_noise_tokens(value_first.group("field")),
                resolution_method="unresolved",
                selection_mode="ambiguous" if _normalize_reference(value_first.group("field")) else None,
            )
        return FilterCondition(field=field_ref, operator=operator, value=value)

    for pattern, operator in (
        (r"^(?P<field>.+?)\s+(?:is|equals?|equal to|=|:)\s*(?P<value>.+)$", "eq"),
        (r"^(?P<field>.+?)\s+(?:not equal to|does not equal|!=|<>)\s*(?P<value>.+)$", "neq"),
        (r"^(?P<field>.+?)\s*(?:>=|at least|not less than)\s*(?P<value>.+)$", "gte"),
        (r"^(?P<field>.+?)\s*(?:<=|at most|not more than)\s*(?P<value>.+)$", "lte"),
        (r"^(?P<field>.+?)\s*(?:>|greater than|above)\s*(?P<value>.+)$", "gt"),
        (r"^(?P<field>.+?)\s*(?:<|less than|below)\s*(?P<value>.+)$", "lt"),
        (r"^(?P<field>.+?)\s+(?:contains?|matching|matches?)\s+(?P<value>.+)$", "contains"),
    ):
        match = re.match(pattern, clause, flags=re.IGNORECASE)
        if not match:
            continue
        field_ref = _ground_column_reference(match.group("field"), source_columns, role_columns)
        value = _coerce_filter_value(match.group("value"))
        if isinstance(value, list) and len(value) > 1 and operator in {"eq", "contains"}:
            operator = "in"
        if field_ref is None:
            field_ref = UnresolvedColumnReference(
                raw_reference=_strip_noise_tokens(match.group("field")),
                resolution_method="unresolved",
                selection_mode="ambiguous" if _normalize_reference(match.group("field")) else None,
            )
        return FilterCondition(field=field_ref, operator=operator, value=value)

    # Bare "customer id 1002" or "status pending" style clauses.
    for column in _ordered_grounded_columns(source_columns, role_columns):
        if not _phrase_in_prompt(column.raw_reference, clause) and column.resolved_column:
            if not _phrase_in_prompt(column.resolved_column, clause):
                continue
        remainder = _remove_column_reference(clause, column.raw_reference)
        if remainder == clause and column.resolved_column:
            remainder = _remove_column_reference(clause, column.resolved_column)
        remainder = _strip_noise_tokens(remainder)
        if not remainder:
            continue
        if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", remainder):
            return FilterCondition(field=column, operator="eq", value=_coerce_value(remainder))
        if remainder:
            value = _coerce_filter_value(remainder)
            operator = "in" if isinstance(value, list) and len(value) > 1 else "contains"
            return FilterCondition(field=column, operator=operator, value=value)
    return None


def _ground_column_reference(
    reference: str,
    source_columns: list[str],
    role_columns: dict[str, list[str]] | None = None,
    *,
    aliases: dict[str, list[str]] | None = None,
) -> UnresolvedColumnReference | None:
    raw_reference = _strip_noise_tokens(reference)
    if not raw_reference:
        return None

    alias_index = aliases or _build_column_alias_index(source_columns)
    normalized_reference = _normalize_reference(raw_reference)
    generic_reference = not normalized_reference or set(normalized_reference.split()) <= _GENERIC_FIELD_REFERENCES
    family_columns = _projection_family_columns(raw_reference, source_columns)

    exact_matches = alias_index.get(normalized_reference, [])
    if exact_matches:
        unique_matches = _dedupe_preserve_order(exact_matches)
        if len(unique_matches) == 1:
            if generic_reference:
                return UnresolvedColumnReference(
                    raw_reference=raw_reference,
                    resolution_method="alias_match",
                    selection_mode="ambiguous",
                    candidate_columns=unique_matches,
                    evidence=[f"Alias {raw_reference!r} maps to a generic column candidate."],
                )
            return UnresolvedColumnReference(
                raw_reference=raw_reference,
                resolved_column=unique_matches[0],
                resolution_method="exact_name" if normalize_semantic_name(raw_reference) == normalize_semantic_name(unique_matches[0]) else "alias_match",
            )
        if generic_reference:
            return UnresolvedColumnReference(
                raw_reference=raw_reference,
                resolution_method="alias_match",
                selection_mode="ambiguous",
                candidate_columns=unique_matches,
                evidence=[f"Alias {raw_reference!r} matches multiple columns."],
            )
        return UnresolvedColumnReference(
            raw_reference=raw_reference,
            resolution_method="alias_match",
            selection_mode="ambiguous",
            candidate_columns=unique_matches,
            resolved_columns=[],
            evidence=[f"Alias {raw_reference!r} matches multiple columns."],
        )

    if role_columns is None:
        role_columns = infer_column_roles(source_columns)
    role_match = _ground_via_roles(normalized_reference, role_columns)
    if role_match is not None:
        return UnresolvedColumnReference(
            raw_reference=raw_reference,
            resolved_column=role_match[0],
            resolution_method="semantic_role",
            selection_mode="single",
        )

    if generic_reference and len(family_columns) > 1:
        return UnresolvedColumnReference(
            raw_reference=raw_reference,
            resolution_method="semantic_family",
            selection_mode="ambiguous",
            resolved_columns=family_columns,
            candidate_columns=family_columns,
            evidence=[f"Generic reference {raw_reference!r} overlaps multiple semantic family columns."],
        )

    best_column, score = _best_column_match(normalized_reference, source_columns)
    if best_column and score >= 0.72:
        if generic_reference and len(family_columns) > 1 and best_column in family_columns:
            return UnresolvedColumnReference(
                raw_reference=raw_reference,
                resolution_method="semantic_family",
                selection_mode="ambiguous",
                resolved_columns=family_columns,
                candidate_columns=family_columns,
                evidence=[f"Generic reference {raw_reference!r} is ambiguous across semantic family columns."],
            )
        method = "normalized_semantic_match" if score >= 0.9 else "fuzzy_match"
        return UnresolvedColumnReference(
            raw_reference=raw_reference,
            resolved_column=best_column,
            resolution_method=method,
            selection_mode="single",
        )

    family_columns = _projection_family_columns(raw_reference, source_columns)
    if family_columns:
        return UnresolvedColumnReference(
            raw_reference=raw_reference,
            resolution_method="semantic_family",
            selection_mode="semantic_family",
            resolved_columns=family_columns,
            candidate_columns=family_columns,
            evidence=[f"Expanded semantic family {raw_reference!r} to matching columns."],
        )

    return UnresolvedColumnReference(
        raw_reference=raw_reference,
        selection_mode="ambiguous" if normalized_reference else None,
    )


def _projection_family_columns(reference: str, source_columns: list[str]) -> list[str]:
    normalized_reference = _normalize_reference(reference)
    family_root = normalized_reference[:-1] if normalized_reference.endswith("s") else normalized_reference
    family_root = family_root.replace(" ", "_")
    if not family_root:
        return []

    candidates: list[str] = []
    for column in source_columns:
        normalized_column = normalize_semantic_name(column)
        if normalized_column == family_root:
            candidates.append(column)
            continue
        if normalized_column.startswith(f"{family_root}_"):
            candidates.append(column)
            continue
        if normalized_column.startswith(family_root) and normalized_column != family_root:
            candidates.append(column)
    return _dedupe_preserve_order(candidates)


def _ground_via_roles(normalized_reference: str, role_columns: dict[str, list[str]]) -> tuple[str, str] | None:
    reference_phrases = {_normalize_reference(normalized_reference), normalized_reference}
    for role, aliases in _ROLE_ALIASES.items():
        if reference_phrases & {_normalize_reference(alias) for alias in aliases}:
            columns = role_columns.get(role)
            if columns and len(columns) == 1:
                return columns[0], role
    return None


def _best_column_match(normalized_reference: str, source_columns: list[str]) -> tuple[str | None, float]:
    best_column: str | None = None
    best_score = 0.0
    for column in source_columns:
        normalized_column = normalize_semantic_name(column)
        score = difflib.SequenceMatcher(None, normalized_reference, normalized_column.replace("_", " ")).ratio()
        if normalized_reference and (
            normalized_reference == normalized_column
            or normalized_reference == normalized_column.replace("_", " ")
        ):
            return column, 1.0
        if _token_overlap_score(normalized_reference, normalized_column) > score:
            score = _token_overlap_score(normalized_reference, normalized_column)
        if score > best_score:
            best_score = score
            best_column = column
    return best_column, best_score


def _token_overlap_score(left: str, right: str) -> float:
    left_tokens = {token for token in _normalize_reference(left).split(" ") if token}
    right_tokens = {token for token in _normalize_reference(right).split(" ") if token}
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = left_tokens & right_tokens
    return len(overlap) / max(len(left_tokens), len(right_tokens))


def _ordered_grounded_columns(source_columns: list[str], role_columns: dict[str, list[str]]) -> list[UnresolvedColumnReference]:
    ordered: list[UnresolvedColumnReference] = []
    for column in source_columns:
        ordered.append(
            UnresolvedColumnReference(
                raw_reference=column,
                resolved_column=column,
                resolution_method="exact_name",
            )
        )
    for role, columns in role_columns.items():
        for column in columns:
            if column not in source_columns:
                ordered.append(
                    UnresolvedColumnReference(
                        raw_reference=role,
                        resolved_column=column,
                        resolution_method="semantic_role",
                    )
                )
    return ordered


def _condition_role(condition: dict[str, Any] | FilterCondition | UnresolvedColumnReference | Any) -> str | None:
    if isinstance(condition, FilterCondition):
        return condition.field.resolved_column or condition.field.raw_reference
    if isinstance(condition, UnresolvedColumnReference):
        return condition.resolved_column or condition.raw_reference
    if isinstance(condition, dict):
        field = condition.get("field")
        if isinstance(field, dict):
            resolved = str(field.get("resolved_column", "")).strip()
            if resolved:
                return resolved
            raw_reference = str(field.get("raw_reference", "")).strip()
            if raw_reference:
                return raw_reference
        role = condition.get("role")
        if role is not None:
            value = str(role).strip()
            return value or None
    return None


def _legacy_condition_role(condition: dict[str, Any]) -> str | None:
    field = condition.get("field")
    raw_reference = ""
    resolved_column = ""
    if isinstance(field, dict):
        raw_reference = str(field.get("raw_reference", "")).strip()
        resolved_column = str(field.get("resolved_column", "")).strip()
    candidate = resolved_column or raw_reference
    candidate_normalized = normalize_semantic_name(candidate)
    if candidate_normalized in {"payment_method", "merchant", "vendor", "provider", "payment_type", "gateway"}:
        return "merchant"
    if candidate_normalized:
        return candidate
    return None


def _serialize_requested_fields(fields: list[Any]) -> list[str]:
    serialized: list[str] = []
    for item in fields:
        if isinstance(item, dict):
            resolved_columns = item.get("resolved_columns")
            if isinstance(resolved_columns, list) and resolved_columns:
                serialized.extend([str(column).strip() for column in resolved_columns if str(column).strip()])
                continue
            resolved = str(item.get("resolved_column", "")).strip()
            raw_reference = str(item.get("raw_reference", "")).strip()
            if resolved:
                serialized.append(resolved)
            elif raw_reference:
                serialized.append(raw_reference)
        elif isinstance(item, UnresolvedColumnReference):
            if item.resolved_columns:
                serialized.extend([column for column in item.resolved_columns if str(column).strip()])
                continue
            serialized.append(item.resolved_column or item.raw_reference)
        else:
            text = str(item).strip()
            if text:
                serialized.append(text)
    return _dedupe_preserve_order(serialized)


def _iter_grounded_references(actions: list[IntentAction]) -> list[dict[str, Any]]:
    grounded: list[dict[str, Any]] = []
    for action in actions:
        if isinstance(action, (ProjectColumnsIntent, DropColumnsIntent)):
            for field in action.requested_fields:
                grounded.append(field.model_dump(mode="json"))
        elif isinstance(action, FilterRowsIntent):
            for condition in action.conditions:
                grounded.append(condition.field.model_dump(mode="json"))
        elif isinstance(action, RenameColumnsIntent):
            for mapping in action.mapping:
                grounded.append(mapping.source.model_dump(mode="json"))
        elif isinstance(action, SortRowsIntent):
            for key in action.sort_keys:
                grounded.append(key.column.model_dump(mode="json"))
        elif isinstance(action, VisualizeIntent):
            for field in action.fields:
                grounded.append(field.model_dump(mode="json"))
    return grounded


def upcast_canonical_intent_payload(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise TypeError("canonical intent payload must be a dictionary")
    version = str(payload.get("schema_version", "")).strip() or "1.0"
    if version not in SUPPORTED_CANONICAL_INTENT_SCHEMA_VERSIONS:
        raise ValueError(f"Unsupported canonical intent schema version: {version}")
    if version == CANONICAL_INTENT_SCHEMA_VERSION:
        return payload

    upgraded = dict(payload)
    upgraded["schema_version"] = CANONICAL_INTENT_SCHEMA_VERSION
    upgraded.setdefault("intent_id", str(uuid.uuid4()))
    upgraded.setdefault("intent_revision", 1)
    upgraded.setdefault("intent_hash", "")
    upgraded.setdefault("parent_intent_id", None)
    upgraded.setdefault("capability_version", CANONICAL_INTENT_CAPABILITY_VERSION)
    upgraded.setdefault("capability_snapshot", build_capability_snapshot().model_dump(mode="json"))
    upgraded.setdefault("created_at", datetime.now(UTC).isoformat())
    upgraded.setdefault("grounded_at", datetime.now(UTC).isoformat())
    return upgraded


def _build_decision_summary(actions: list[IntentAction]) -> str:
    if not actions:
        return ""
    parts: list[str] = []
    for action in actions:
        if isinstance(action, BaseModel):
            kind = getattr(action, "kind", "")
        elif isinstance(action, dict):
            kind = str(action.get("kind", "")).strip()
        else:
            kind = ""
        if kind:
            parts.append(str(kind))
    return " + ".join(parts) if parts else ""


def _build_column_alias_index(source_columns: list[str]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for column in source_columns:
        for alias in _column_aliases(column):
            index.setdefault(alias, []).append(column)
    return index


def _column_aliases(column: str) -> set[str]:
    normalized = normalize_semantic_name(column)
    readable = normalized.replace("_", " ")
    aliases = {
        str(column).strip().lower(),
        normalized,
        readable,
        normalized.replace("_", ""),
        readable.replace(" ", ""),
    }
    aliases.update(_synonym_variants(normalized))
    aliases.update(_semantic_role_aliases(normalized))
    return {alias for alias in aliases if alias}


def _semantic_role_aliases(normalized: str) -> set[str]:
    aliases: set[str] = set()
    if normalized.endswith("_id"):
        base = normalized[:-3].strip("_")
        if base:
            aliases.add(f"{base} id")
            aliases.add(f"{base} identifier")
    if normalized.endswith("_name"):
        base = normalized[:-5].strip("_")
        if base:
            aliases.add(f"{base} name")
            aliases.add("name")
    if "payment_method" in normalized:
        aliases.update({"merchant", "vendor", "provider", "payment method"})
    if "status" in normalized:
        aliases.update({"status", "state"})
    return aliases


def _synonym_variants(normalized: str) -> set[str]:
    variants = {normalized}
    variants.add(normalized.replace("identifier", "id"))
    variants.add(normalized.replace("id", "identifier"))
    variants.add(normalized.replace("number", "no"))
    return variants


def _normalize_reference(value: str) -> str:
    text = _strip_noise_tokens(value)
    text = text.replace("_", " ")
    text = re.sub(r"\bidentifier\b", "id", text, flags=re.IGNORECASE)
    text = re.sub(r"\bidentifiers\b", "id", text, flags=re.IGNORECASE)
    text = re.sub(r"\bids\b", "id", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _strip_noise_tokens(value: str) -> str:
    text = str(value or "").strip().strip(",.;:!?")
    text = re.sub(
        r"^(?:the|a|an|and|or|just|only|return|give|show|keep|output|extract|which|that|where|with|rows?|records?|field|fields|column|columns)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+(?:column|columns)$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+(?:only|nothing else|and nothing else)$", "", text, flags=re.IGNORECASE)
    return text.strip().strip(",.;:!?")


def _strip_filter_prefix(text: str) -> str:
    cleaned = str(text or "").strip()
    previous = None
    while cleaned and cleaned != previous:
        previous = cleaned
        cleaned = _FILTER_INTENT_PREFIX.sub("", cleaned, count=1).strip()
    return cleaned


def _phrase_in_prompt(phrase: str, normalized_prompt: str) -> bool:
    phrase = _normalize_reference(phrase)
    if not phrase:
        return False
    pattern = rf"(?<!\w){re.escape(phrase)}(?!\w)"
    return bool(re.search(pattern, normalized_prompt))


def _remove_column_reference(text: str, reference: str) -> str:
    normalized_reference = _normalize_reference(reference)
    if not normalized_reference:
        return text
    pattern = rf"(?<!\w){re.escape(normalized_reference)}(?!\w)"
    return re.sub(pattern, "", text, count=1).strip()


def _split_filter_clauses(text: str) -> tuple[list[str], list[str]]:
    text = re.sub(r"\s*,\s*", " and ", text)
    clauses: list[str] = []
    connectors: list[str] = []
    buffer: list[str] = []
    tokens = re.split(r"(\s+(?:and|or)\s+)", text)
    for token in tokens:
        if not token:
            continue
        connector_match = re.fullmatch(r"\s+(and|or)\s+", token, flags=re.IGNORECASE)
        if connector_match:
            clause = _strip_noise_tokens("".join(buffer))
            if clause:
                clauses.append(clause)
                connectors.append(connector_match.group(1).lower())
            buffer = []
            continue
        buffer.append(token)
    trailing = _strip_noise_tokens("".join(buffer))
    if trailing:
        clauses.append(trailing)
    if len(connectors) >= len(clauses):
        connectors = connectors[: max(len(clauses) - 1, 0)]
    return clauses, connectors


def _coerce_value(value: str) -> Any:
    cleaned = _strip_noise_tokens(value).strip().strip("\"'")
    cleaned = re.sub(r"^(?:is|equals?|equal to|value is)\s+", "", cleaned, flags=re.IGNORECASE).strip()
    if re.fullmatch(r"-?\d+", cleaned):
        try:
            return int(cleaned)
        except ValueError:
            return cleaned
    if re.fullmatch(r"-?\d+\.\d+", cleaned):
        try:
            return float(cleaned)
        except ValueError:
            return cleaned
    return cleaned


def _coerce_filter_value(value: Any) -> Any:
    if isinstance(value, list):
        flattened: list[Any] = []
        for item in value:
            coerced = _coerce_filter_value(item)
            if isinstance(coerced, list):
                flattened.extend(coerced)
            elif coerced not in {None, ""}:
                flattened.append(coerced)
        deduped = _dedupe_preserve_order(flattened)
        return deduped

    text = str(value or "").strip()
    if not text:
        return ""

    parts = [
        part.strip()
        for part in re.split(r"\s+(?:or|and)\s+|[,;]", text, flags=re.IGNORECASE)
        if part.strip()
    ]
    if len(parts) > 1:
        coerced_parts = [_coerce_value(part) for part in parts]
        deduped = [item for item in _dedupe_preserve_order(coerced_parts) if item not in {None, ""}]
        if deduped:
            return deduped
    return _coerce_value(text)


def _dedupe_references(items: list[UnresolvedColumnReference]) -> list[UnresolvedColumnReference]:
    seen: set[tuple[str, str | None]] = set()
    deduped: list[UnresolvedColumnReference] = []
    for item in items:
        key = (item.raw_reference.lower(), item.resolved_column)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _dedupe_preserve_order(items: list[Any]) -> list[Any]:
    seen: set[Any] = set()
    result: list[Any] = []
    for item in items:
        marker = item if isinstance(item, str) else repr(item)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(item)
    return result


def _reference_appears_in_prompt(reference: str, prompt: str) -> bool:
    normalized = _normalize_reference(reference)
    if not normalized:
        return False
    return _phrase_in_prompt(normalized, prompt)
