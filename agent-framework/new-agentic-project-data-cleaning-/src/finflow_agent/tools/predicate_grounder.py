"""Semantic predicate grounding for filter execution.

This module sits between the dataframe profile and the deterministic filter
executor. It resolves ambiguous filter clauses using semantic evidence from
the dataframe profile, not by requiring the requested literal to already be
observed in preview rows.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Iterable, Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field

from finflow_agent.llm import get_chat_groq
from finflow_agent.tools.column_resolver import CONFIDENCE_THRESHOLD
from finflow_agent.tools.dataframe_profile import DataFrameProfile
from finflow_agent.tools.semantic_column_profiler import (
    BroadSemanticType,
    SemanticColumnProfile,
    profile_semantic_columns,
)

logger = logging.getLogger(__name__)

ENABLE_LLM_PREDICATE_GROUNDING_VAR = "ENABLE_LLM_PREDICATE_GROUNDING"
LEGACY_ENABLE_LLM_COLUMN_RESOLUTION_VAR = "ENABLE_LLM_COLUMN_RESOLUTION"
GROUNDING_VERSION = "1"
# Keep predicate grounding aligned with the resolver's low-confidence gate:
# anything below the shared confidence floor should get an LLM chance.
GROUNDING_THRESHOLD = CONFIDENCE_THRESHOLD
AMBIGUITY_MARGIN = 0.1

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_PAYMENT_VALUE_HINTS = frozenset(
    {
        "cash",
        "card",
        "credit",
        "debit",
        "paypal",
        "pay",
        "upi",
        "wallet",
        "bank",
        "transfer",
        "merchant",
        "visa",
        "mastercard",
    }
)

_STATUS_VALUE_HINTS = frozenset(
    {
        "pending",
        "completed",
        "complete",
        "paid",
        "unpaid",
        "failed",
        "success",
        "successful",
        "processed",
        "approved",
        "rejected",
        "declined",
        "processing",
        "open",
        "closed",
        "cancelled",
        "canceled",
    }
)

_IDENTIFIER_VALUE_HINTS = frozenset(
    {
        "id",
        "uuid",
        "ref",
        "reference",
        "code",
        "key",
        "txn",
        "transaction",
    }
)


class UnresolvedFilterClause(BaseModel):
    requested_field: str
    operator: str
    value: Any = None
    value_to: Any = None
    case_sensitive: bool = False


class GroundingCandidate(BaseModel):
    column: str
    score: float = Field(ge=0.0, le=1.0)
    broad_type: BroadSemanticType
    positive_evidence: list[str] = Field(default_factory=list)
    negative_evidence: list[str] = Field(default_factory=list)
    semantic_tags: list[str] = Field(default_factory=list)


class GroundedFilterClause(BaseModel):
    requested_field: str
    resolved_column: str
    operator: str
    value: Any = None
    value_to: Any = None
    case_sensitive: bool = False
    confidence: float = Field(ge=0.0, le=1.0)
    grounding_method: Literal["deterministic", "llm", "manual"] = "deterministic"
    positive_evidence: list[str] = Field(default_factory=list)
    negative_evidence: list[str] = Field(default_factory=list)
    candidate_scores: list[GroundingCandidate] = Field(default_factory=list)


class PredicateGroundingResult(BaseModel):
    status: Literal["grounded", "needs_review", "quarantined"]
    grounded_clauses: list[GroundedFilterClause] = Field(default_factory=list)
    unresolved_clauses: list[UnresolvedFilterClause] = Field(default_factory=list)
    candidate_scores: list[GroundingCandidate] = Field(default_factory=list)
    reason: str = ""
    contract_violation: dict[str, Any] | None = None


class LLMGroundingDecision(BaseModel):
    selected_column: Optional[str] = None
    reason: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokenize(value: Any) -> set[str]:
    return {token for token in _TOKEN_RE.findall(_normalize_text(value)) if token}


def _is_llm_enabled() -> bool:
    raw = os.environ.get(ENABLE_LLM_PREDICATE_GROUNDING_VAR)
    if raw is None:
        raw = os.environ.get(LEGACY_ENABLE_LLM_COLUMN_RESOLUTION_VAR, "true")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _value_concepts(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, bool):
        return {"boolean"}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return {"numeric"}

    text = _normalize_text(value)
    if not text:
        return set()

    concepts: set[str] = {"text"}
    compact = re.sub(r"[^a-z0-9]+", "", text)
    if any(token in _PAYMENT_VALUE_HINTS for token in text.split()):
        concepts.add("payment")
    if any(token in _STATUS_VALUE_HINTS for token in text.split()):
        concepts.add("status")
    if any(token in _IDENTIFIER_VALUE_HINTS for token in text.split()):
        concepts.add("identifier")
    if compact.isdigit():
        concepts.add("numeric")
        concepts.discard("text")
    if "@" in str(value):
        concepts.add("identifier")
    try:
        parsed = pd.to_datetime([value], errors="coerce")
        if parsed.notna().any():
            concepts.add("date")
    except Exception:  # pragma: no cover - defensive
        pass
    return concepts


def _field_tokens(requested_field: str) -> set[str]:
    field = requested_field.strip()
    if field.startswith("__") and field.endswith("_column__"):
        field = field[2:-9]
    return _tokenize(field)


def _score_candidate(
    clause: UnresolvedFilterClause,
    semantic: SemanticColumnProfile,
) -> GroundingCandidate:
    requested_tokens = _field_tokens(clause.requested_field)
    value_concepts = _value_concepts(clause.value)
    broad_type = semantic.broad_type
    positive: list[str] = []
    negative: list[str] = []
    score = 0.0

    if requested_tokens:
        overlap = requested_tokens & set(semantic.semantic_tags)
        if overlap:
            score += 0.35
            positive.append(f"field_overlap={sorted(overlap)}")

    # Value-tag overlap: if the filter VALUE appears in this column's
    # semantic tags, it's strong evidence this is the right column.
    # This breaks ties like education_level vs loan_purpose where both
    # have "education" in tags but only education_level has "phd".
    if clause.value is not None:
        value_tokens = _tokenize(clause.value)
        value_tag_overlap = value_tokens & set(semantic.semantic_tags)
        if value_tag_overlap:
            score += 0.25
            positive.append(f"filter_value_in_tags={sorted(value_tag_overlap)}")

    if broad_type == BroadSemanticType.product:
        if "text" in value_concepts:
            score += 0.35
            positive.append("value looks like product/free-text content")
        if "payment" in value_concepts or "status" in value_concepts:
            score -= 0.45
            negative.append("value concept conflicts with product column")
    elif broad_type == BroadSemanticType.payment:
        if "payment" in value_concepts:
            score += 0.55
            positive.append("value looks payment-related")
        if "text" in value_concepts and "payment" not in value_concepts:
            score -= 0.45
            negative.append("value is generic text, not payment-like")
    elif broad_type == BroadSemanticType.status:
        if "status" in value_concepts:
            score += 0.60
            positive.append("value looks status-like")
        if "text" in value_concepts and "status" not in value_concepts:
            score -= 0.35
            negative.append("value is generic text, not status-like")
    elif broad_type == BroadSemanticType.identifier:
        if "identifier" in value_concepts or "numeric" in value_concepts:
            score += 0.4
            positive.append("value looks identifier-like")
        if "text" in value_concepts and "identifier" not in value_concepts:
            score -= 0.25
            negative.append("value is not identifier-like")
    elif broad_type == BroadSemanticType.date:
        if "date" in value_concepts:
            score += 0.45
            positive.append("value looks date-like")
        if "text" in value_concepts and "date" not in value_concepts:
            score -= 0.25
            negative.append("value is not date-like")
    elif broad_type == BroadSemanticType.currency:
        if "numeric" in value_concepts:
            score += 0.35
            positive.append("value looks numeric/currency-like")
    elif broad_type == BroadSemanticType.numeric:
        if "numeric" in value_concepts:
            score += 0.4
            positive.append("value looks numeric")
    elif broad_type == BroadSemanticType.categorical:
        if "text" in value_concepts:
            score += 0.15
            positive.append("value is text against categorical column")
        # When the column is generic "categorical" but the value matches
        # a specific domain (status, payment), boost significantly.
        # This handles the c1/c2/c3/c4 case where column names are
        # meaningless but values reveal the domain.
        if "status" in value_concepts:
            score += 0.45
            positive.append("value is status-like against categorical column (likely a status column)")
        if "payment" in value_concepts:
            score += 0.40
            positive.append("value is payment-like against categorical column (likely a payment column)")

    # Column-name semantics and descriptive evidence.
    if semantic.broad_type in {
        BroadSemanticType.product,
        BroadSemanticType.payment,
        BroadSemanticType.status,
        BroadSemanticType.identifier,
    }:
        score += 0.2
        positive.append(f"semantic_type={semantic.broad_type.value}")
    elif semantic.broad_type == BroadSemanticType.free_text:
        score += 0.12
        positive.append("free-text column")
    elif semantic.broad_type == BroadSemanticType.categorical:
        # Categorical columns with tags that overlap value domain hints
        # get a stronger bonus — this is the path for c1/c2/c3/c4 columns
        # where the semantic profiler couldn't determine a specific type
        # but the tag content reveals the domain.
        tag_set = set(semantic.semantic_tags)
        if tag_set & _STATUS_VALUE_HINTS:
            score += 0.18
            positive.append("categorical column with status-like tags")
        elif tag_set & _PAYMENT_VALUE_HINTS:
            score += 0.18
            positive.append("categorical column with payment-like tags")
        else:
            score += 0.08
            positive.append("categorical column")

    if clause.operator in {"eq", "neq", "contains", "not_contains", "starts_with", "ends_with", "in", "not_in"}:
        score += 0.05
        positive.append(f"text-operator={clause.operator}")

    name_overlap = requested_tokens & set(semantic.semantic_tags)
    if name_overlap:
        score += 0.15
        positive.append("column name overlaps requested field")

    if not requested_tokens and "text" in value_concepts and semantic.broad_type in {
        BroadSemanticType.product,
        BroadSemanticType.free_text,
    }:
        score += 0.15
        positive.append("generic text prefers product/free-text columns")

    score = max(0.0, min(1.0, score + semantic.match_score * 0.2))
    return GroundingCandidate(
        column=semantic.column,
        score=score,
        broad_type=semantic.broad_type,
        positive_evidence=positive,
        negative_evidence=negative,
        semantic_tags=semantic.semantic_tags,
    )


def _llm_ground_clause(
    *,
    clause: UnresolvedFilterClause,
    semantic_profiles: list[SemanticColumnProfile],
    candidates: list[GroundingCandidate],
) -> Optional[LLMGroundingDecision]:
    if not _is_llm_enabled() or not os.environ.get("GROQ_API_KEY"):
        return None
    if not candidates:
        return None

    # Only send the top-N candidates to the LLM to avoid token limits.
    # For a 182-column dataset, sending all candidates hits the 12k TPM.
    # The top 10 candidates sorted by score are sufficient context.
    top_candidates = candidates[:10]

    # --- Telemetry: log call start ---
    _telemetry_ctx = None
    try:
        from finflow_agent.llm_telemetry import log_llm_started, log_llm_completed, log_llm_failed, log_runtime_event
        log_runtime_event(
            "predicate_grounding_llm_entered",
            service="agent-service",
            trigger="worker",
            instruction_present=False,
            canonical_intent_present=True,
            legacy_schema_state_present=False,
            requested_field=clause.requested_field,
            prompt_text=f"requested_field={clause.requested_field}, operator={clause.operator}, value={clause.value}",
            model="llama-3.3-70b-versatile",
            api_key=os.environ.get("GROQ_API_KEY", ""),
            api_key_source="GROQ_API_KEY",
        )
        _telemetry_ctx = log_llm_started(
            service="agent-service",
            operation="predicate_grounding",
            caller_file="predicate_grounder.py",
            caller_function="_llm_ground_clause",
            model="llama-3.3-70b-versatile",
            api_key_source="GROQ_API_KEY",
            api_key=os.environ.get("GROQ_API_KEY", ""),
            attempt=1,
            trigger=f"ground:{clause.requested_field}",
            messages=[{"role": "user", "content": f"requested_field={clause.requested_field}, operator={clause.operator}, value={clause.value}"}],
        )
    except Exception:
        pass
    # --- End telemetry start ---

    try:
        llm = get_chat_groq(model_name="llama-3.3-70b-versatile", temperature=0)
        structured_llm = llm.with_structured_output(LLMGroundingDecision)
        from langchain_core.prompts import PromptTemplate

        candidate_lines = []
        for candidate in top_candidates:
            candidate_lines.append(
                f"- {candidate.column} | type={candidate.broad_type.value} | score={candidate.score:.3f} | "
                f"positive={candidate.positive_evidence[:3]} | negative={candidate.negative_evidence[:3]}"
            )
        # Only include semantic profiles for the top candidates
        top_column_names = {c.column for c in top_candidates}
        semantic_lines = []
        for semantic in semantic_profiles:
            if semantic.column in top_column_names:
                semantic_lines.append(
                    f"- {semantic.column}: {semantic.semantic_description}"
                )

        prompt = PromptTemplate.from_template(
            "You are grounding a filter clause against a sanitized dataframe profile.\n"
            "Choose exactly one column from the candidate list, or null if none fit.\n"
            "The requested literal may be absent from the chosen column and that is not an error.\n"
            "Do not invent columns.\n\n"
            "Clause:\n"
            "requested_field={requested_field}\n"
            "operator={operator}\n"
            "value={value}\n"
            "value_to={value_to}\n\n"
            "Semantic profile:\n{semantic_profile}\n\n"
            "Candidates:\n{candidates}\n\n"
            "Return the best matching actual column name only."
        )
        chain = prompt | structured_llm
        result = chain.invoke(
            {
                "requested_field": clause.requested_field,
                "operator": clause.operator,
                "value": clause.value,
                "value_to": clause.value_to,
                "semantic_profile": "\n".join(semantic_lines),
                "candidates": "\n".join(candidate_lines),
            }
        )

        # --- Telemetry: log success (LangChain doesn't expose token usage directly) ---
        if _telemetry_ctx:
            try:
                log_llm_completed(
                    _telemetry_ctx,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    finish_reason="stop",
                )
            except Exception:
                pass
        # --- End telemetry success ---

        if isinstance(result, LLMGroundingDecision):
            return result
        return LLMGroundingDecision.model_validate(result)
    except Exception as exc:  # pragma: no cover - defensive
        # --- Telemetry: log failure ---
        if _telemetry_ctx:
            try:
                log_llm_failed(
                    _telemetry_ctx,
                    status_code=0,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            except Exception:
                pass
        # --- End telemetry failure ---
        logger.warning("Predicate grounding LLM fallback failed: %s", exc)
        return None


def _ground_single_clause(
    clause: UnresolvedFilterClause,
    semantic_profiles: list[SemanticColumnProfile],
) -> tuple[Optional[GroundedFilterClause], list[GroundingCandidate], str | None]:
    candidate_scores = [_score_candidate(clause, semantic) for semantic in semantic_profiles]
    candidate_scores.sort(key=lambda item: item.score, reverse=True)
    if not candidate_scores:
        return None, [], "no semantic candidates available"

    best = candidate_scores[0]
    second = candidate_scores[1] if len(candidate_scores) > 1 else None
    ambiguous = (
        best.score < GROUNDING_THRESHOLD
        or (second is not None and abs(best.score - second.score) < AMBIGUITY_MARGIN)
    )

    if ambiguous:
        llm_decision = _llm_ground_clause(
            clause=clause,
            semantic_profiles=semantic_profiles,
            candidates=candidate_scores,
        )
        if llm_decision is not None and llm_decision.selected_column:
            selected = next(
                (candidate for candidate in candidate_scores if candidate.column == llm_decision.selected_column),
                None,
            )
            if selected is not None:
                return (
                    GroundedFilterClause(
                        requested_field=clause.requested_field,
                        resolved_column=selected.column,
                        operator=clause.operator,
                        value=clause.value,
                        value_to=clause.value_to,
                        case_sensitive=clause.case_sensitive,
                        confidence=max(selected.score, llm_decision.confidence, 0.8),
                        grounding_method="llm",
                        positive_evidence=selected.positive_evidence,
                        negative_evidence=selected.negative_evidence,
                        candidate_scores=candidate_scores,
                    ),
                    candidate_scores,
                    None,
                )
        return None, candidate_scores, "ambiguous semantic grounding"

    if best.score <= 0.0:
        return None, candidate_scores, "no plausible semantic match"

    return (
        GroundedFilterClause(
            requested_field=clause.requested_field,
            resolved_column=best.column,
            operator=clause.operator,
            value=clause.value,
            value_to=clause.value_to,
            case_sensitive=clause.case_sensitive,
            confidence=best.score,
            grounding_method="deterministic",
            positive_evidence=best.positive_evidence,
            negative_evidence=best.negative_evidence,
            candidate_scores=candidate_scores,
        ),
        candidate_scores,
        None,
    )


def ground_filter_clauses(
    clauses: Iterable[UnresolvedFilterClause],
    *,
    profile: DataFrameProfile,
    semantic_profiles: list[SemanticColumnProfile] | None = None,
) -> PredicateGroundingResult:
    """Ground unresolved filter clauses to concrete dataframe columns."""
    semantic_profiles = semantic_profiles or profile_semantic_columns(profile)
    grounded: list[GroundedFilterClause] = []
    unresolved: list[UnresolvedFilterClause] = []
    all_candidates: list[GroundingCandidate] = []

    for clause in clauses:
        grounded_clause, candidate_scores, reason = _ground_single_clause(
            clause, semantic_profiles
        )
        all_candidates.extend(candidate_scores)
        if grounded_clause is None:
            unresolved.append(clause)
            continue
        grounded.append(grounded_clause)

    if unresolved:
        return PredicateGroundingResult(
            status="needs_review",
            grounded_clauses=grounded,
            unresolved_clauses=unresolved,
            candidate_scores=all_candidates,
            reason="; ".join(
                [
                    f"{clause.requested_field}: {_ground_reason_for_clause(clause, semantic_profiles)}"
                    for clause in unresolved
                ]
            ),
        )

    return PredicateGroundingResult(
        status="grounded",
        grounded_clauses=grounded,
        unresolved_clauses=[],
        candidate_scores=all_candidates,
        reason="All filter clauses grounded semantically.",
    )


def _ground_reason_for_clause(
    clause: UnresolvedFilterClause,
    semantic_profiles: list[SemanticColumnProfile],
) -> str:
    candidates = [_score_candidate(clause, semantic) for semantic in semantic_profiles]
    candidates.sort(key=lambda item: item.score, reverse=True)
    if not candidates:
        return "no semantic candidates available"
    best = candidates[0]
    if best.score < GROUNDING_THRESHOLD:
        return f"best score {best.score:.3f} below grounding threshold"
    return "ambiguous semantic grounding"


__all__ = [
    "ENABLE_LLM_PREDICATE_GROUNDING_VAR",
    "GROUNDING_THRESHOLD",
    "AMBIGUITY_MARGIN",
    "UnresolvedFilterClause",
    "GroundingCandidate",
    "GroundedFilterClause",
    "PredicateGroundingResult",
    "LLMGroundingDecision",
    "ground_filter_clauses",
]
