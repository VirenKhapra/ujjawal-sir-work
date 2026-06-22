"""Intent envelope and pipeline status models for FinFlow's semantic pipeline.

Defines the IntentEnvelope container that tracks pipeline state, along with
supporting models for resolution audit records and shadow comparison metrics.

Requirements: 4.4, 20.1
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from finflow_agent.models.provenance import ProvenanceRef

if TYPE_CHECKING:
    from finflow_agent.models.canonical import CanonicalIntent
    from finflow_agent.models.draft import SemanticIntentDraft


class PipelineStatus(str, Enum):
    """Pipeline processing status for an intent envelope."""

    PROCESSING = "processing"
    NEEDS_CLARIFICATION = "needs_clarification"
    INTERPRETATION_FAILED = "interpretation_failed"
    UNSUPPORTED = "unsupported"
    INVALID = "invalid"
    RESOLVED = "resolved"


class ResolutionRecord(BaseModel):
    """Audit record for a resolution decision.

    Captures the timestamp, stage, decision owner, and evidence for each
    resolution applied during pipeline processing.

    Requirements: 20.1 - resolution_status and resolution_origin separation
    """

    model_config = ConfigDict(strict=True)

    timestamp: datetime
    stage: str
    decision_owner: str
    element_path: str
    resolution: str
    confidence: float
    evidence: list[str]
    provenance: list[ProvenanceRef]


class ShadowComparisonMetric(BaseModel):
    """Comparison between deterministic and LLM coverage results.

    Used by the shadow LLM mode to record agreement/disagreement between
    the authoritative deterministic path and the non-authoritative LLM path.
    """

    model_config = ConfigDict(strict=True)

    deterministic_result: bool
    llm_result: bool | None
    agreement_status: Literal["agree", "disagree", "llm_unavailable"]
    deterministic_gaps: list[str]
    llm_gaps: list[str]


class IntentEnvelope(BaseModel):
    """Container tracking pipeline state.

    Holds either a draft or canonical intent (never both simultaneously) along
    with pipeline metadata including status, stage, model version, and feature
    flags.

    Requirements: 4.4 - patch application produces new revision via envelope
    """

    model_config = ConfigDict(strict=True, populate_by_name=True)

    submission_id: str
    pipeline_status: PipelineStatus
    current_stage: str

    # Holds either draft or canonical (never both)
    draft: Any = None  # SemanticIntentDraft (forward ref, resolved at runtime)
    canonical: Any = None  # CanonicalIntent (forward ref, resolved at runtime)

    # Metadata
    pipeline_model_version: str = Field(alias="model_version")
    feature_flags: dict[str, bool]
    created_at: datetime
    updated_at: datetime
