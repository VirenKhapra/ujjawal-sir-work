import React, { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FiCheck, FiChevronDown, FiDownload, FiRefreshCw, FiX } from "react-icons/fi";
import { useParams } from "react-router-dom";
import DataTable from "../components/DataTable.jsx";
import ClarificationPanel from "../components/ClarificationPanel.jsx";
import {
  confirmInterpretation,
  rejectInterpretation,
  replaceColumnMapping,
  downloadJobOutput,
  fetchClarificationStatus,
  fetchJobDetail,
  retryJob,
  confirmExtraction,
} from "../api/finflow.js";
import { useLiveJobRefresh } from "../hooks/useLiveJobRefresh.js";
import { useClarificationSocket } from "../hooks/useClarificationSocket.js";
import {
  formatDateTime,
  formatJobStatus,
  formatStepStatus,
} from "../utils/finflowFormatters.js";

export default function AuditPage() {
  const { jobId } = useParams();
  const [openIndex, setOpenIndex] = useState(0);
  const queryClient = useQueryClient();
  const {
    data: job,
    isLoading,
    isError,
  } = useQuery({
    queryKey: ["jobs", "detail", jobId],
    queryFn: () => fetchJobDetail(jobId),
    enabled: Boolean(jobId),
  });
  useLiveJobRefresh(jobId);

  // Clarification WebSocket integration
  const {
    session: clarificationSession,
    isActive: clarificationActive,
    isResolved: clarificationResolved,
    isExpired: clarificationExpired,
    refreshSession: refreshClarificationSession,
    dismiss: dismissClarification,
  } = useClarificationSocket(jobId);

  // REST fallback: fetch clarification session state on page load when job is
  // already in "awaiting_clarification" status but WebSocket hasn't delivered the event yet
  const showClarificationFromStatus = job?.rawStatus === "awaiting_clarification" && !clarificationActive;
  const { data: restClarificationSession } = useQuery({
    queryKey: ["clarification-status", jobId],
    queryFn: () => fetchClarificationStatus(jobId),
    enabled: Boolean(jobId) && showClarificationFromStatus,
    retry: 1,
    staleTime: 30000,
  });

  // Determine the active clarification data: prefer WebSocket, fallback to REST
  const activeClarificationData = clarificationActive
    ? clarificationSession
    : showClarificationFromStatus && restClarificationSession
      ? {
          sessionId: restClarificationSession.session_id,
          submissionId: restClarificationSession.submission_id || jobId,
          questions: restClarificationSession.questions || [],
          roundCount: restClarificationSession.round_count ?? 0,
          maxRounds: restClarificationSession.max_rounds ?? 2,
          expiresAt: restClarificationSession.expires_at,
          revisionToken: restClarificationSession.revision_token,
          intentVersion: restClarificationSession.intent_version,
        }
      : null;
  const shouldShowClarificationPanel = clarificationActive || Boolean(activeClarificationData);

  const retryMutation = useMutation({
    mutationFn: retryJob,
    onSuccess: async () => {
      await refreshJobViews(queryClient, jobId);
    },
  });

  const confirmInterpretationMutation = useMutation({
    mutationFn: confirmInterpretation,
    onSuccess: async () => {
      await refreshJobViews(queryClient, jobId);
    },
  });

  const rejectInterpretationMutation = useMutation({
    mutationFn: rejectInterpretation,
    onSuccess: async () => {
      await refreshJobViews(queryClient, jobId);
    },
  });

  const confirmExtractionMutation = useMutation({
    mutationFn: () => confirmExtraction(jobId, job?.previewToken),
    onSuccess: async () => {
      await refreshJobViews(queryClient, jobId);
    },
  });

  const dataProfile = job?.dataProfile || {};
  const canonicalIntent = job?.canonicalIntent || {};
  const clarification = job?.clarification || null;
  const intentConfirmationVisible = clarification?.mode === "intent_confirmation";
  const clarificationModeVisible = clarification?.mode === "clarification";
  const interpretationReviewVisible =
    intentConfirmationVisible || clarificationModeVisible;
  const extractionPreview = clarification?.extraction_preview || job?.extractionPreview || null;
  const extractionPreviewVisible = clarification?.mode === "extraction_confirmation" || Boolean(extractionPreview);
  const profileColumns = Array.isArray(dataProfile.columns)
    ? dataProfile.columns.filter(f => f && typeof f === "object")
    : [];
  const extractionFields = Array.isArray(extractionPreview?.proposed_fields)
    ? extractionPreview.proposed_fields.filter(f => f && typeof f === "object")
    : [];
  const validationWarnings = Array.isArray(extractionPreview?.validation_warnings)
    ? extractionPreview.validation_warnings
    : [];
  const rawActions = canonicalIntent.actions;
  const actionSchema = Array.isArray(rawActions) ? rawActions.filter(a => a && typeof a === "object") : [];
  const schemaColumns = Array.isArray(dataProfile.source_columns) ? dataProfile.source_columns : (Array.isArray(job?.columns) ? job.columns : []);
  const schemaRows = Array.isArray(dataProfile.preview_rows) ? dataProfile.preview_rows : (Array.isArray(job?.previewRows) ? job.previewRows : []);
  const detectedTypes = dataProfile.detected_types && typeof dataProfile.detected_types === "object"
    ? Object.entries(dataProfile.detected_types)
    : [];
  const extractionColumns = Array.isArray(extractionPreview?.source_columns) ? extractionPreview.source_columns : schemaColumns;
  const extractionRows = Array.isArray(extractionPreview?.preview_rows) ? extractionPreview.preview_rows : schemaRows;
  const extractionAnchor = String(extractionPreview?.anchor_column || "");
  const extractionCompleteCount = Number(extractionPreview?.complete_count || 0);
  const extractionPartialCount = Number(extractionPreview?.partial_count || 0);
  const extractionInvalidCount = Number(extractionPreview?.invalid_count || 0);
  const extractionLlmOnlyCount = Number(extractionPreview?.llm_only_count || 0);
  const extractionRecoveredCount = Number(extractionPreview?.recovered_count || 0);
  const extractionMergedCount = Number(extractionPreview?.merged_count || 0);
  const extractionAmbiguousDateCount = Number(extractionPreview?.ambiguous_date_count || 0);
  const extractionAssumedDateConvention = String(extractionPreview?.assumed_date_convention || "");
  const availableMappingColumns = useMemo(
    () => collectAvailableMappingColumns(profileColumns, schemaColumns),
    [profileColumns, schemaColumns]
  );
  const unresolvedMappingFields = useMemo(
    () => collectUnresolvedMappingFields(canonicalIntent),
    [canonicalIntent]
  );
  const mappingInputId = `column-mapping-options-${jobId}`;
  const [mappingDraft, setMappingDraft] = useState({});
  const [mappingError, setMappingError] = useState("");

  useEffect(() => {
    const nextDraft = {};
    unresolvedMappingFields.forEach((field) => {
      nextDraft[field.sourceKey] = field.defaultValue || "";
    });
    setMappingDraft(nextDraft);
    setMappingError("");
  }, [jobId, unresolvedMappingFields]);

  const replaceMappingMutation = useMutation({
    mutationFn: ({ mapping, reason }) => replaceColumnMapping(job.backendId, mapping, reason),
    onSuccess: async () => {
      setMappingError("");
      await refreshJobViews(queryClient, jobId);
    },
  });

  const handleMappingSubmit = () => {
    const mapping = {};
    unresolvedMappingFields.forEach((field) => {
      const value = parseMappingSubmissionValue(mappingDraft[field.sourceKey]);
      if (value) {
        mapping[field.sourceKey] = value;
      }
    });

    if (!Object.keys(mapping).length) {
      setMappingError("Choose at least one column mapping before applying.");
      return;
    }

    setMappingError("");
    replaceMappingMutation.mutate({
      mapping,
      reason: "Resolved through the manual clarification editor.",
    });
  };

  if (isLoading) {
    return (
      <div className="ff-page-grid">
        <section className="ff-panel">
          <h2>Loading job detail...</h2>
        </section>
      </div>
    );
  }

  if (isError || !job) {
    return (
      <div className="ff-page-grid">
        <section className="ff-panel">
          <h2>We could not load this job.</h2>
          <p className="ff-copy-muted">
            The workflow may have been removed or the backend may be
            unavailable.
          </p>
        </section>
      </div>
    );
  }

  return (
    <div className="ff-page-grid">
      <section className="ff-panel">
        <div className="ff-panel__head">
          <div>
            <p className="ff-eyebrow">Job detail and audit</p>
            <h2>{job.title}</h2>
            <p className="ff-copy-muted">{job.instruction}</p>
          </div>
          <div className="ff-detail-actions">
            <span className={`ff-status ff-status--${job.status}`}>
              {formatJobStatus(job.status)}
            </span>
            {job.status === "complete" && job.outputReady ? (
              <button
                type="button"
                className="ff-secondary-button"
                onClick={() => downloadJobOutput(job.backendId)}
              >
                <FiDownload size={15} />
                Download output
              </button>
            ) : null}
            {job.status === "failed" ? (
              <button
                type="button"
                className="ff-secondary-button"
                onClick={() => retryMutation.mutate(job.backendId)}
                disabled={retryMutation.isPending}
              >
                <FiRefreshCw size={15} />
                {retryMutation.isPending ? "Requeueing..." : "Retry job"}
              </button>
            ) : null}
            {intentConfirmationVisible ? (
              <>
                <button
                  type="button"
                  className="ff-secondary-button"
                  onClick={() => rejectInterpretationMutation.mutate(job.backendId)}
                  disabled={confirmInterpretationMutation.isPending || rejectInterpretationMutation.isPending}
                >
                  <FiX size={15} />
                  {rejectInterpretationMutation.isPending ? "Rejecting..." : "Reject interpretation"}
                </button>
                <button
                  type="button"
                  className="ff-primary-button"
                  onClick={() => confirmInterpretationMutation.mutate(job.backendId)}
                  disabled={confirmInterpretationMutation.isPending || rejectInterpretationMutation.isPending}
                >
                  <FiCheck size={15} />
                  {confirmInterpretationMutation.isPending ? "Confirming..." : "Confirm intent"}
                </button>
              </>
            ) : null}
            {extractionPreviewVisible ? (
              <>
                <button
                  type="button"
                  className="ff-primary-button"
                  onClick={() => confirmExtractionMutation.mutate()}
                  disabled={confirmExtractionMutation.isPending}
                >
                  <FiDownload size={15} />
                  {confirmExtractionMutation.isPending ? "Confirming..." : "Confirm & Download Excel"}
                </button>
              </>
            ) : null}
          </div>
        </div>

        <div className="ff-key-metrics" style={{ margin: "2rem 0" }}>
          <div>
            <span>Job ID</span>
            <strong>{job.id}</strong>
          </div>
          <div>
            <span>Submitted by</span>
            <strong>{job.submittedBy}</strong>
          </div>
          <div>
            <span>Submitted at</span>
            <strong>{formatDateTime(job.submittedAt)}</strong>
          </div>
          <div>
            <span>Completed at</span>
            <strong>{formatDateTime(job.completedAt)}</strong>
          </div>
        </div>

        {/* Clarification Panel — rendered when WebSocket delivers clarification events or status is awaiting_clarification */}
        {shouldShowClarificationPanel && activeClarificationData && (
          <ClarificationPanel
            submissionId={activeClarificationData.submissionId}
            sessionId={activeClarificationData.sessionId}
            questions={activeClarificationData.questions}
            roundCount={activeClarificationData.roundCount}
            maxRounds={activeClarificationData.maxRounds}
            expiresAt={activeClarificationData.expiresAt}
            revisionToken={activeClarificationData.revisionToken}
            intentVersion={activeClarificationData.intentVersion}
            onResolved={() => {
              dismissClarification();
              queryClient.invalidateQueries({ queryKey: ["jobs", "detail", jobId] });
              queryClient.invalidateQueries({ queryKey: ["clarification-status", jobId] });
            }}
            onSessionExpired={() => {
              dismissClarification();
              queryClient.invalidateQueries({ queryKey: ["jobs", "detail", jobId] });
              queryClient.invalidateQueries({ queryKey: ["clarification-status", jobId] });
            }}
            onSessionRefresh={refreshClarificationSession}
          />
        )}
        {clarificationResolved && (
          <section className="ff-panel ff-panel--dense" aria-live="polite" aria-label="Clarification resolved">
            <div className="ff-clarification-success">
              <h3>Ambiguities resolved</h3>
              <p className="ff-copy-muted">Your answers have been applied. The job is now running.</p>
            </div>
          </section>
        )}
        {clarificationExpired && (
          <section className="ff-panel ff-panel--dense" aria-live="polite" aria-label="Session expired">
            <div className="ff-clarification-expired">
              <h3>Session expired</h3>
              <p className="ff-copy-muted">
                The clarification session has expired. This job has been moved to quarantine.
              </p>
            </div>
          </section>
        )}
        <section className="ff-panel ff-panel--dense">
          <div className="ff-panel__head">
            <div>
              <p className="ff-eyebrow">Job summary</p>
              <h3>Concise outcome</h3>
            </div>
          </div>
          <p className="ff-job-summary">
            {job.jobSummary || "A concise summary will appear here once the workflow finishes."}
          </p>
        </section>
        {job.status === "quarantined" ? (
          <div className="ff-panel ff-panel--dense">
            <div className="ff-panel__head">
              <div>
                <p className="ff-eyebrow">Quarantine outcome</p>
                <h3>Part of this workflow requires review</h3>
              </div>
            </div>
            <div className="ff-key-metrics">
              <div>
                <span>Reason</span>
                <strong>{job.reason || "Part of the workflow is not supported by the current agent coverage."}</strong>
              </div>
              <div>
                <span>Next step</span>
                <strong>{job.suggestion || "The unsupported portion is quarantined until coverage is added or the request is adjusted."}</strong>
              </div>
              <div>
                <span>Quarantine status</span>
                <strong>{formatJobStatus(job.quarantineStatus || "queued_for_review")}</strong>
              </div>
              <div>
                <span>Available agents</span>
                <strong>{Array.isArray(job?.availableAgents) && job.availableAgents.length ? job.availableAgents.join(", ") : "None registered for this intent"}</strong>
              </div>
              <div>
                <span>Preferred agent</span>
                <strong>{job.preferredAgentName || "Not assigned"}</strong>
              </div>
            </div>
          </div>
        ) : null}
        {interpretationReviewVisible ? (
          <div className="ff-panel ff-panel--dense">
            <div className="ff-panel__head">
              <div>
                <p className="ff-eyebrow">Interpretation review</p>
                <h3>
                  {intentConfirmationVisible
                    ? "Review the canonical interpretation before processing starts"
                    : "This job still needs clarification before processing can continue"}
                </h3>
              </div>
            </div>
            <div className="ff-key-metrics ff-key-metrics--compact">
              <div>
                <span>Profile status</span>
                <strong>{job.profileStatus || "unknown"}</strong>
              </div>
              <div>
                <span>Suggested next step</span>
                <strong>
                  {clarification?.reason
                    || job.suggestion
                    || (intentConfirmationVisible
                      ? "Confirm if the interpretation looks correct."
                      : "Replace the mapping or add clarification before confirming intent.")}
                </strong>
              </div>
              <div>
                <span>Detected fields</span>
                <strong>{schemaColumns.length || profileColumns.length}</strong>
              </div>
              <div>
                <span>Preview rows</span>
                <strong>{schemaRows.length}</strong>
              </div>
              <div>
                <span>Validation warnings</span>
                <strong>{validationWarnings.length}</strong>
              </div>
            </div>
            {clarificationModeVisible && !shouldShowClarificationPanel ? (
              <div className="ff-template-card" style={{ marginTop: 18 }}>
                <div>
                  <strong>Resolve missing mapping</strong>
                  <span>
                    No live clarification session is active. Choose the source column for each unresolved reference below.
                  </span>
                </div>

                {availableMappingColumns.length ? (
                  <datalist id={mappingInputId}>
                    {availableMappingColumns.map((column) => (
                      <option key={column} value={column} />
                    ))}
                  </datalist>
                ) : null}

                <div className="ff-field-stack">
                  {unresolvedMappingFields.length ? unresolvedMappingFields.map((field) => (
                    <label key={field.key} className="ff-field">
                      <span>
                        {field.sourceLabel}
                        {field.contextLabel ? ` - ${field.contextLabel}` : ""}
                      </span>
                      <div className="ff-search">
                        <input
                          type="text"
                          list={availableMappingColumns.length ? mappingInputId : undefined}
                          value={mappingDraft[field.sourceKey] || ""}
                          onChange={(event) => {
                            const value = event.target.value;
                            setMappingDraft((prev) => ({
                              ...prev,
                              [field.sourceKey]: value,
                            }));
                            if (mappingError) setMappingError("");
                          }}
                          placeholder={field.placeholder}
                        />
                      </div>
                      <small className="ff-copy-muted">{field.helperText}</small>
                    </label>
                  )) : (
                    <p className="ff-copy-muted">
                      No unresolved field references were detected in the canonical intent.
                    </p>
                  )}
                </div>

                <div
                  className="ff-submit-controls"
                  style={{ display: "flex", gap: "16px", flexWrap: "wrap", alignItems: "center" }}
                >
                  <button
                    type="button"
                    className="ff-primary-button"
                    onClick={handleMappingSubmit}
                    disabled={replaceMappingMutation.isPending || !unresolvedMappingFields.length}
                  >
                    <FiCheck size={15} />
                    {replaceMappingMutation.isPending ? "Applying..." : "Apply mapping"}
                  </button>
                  <span className="ff-copy-muted">
                    This updates the canonical intent and re-evaluates the job.
                  </span>
                </div>

                {mappingError ? (
                  <p className="ff-copy-muted" role="alert">
                    {mappingError}
                  </p>
                ) : null}
              </div>
            ) : null}
          </div>
        ) : null}
        {extractionPreviewVisible ? (
          <div className="ff-panel ff-panel--dense">
            <div className="ff-panel__head">
              <div>
                <p className="ff-eyebrow">Extraction preview</p>
                <h3>Review extracted columns and rows before confirmation</h3>
              </div>
            </div>
            <div className="ff-key-metrics ff-key-metrics--compact">
              <div>
                <span>Anchor column</span>
                <strong>{extractionAnchor || "Not inferred"}</strong>
              </div>
              <div>
                <span>Complete rows</span>
                <strong>{extractionCompleteCount}</strong>
              </div>
              <div>
                <span>Partial rows</span>
                <strong>{extractionPartialCount}</strong>
              </div>
              <div>
                <span>Invalid rows</span>
                <strong>{extractionInvalidCount}</strong>
              </div>
              <div>
                <span>Extracted fields</span>
                <strong>{extractionColumns.length}</strong>
              </div>
              <div>
                <span>Recovered rows</span>
                <strong>{extractionRecoveredCount}</strong>
              </div>
              <div>
                <span>Merged rows</span>
                <strong>{extractionMergedCount}</strong>
              </div>
              <div>
                <span>LLM-only rows</span>
                <strong>{extractionLlmOnlyCount}</strong>
              </div>
            </div>
            {extractionAmbiguousDateCount ? (
              <p className="ff-copy-muted" style={{ marginTop: 16 }}>
                {extractionAmbiguousDateCount} row(s) had ambiguous slash dates. The preview assumed{" "}
                {extractionAssumedDateConvention || "DD/MM/YYYY"}.
              </p>
            ) : null}
            {extractionFields.length ? (
              <div className="ff-schema-grid" style={{ marginTop: 18 }}>
                {extractionFields.map((field) => (
                  <div key={`${field.source}-${field.target}`} className="ff-schema-card">
                    <div className="ff-schema-card__head">
                      <strong>{field.target || field.source || "Unmapped field"}</strong>
                      <span>{formatConfidence(field.confidence)}</span>
                    </div>
                    <p>
                      <span>Source:</span> {field.source || "Unknown"}
                    </p>
                    <p>
                      <span>Detected type:</span> {field.detected_type || "Unknown"}
                    </p>
                    <p>
                      <span>Why:</span> {field.reason || "Extracted from the uploaded file."}
                    </p>
                  </div>
                ))}
              </div>
            ) : null}
          </div>
        ) : null}
      </section>

      <section className="ff-detail-layout">
        {interpretationReviewVisible ? (
          <>
            <article className="ff-panel" style={{ gridRow: "span 2", marginBottom: "2rem" }}>
              <div className="ff-panel__head">
                <div>
                  <p className="ff-eyebrow">Data profile</p>
                  <h3>Source fields and deterministic profile facts</h3>
                </div>
              </div>
              <div className="ff-schema-grid">
                {profileColumns.length ? profileColumns.map((field) => (
                  <div key={`${field.name}-${field.normalized_name}`} className="ff-schema-card">
                    <div className="ff-schema-card__head">
                      <strong>{field.name || field.normalized_name || "Unnamed field"}</strong>
                      <span>{field.detected_type || field.physical_dtype || "unknown"}</span>
                    </div>
                    <p>
                      <span>Normalized:</span> {field.normalized_name || "Unknown"}
                    </p>
                    <p>
                      <span>Semantic hint:</span> {field.semantic_type_hint || "None"}
                    </p>
                    <p>
                      <span>Distinct values:</span> {field.distinct_count ?? "Unknown"}
                    </p>
                  </div>
                )) : (
                  <div className="ff-copy-muted">
                    No proposed mapping fields are available for this upload.
                  </div>
                )}
              </div>
              {detectedTypes.length ? (
                <div className="ff-tag-list" style={{ marginTop: 18 }}>
                  {detectedTypes.map(([field, type]) => (
                    <span key={field}>{field}: {String(type)}</span>
                  ))}
                </div>
              ) : null}
            </article>

            <article className="ff-panel" style={{ marginBottom: "2rem" }}>
              <div className="ff-panel__head">
                <div>
                  <p className="ff-eyebrow">Canonical intent</p>
                  <h3>Deterministic actions derived from the instruction</h3>
                </div>
              </div>
              <div className="ff-schema-grid">
                {actionSchema.length ? (
                  actionSchema.map((action, idx) => (
                    <div key={idx} className="ff-schema-card">
                      <div className="ff-schema-card__head">
                        <strong>{action?.action || "Unknown Action"}</strong>
                        <span className={`ff-status ff-status--info`}>
                          action
                        </span>
                      </div>
                      <p>
                        <span>Target Roles:</span> {Array.isArray(action?.roles) ? action.roles.join(", ") : "N/A"}
                      </p>
                      {action?.condition_tree && (
                        <p>
                          <span>Condition:</span> {JSON.stringify(action.condition_tree)}
                        </p>
                      )}
                      {action?.mapping && (
                        <p>
                          <span>Mapping:</span> {JSON.stringify(action.mapping)}
                        </p>
                      )}
                    </div>
                  ))
                ) : (
                  <div className="ff-copy-muted">
                    No execution actions detected.
                  </div>
                )}
              </div>
            </article>
          </>
        ) : null}

        <article className="ff-panel">
          <div className="ff-panel__head">
            <div>
              <p className="ff-eyebrow">Execution plan</p>
              <h3>Step-by-step timeline</h3>
            </div>
          </div>
          <div className="ff-timeline">
            {Array.isArray(job?.steps) ? job.steps.map((step, index) => (
              <div key={step.name} className="ff-timeline__item">
                <div className={`ff-timeline__dot is-${step.status}`} />
                <div className="ff-timeline__content">
                  <div className="ff-timeline__head">
                    <div>
                      <strong>
                        {index + 1}. {step.name}
                      </strong>
                      <span>{formatStepStatus(step.status)}</span>
                    </div>
                    <small>{step.time || "Pending"}</small>
                  </div>
                  <p>{step.summary}</p>
                </div>
              </div>
            )) : null}
          </div>
        </article>
        {interpretationReviewVisible || extractionPreviewVisible ? (
          <article className="ff-panel" style={{ gridColumn: "1 / -1" }}>
            <div className="ff-panel__head">
              <div>
                <p className="ff-eyebrow">Data preview</p>
                <h3>Sample rows captured from the uploaded file</h3>
              </div>
            </div>
            {(extractionPreviewVisible ? extractionColumns : schemaColumns).length ? (
              <DataTable
                columns={extractionPreviewVisible ? extractionColumns : schemaColumns}
                rows={extractionPreviewVisible ? extractionRows : schemaRows}
                pageSize={6}
                title="Uploaded data preview"
              />
            ) : (
              <div className="ff-copy-muted">
                No preview rows are available for this upload yet.
              </div>
            )}
          </article>
        ) : (
          <article className="ff-panel">
            <div className="ff-panel__head">
              <div>
                <p className="ff-eyebrow">Agent summaries</p>
                <h3>Expandable summaries from each agent</h3>
              </div>
            </div>
            <div className="ff-log-stack">
              {Array.isArray(job?.agentSummaries) && job.agentSummaries.length ? job.agentSummaries.map((entry, index) => {
                const open = openIndex === index;
                return (
                  <button
                    key={`${entry.agentId}-${entry.agentName}`}
                    type="button"
                    className={`ff-log-card${open ? " is-open" : ""}`}
                    onClick={() => setOpenIndex(open ? -1 : index)}
                  >
                    <div className="ff-log-card__head">
                      <div>
                        <strong>{entry.agentName}</strong>
                        <span>{formatJobStatus(entry.status)}</span>
                      </div>
                      <FiChevronDown size={16} />
                    </div>
                    {open && (
                      <div className="ff-log-card__body">
                        <p>{entry.summary}</p>
                        {entry.bullets.length ? (
                          <ul className="ff-summary-bullets">
                            {entry.bullets.map((bullet) => (
                              <li key={bullet}>{bullet}</li>
                            ))}
                          </ul>
                        ) : null}
                      </div>
                    )}
                  </button>
                );
              }) : (
                <div className="ff-copy-muted">
                  No agent summaries are available for this workflow yet.
                </div>
              )}
            </div>
          </article>
        )}
      </section>

      <section className="ff-panel">
        <div className="ff-panel__head">
          <div>
            <p className="ff-eyebrow">Audit trail</p>
            <h3>Chronological chain of custody</h3>
          </div>
        </div>
        <div className="ff-audit-list">
          {Array.isArray(job?.audit) ? job.audit.map((entry) => (
            <div key={`${entry.time}-${entry.action}`} className="ff-audit-row">
              <strong>{entry.time}</strong>
              <div>
                <span>{entry.action}</span>
                <p>{entry.detail}</p>
              </div>
            </div>
          )) : null}
          {(!Array.isArray(job?.audit) || !job.audit.length) && (
            <div className="ff-copy-muted">
              No audit entries are available for this workflow yet.
            </div>
          )}
        </div>
      </section>
    </div>
  );
}

async function refreshJobViews(queryClient, jobId) {
  await queryClient.invalidateQueries({ queryKey: ["jobs"] });
  await queryClient.invalidateQueries({
    queryKey: ["jobs", "detail", jobId],
  });
  await queryClient.invalidateQueries({ queryKey: ["manager-dashboard"] });
}

function formatConfidence(value) {
  const numeric = Number(value);
  if (Number.isNaN(numeric)) return "Unscored";
  return `${Math.round(numeric * 100)}% confidence`;
}

function collectAvailableMappingColumns(profileColumns, schemaColumns) {
  const values = [];
  const seen = new Set();
  const pushValue = (column) => {
    const label = formatColumnLabel(column);
    if (!label) return;
    const key = label.toLowerCase();
    if (seen.has(key)) return;
    seen.add(key);
    values.push(label);
  };

  if (Array.isArray(schemaColumns)) schemaColumns.forEach(pushValue);
  if (Array.isArray(profileColumns)) profileColumns.forEach(pushValue);
  return values;
}

function collectUnresolvedMappingFields(canonicalIntent) {
  const actions = Array.isArray(canonicalIntent?.actions) ? canonicalIntent.actions : [];
  const seen = new Set();
  const fields = [];

  const pushField = ({ sourceField, sourceKey, contextLabel, helperText, placeholder }) => {
    const label = formatFieldLabel(sourceField);
    const key = String(sourceKey || label || "").trim();
    if (!key) return;
    if (seen.has(key.toLowerCase())) return;
    seen.add(key.toLowerCase());
    fields.push({
      key,
      sourceKey: key,
      sourceLabel: label,
      contextLabel,
      helperText,
      placeholder,
      defaultValue: sourceField?.candidate_columns?.length === 1 ? formatColumnLabel(sourceField.candidate_columns[0]) : "",
    });
  };

  actions.forEach((action, actionIndex) => {
    if (!action || typeof action !== "object") return;
    const kind = String(action.kind || action.action || "").trim();
    if (kind === "project_columns" || kind === "drop_columns") {
      (Array.isArray(action.requested_fields) ? action.requested_fields : []).forEach((field, fieldIndex) => {
        if (!field || typeof field !== "object") return;
        if (field.resolved_column || (Array.isArray(field.resolved_columns) && field.resolved_columns.length)) return;
        pushField({
          sourceField: field,
          sourceKey: formatMappingSourceKey(field) || `${kind}-${actionIndex}-${fieldIndex}`,
          contextLabel: formatActionKindLabel(kind),
          helperText: "Choose one or more matching columns. Separate multiple values with commas.",
          placeholder: "Type a column name or comma-separated list",
        });
      });
      return;
    }

    if (kind === "filter_rows") {
      (Array.isArray(action.conditions) ? action.conditions : []).forEach((condition, conditionIndex) => {
        const field = condition?.field;
        if (!field || typeof field !== "object") return;
        if (field.resolved_column || (Array.isArray(field.resolved_columns) && field.resolved_columns.length)) return;
        pushField({
          sourceField: field,
          sourceKey: formatMappingSourceKey(field) || `${kind}-${actionIndex}-${conditionIndex}`,
          contextLabel: formatActionKindLabel(kind),
          helperText: "Choose the exact column that should satisfy this filter.",
          placeholder: "Type a single column name",
        });
      });
      return;
    }

    if (kind === "sort_rows") {
      (Array.isArray(action.sort_keys) ? action.sort_keys : []).forEach((sortKey, sortIndex) => {
        const field = sortKey?.column;
        if (!field || typeof field !== "object") return;
        if (field.resolved_column || (Array.isArray(field.resolved_columns) && field.resolved_columns.length)) return;
        pushField({
          sourceField: field,
          sourceKey: formatMappingSourceKey(field) || `${kind}-${actionIndex}-${sortIndex}`,
          contextLabel: formatActionKindLabel(kind),
          helperText: "Choose the exact column used for sorting.",
          placeholder: "Type a single column name",
        });
      });
      return;
    }

    if (kind === "visualize") {
      (Array.isArray(action.fields) ? action.fields : []).forEach((field, fieldIndex) => {
        if (!field || typeof field !== "object") return;
        if (field.resolved_column || (Array.isArray(field.resolved_columns) && field.resolved_columns.length)) return;
        pushField({
          sourceField: field,
          sourceKey: formatMappingSourceKey(field) || `${kind}-${actionIndex}-${fieldIndex}`,
          contextLabel: formatActionKindLabel(kind),
          helperText: "Choose the columns that should appear in the visualization.",
          placeholder: "Type a column name or comma-separated list",
        });
      });
      return;
    }

    if (kind === "rename_columns") {
      (Array.isArray(action.mapping) ? action.mapping : []).forEach((mappingItem, mappingIndex) => {
        const field = mappingItem?.source;
        if (!field || typeof field !== "object") return;
        if (field.resolved_column || (Array.isArray(field.resolved_columns) && field.resolved_columns.length)) return;
        pushField({
          sourceField: field,
          sourceKey: formatMappingSourceKey(field) || `${kind}-${actionIndex}-${mappingIndex}`,
          contextLabel: formatActionKindLabel(kind),
          helperText: "Choose the source column that should be renamed.",
          placeholder: "Type a single column name",
        });
      });
    }
  });

  return fields;
}

function parseMappingSubmissionValue(value) {
  const raw = String(value || "").trim();
  if (!raw) return null;
  const items = raw
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
  if (!items.length) return null;
  return items.length === 1 ? items[0] : items;
}

function formatMappingSourceKey(field) {
  if (!field || typeof field !== "object") return "";
  const candidates = [
    field.raw_reference,
    field.resolved_column,
    ...(Array.isArray(field.candidate_columns) ? field.candidate_columns : []),
    field.name,
    field.column,
    field.target,
    field.source,
    field.label,
  ];
  return candidates.map((value) => String(value || "").trim()).find(Boolean) || "";
}

function formatFieldLabel(field) {
  if (!field || typeof field !== "object") return "Unresolved field";
  return (
    String(field.raw_reference || "").trim()
    || String(field.resolved_column || "").trim()
    || String(field.name || "").trim()
    || String(field.column || "").trim()
    || String(field.target || "").trim()
    || String(field.source || "").trim()
    || String(field.label || "").trim()
    || "Unresolved field"
  );
}

function formatActionKindLabel(kind) {
  const normalized = String(kind || "").trim();
  if (normalized === "project_columns") return "project columns";
  if (normalized === "drop_columns") return "drop columns";
  if (normalized === "filter_rows") return "filter rows";
  if (normalized === "sort_rows") return "sort rows";
  if (normalized === "rename_columns") return "rename columns";
  if (normalized === "visualize") return "visualize";
  return normalized || "clarification";
}

function formatColumnLabel(column) {
  if (typeof column === "string") return column.trim();
  if (!column || typeof column !== "object") return "";
  return (
    String(column.name || "").trim()
    || String(column.normalized_name || "").trim()
    || String(column.label || "").trim()
    || String(column.source || "").trim()
    || String(column.field || "").trim()
    || String(column.value || "").trim()
    || String(column.column || "").trim()
  );
}
