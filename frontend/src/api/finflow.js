import { api } from "./client.js";

export function mapWorkflowStatus(status) {
  const normalized = String(status || "").trim().toLowerCase();
  if (normalized === "awaiting_clarification") return "clarification";
  if (normalized === "awaiting_confirmation") return "clarification";
  if (["pending", "queued", "planning", "running"].includes(normalized)) return "running";
  if (normalized === "awaiting_schema_approval") return "clarification";
  if (normalized === "quarantined") return "quarantined";
  if (["succeeded", "complete"].includes(normalized)) return "complete";
  if (["failed", "callback_failed", "declined"].includes(normalized)) return "failed";
  // Validation fallback: ensure "awaiting_clarification" is always caught
  // even if normalization is bypassed or status arrives in unexpected format
  if (String(status || "").includes("clarification")) return "clarification";
  return "running";
}

function deriveWorkflowStatus(item) {
  const rawStatus = String(item?.status || "").trim().toLowerCase();
  return mapWorkflowStatus(rawStatus);
}

function isCompletedWorkflowStatus(status) {
  const normalized = String(status || "").trim().toLowerCase();
  return normalized === "complete" || normalized === "succeeded";
}

export function formatDisplayJobId(subId, backendId) {
  if (subId) return `FF-${String(subId).padStart(5, "0")}`;
  return `FF-${String(backendId).slice(0, 8).toUpperCase()}`;
}

function buildTitle(instruction, filename) {
  const trimmedInstruction = String(instruction || "").trim();
  if (!trimmedInstruction) return filename;
  return trimmedInstruction.length > 72 ? `${trimmedInstruction.slice(0, 69)}...` : trimmedInstruction;
}

function buildSummarySteps(status, workflowStatus) {
  if (workflowStatus === "running") {
    return [
      { name: "Ingestion", status: "complete" },
      { name: "Agent execution", status: "running" },
      { name: "Output", status: "running" },
    ];
  }
  if (workflowStatus === "complete") {
    return [
      { name: "Ingestion", status: "complete" },
      { name: "Agent execution", status: "complete" },
      { name: "Output", status: "complete" },
    ];
  }
  if (workflowStatus === "failed") {
    return [
      { name: "Ingestion", status: "complete" },
      { name: "Agent execution", status: "failed" },
      { name: "Output", status: "failed" },
    ];
  }
  if (workflowStatus === "quarantined") {
    return [
      { name: "Ingestion", status: "complete" },
      { name: "Quarantine review", status: "blocked" },
      { name: "Output", status: "blocked" },
    ];
  }
  if (workflowStatus === "clarification") {
    return [
      { name: "Ingestion", status: "complete" },
      { name: "Interpretation review", status: "blocked" },
      { name: "Output", status: "blocked" },
    ];
  }
  return [
    { name: "Ingestion", status: "complete" },
    { name: "Agent execution", status: "running" },
    { name: "Output", status: "running" },
  ];
}

function mapAgentSummaries(entries) {
  if (!Array.isArray(entries)) return [];
  return entries
    .filter((entry) => entry && typeof entry === "object")
    .map((entry) => ({
      agentId: String(entry.agent_id || entry.agentId || entry.agent || "agent"),
      agentName: String(entry.agent_name || entry.agentName || entry.label || entry.agent_id || "Agent"),
      status: String(entry.status || "complete"),
      summary: String(entry.summary || entry.detail || entry.description || "No summary available."),
      bullets: Array.isArray(entry.bullets) ? entry.bullets.map((bullet) => String(bullet).trim()).filter(Boolean) : [],
    }));
}

export function mapUploadSummaryToJob(upload) {
  const completedAt = upload.reviewed_at || (isCompletedWorkflowStatus(upload.status) ? upload.created_at : null);
  const workflowStatus = deriveWorkflowStatus(upload);
  const summary = upload.summary || {};
  return {
    id: formatDisplayJobId(upload.sub_id, upload.id),
    backendId: upload.id,
    title: buildTitle(upload.instruction, upload.filename),
    instruction: upload.instruction || "No prompt captured for this workflow.",
    fileName: upload.filename,
    outputFormat: upload.output_format || "XLSX",
    status: workflowStatus,
    rawStatus: upload.status,
    agentStatus: upload.status,
    reason: summary.reason || "",
    suggestion: summary.suggestion || "",
    quarantineStatus: "",
    availableAgents: Array.isArray(upload.available_agents) ? upload.available_agents : [],
    agentError: summary.error || "",
    submittedAt: upload.created_at,
    completedAt,
    submittedBy: upload.uploader_name,
    preferredAgentName: upload.preferred_agent_name || "",
    outputReady: Boolean(upload.output_ready),
    jobSummary: upload.job_summary || "",
    agentSummaries: mapAgentSummaries(upload.agent_summaries),
    steps: buildSummarySteps(upload.status, workflowStatus),
  };
}

export function mapJobDetail(detail) {
  const completedAt = detail.completed_at || (isCompletedWorkflowStatus(detail.status) ? detail.submitted_at : null);
  const workflowStatus = deriveWorkflowStatus(detail);
  const summary = detail.summary || {};
  const dataProfile = detail.data_profile || {};
  const canonicalIntent = detail.canonical_intent || {};
  const clarification = detail.clarification || null;
  const execution = detail.execution || null;
  return {
    id: formatDisplayJobId(detail.sub_id, detail.id),
    backendId: detail.id,
    title: detail.title,
    instruction: detail.instruction || "No prompt captured for this workflow.",
    fileName: detail.file_name,
    outputFormat: detail.output_format || "XLSX",
    status: workflowStatus,
    rawStatus: detail.status,
    agentStatus: detail.status,
    reason: summary.reason || "",
    suggestion: summary.suggestion || "",
    quarantineStatus: "",
    availableAgents: Array.isArray(detail.available_agents) ? detail.available_agents : [],
    agentError: summary.error || "",
    submittedBy: detail.submitted_by,
    submittedAt: detail.submitted_at,
    completedAt,
    preferredAgentName: detail.preferred_agent_name || "",
    outputReady: Boolean(detail.output_ready),
    jobSummary: detail.job_summary || "",
    agentSummaries: mapAgentSummaries(detail.agent_summaries),
    steps: detail.steps || [],
    audit: detail.audit || [],
    columns: Array.isArray(detail.columns) ? detail.columns : [],
    detectedTypes: detail.detected_types || {},
    validation: detail.validation || {},
    previewRows: Array.isArray(detail.preview_rows) ? detail.preview_rows : [],
    dataProfile,
    profileStatus: detail.profile_status || "",
    canonicalIntent,
    intentStatus: detail.intent_status || "",
    clarification,
    execution,
    repairAvailable: Boolean(detail.repair_available),
    extractionPreview: clarification?.extraction_preview || null,
    previewToken: detail.preview_token || "",
  };
}

export async function fetchJobs() {
  const response = await api.get("/uploads");
  return response.data.map(mapUploadSummaryToJob);
}

export async function fetchJobDetail(jobId) {
  const response = await api.get(`/uploads/${jobId}/job-detail`);
  return mapJobDetail(response.data);
}

export async function submitJob(payload) {
  const formData = new FormData();
  formData.append("file", payload.file);
  formData.append("instruction", payload.instruction);
  formData.append("output_format", payload.outputFormat);

  const response = await api.post("/uploads", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
  return response.data;
}

export async function confirmExtraction(jobId, previewToken) {
  const response = await api.post(`/uploads/${jobId}/confirm-extraction`, {
    preview_token: previewToken,
  });
  return response.data;
}

export async function fetchRegisteredAgents() {
  const response = await api.get("/agent/registry");
  return response.data;
}

export async function fetchQuarantinedJobs() {
  const response = await api.get("/admin/quarantined-jobs");
  return response.data.map(mapUploadSummaryToJob);
}

export async function fetchManagers() {
  const response = await api.get("/admin/managers");
  return Array.isArray(response.data) ? response.data : [];
}

export async function fetchEmployees() {
  const response = await api.get("/admin/employees");
  return Array.isArray(response.data) ? response.data : [];
}

export async function assignEmployee(employeeId, managerId) {
  const response = await api.post("/admin/assign", {
    employee_id: employeeId,
    manager_id: managerId,
  });
  return response.data;
}

export async function reassignEmployee(employeeId, managerId) {
  const response = await api.post("/admin/reassign", {
    employee_id: employeeId,
    manager_id: managerId,
  });
  return response.data;
}

export async function retryQuarantinedJob(jobId) {
  const response = await api.post(`/admin/quarantined-jobs/${jobId}/retry`);
  return response.data;
}

export async function assignQuarantinedJob(jobId, preferredAgentName) {
  const response = await api.post(`/admin/quarantined-jobs/${jobId}/assign`, {
    preferred_agent_name: preferredAgentName,
  });
  return response.data;
}

export async function fetchAnalyticsKpis() {
  const response = await api.get("/analytics/kpis");
  return response.data;
}

export async function fetchUploadMetadata() {
  const response = await api.get("/uploads/metadata");
  return response.data;
}

export async function downloadJobOutput(jobId) {
  const response = await api.get(`/uploads/${jobId}/download`, {
    responseType: "blob",
  });

  const contentDisposition = response.headers["content-disposition"] || "";
  const filename = parseDownloadFilename(contentDisposition) || `finflow-output-${jobId}`;
  const blobUrl = window.URL.createObjectURL(response.data);
  const link = document.createElement("a");
  link.href = blobUrl;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.URL.revokeObjectURL(blobUrl);
}

export async function retryJob(jobId) {
  const response = await api.post(`/uploads/${jobId}/retry`);
  return response.data;
}

export async function fetchClarificationStatus(submissionId) {
  const response = await api.get(`/uploads/${submissionId}/clarification-status`);
  return response.data;
}

export async function confirmInterpretation(jobId, reason = null) {
  const response = await api.post(`/uploads/${jobId}/confirm-interpretation`, { reason });
  return response.data;
}

export async function rejectInterpretation(jobId, reason = null) {
  const response = await api.post(`/uploads/${jobId}/reject-interpretation`, { reason });
  return response.data;
}

export async function replaceColumnMapping(jobId, mapping, reason = null) {
  const response = await api.post(`/uploads/${jobId}/replace-column-mapping`, {
    mapping,
    reason,
  });
  return response.data;
}

function parseDownloadFilename(contentDisposition) {
  if (!contentDisposition) return "";

  const utfMatch = contentDisposition.match(/filename\*\s*=\s*UTF-8''([^;]+)/i);
  if (utfMatch?.[1]) {
    try {
      return decodeURIComponent(utfMatch[1]);
    } catch {
      return utfMatch[1];
    }
  }

  const basicMatch = contentDisposition.match(/filename\s*=\s*"([^"]+)"/i)
    || contentDisposition.match(/filename\s*=\s*([^;]+)/i);
  if (!basicMatch?.[1]) return "";
  return basicMatch[1].trim();
}
