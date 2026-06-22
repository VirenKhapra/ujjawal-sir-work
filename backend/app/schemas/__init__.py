from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    role: str = Field(default="employee", pattern="^(employee|manager)$")


class UserRead(BaseModel):
    id: UUID
    name: str
    email: EmailStr
    role: str
    manager_id: UUID | None = None
    manager_name: str | None = None


class AdminUserRead(UserRead):
    assigned_employee_count: int = 0


class AdminEmployeeRead(UserRead):
    assignment_status: str


class AssignmentRequest(BaseModel):
    employee_id: UUID
    manager_id: UUID


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class AccountUpdateRequest(BaseModel):
    name: str = Field(min_length=2, max_length=120)


class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str = Field(min_length=8, max_length=128)


class PasswordChangeVerifyRequest(BaseModel):
    token: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserRead


class AgentTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UploadVersionRead(BaseModel):
    id: UUID
    filename: str
    status: str
    version_number: int
    created_at: datetime
    reviewed_at: datetime | None = None


class UploadPreview(BaseModel):
    upload_id: UUID
    sub_id: int | None = None
    filename: str
    instruction: str = ""
    output_format: str = "XLSX"
    status: str
    version_number: int = 1
    parent_submission_id: UUID | None = None
    total_rows: int
    total_columns: int
    created_at: datetime | None = None
    reviewed_at: datetime | None = None
    columns: list[str]
    detected_types: dict = Field(default_factory=dict)
    validation: dict = Field(default_factory=dict)
    preview_rows: list[dict]
    version_history: list[UploadVersionRead] = Field(default_factory=list)
    preferred_agent_name: str | None = None
    job_summary: str | None = None
    agent_summaries: list["JobAgentSummaryRead"] = Field(default_factory=list)
    data_profile: dict = Field(default_factory=dict)
    profile_status: str | None = None
    canonical_intent: dict = Field(default_factory=dict)
    intent_status: str | None = None
    clarification: dict | None = None
    execution: dict | None = None
    repair_available: bool = False


class UploadSummary(BaseModel):
    id: UUID
    sub_id: int | None = None
    filename: str
    instruction: str = ""
    id: UUID
    sub_id: int | None = None
    filename: str
    instruction: str = ""
    output_format: str = "XLSX"
    status: str
    version_number: int = 1
    parent_submission_id: UUID | None = None
    total_rows: int
    total_columns: int
    uploader_name: str | None = None
    validation_passed: bool = True
    created_at: datetime
    reviewed_at: datetime | None = None
    summary: dict | None = None
    available_agents: list[str] = Field(default_factory=list)
    preferred_agent_name: str | None = None
    output_ready: bool = False
    job_summary: str | None = None
    agent_summaries: list["JobAgentSummaryRead"] = Field(default_factory=list)


class UploadMetadataRead(BaseModel):
    accepted_file_types: list[str] = Field(default_factory=list)
    output_format_options: list[str] = Field(default_factory=list)
    max_upload_size_mb: int


class ApprovalRequest(BaseModel):
    manager_id: UUID | None = None
    comment: str | None = Field(default=None, max_length=2000)


class ApprovalActionRequest(ApprovalRequest):
    upload_id: UUID


class RejectRequest(ApprovalRequest):
    pass


class RejectActionRequest(RejectRequest):
    upload_id: UUID


class SubmissionCommentCreate(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class SubmissionCommentRead(BaseModel):
    id: UUID
    submission_id: UUID
    user_id: UUID
    user_name: str
    user_role: str
    message: str
    created_at: datetime


class AlertCreate(BaseModel):
    type: str = Field(default="dtcd_validation")
    entry_no: str = Field(alias="Entry no", min_length=1, max_length=80)
    account_code: str = Field(alias="Account code", min_length=1, max_length=80)
    sub_account: str = Field(alias="Sub Account", min_length=1, max_length=255)
    difference: float
    status: str = Field(default="FAILED", max_length=40)


class AlertRead(BaseModel):
    id: UUID
    alert_type: str = "transaction_validation"
    title: str | None = None
    message: str | None = None
    upload_id: UUID | None = None
    entry_no: str
    account_code: str
    sub_account: str
    difference: float
    status: str
    is_read: bool
    created_at: datetime
    transaction_id: str | None = None
    upload_id: UUID | None = None
    debit_account_name: str | None = None
    debit_account_code: str | None = None
    debit_amount: float | None = None
    credit_account_name: str | None = None
    credit_account_code: str | None = None
    credit_amount: float | None = None


class AgentRegisterRequest(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    description: str = Field(min_length=4, max_length=2000)
    capability_tags: list[str] = Field(default_factory=list)
    input_formats: list[str] = Field(default_factory=list)
    output_formats: list[str] = Field(default_factory=list)
    endpoint_url: str | None = None
    status: str = Field(default="active", max_length=40)


class QuarantineAssignRequest(BaseModel):
    preferred_agent_name: str = Field(min_length=2, max_length=120)


class AgentRead(BaseModel):
    id: UUID
    name: str
    description: str
    capability_tags: list[str] = Field(default_factory=list)
    input_formats: list[str] = Field(default_factory=list)
    output_formats: list[str] = Field(default_factory=list)
    endpoint_url: str | None = None
    status: str
    last_heartbeat: datetime | None = None
    total_invocations: int = 0
    registered_at: datetime


class JobStepRead(BaseModel):
    name: str
    status: str
    summary: str
    time: str | None = None


class JobAuditEntryRead(BaseModel):
    time: str
    action: str
    detail: str


class JobAgentSummaryRead(BaseModel):
    agent_id: str
    agent_name: str
    status: str
    summary: str
    bullets: list[str] = Field(default_factory=list)


class JobDetailRead(BaseModel):
    id: UUID
    sub_id: int | None = None
    title: str
    instruction: str
    file_name: str
    output_format: str
    status: str
    submitted_by: str | None = None
    submitted_at: datetime | None = None
    completed_at: datetime | None = None
    summary: dict | None = None
    available_agents: list[str] = Field(default_factory=list)
    preferred_agent_name: str | None = None
    output_ready: bool = False
    job_summary: str | None = None
    agent_summaries: list[JobAgentSummaryRead] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    detected_types: dict = Field(default_factory=dict)
    validation: dict = Field(default_factory=dict)
    preview_rows: list[dict] = Field(default_factory=list)
    data_profile: dict = Field(default_factory=dict)
    profile_status: str | None = None
    canonical_intent: dict = Field(default_factory=dict)
    intent_status: str | None = None
    clarification: dict | None = None
    execution: dict | None = None
    repair_available: bool = False
    preview_token: str | None = None
    steps: list[JobStepRead] = Field(default_factory=list)
    audit: list[JobAuditEntryRead] = Field(default_factory=list)
