from typing import Literal
import os
import pandas as pd
from pydantic import BaseModel, ValidationError
from finflow_agent.registry import registry, AgentSpec
from finflow_agent.state import AgentResult
from finflow_agent.tools.path_safety import get_safe_input_path
from finflow_agent.operations.errors import UnsafeInputPathError

class IngestionAgentParams(BaseModel):
    resolved_file_path: str
    file_type: Literal["xlsx", "xls", "csv"]

@registry.register
class IngestionAgent:
    spec = AgentSpec(
        name="ingestion_agent",
        description="Parses XLSX, XLS, and CSV files into a structured dataframe.",
        stage="ingest",
        accepts=["file"],
        produces=["dataframe"],
        params_schema={
            "resolved_file_path": {"type": "string"},
            "file_type": {"type": "string"}
        }
    )
    # Pydantic params model picked up by the registry so the validator and
    # engine can re-validate `step.params` before this agent is invoked.
    params_model = IngestionAgentParams

    def execute(self, params: dict, input_data: dict) -> AgentResult:
        # Strict parameter validation
        try:
            validated = IngestionAgentParams.model_validate(params)
        except ValidationError as e:
            return AgentResult(
                status="failed",
                error_message=f"Invalid parameter schema for IngestionAgent: {str(e)}"
            )

        resolved_file_path = validated.resolved_file_path
        file_type = validated.file_type.lower()

        if file_type in ["png", "jpg", "jpeg", "gif"]:
            return AgentResult(status="failed", error_message="Image files are not supported.")

        if file_type not in ["xlsx", "xls", "csv"]:
            return AgentResult(status="failed", error_message=f"Unsupported file type: {file_type}")

        # Path safety: when UPLOAD_DIR is configured, enforce a sandbox boundary
        # so a malformed or malicious upstream cannot make us read files outside
        # the configured upload directory (e.g. via "..", an absolute path to
        # /etc/passwd, or a Windows system file). When UPLOAD_DIR is unset we
        # fall back to the legacy existence check for back-compat with callers
        # that have not been migrated yet.
        upload_dir = os.environ.get("UPLOAD_DIR")
        if upload_dir:
            try:
                safe_path = get_safe_input_path(upload_dir, resolved_file_path)
            except UnsafeInputPathError as exc:
                return AgentResult(
                    status="failed",
                    error_message=f"Unsafe input path: {exc}",
                )
            resolved_file_path = str(safe_path)
        else:
            if not os.path.exists(resolved_file_path):
                return AgentResult(status="failed", error_message=f"File not found: {resolved_file_path}")

        try:
            if file_type == "csv":
                df = pd.read_csv(resolved_file_path)
            else:
                df = pd.read_excel(resolved_file_path)

            row_count = len(df)
            column_count = len(df.columns)

            from finflow_agent.tools.dataframe_profile import profile_dataframe
            profile = profile_dataframe(df, include_samples=False)

            return AgentResult(
                status="success",
                data=df,
                summary=f"Successfully ingested {file_type.upper()} file with {row_count} rows and {column_count} columns.",
                metrics={
                    "row_count": row_count,
                    "column_count": column_count,
                    "profile": profile.model_dump(mode="json")
                }
            )
        except Exception as e:
            return AgentResult(status="failed", error_message=f"Failed to parse file: {str(e)}")
