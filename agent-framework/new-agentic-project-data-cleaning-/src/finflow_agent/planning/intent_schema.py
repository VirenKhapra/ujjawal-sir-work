from typing import Literal, Optional

from pydantic import BaseModel, model_validator
from finflow_agent.operations.schemas import (
    CleaningOperationPlan,
    FilterOperationPlan,
    CalculationOperationPlan,
    VisualizationOperationPlan
)
from finflow_agent.planning.normalizer import PLAN_SCHEMA_VERSION

class PlanIntent(BaseModel):
    schema_version: str = PLAN_SCHEMA_VERSION
    is_quarantined: bool = False
    quarantine_reason: Optional[str] = None
    needs_cleaning: bool = False
    needs_filtering: bool = False
    needs_calculation: bool = False
    needs_visualization: bool = False
    output_format: Literal["xlsx", "csv", "json", "txt"] = "xlsx"

    cleaning_plan: Optional[CleaningOperationPlan] = None
    filter_plan: Optional[FilterOperationPlan] = None
    calculation_plan: Optional[CalculationOperationPlan] = None
    visualization_plan: Optional[VisualizationOperationPlan] = None
    reporting_title: Optional[str] = None
    sheet_name: Optional[str] = None

    @model_validator(mode="after")
    def _require_plans_for_requested_capabilities(self) -> "PlanIntent":
        if self.is_quarantined:
            return self

        required_plan_pairs = [
            ("needs_cleaning", "cleaning_plan"),
            ("needs_filtering", "filter_plan"),
            ("needs_calculation", "calculation_plan"),
            ("needs_visualization", "visualization_plan"),
        ]
        for flag_name, plan_name in required_plan_pairs:
            if getattr(self, flag_name) and getattr(self, plan_name) is None:
                raise ValueError(f"{flag_name} is true but {plan_name} is missing")

        return self
