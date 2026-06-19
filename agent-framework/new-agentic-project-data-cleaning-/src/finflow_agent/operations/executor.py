import pandas as pd
import time
import uuid
from typing import Dict, Any

from finflow_agent.state import ExecutionOutput
from finflow_agent.operations.schemas import (
    CleaningOperationPlan, FilterOperationPlan, CalculationOperationPlan,
    VisualizationOperationPlan, ReportingOperationPlan
)
from finflow_agent.operations.errors import UnsupportedOperationError
from finflow_agent.operations.cleaning_handlers import CLEANING_HANDLERS
from finflow_agent.operations.filter_handlers import FILTER_HANDLERS
from finflow_agent.operations.calculation_handlers import CALCULATION_HANDLERS
from finflow_agent.operations.visualization_handlers import VISUALIZATION_HANDLERS
from finflow_agent.operations.reporting_handlers import REPORTING_HANDLERS
from finflow_agent.operations.validators import required_columns_for_operation, validate_columns_exist, hash_operation_params

def execute_cleaning_plan(df: pd.DataFrame, plan: CleaningOperationPlan) -> ExecutionOutput:
    output = ExecutionOutput(data=df.copy())
    
    for op in plan.operations:
        req_cols = required_columns_for_operation(op)
        validate_columns_exist(output.data, req_cols)
        
        started_at = int(time.time() * 1000)
        initial_rows = len(output.data)
        input_cols = list(output.data.columns)
        
        handler = CLEANING_HANDLERS.get(op.type)
        if not handler:
            raise UnsupportedOperationError(f"No cleaning handler found for {op.type}")
            
        metrics = handler(output.data, op)
        if metrics is None:
            metrics = {}
            
        finished_at = int(time.time() * 1000)
        output_cols = list(output.data.columns)
        
        cols_mod = list(set(input_cols) ^ set(output_cols))
        targeted = []
        if hasattr(op, "column") and getattr(op, "column"):
            targeted.append(op.column)
        if hasattr(op, "columns"):
            cols = op.columns
            if cols != "__all_string_columns__":
                if isinstance(cols, str):
                    targeted.append(cols)
                elif isinstance(cols, list):
                    targeted.extend(cols)
        for c in targeted:
            if c in output_cols and c not in cols_mod:
                cols_mod.append(c)
                
        op_warnings = metrics.get("warnings", [])
        if op_warnings:
            output.warnings.extend(op_warnings)
            
        output.operations_applied.append({
            "operation_id": f"op_{uuid.uuid4().hex[:8]}",
            "operation_type": op.type,
            "type": op.type,
            "input_row_count": initial_rows,
            "output_row_count": len(output.data),
            "input_columns": input_cols,
            "output_columns": output_cols,
            "columns_modified": cols_mod,
            "warnings": op_warnings,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_ms": finished_at - started_at,
            "params_hash": hash_operation_params(op),
            "changed_values": initial_rows - len(output.data) if initial_rows != len(output.data) else 0,
            **{k: v for k, v in metrics.items() if k not in ["warnings"]}
        })
        
    output.summary = f"Successfully applied {len(plan.operations)} cleaning operations."
    return output

def execute_filter_plan(df: pd.DataFrame, plan: FilterOperationPlan) -> ExecutionOutput:
    output = ExecutionOutput(data=df.copy())
    initial_rows = len(output.data)
    input_cols = list(output.data.columns)
    
    started_at = int(time.time() * 1000)
    
    if plan.select_columns:
        validate_columns_exist(output.data, plan.select_columns)
        
    if not plan.conditions:
        if plan.select_columns:
            output.data = output.data[plan.select_columns]
        if plan.limit:
            output.data = output.data.head(plan.limit)
            
        finished_at = int(time.time() * 1000)
        output_cols = list(output.data.columns)
        cols_mod = list(set(input_cols) ^ set(output_cols))
        
        output.operations_applied.append({
            "operation_id": f"op_{uuid.uuid4().hex[:8]}",
            "operation_type": "filter_select",
            "type": "filter_select",
            "input_row_count": initial_rows,
            "output_row_count": len(output.data),
            "input_columns": input_cols,
            "output_columns": output_cols,
            "columns_modified": cols_mod,
            "warnings": [],
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_ms": finished_at - started_at,
            "params_hash": hash_operation_params(plan.model_dump())
        })
        output.summary = "No filter conditions applied."
        return output
        
    masks = []
    for cond in plan.conditions:
        validate_columns_exist(output.data, cond.column)
            
        handler = FILTER_HANDLERS.get(cond.operator)
        if not handler:
            raise UnsupportedOperationError(f"No filter handler found for {cond.operator}")
            
        mask = handler(output.data[cond.column], cond)
        masks.append(mask)
        
    if masks:
        final_mask = masks[0]
        if plan.logic == "and":
            for m in masks[1:]:
                final_mask = final_mask & m
        elif plan.logic == "or":
            for m in masks[1:]:
                final_mask = final_mask | m
        output.data = output.data[final_mask]
        
    if plan.select_columns:
        output.data = output.data[plan.select_columns]
        
    if plan.limit:
        output.data = output.data.head(plan.limit)
        
    finished_at = int(time.time() * 1000)
    output_cols = list(output.data.columns)
    cols_mod = list(set(input_cols) ^ set(output_cols))
    if plan.select_columns:
        cols_mod.extend([c for c in plan.select_columns if c not in cols_mod])
        
    output.operations_applied.append({
        "operation_id": f"op_{uuid.uuid4().hex[:8]}",
        "operation_type": "filter",
        "type": "filter",
        "input_row_count": initial_rows,
        "output_row_count": len(output.data),
        "input_columns": input_cols,
        "output_columns": output_cols,
        "columns_modified": cols_mod,
        "warnings": [],
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_ms": finished_at - started_at,
        "params_hash": hash_operation_params(plan.model_dump())
    })
    
    output.summary = f"Filtered from {initial_rows} to {len(output.data)} rows using {plan.logic} logic."
    return output

def execute_calculation_plan(df: pd.DataFrame, plan: CalculationOperationPlan) -> ExecutionOutput:
    output = ExecutionOutput(data=df.copy())
    
    for op in plan.operations:
        req_cols = required_columns_for_operation(op)
        validate_columns_exist(output.data, req_cols)
        
        started_at = int(time.time() * 1000)
        initial_rows = len(output.data)
        input_cols = list(output.data.columns)
        
        handler = CALCULATION_HANDLERS.get(op.type)
        if not handler:
            raise UnsupportedOperationError(f"No calculation handler found for {op.type}")
             
        res = handler(output.data, op)
        if res is None:
            res = {}
            
        if "metrics" in res:
            output.metrics.update(res["metrics"])
        if "df" in res:
            output.data = res["df"]
        if "warnings" in res:
            output.warnings.extend(res["warnings"])
            
        finished_at = int(time.time() * 1000)
        output_cols = list(output.data.columns)
        
        cols_mod = list(set(input_cols) ^ set(output_cols))
        if op.output_column and op.output_column in output_cols and op.output_column not in cols_mod:
            cols_mod.append(op.output_column)
            
        output.operations_applied.append({
            "operation_id": f"op_{uuid.uuid4().hex[:8]}",
            "operation_type": op.type,
            "type": op.type,
            "input_row_count": initial_rows,
            "output_row_count": len(output.data),
            "input_columns": input_cols,
            "output_columns": output_cols,
            "columns_modified": cols_mod,
            "warnings": res.get("warnings", []),
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_ms": finished_at - started_at,
            "params_hash": hash_operation_params(op),
            "column": op.column
        })
        
    output.summary = f"Successfully calculated {len(plan.operations)} metrics."
    return output

def execute_visualization_plan(df: pd.DataFrame, plan: VisualizationOperationPlan) -> ExecutionOutput:
    output = ExecutionOutput(data=df.copy())
    
    for chart in plan.charts:
        validate_columns_exist(output.data, chart.x)
        validate_columns_exist(output.data, chart.y)
        
        started_at = int(time.time() * 1000)
        initial_rows = len(output.data)
        input_cols = list(output.data.columns)
        
        handler = VISUALIZATION_HANDLERS.get(chart.type)
        if not handler:
            raise UnsupportedOperationError(f"No visualization handler found for {chart.type}")
            
        res = handler(output.data, chart)
        output.artifacts[res["chart_id"]] = res["spec"]
        
        finished_at = int(time.time() * 1000)
        output_cols = list(output.data.columns)
        
        output.operations_applied.append({
            "operation_id": f"op_{uuid.uuid4().hex[:8]}",
            "operation_type": "visualization",
            "type": "visualization",
            "input_row_count": initial_rows,
            "output_row_count": len(output.data),
            "input_columns": input_cols,
            "output_columns": output_cols,
            "columns_modified": [],
            "warnings": [],
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_ms": finished_at - started_at,
            "params_hash": hash_operation_params(chart),
            "chart_type": chart.type
        })
        
    output.summary = f"Successfully generated {len(plan.charts)} chart specifications."
    return output

def execute_reporting_plan(df: pd.DataFrame, plan: ReportingOperationPlan, output_dir: str, file_prefix: str, chart_configs: list = None) -> ExecutionOutput:
    output = ExecutionOutput(data=df.copy())
    
    started_at = int(time.time() * 1000)
    initial_rows = len(output.data)
    input_cols = list(output.data.columns)
    
    handler = REPORTING_HANDLERS.get(plan.output_format)
    if not handler:
        raise UnsupportedOperationError(f"No reporting handler found for format {plan.output_format}")
        
    res = handler(output.data, plan, output_dir, file_prefix, chart_configs=chart_configs)
    output.artifacts.update(res)
    
    finished_at = int(time.time() * 1000)
    output_cols = list(output.data.columns)
    
    output.operations_applied.append({
        "operation_id": f"op_{uuid.uuid4().hex[:8]}",
        "operation_type": "reporting",
        "type": "reporting",
        "input_row_count": initial_rows,
        "output_row_count": len(output.data),
        "input_columns": input_cols,
        "output_columns": output_cols,
        "columns_modified": [],
        "warnings": [],
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_ms": finished_at - started_at,
        "params_hash": hash_operation_params(plan),
        "format": plan.output_format
    })
    
    output.summary = f"Successfully exported {plan.output_format} report to {res.get('output_file_path')}."
    return output
