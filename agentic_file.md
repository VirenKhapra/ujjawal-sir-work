# FinFlow — Agentic Data Processing Platform

## Complete Architecture & Code Reference

---

## 1. What is FinFlow?

FinFlow is an AI-powered data processing platform. Users upload files (CSV, Excel, PDF), write natural language instructions, and receive cleaned/transformed/visualized outputs. The system uses a multi-agent pipeline powered by Groq LLM to interpret instructions and execute them.

**Key capabilities:**
- Data cleaning (trim, normalize, deduplicate, handle nulls, remove negatives)
- Row filtering with aggregate comparisons (e.g., "rows where amount > average")
- Calculations (sum, mean, group_by, percentage, quarterly aggregation, cross-tabulation)
- Multi-chart visualization (pie, bar, line, scatter)
- PDF table extraction
- Multi-step natural language processing

---

## 2. System Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────┐     ┌──────────────────────┐
│   Frontend  │────▶│  Backend (8000)   │────▶│  Redis  │────▶│  Agent Service (8001)│
│  React+Vite │◀────│  FastAPI+Postgres │◀────│  Queue  │◀────│  ARQ Worker + Engine │
└─────────────┘     └──────────────────┘     └─────────┘     └──────────────────────┘
                            │                                           │
                            ▼                                           ▼
                     PostgreSQL (5433)                        Shared Docker Volume
                     - submissions                           /app/storage/uploads/
                     - job_visualizations                    /app/storage/outputs/
                     - data_profiles
```

---

## 3. Complete Flow: Prompt → Output

### Step 1: Upload & Intent Extraction
- **File:** `backend/app/api/uploads.py` → `create_upload()`
- User uploads file + instruction
- File saved to `/app/storage/uploads/{id}.csv`
- `build_canonical_intent()` called → LLM or regex extracts structured actions

### Step 2: Dispatch to Redis
- **File:** `backend/app/services/agent_dispatcher.py` → `enqueue_submission_dispatch()`
- Payload (submission_id, file_path, canonical_intent) pushed to Redis
- Submission status → "planning"

### Step 3: Agent Worker Picks Up Job
- **File:** `agent-framework/.../src/finflow_agent/api.py` → `process_job_task()`
- ARQ worker consumes from Redis
- Validates payload, resolves file path

### Step 4: Compile Execution Plan
- **File:** `agent-framework/.../src/finflow_agent/planning/compiler.py`
- `compile_canonical_intent()` → `_canonical_intent_to_plan_intent()` → `compile_intent_to_plan()`
- Produces ordered `ExecutionPlan`: [ingest → clean → filter → calculate → visualize → report]

### Step 5: Execute DAG
- **File:** `agent-framework/.../src/finflow_agent/execution/engine.py`
- `ExecutionEngine.execute()` builds LangGraph state machine
- Runs agents in topological order via `PipelineState`

### Step 6: Individual Agents Execute
- Each agent: validates params → builds input from predecessors → executes → returns `AgentResult`
- DataFrame flows in-memory between agents

### Step 7: Callback to Backend
- **File:** `agent-framework/.../src/finflow_agent/jobs/callbacks.py`
- HTTP POST to backend with status, output_path, summary, visualizations

### Step 8: Frontend Displays Result
- Backend updates submission → frontend polls job-detail → shows download + charts

---

## 4. File Reference: Agent Framework

### Core Files

| File | Function |
|------|----------|
| `api.py` | Real orchestrator. ARQ worker entry point, FastAPI app, `process_job_task()` |
| `bootstrap.py` | Imports all agents (triggers registration), validates required agents present |
| `registry.py` | Agent registry singleton, `@register` decorator, `get_agent_class(name)` |
| `state.py` | Shared models: `PlanStep`, `ExecutionPlan`, `PipelineState`, `AgentResult`, `ExecutionOutput` |
| `llm.py` | All Groq API calls: `get_groq_client()`, `call_groq_json()`, role normalization |
| `llm_telemetry.py` | Structured logging for every LLM call (timing, tokens, errors) |

### Agents (`agents/`)

| Agent | Stage | Function |
|-------|-------|----------|
| `ingestion_agent.py` | ingest | Reads CSV/XLSX/PDF → DataFrame |
| `cleaning_agent.py` | transform | Trim, normalize, drop nulls, absolute value, dedup. Has `safe_default` and `explicit` modes |
| `filter_agent.py` | transform | Row filtering with predicate grounding, value resolution, case-insensitive matching |
| `calculation_agent.py` | analyze | Sum, mean, group_by, cross-tab, conditional percentage, quarterly aggregation |
| `visualization_agent.py` | visualize | Per-chart independent aggregation, produces VisualizationSpec for frontend |
| `reporting_agent.py` | deliver | Writes final XLSX/CSV with sheets: cleaned_data, audit_log, warnings, column_mapping |

### Planning (`planning/`)

| File | Function |
|------|----------|
| `compiler.py` | The brain. Converts canonical intent → PlanIntent → ExecutionPlan. Handles all step insertion logic |
| `canonical_intent.py` | Pydantic models: CanonicalIntent, FilterRowsIntent, CleanIntent, CalculateIntent, VisualizeIntent |
| `intent_schema.py` | `PlanIntent` model — intermediate representation with needs_cleaning/filtering/calculation/visualization flags |
| `intent_enricher.py` | Multi-chart detection, field grounding against source_columns, `normalize_identifier()` |
| `trigger_detector.py` | Regex detection of chart/graph/plot keywords |
| `validators.py` | Plan validation: stage ordering, cycles, input_from references |

### Execution (`execution/`)

| File | Function |
|------|----------|
| `engine.py` | `ExecutionEngine` — LangGraph DAG runner, per-step param validation, aggregate value resolution |
| `visualization/executor.py` | Produces VisualizationSpec from OperationResult, auto-selects chart type, maps encoding |
| `visualization/validators.py` | Per-chart validation (pie needs category+measure, scatter needs 2 numeric) |
| `visualization/spec.py` | `VisualizationSpec` model — JSON contract for frontend rendering |
| `visualization_runner.py` | Visualization DAG node lifecycle, concurrent execution |

### Operations (`operations/`)

| File | Function |
|------|----------|
| `schemas.py` | All operation schemas: FilterCondition, CalculationOperation, ChartSpec, CleaningOperationType |
| `executor.py` | `execute_cleaning_plan()`, `execute_filter_plan()`, `execute_calculation_plan()`, `_resolve_aggregate_value()` |
| `filter_handlers.py` | eq, neq, gt, lt, contains, in — case-insensitive text matching for eq/neq |
| `calculation_handlers.py` | 23 handlers: sum, mean, group_sum, cross_tab_sum, conditional_percentage, quarterly_sum, etc. |
| `cleaning_handlers.py` | 19 handlers: trim, normalize, drop_nulls, absolute_value, normalize_categorical_values, etc. |
| `result_contract.py` | `OperationResult` — standardized versioned output contract for all calculations |
| `result_builder.py` | Normalizes handler outputs → OperationResult with proper serialization |

### Tools (`tools/`)

| File | Function |
|------|----------|
| `predicate_grounder.py` | Semantic column resolution for filters. Scores candidates, handles aggregate value detection |
| `value_resolver.py` | Checks if filter values exist in column data |
| `column_resolver.py` | Fuzzy column name matching |
| `dataframe_profile.py` | Profiles DataFrame: column types, nulls, distinct values, semantic guesses |
| `config.py` | Feature flags: ENABLE_VISUALIZATION, confidence thresholds |
| `path_safety.py` | Directory traversal prevention |

### Grounding (`grounding/`)

| File | Function |
|------|----------|
| `semantic_extractor.py` | LLM prompt → SemanticIntentDraft. Contains `_EXTRACTION_SYSTEM_PROMPT` |
| `llm_adapter.py` | LLM call interface: error handling, rate limits, timeouts |
| `column_grounder.py` | Resolves vague column references to actual columns via semantic scoring |
| `predicate_grounder.py` | Grounds filter predicates specifically |
| `candidate_generator.py` | Generates ranked candidate column matches |
| `schema_service.py` | Infers column roles (date, category, numeric) |
| `preflight_loader.py` | Profiles data file before execution starts |

### Jobs (`jobs/`)

| File | Function |
|------|----------|
| `callbacks.py` | `send_backend_callback()` — HTTP POST result to backend with retries |
| `repository.py` | Local JSON job state tracking (QUEUED→PLANNING→RUNNING→SUCCEEDED/FAILED) |

---

## 5. File Reference: Backend

| File | Function |
|------|----------|
| `app/api/uploads.py` | Upload endpoint, job-detail, download, field mapping UI |
| `app/api/agent.py` | Agent callback handler, visualization persistence |
| `app/services/canonical_intent.py` | `build_canonical_intent()`, LLM extraction, regex fallback, all repair functions |
| `app/services/agent_dispatcher.py` | `enqueue_submission_dispatch()` — pushes job to Redis |
| `app/services/new_pipeline_bridge.py` | Bridge between backend and agent-framework's semantic extractor |
| `app/services/data_profile.py` | File profiling: builds column metadata from CSV/XLSX/PDF |
| `app/models/visualization.py` | `JobVisualization` SQLAlchemy model — persists chart specs |
| `app/schemas/visualization.py` | `VisualizationSpecRead` — API response schema for charts |

---

## 6. Repair Pipeline (Backend)

When the LLM produces imperfect or incomplete intent, these repair functions fix it:

| Function | What it fixes |
|----------|---------------|
| `_repair_select_all_projection` | Removes single-column projection when user wants all data |
| `_repair_missing_clean_action` | Injects clean action when "clean" is in prompt but missing from intent |
| `_repair_missing_clean_operations` | Adds absolute_value/dedup/etc. operations to existing clean action |
| `_repair_missing_calculate_action` | Injects calculate action for "average X by Y" patterns |
| `_repair_filter_mode` | Flips keep→drop when user says "remove rows", handles double negation |
| `_repair_null_row_cleanup` | Converts drop_columns to drop_nulls when user says "remove rows with missing values" |
| `_repair_profile_grounded_references` | Resolves column references against data profile |

---

## 7. Key Design Decisions

| Decision | Why |
|----------|-----|
| Redis queue (not direct HTTP) | Non-blocking upload, retry capability, scalability |
| LangGraph (not AgentExecutor) | Deterministic DAG, debuggable, explicit |
| Canonical intent contract | Decouples NLP layer from execution layer |
| Per-agent single responsibility | Traceable failures — one agent = one job |
| Case-insensitive filter_eq | Real data has inconsistent casing (Male vs male vs M) |
| Aggregate value resolution at filter time | Supports "rows > average" without separate calc step |
| Per-chart independent aggregation | Multiple charts can group by different columns |
| Field grounding before compilation | Compiler only consumes resolved columns, never interprets NL |
| No hardcoded column fallbacks | Wrong chart > incorrect chart. Fail visibly, not silently |

---

## 8. Docker Services

| Service | Port | Image |
|---------|------|-------|
| `personalagent-frontend-2` | 5173 | React Nginx |
| `personalagent-backend-2` | 8000 | Python 3.12 FastAPI |
| `personalagent-agent-service-2` | 8001 | Python 3.11 ARQ + Uvicorn |
| `personalagent-postgres-2` | 5433 | PostgreSQL 16 |
| `personalagent-redis-2` | 6379 | Redis 7 |

Shared volume: `backend_storage` mounted at `/app/storage` in both backend and agent-service.

---

## 9. How Data Travels

| What | Path |
|------|------|
| Raw file | User → Backend disk (`/app/storage/uploads/`) — stays here forever |
| Job message | Backend → Redis → Agent Service (JSON only, no file data) |
| DataFrame | In-memory only during agent execution. Never in DB. |
| Output file | Agent writes to `/app/storage/outputs/` → Backend serves to frontend |
| Visualizations | Agent generates JSON specs → stored in `job_visualizations` table → rendered by frontend |

---

## 10. Supported Operations

### Cleaning (19 handlers)
trim_whitespace, normalize_column_names, drop_duplicates, fill_nulls, drop_nulls, normalize_date, normalize_currency, normalize_number, normalize_text_case, replace_values, strip_currency_symbols, remove_commas_from_numbers, coerce_column_type, remove_empty_rows, remove_empty_columns, rename_columns, reorder_columns, absolute_value, normalize_categorical_values

### Calculation (23 handlers)
sum, mean, median, min, max, count, count_distinct, variance, standard_deviation, group_sum, group_mean, group_count, running_total, percentage_change, difference, ratio, absolute_value, conditional_percentage, quarterly_sum, quarterly_mean, quarterly_count, cross_tab_sum, cross_tab_mean, cross_tab_count

### Filter (15 operators)
eq, neq, gt, gte, lt, lte, contains, not_contains, starts_with, ends_with, between, in, not_in, is_null, is_not_null

### Visualization (5 chart types)
pie, bar, line, scatter, histogram

---

## 11. Environment Variables

| Variable | Service | Purpose |
|----------|---------|---------|
| `GROQ_API_KEY` | Agent Service + Backend | LLM API authentication |
| `DATABASE_URL` | Backend | PostgreSQL connection |
| `REDIS_URL` | Both | Redis queue connection |
| `UPLOAD_DIR` | Both | File storage path |
| `OUTPUT_DIR` | Agent Service | Output file path |
| `ENABLE_VISUALIZATION` | Agent Service | Feature flag for charts (default: true) |
| `BACKEND_BASE_URL` | Agent Service | Callback URL target |

---

## 12. Error Handling

| Scenario | Behavior |
|----------|----------|
| Groq rate limited | Falls back to regex extraction — still works |
| File can't be parsed | Ingestion agent returns failed → job fails immediately |
| Column not found | Predicate grounder rejects → contract violation → quarantined |
| Filter value is aggregate dict | `_resolve_aggregate_value()` computes it at runtime |
| Null value for eq/gt operator | Compiler skips the condition (no crash) |
| Sort action in plan | Skipped gracefully (not supported yet) |
| Visualization field unresolvable | Returns error instead of defaulting to wrong column |
