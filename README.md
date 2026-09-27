# Healthcare Claims Multi-Agent System (`google-adk` on Vertex AI Agent Engine)

An enterprise-grade **Google Agent Development Kit (ADK)** multi-agent system deployed on **Vertex AI Agent Engine** (`reasoningEngines`) in Google Cloud project **`arnbtest`**, engineered for healthcare claims adjudication, member eligibility verification, clinical policy vector search, and Human-in-the-Loop (HITL) appeal workflows.

---

## 1. Enterprise Architecture & Evaluation Pillars

### I. Tool & Interface Design (`schemas.py` & `main.py`)
- **Strict Pydantic v2 Input & Output Schemas:** Every ADK tool validates incoming arguments against a `ConfigDict(extra="forbid")` Pydantic `BaseModel` (`MemberLookupRequest`, `ListClaimsRequest`, `ClaimDetailsRequest`, `PriorAuthLookupRequest`, `CostEstimateRequest`, `SubmitAppealRequest`, `PolicySearchRequest`) with regex patterns (`^MEM-\d{4}$`, `^CLM-\d{4}-\d{4}$`), numeric bounds (`gt=0.0`), and `@field_validator` normalizers.
- **Constrained Output Envelopes:** All tools return explicit Pydantic response models (`MemberEligibilityResponse`, `MemberClaimsListResponse`, `ClaimDetailsResponse`, `PriorAuthLookupResponse`, `CostEstimateResponse`, `ClaimAppealResponse`, `PolicySearchResponse`) with actionable recovery hints on validation or lookup errors.

### II. Context & Memory Management (`database.py` & `memory_manager.py`)
- **Persistent Relational Database (SQLite WAL):** Stores normalized `members`, `claims`, `prior_authorizations`, `claim_appeals`, and `long_term_memories` tables with ACID transactions.
- **Clinical Policy Vector Store (`clinical_policy_vectors`):** Indexes payer adjudication policies (`POL-CARC-197`, `POL-CARC-50-GENOMICS`) and CMS Local Coverage Determinations (`POL-LCD-L33965`, `POL-LCD-L34636`) with L2-normalized vector embeddings and cosine similarity retrieval (`search_clinical_policies_and_guidelines`).
- **Context Bloat Management & Sliding-Window Compaction:** Combines ADK's `ContextFilterPlugin(num_invocations_to_keep=6)` with `ContextCompactionManager`, which enforces token/character budgets (`MAX_ESTIMATED_CONTEXT_TOKENS = 6000`) and compacts evicted conversation turns into `callback_context.state["compacted_session_memory"]`.
- **Asynchronous Background Memory Operations:** `AsyncBackgroundMemoryWorker` uses `asyncio.create_task` and a background `ThreadPoolExecutor` to asynchronously extract and persist long-term clinical interaction facts after each turn without blocking the user response.

### III. Multi-Agent Orchestration, Model Routing, & Guardrails (`main.py` & `guardrails.py`)
- **Hierarchical Multi-Agent System & Strategic Model Routing:**
  1. **`Healthcare_Claims_Agent` (Coordinator, `gemini-2.5-flash`)** — Triages requests, enforces global governance, and delegates to specialized sub-agents.
  2. **`Eligibility_And_Benefits_Agent` (`gemini-2.5-flash`)** — Low-latency tier for member eligibility verification and out-of-pocket cost estimation.
  3. **`Clinical_Denial_Analyst_Agent` (`gemini-2.5-pro`)** — High-reasoning tier for complex CPT/ICD-10 medical necessity analysis, CARC/RARC denial root-cause investigation, and clinical policy vector retrieval.
  4. **`Appeals_And_Grievances_Specialist_Agent` (`gemini-2.5-pro`)** — Regulated adjudication specialist for formal claim appeals.
- **Dynamic Complexity Router (`StrategicModelRouter`):** Inspects classified intents in `before_model_callback` to route complex clinical/appeal workflows to `gemini-2.5-pro` and routine lookups to `gemini-2.5-flash`.
- **Security Guardrails (`HealthcareSecurityGuardrail` & `HealthcareGovernancePlugin`):** Blocks prompt injection, system prompt extraction, SQL injection, and bulk PHI exfiltration before reaching the model.
- **Human-in-the-Loop (HITL) Confirmation Hooks:** Wraps `submit_claim_appeal` in `FunctionTool(..., require_confirmation=requires_appeal_hitl_confirmation)` and enforces `evaluate_hitl_gate` so high-dollar appeals ($\ge \$1,000$) require explicit human confirmation (`human_confirmed=True`).

### IV. Observability, Tracing, & HIPAA Redaction (`observability.py`)
- **OpenTelemetry Distributed Tracing:** Enabled on `AdkApp(enable_tracing=True)` with span enrichment (`gen_ai.intent.categories`, `gen_ai.outcome.status`, `gen_ai.outcome.goal_alignment_score`, `gen_ai.model.routed`).
- **Cloud Logging Structured JSON Logs (`StructuredJsonFormatter`):** Emits single-line JSON records with `severity`, `timestamp`, `logging.googleapis.com/trace`, `logging.googleapis.com/spanId`, `event_type`, and `latency_ms`.
- **Intent vs. Outcome Auditing (`IntentOutcomeTracker`):** Classifies user intent at turn start and logs a structured `INTENT_VS_OUTCOME_AUDIT` record comparing expected domain goals against actual tools invoked and outcome status.
- **Deterministic PII/PHI Redaction (`PiiPhiRedactor`):** Scrubs SSNs (`[REDACTED-SSN]`), phone numbers, emails, payment cards, Medicare MBIs, and masks DOBs (`YYYY-**-** [REDACTED-DOB]`) across prompts, model outputs, and logs.

### V. Infrastructure-as-Code, Secret Manager, & CI/CD (`terraform/`, `secrets_manager.py`, `evals/`, `.github/workflows/`)
- **Terraform IaC (`terraform/`):** Provisions Google Cloud APIs, least-privilege Service Account & IAM bindings, Secret Manager secrets (`gcp-project-id`, `reasoning-engine-id`), and versioned GCS artifact staging bucket.
- **Google Cloud Secret Manager & ADC (`secrets_manager.py`):** Retrieves secrets via `google.cloud.secretmanager.SecretManagerServiceClient` with TTL caching and authenticates via Application Default Credentials (`google.auth.default()`).
- **Automated Golden Dataset Evaluation Suite (`evals/golden_dataset.json` & `evals/run_evaluation.py`):** Evaluates 8 golden healthcare scenarios across intent classification, model routing, Pydantic schema compliance, factual accuracy, HITL gating, and prompt-injection rejection (`100%` score gate in CI).
- **GitHub Actions CI/CD Pipeline (`.github/workflows/ci_cd.yml`):** Runs unit tests, executes the Golden Dataset Evaluation Gate (`--min-score 0.95`), validates Terraform configurations, and deploys to Vertex AI Agent Engine via Workload Identity Federation.

---

## 2. Repository Structure

```text
.
├── main.py                        # Multi-agent ADK definitions, 7 schema-validated tools, AdkApp
├── schemas.py                     # Strict Pydantic v2 input/output & domain JSON schemas
├── database.py                    # Persistent SQLite relational store & Clinical Policy Vector Store
├── memory_manager.py              # Sliding-window context compaction & async background memory worker
├── guardrails.py                  # Security guardrails, StrategicModelRouter, HITL gate, & ADK BasePlugin
├── observability.py               # Structured JSON logger, PII/PHI redactor, & Intent-vs-Outcome tracker
├── secrets_manager.py             # Google Cloud Secret Manager & ADC integration
├── deploy_agent.py                # Source-based Vertex AI Reasoning Engine deployment script
├── test_agent.py                  # Live SSE streaming verification client (:streamQuery)
├── requirements.txt               # Runtime dependencies
├── evals/
│   ├── golden_dataset.json        # 8 golden benchmark scenarios & expected tool trajectories
│   └── run_evaluation.py          # Automated evaluation runner & CI quality gate
├── tests/
│   └── test_agent_suite.py        # Unit & integration test suite
├── terraform/
│   ├── versions.tf                # Terraform & Google provider constraints
│   ├── variables.tf               # Input variables for project, region, and engine ID
│   ├── main.tf                    # APIs, Service Account IAM, Secret Manager, and GCS staging
│   └── outputs.tf                 # Provisioned infrastructure outputs
├── .github/workflows/
│   └── ci_cd.yml                  # Automated test, golden eval, Terraform validate, & WIF deploy pipeline
└── web_ui/                        # Interactive Adjudication & Member Resolution Web Chat Console
    ├── server.py
    ├── index.html
    ├── styles.css
    └── app.js
```

---

## 3. Running Tests, Golden Evaluations, & Web Console

### Run Unit & Integration Tests
```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

### Run the Golden Dataset Evaluation Suite
```bash
python3 evals/run_evaluation.py --min-score 0.95
```

### Launch the Interactive Web Chat Console
```bash
python3 web_ui/server.py
```
