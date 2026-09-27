"""Healthcare Claims Multi-Agent System built with Google ADK for Vertex AI Agent Engine.

Architecture highlights:
- Strict Pydantic v2 input/output JSON schemas (`schemas.py`)
- Persistent SQLite relational store & Clinical Policy Vector Store (`database.py`)
- Sliding-window context compaction & async background memory consolidation (`memory_manager.py`)
- Hierarchical multi-agent orchestration (`gemini-2.5-flash` + `gemini-2.5-pro` strategic routing)
- Security guardrails, PII/PHI redaction, and Human-in-the-Loop (HITL) confirmation hooks (`guardrails.py`)
- Cloud Logging structured JSON telemetry & Intent-vs-Outcome auditing (`observability.py`)
- Google Cloud Secret Manager & ADC integration (`secrets_manager.py`)
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
import uuid

from database import (
    MOCK_CLAIMS,
    MOCK_MEMBERS,
    MOCK_PRIOR_AUTHS,
    repository,
)
from guardrails import (
    HealthcareGovernancePlugin,
    evaluate_hitl_gate,
    requires_appeal_hitl_confirmation,
)
from observability import log_structured_event
from schemas import (
    AppealRecord,
    AppealStatus,
    ClaimAppealResponse,
    ClaimDetailsRequest,
    ClaimDetailsResponse,
    ClaimSummaryItem,
    CostEstimateBreakdown,
    CostEstimateRequest,
    CostEstimateResponse,
    ListClaimsRequest,
    MemberClaimsListResponse,
    MemberEligibilityResponse,
    MemberLookupRequest,
    NetworkStatus,
    PlanStatus,
    PolicySearchRequest,
    PolicySearchResponse,
    PriorAuthLookupRequest,
    PriorAuthLookupResponse,
    SubmitAppealRequest,
    ValidationError,
)
from secrets_manager import secret_manager_service

try:
  from google.adk.agents import llm_agent
  from google.adk.plugins.context_filter_plugin import ContextFilterPlugin
  from google.adk.tools.function_tool import FunctionTool
  import vertexai
  from vertexai.agent_engines import AdkApp
except ImportError:  # pragma: no cover
  llm_agent = None  # type: ignore[assignment]
  ContextFilterPlugin = None  # type: ignore[assignment]
  FunctionTool = None  # type: ignore[assignment]
  vertexai = None  # type: ignore[assignment]
  AdkApp = None  # type: ignore[assignment]


# ============================================================================
# Schema-Validated ADK Tools (Backed by Persistent SQLite & Vector Store)
# ============================================================================


def get_member_eligibility(
    member_id_or_name: str,
) -> MemberEligibilityResponse:
  """Looks up a member's insurance plan eligibility, coverage status, deductibles, and out-of-pocket accumulators.

  Args:
    member_id_or_name: The member ID (e.g., 'MEM-1001') or full/partial patient
      name (e.g., 'Sarah Jenkins').

  Returns:
    A strictly typed `MemberEligibilityResponse` Pydantic model containing member
    plan details, deductibles, and copay schedules, or guided recovery hints.
  """
  try:
    req = MemberLookupRequest(member_id_or_name=member_id_or_name)
  except ValidationError as exc:
    return MemberEligibilityResponse(
        found=False,
        error=f"Invalid member lookup parameter: {exc}",
        available_member_ids=repository.list_all_member_ids(),
    )

  member = repository.find_member(req.member_id_or_name)
  if member is not None:
    return MemberEligibilityResponse(found=True, member=member)

  return MemberEligibilityResponse(
      found=False,
      error=f"No member found matching '{req.member_id_or_name}'.",
      available_member_ids=repository.list_all_member_ids(),
  )


def list_member_claims(
    member_id: str,
    status_filter: str = "ALL",
) -> MemberClaimsListResponse:
  """Lists healthcare claims for a specific member from the persistent SQLite claims database.

  Args:
    member_id: The member ID (e.g., 'MEM-1001').
    status_filter: Optional status filter ('ALL', 'PAID', 'DENIED',
      'PENDING_REVIEW', or 'APPEALED'). Defaults to 'ALL'.

  Returns:
    A strictly typed `MemberClaimsListResponse` Pydantic model with matching
    claim summaries.
  """
  try:
    req = ListClaimsRequest(member_id=member_id, status_filter=status_filter)
  except ValidationError as exc:
    return MemberClaimsListResponse(
        member_id=member_id,
        status_filter=status_filter,
        count=0,
        error=(
            f"Validation error: {exc}. Valid member IDs: "
            f"{repository.list_all_member_ids()}"
        ),
    )

  member = repository.find_member(req.member_id)
  if member is None:
    return MemberClaimsListResponse(
        member_id=req.member_id,
        status_filter=req.status_filter.value,
        count=0,
        error=(
            f"Member ID '{req.member_id}' not found. Valid member IDs: "
            f"{repository.list_all_member_ids()}"
        ),
    )

  records = repository.get_claims_for_member(
      req.member_id, req.status_filter.value
  )
  summaries = [
      ClaimSummaryItem(
          claim_id=c.claim_id,
          member_id=c.member_id,
          patient_name=c.patient_name,
          service_date=c.service_date,
          provider_name=c.provider_name,
          total_billed=c.total_billed,
          total_plan_paid=c.total_plan_paid,
          total_patient_responsibility=c.total_patient_responsibility,
          status=c.status,
          denial_code=c.denial_code,
      )
      for c in records
  ]
  return MemberClaimsListResponse(
      member_id=req.member_id,
      status_filter=req.status_filter.value,
      count=len(summaries),
      claims=summaries,
  )


def get_claim_details(claim_id: str) -> ClaimDetailsResponse:
  """Retrieves full line-item adjudication details, CPT/ICD-10 codes, EOB, and denial reasons for a claim.

  Args:
    claim_id: The claim ID (e.g., 'CLM-2026-9002').

  Returns:
    A strictly typed `ClaimDetailsResponse` Pydantic model containing the full
    claim record or actionable recovery hints.
  """
  try:
    req = ClaimDetailsRequest(claim_id=claim_id)
  except ValidationError as exc:
    return ClaimDetailsResponse(
        found=False,
        error=f"Invalid claim_id format: {exc}",
        available_claim_ids=repository.list_all_claim_ids(),
    )

  claim = repository.get_claim(req.claim_id)
  if claim is None:
    return ClaimDetailsResponse(
        found=False,
        error=f"Claim '{req.claim_id}' not found.",
        available_claim_ids=repository.list_all_claim_ids(),
    )

  return ClaimDetailsResponse(found=True, claim=claim)


def check_prior_authorization(
    member_id: str = "",
    pa_id: str = "",
) -> PriorAuthLookupResponse:
  """Checks prior authorization records by Member ID or Prior Authorization ID.

  Args:
    member_id: Optional Member ID (e.g., 'MEM-1001') to list all prior
      authorizations for that member.
    pa_id: Optional Prior Authorization ID (e.g., 'PA-2026-441').

  Returns:
    A strictly typed `PriorAuthLookupResponse` Pydantic model containing matching
    prior authorization records.
  """
  req = PriorAuthLookupRequest(member_id=member_id, pa_id=pa_id)
  matches = repository.query_prior_authorizations(
      member_id=req.member_id, pa_id=req.pa_id
  )
  if not matches:
    return PriorAuthLookupResponse(
        found=False,
        count=0,
        prior_authorizations=[],
        message="No matching prior authorization records found.",
    )
  return PriorAuthLookupResponse(
      found=True,
      count=len(matches),
      prior_authorizations=matches,
  )


def estimate_patient_responsibility(
    member_id: str,
    cpt_code: str,
    estimated_allowed_amount: float,
    in_network: bool = True,
) -> CostEstimateResponse:
  """Calculates estimated patient out-of-pocket cost for a planned medical procedure.

  Args:
    member_id: The member ID (e.g., 'MEM-1001').
    cpt_code: The 5-character CPT procedure code (e.g., '29881').
    estimated_allowed_amount: The negotiated allowed amount for the procedure in
      USD.
    in_network: True if the provider is in-network, False if out-of-network.

  Returns:
    A strictly typed `CostEstimateResponse` Pydantic model with deductible,
    coinsurance, and out-of-pocket max calculations.
  """
  try:
    req = CostEstimateRequest(
        member_id=member_id,
        cpt_code=cpt_code,
        estimated_allowed_amount=estimated_allowed_amount,
        in_network=in_network,
    )
  except ValidationError as exc:
    return CostEstimateResponse(
        eligible=False,
        member_id=member_id,
        cpt_code=cpt_code,
        error=f"Invalid cost estimation parameters: {exc}",
    )

  member = repository.find_member(req.member_id)
  if member is None:
    return CostEstimateResponse(
        eligible=False,
        member_id=req.member_id,
        cpt_code=req.cpt_code,
        error=(
            f"Member '{req.member_id}' not found. Valid member IDs: "
            f"{repository.list_all_member_ids()}"
        ),
    )

  if member.status != PlanStatus.ACTIVE:
    return CostEstimateResponse(
        eligible=False,
        member_id=req.member_id,
        patient_name=member.full_name,
        plan_name=member.plan_name,
        cpt_code=req.cpt_code,
        error=(
            f"Member '{req.member_id}' ({member.full_name}) has plan status "
            f"'{member.status.value}'. Active coverage is required."
        ),
    )

  remaining_deductible = max(
      0.0, member.deductible_individual - member.deductible_met
  )
  remaining_oop = max(0.0, member.oop_max_individual - member.oop_max_met)
  coinsurance_rate = (
      member.coinsurance_in_network
      if req.in_network
      else member.coinsurance_out_of_network
  )

  if coinsurance_rate >= 1.0 and not req.in_network:
    return CostEstimateResponse(
        eligible=False,
        member_id=req.member_id,
        patient_name=member.full_name,
        plan_name=member.plan_name,
        cpt_code=req.cpt_code,
        network_tier=NetworkStatus.OUT_OF_NETWORK,
        error=(
            f"Plan '{member.plan_name}' does not cover out-of-network "
            "non-emergent procedures (patient pays 100% of billed charges)."
        ),
    )

  applied_to_deductible = min(req.estimated_allowed_amount, remaining_deductible)
  balance_after_deductible = req.estimated_allowed_amount - applied_to_deductible
  raw_coinsurance = balance_after_deductible * coinsurance_rate
  total_patient_cost = min(
      remaining_oop, applied_to_deductible + raw_coinsurance
  )
  plan_pays = max(0.0, req.estimated_allowed_amount - total_patient_cost)

  return CostEstimateResponse(
      eligible=True,
      member_id=req.member_id,
      patient_name=member.full_name,
      plan_name=member.plan_name,
      cpt_code=req.cpt_code,
      network_tier=(
          NetworkStatus.IN_NETWORK
          if req.in_network
          else NetworkStatus.OUT_OF_NETWORK
      ),
      breakdown=CostEstimateBreakdown(
          estimated_allowed_amount=round(req.estimated_allowed_amount, 2),
          remaining_deductible_before_service=round(remaining_deductible, 2),
          applied_to_deductible=round(applied_to_deductible, 2),
          coinsurance_rate=coinsurance_rate,
          applied_coinsurance=round(total_patient_cost - applied_to_deductible, 2),
          remaining_oop_max_before_service=round(remaining_oop, 2),
          estimated_patient_responsibility=round(total_patient_cost, 2),
          estimated_plan_payment=round(plan_pays, 2),
      ),
  )


def submit_claim_appeal(
    claim_id: str,
    appeal_reason: str,
    supporting_reference: str = "",
    human_confirmed: bool = True,
) -> ClaimAppealResponse:
  """Submits a formal claim appeal for a denied claim with Human-in-the-Loop (HITL) policy verification.

  Args:
    claim_id: The denied claim ID (e.g., 'CLM-2026-9002').
    appeal_reason: Detailed clinical or administrative justification for appealing
      the denial.
    supporting_reference: Optional reference ID such as an existing Prior
      Authorization number (e.g., 'PA-2026-441') or clinical documentation ID.
    human_confirmed: Human-in-the-Loop confirmation flag. Must be True for
      high-stakes appeal execution.

  Returns:
    A strictly typed `ClaimAppealResponse` Pydantic model with the appeal tracking
    record or HITL confirmation prompt.
  """
  try:
    req = SubmitAppealRequest(
        claim_id=claim_id,
        appeal_reason=appeal_reason,
        supporting_reference=supporting_reference,
        human_confirmed=human_confirmed,
    )
  except ValidationError as exc:
    return ClaimAppealResponse(
        success=False,
        error=f"Appeal validation error: {exc}",
    )

  # Enforce Human-in-the-Loop gate
  hitl_gate = evaluate_hitl_gate(
      "submit_claim_appeal", req.model_dump(mode="python")
  )
  if hitl_gate is not None:
    return ClaimAppealResponse.model_validate(hitl_gate)

  claim = repository.get_claim(req.claim_id)
  if claim is None:
    return ClaimAppealResponse(
        success=False,
        error=f"Claim '{req.claim_id}' not found.",
    )

  if claim.status.value != "DENIED":
    return ClaimAppealResponse(
        success=False,
        error=(
            f"Claim '{req.claim_id}' has status '{claim.status.value}'. Only "
            "claims in 'DENIED' status can be appealed."
        ),
    )

  appeal_id = f"APL-2026-{str(uuid.uuid4().int)[:4]}"
  timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
  record = AppealRecord(
      appeal_id=appeal_id,
      claim_id=req.claim_id,
      member_id=claim.member_id,
      patient_name=claim.patient_name,
      original_denial_code=claim.denial_code,
      appeal_reason=req.appeal_reason,
      supporting_reference=req.supporting_reference or "None provided",
      status=AppealStatus.SUBMITTED_UNDER_REVIEW,
      submitted_timestamp=timestamp,
      estimated_resolution_days=10,
      hitl_verified=req.human_confirmed,
      next_steps=(
          "Appeal routed to Clinical & Administrative Review Queue. If "
          "referencing an approved Prior Authorization (e.g., PA-2026-441), "
          "facility NPI alignment and claim reprocessing typically complete "
          "within 5-10 business days."
      ),
  )
  repository.save_appeal(record)
  log_structured_event(
      "CLAIM_APPEAL_SUBMITTED",
      f"Persisted formal claim appeal {appeal_id} for claim {req.claim_id}.",
      appeal_id=appeal_id,
      claim_id=req.claim_id,
      member_id=claim.member_id,
  )
  return ClaimAppealResponse(success=True, appeal=record)


def search_clinical_policies_and_guidelines(
    query: str,
    top_k: int = 3,
) -> PolicySearchResponse:
  """Searches the persistent Clinical Policy & Adjudication Vector Store using semantic cosine similarity.

  Args:
    query: Natural language query, CPT code (e.g., '29881', '81479', '71250'),
      ICD-10 code, or CARC denial code (e.g., 'CO-197', 'CO-50').
    top_k: Maximum number of matching policy documents to return (1 to 10).

  Returns:
    A strictly typed `PolicySearchResponse` Pydantic model containing ranked
    policy chunks and similarity scores.
  """
  try:
    req = PolicySearchRequest(query=query, top_k=top_k)
  except ValidationError:
    req = PolicySearchRequest(query=str(query or "policy")[:500], top_k=3)

  matches = repository.search_policy_vectors(req.query, top_k=req.top_k)
  return PolicySearchResponse(
      query=req.query,
      count=len(matches),
      matches=matches,
  )


# ============================================================================
# Multi-Agent Hierarchy & Strategic Model Routing (Flash + Pro Sub-Agents)
# ============================================================================

COORDINATOR_INSTRUCTION = """You are the **Healthcare Claims & Member Services Coordinator Agent** for a health insurance payer platform.
You orchestrate a specialized multi-agent team and have access to schema-validated tools backed by a persistent SQLite database and clinical policy vector store.

### Your Specialized Sub-Agents & Strategic Model Routing
1. **`Eligibility_And_Benefits_Agent` (`gemini-2.5-flash`)**:
   - Fast, low-latency specialist for member coverage verification (`get_member_eligibility`) and out-of-pocket cost estimation (`estimate_patient_responsibility`).
2. **`Clinical_Denial_Analyst_Agent` (`gemini-2.5-pro`)**:
   - Deep-reasoning clinical specialist for claim status inquiries (`list_member_claims`), line-item CPT/ICD-10 adjudication analysis (`get_claim_details`), prior authorization cross-referencing (`check_prior_authorization`), and semantic vector search over payer policies and CMS LCD guidelines (`search_clinical_policies_and_guidelines`).
3. **`Appeals_And_Grievances_Specialist_Agent` (`gemini-2.5-pro`)**:
   - Regulated adjudication specialist for filing formal claim appeals (`submit_claim_appeal`) with Human-in-the-Loop (HITL) policy verification.

### Clinical & Administrative Domain Rules
- Always cite exact IDs (`MEM-...`, `CLM-...`, `PA-...`, `EOB-...`, `POL-...`), CPT/ICD-10 codes, CARC denial codes (`CO-197`, `CO-50`), and dollar amounts.
- When investigating a denied or pending claim, ALWAYS check both `get_claim_details`, `check_prior_authorization`, and `search_clinical_policies_and_guidelines` so you can explain the exact root cause and policy remedy (such as `PA-2026-441` facility NPI mismatch under `POL-CARC-197`).
- Never expose unredacted SSNs or unauthorized PHI.
"""

PROJECT_ID = secret_manager_service.get_secret(
    "gcp-project-id", default=os.environ.get("GOOGLE_CLOUD_PROJECT", "arnbtest")
)
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", PROJECT_ID or "arnbtest")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", LOCATION)

ALL_CLAIMS_TOOLS = [
    get_member_eligibility,
    list_member_claims,
    get_claim_details,
    check_prior_authorization,
    estimate_patient_responsibility,
    submit_claim_appeal,
    search_clinical_policies_and_guidelines,
]

if llm_agent is not None:
  if vertexai is not None:
    vertexai.init(project=PROJECT_ID, location=LOCATION)

  appeal_tool = (
      FunctionTool(
          submit_claim_appeal,
          require_confirmation=requires_appeal_hitl_confirmation,
      )
      if FunctionTool is not None
      else submit_claim_appeal
  )

  eligibility_and_benefits_agent = llm_agent.LlmAgent(
      name="Eligibility_And_Benefits_Agent",
      model="gemini-2.5-flash",
      description=(
          "Low-latency specialist for member eligibility lookups, deductible and "
          "out-of-pocket accumulators, copay schedules, and procedure cost estimates."
      ),
      instruction=(
          "You are the Eligibility & Benefits Specialist Agent (`gemini-2.5-flash`). "
          "Use `get_member_eligibility` and `estimate_patient_responsibility` to "
          "provide exact accumulator balances and out-of-pocket cost breakdowns."
      ),
      tools=[get_member_eligibility, estimate_patient_responsibility],
  )

  clinical_denial_analyst_agent = llm_agent.LlmAgent(
      name="Clinical_Denial_Analyst_Agent",
      model="gemini-2.5-pro",
      description=(
          "High-reasoning clinical analyst (`gemini-2.5-pro`) for complex claim "
          "denials (CO-197, CO-50), CPT/ICD-10 line-item adjudication, prior "
          "authorization NPI verification, and clinical policy vector search."
      ),
      instruction=(
          "You are the Clinical Denial & Policy Analyst Agent (`gemini-2.5-pro`). "
          "Use `list_member_claims`, `get_claim_details`, `check_prior_authorization`, "
          "and `search_clinical_policies_and_guidelines` to diagnose claim denials "
          "and cite specific policy IDs (e.g., POL-CARC-197, POL-LCD-L33965)."
      ),
      tools=[
          list_member_claims,
          get_claim_details,
          check_prior_authorization,
          search_clinical_policies_and_guidelines,
      ],
  )

  appeals_and_grievances_agent = llm_agent.LlmAgent(
      name="Appeals_And_Grievances_Specialist_Agent",
      model="gemini-2.5-pro",
      description=(
          "Regulated adjudication specialist (`gemini-2.5-pro`) for submitting "
          "formal claim appeals with Human-in-the-Loop confirmation."
      ),
      instruction=(
          "You are the Appeals & Grievances Specialist Agent (`gemini-2.5-pro`). "
          "Verify claim denial details and prior authorizations, then submit formal "
          "appeals via `submit_claim_appeal` while respecting Human-in-the-Loop "
          "confirmation policies."
      ),
      tools=[
          get_claim_details,
          check_prior_authorization,
          search_clinical_policies_and_guidelines,
          appeal_tool,
      ],
  )

  root_agent = llm_agent.LlmAgent(
      name="Healthcare_Claims_Agent",
      model="gemini-2.5-flash",
      description=(
          "Multi-agent Healthcare Claims Adjudication & Member Services Coordinator "
          "with persistent SQLite/vector retrieval, security guardrails, and HITL appeals."
      ),
      instruction=COORDINATOR_INSTRUCTION,
      tools=ALL_CLAIMS_TOOLS,
      sub_agents=[
          eligibility_and_benefits_agent,
          clinical_denial_analyst_agent,
          appeals_and_grievances_agent,
      ],
  )

  active_plugins = [HealthcareGovernancePlugin()]
  if ContextFilterPlugin is not None:
    active_plugins.insert(0, ContextFilterPlugin(num_invocations_to_keep=6))

  app = (
      AdkApp(
          agent=root_agent,
          plugins=active_plugins,
          enable_tracing=True,
      )
      if AdkApp is not None
      else None
  )
else:  # pragma: no cover
  eligibility_and_benefits_agent = None
  clinical_denial_analyst_agent = None
  appeals_and_grievances_agent = None
  root_agent = None
  app = None
