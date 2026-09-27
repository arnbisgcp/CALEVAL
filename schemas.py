"""Strict Pydantic v2 JSON schemas for Healthcare Claims Agent tools, state, and outputs."""

from __future__ import annotations

from enum import Enum
import json
import re
from typing import Any, Optional, get_args, get_origin, get_type_hints

try:
  from pydantic import (
      BaseModel,
      ConfigDict,
      Field,
      ValidationError,
      field_validator,
  )
except ImportError:  # pragma: no cover
  # Lightweight Pydantic v2-compatible validation engine for bare system Python environments
  class ValidationError(ValueError):
    """Pydantic-compatible validation error."""

  def ConfigDict(**kwargs: Any) -> dict[str, Any]:  # noqa: N802
    return dict(kwargs)

  class _FieldInfo:

    def __init__(self, default: Any = ..., **kwargs: Any) -> None:
      self.default = default
      self.default_factory = kwargs.get("default_factory")
      self.pattern = kwargs.get("pattern")
      self.gt = kwargs.get("gt")
      self.ge = kwargs.get("ge")
      self.lt = kwargs.get("lt")
      self.le = kwargs.get("le")
      self.min_length = kwargs.get("min_length")
      self.max_length = kwargs.get("max_length")
      self.description = kwargs.get("description", "")

  def Field(default: Any = ..., **kwargs: Any) -> Any:  # noqa: N802
    return _FieldInfo(default, **kwargs)

  def field_validator(*fields: str, mode: str = "after") -> Any:
    def decorator(fn: Any) -> Any:
      target = fn.__func__ if isinstance(fn, classmethod) else fn
      target._validator_fields = fields
      target._validator_mode = mode
      return classmethod(target)

    return decorator

  class BaseModel:  # type: ignore[no-redef]
    """Pydantic v2 BaseModel fallback enforcing field constraints, regex patterns, and JSON serialization."""

    model_config: dict[str, Any] = {}

    def __init__(self, **data: Any) -> None:
      cls = self.__class__
      hints = get_type_hints(cls)
      extra_policy = cls.model_config.get("extra", "ignore")
      if extra_policy == "forbid":
        unknown = set(data.keys()) - set(hints.keys())
        if unknown:
          raise ValidationError(f"Extra inputs are not permitted: {sorted(unknown)}")

      # Run 'before' field validators
      for attr_name in dir(cls):
        attr = getattr(cls, attr_name)
        func = getattr(attr, "__func__", attr)
        if hasattr(func, "_validator_fields"):
          for f_name in func._validator_fields:
            if f_name in data:
              data[f_name] = attr(data[f_name])

      for name, hint in hints.items():
        if name == "model_config":
          continue
        class_default = getattr(cls, name, ...)
        field_info = class_default if isinstance(class_default, _FieldInfo) else None

        if name in data:
          val = data[name]
        elif field_info is not None:
          if field_info.default_factory is not None:
            val = field_info.default_factory()
          elif field_info.default is not ...:
            val = field_info.default
          else:
            raise ValidationError(f"Field '{name}' is required.")
        elif class_default is not ...:
          val = class_default
        else:
          raise ValidationError(f"Field '{name}' is required.")

        if isinstance(val, str) and cls.model_config.get("str_strip_whitespace"):
          val = val.strip()

        val = self._coerce_and_validate(name, val, hint, field_info)
        setattr(self, name, val)

    @classmethod
    def _coerce_and_validate(
        cls, name: str, val: Any, hint: Any, field_info: Optional[_FieldInfo]
    ) -> Any:
      if val is None:
        return None
      origin = get_origin(hint)
      args = get_args(hint)
      if origin is not None and type(None) in args:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
          hint = non_none[0]
          origin = get_origin(hint)
          args = get_args(hint)

      if isinstance(hint, type) and issubclass(hint, Enum):
        try:
          val = hint(val)
        except ValueError as exc:
          raise ValidationError(f"Invalid enum value for '{name}': {val}") from exc
      elif isinstance(hint, type) and issubclass(hint, BaseModel):
        if isinstance(val, dict):
          val = hint.model_validate(val)
      elif origin is list and args:
        inner = args[0]
        if isinstance(inner, type) and issubclass(inner, BaseModel):
          val = [
              inner.model_validate(item) if isinstance(item, dict) else item
              for item in val
          ]

      if field_info is not None:
        if field_info.pattern and isinstance(val, str):
          if not re.match(field_info.pattern, val):
            raise ValidationError(
                f"Field '{name}' value '{val}' does not match pattern '{field_info.pattern}'"
            )
        if field_info.min_length is not None and isinstance(val, str):
          if len(val) < field_info.min_length:
            raise ValidationError(
                f"Field '{name}' shorter than min_length={field_info.min_length}"
            )
        if field_info.max_length is not None and isinstance(val, str):
          if len(val) > field_info.max_length:
            raise ValidationError(
                f"Field '{name}' longer than max_length={field_info.max_length}"
            )
        if isinstance(val, (int, float)):
          if field_info.gt is not None and not (val > field_info.gt):
            raise ValidationError(f"Field '{name}' must be > {field_info.gt}")
          if field_info.ge is not None and not (val >= field_info.ge):
            raise ValidationError(f"Field '{name}' must be >= {field_info.ge}")
          if field_info.lt is not None and not (val < field_info.lt):
            raise ValidationError(f"Field '{name}' must be < {field_info.lt}")
          if field_info.le is not None and not (val <= field_info.le):
            raise ValidationError(f"Field '{name}' must be <= {field_info.le}")
      return val

    @classmethod
    def model_validate(cls, obj: Any) -> Any:
      if isinstance(obj, cls):
        return obj
      if isinstance(obj, dict):
        return cls(**obj)
      raise ValidationError(f"Cannot validate {type(obj)} as {cls.__name__}")

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
      def _conv(v: Any) -> Any:
        if isinstance(v, BaseModel):
          return v.model_dump(mode=mode)
        if isinstance(v, Enum):
          return v.value if mode == "json" else v
        if isinstance(v, list):
          return [_conv(i) for i in v]
        if isinstance(v, dict):
          return {k: _conv(val) for k, val in v.items()}
        return v

      return {k: _conv(v) for k, v in self.__dict__.items()}

    def model_dump_json(self) -> str:
      return json.dumps(self.model_dump(mode="json"))


class PlanStatus(str, Enum):
  """Member insurance coverage status."""

  ACTIVE = "ACTIVE"
  INACTIVE = "INACTIVE"
  COBRA_PENDING = "COBRA_PENDING"


class ClaimStatus(str, Enum):
  """Adjudication status of a healthcare claim."""

  PAID = "PAID"
  DENIED = "DENIED"
  PENDING_REVIEW = "PENDING_REVIEW"
  APPEALED = "APPEALED"


class ClaimStatusFilter(str, Enum):
  """Filter options when listing claims for a member."""

  ALL = "ALL"
  PAID = "PAID"
  DENIED = "DENIED"
  PENDING_REVIEW = "PENDING_REVIEW"
  APPEALED = "APPEALED"


class NetworkStatus(str, Enum):
  """Provider network participation tier."""

  IN_NETWORK = "IN_NETWORK"
  OUT_OF_NETWORK = "OUT_OF_NETWORK"


class PriorAuthStatus(str, Enum):
  """Clinical prior authorization adjudication status."""

  APPROVED = "APPROVED"
  DENIED = "DENIED"
  PENDING_CLINICAL_REVIEW = "PENDING_CLINICAL_REVIEW"
  EXPIRED = "EXPIRED"


class AppealStatus(str, Enum):
  """Lifecycle status of a formal claim appeal."""

  SUBMITTED_UNDER_REVIEW = "SUBMITTED_UNDER_REVIEW"
  AWAITING_HITL_CONFIRMATION = "AWAITING_HITL_CONFIRMATION"
  APPROVED = "APPROVED"
  REJECTED = "REJECTED"


class IntentCategory(str, Enum):
  """Classified user intent for observability intent-vs-outcome tracking."""

  ELIGIBILITY_CHECK = "ELIGIBILITY_CHECK"
  CLAIM_STATUS_INQUIRY = "CLAIM_STATUS_INQUIRY"
  DENIAL_ROOT_CAUSE_ANALYSIS = "DENIAL_ROOT_CAUSE_ANALYSIS"
  PRIOR_AUTH_VERIFICATION = "PRIOR_AUTH_VERIFICATION"
  COST_ESTIMATION = "COST_ESTIMATION"
  APPEAL_SUBMISSION = "APPEAL_SUBMISSION"
  CLINICAL_POLICY_SEARCH = "CLINICAL_POLICY_SEARCH"
  SECURITY_VIOLATION = "SECURITY_VIOLATION"
  GENERAL_INQUIRY = "GENERAL_INQUIRY"


# ============================================================================
# Tool Input / Argument Validation Schemas
# ============================================================================


class MemberLookupRequest(BaseModel):
  """Validated input schema for `get_member_eligibility`."""

  model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

  member_id_or_name: str = Field(
      ...,
      min_length=2,
      max_length=120,
      description=(
          "Member ID (e.g., 'MEM-1001') or full/partial patient name "
          "(e.g., 'Sarah Jenkins')."
      ),
  )


class ListClaimsRequest(BaseModel):
  """Validated input schema for `list_member_claims`."""

  model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

  member_id: str = Field(
      ...,
      pattern=r"^MEM-\d{4}$",
      description="Canonical Member ID matching '^MEM-\\d{4}$' (e.g., 'MEM-1001').",
  )
  status_filter: ClaimStatusFilter = Field(
      default=ClaimStatusFilter.ALL,
      description="Filter claims by status: ALL, PAID, DENIED, PENDING_REVIEW, or APPEALED.",
  )

  @field_validator("member_id", mode="before")
  @classmethod
  def normalize_member_id(cls, v: str) -> str:
    return str(v).strip().upper()

  @field_validator("status_filter", mode="before")
  @classmethod
  def normalize_status_filter(
      cls, v: str | ClaimStatusFilter
  ) -> str | ClaimStatusFilter:
    if isinstance(v, str):
      return v.strip().upper()
    return v


class ClaimDetailsRequest(BaseModel):
  """Validated input schema for `get_claim_details`."""

  model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

  claim_id: str = Field(
      ...,
      pattern=r"^CLM-\d{4}-\d{4}$",
      description="Canonical Claim ID matching '^CLM-\\d{4}-\\d{4}$' (e.g., 'CLM-2026-9002').",
  )

  @field_validator("claim_id", mode="before")
  @classmethod
  def normalize_claim_id(cls, v: str) -> str:
    return str(v).strip().upper()


class PriorAuthLookupRequest(BaseModel):
  """Validated input schema for `check_prior_authorization`."""

  model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

  member_id: str = Field(
      default="",
      description="Optional Member ID (e.g., 'MEM-1001') to list prior authorizations.",
  )
  pa_id: str = Field(
      default="",
      description="Optional Prior Authorization ID (e.g., 'PA-2026-441').",
  )


class CostEstimateRequest(BaseModel):
  """Validated input schema for `estimate_patient_responsibility`."""

  model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

  member_id: str = Field(
      ...,
      pattern=r"^MEM-\d{4}$",
      description="Canonical Member ID (e.g., 'MEM-1001').",
  )
  cpt_code: str = Field(
      ...,
      pattern=r"^\d{4,5}[A-Z]?$",
      description="5-character CPT/HCPCS procedure code (e.g., '29881', '73721').",
  )
  estimated_allowed_amount: float = Field(
      ...,
      gt=0.0,
      le=1_000_000.0,
      description="Negotiated plan allowed amount in USD (must be > 0).",
  )
  in_network: bool = Field(
      default=True,
      description="True if the rendering provider is in-network, False if out-of-network.",
  )

  @field_validator("member_id", "cpt_code", mode="before")
  @classmethod
  def normalize_codes(cls, v: str) -> str:
    return str(v).strip().upper()


class SubmitAppealRequest(BaseModel):
  """Validated input schema for `submit_claim_appeal` with HITL confirmation support."""

  model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

  claim_id: str = Field(
      ...,
      pattern=r"^CLM-\d{4}-\d{4}$",
      description="Denied Claim ID to appeal (e.g., 'CLM-2026-9002').",
  )
  appeal_reason: str = Field(
      ...,
      min_length=10,
      max_length=2000,
      description="Detailed clinical or administrative rationale for the appeal.",
  )
  supporting_reference: str = Field(
      default="",
      max_length=500,
      description="Supporting Prior Authorization ID (e.g., 'PA-2026-441') or clinical guideline ID.",
  )
  human_confirmed: bool = Field(
      default=True,
      description=(
          "Human-in-the-loop confirmation flag. High-dollar appeals (>$1,000) "
          "are gated by HITL policy verification."
      ),
  )

  @field_validator("claim_id", mode="before")
  @classmethod
  def normalize_claim_id(cls, v: str) -> str:
    return str(v).strip().upper()


class PolicySearchRequest(BaseModel):
  """Validated input schema for `search_clinical_policies_and_guidelines`."""

  model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

  query: str = Field(
      ...,
      min_length=3,
      max_length=500,
      description="Natural language query, CPT code, ICD-10 code, or CARC denial code to search in the policy vector store.",
  )
  top_k: int = Field(
      default=3,
      ge=1,
      le=10,
      description="Number of top matching policy documents to retrieve (1-10).",
  )


# ============================================================================
# Domain & Tool Output / Response Schemas
# ============================================================================


class CopaySchedule(BaseModel):
  """Fixed copayment amounts in USD by service category."""

  model_config = ConfigDict(extra="forbid")

  pcp: float = Field(..., ge=0.0, description="Primary Care Provider visit copay in USD.")
  specialist: float = Field(..., ge=0.0, description="Specialist visit copay in USD.")
  urgent_care: float = Field(..., ge=0.0, description="Urgent Care visit copay in USD.")
  emergency_room: float = Field(..., ge=0.0, description="Emergency Room visit copay in USD.")


class MemberRecord(BaseModel):
  """Canonical member eligibility and accumulator record."""

  model_config = ConfigDict(extra="forbid")

  member_id: str = Field(..., pattern=r"^MEM-\d{4}$")
  full_name: str
  dob: str = Field(..., description="Date of birth (YYYY-MM-DD or masked in logs).")
  plan_name: str
  group_number: str
  status: PlanStatus
  effective_date: str
  termination_date: Optional[str] = None
  cobra_eligible: Optional[bool] = None
  deductible_individual: float = Field(..., ge=0.0)
  deductible_met: float = Field(..., ge=0.0)
  oop_max_individual: float = Field(..., ge=0.0)
  oop_max_met: float = Field(..., ge=0.0)
  coinsurance_in_network: float = Field(..., ge=0.0, le=1.0)
  coinsurance_out_of_network: float = Field(..., ge=0.0, le=1.0)
  copays: CopaySchedule


class DiagnosisCode(BaseModel):
  """ICD-10-CM diagnosis code and clinical description."""

  model_config = ConfigDict(extra="forbid")

  icd10: str
  description: str


class ClaimLineItem(BaseModel):
  """Individual CPT/HCPCS service line item on a claim."""

  model_config = ConfigDict(extra="forbid")

  cpt: str
  description: str
  billed: float = Field(..., ge=0.0)
  allowed: float = Field(..., ge=0.0)
  plan_paid: float = Field(..., ge=0.0)
  patient_resp: float = Field(..., ge=0.0)
  notes: str


class ClaimRecord(BaseModel):
  """Full adjudicated healthcare claim record."""

  model_config = ConfigDict(extra="forbid")

  claim_id: str = Field(..., pattern=r"^CLM-\d{4}-\d{4}$")
  member_id: str = Field(..., pattern=r"^MEM-\d{4}$")
  patient_name: str
  service_date: str
  received_date: str
  processed_date: Optional[str] = None
  provider_name: str
  provider_npi: str
  network_status: NetworkStatus
  diagnosis_codes: list[DiagnosisCode]
  line_items: list[ClaimLineItem]
  total_billed: float = Field(..., ge=0.0)
  total_allowed: float = Field(..., ge=0.0)
  total_plan_paid: float = Field(..., ge=0.0)
  total_patient_responsibility: float = Field(..., ge=0.0)
  status: ClaimStatus
  eob_number: Optional[str] = None
  denial_code: Optional[str] = None
  denial_reason: Optional[str] = None
  appeal_eligible: bool = False
  appeal_deadline: Optional[str] = None


class ClaimSummaryItem(BaseModel):
  """Compact claim summary item for list views."""

  model_config = ConfigDict(extra="forbid")

  claim_id: str
  member_id: str
  patient_name: str
  service_date: str
  provider_name: str
  total_billed: float
  total_plan_paid: float
  total_patient_responsibility: float
  status: ClaimStatus
  denial_code: Optional[str] = None


class PriorAuthorizationRecord(BaseModel):
  """Clinical prior authorization record."""

  model_config = ConfigDict(extra="forbid")

  pa_id: str = Field(..., pattern=r"^PA-\d{4}-\d{3,4}$")
  member_id: str = Field(..., pattern=r"^MEM-\d{4}$")
  patient_name: str
  cpt_code: str
  procedure_description: str
  status: PriorAuthStatus
  approved_provider: str
  approved_npi: str
  valid_from: Optional[str] = None
  valid_to: Optional[str] = None
  notes: str


class AppealRecord(BaseModel):
  """Submitted claim appeal tracking record."""

  model_config = ConfigDict(extra="forbid")

  appeal_id: str
  claim_id: str
  member_id: str
  patient_name: str
  original_denial_code: Optional[str] = None
  appeal_reason: str
  supporting_reference: str
  status: AppealStatus
  submitted_timestamp: str
  estimated_resolution_days: int
  hitl_verified: bool = True
  next_steps: str


class PolicyDocumentMatch(BaseModel):
  """Retrieved policy document chunk from the vector store."""

  model_config = ConfigDict(extra="forbid")

  policy_id: str
  title: str
  category: str
  cpt_codes: list[str]
  carc_codes: list[str]
  similarity_score: float = Field(..., ge=0.0, le=1.0)
  content: str


# ============================================================================
# Constrained Tool Response Envelope Schemas
# ============================================================================


class MemberEligibilityResponse(BaseModel):
  """Strict response schema for `get_member_eligibility`."""

  model_config = ConfigDict(extra="forbid")

  found: bool
  member: Optional[MemberRecord] = None
  error: Optional[str] = None
  available_member_ids: Optional[list[str]] = None


class MemberClaimsListResponse(BaseModel):
  """Strict response schema for `list_member_claims`."""

  model_config = ConfigDict(extra="forbid")

  member_id: str
  status_filter: str
  count: int = Field(..., ge=0)
  claims: list[ClaimSummaryItem] = Field(default_factory=list)
  error: Optional[str] = None


class ClaimDetailsResponse(BaseModel):
  """Strict response schema for `get_claim_details`."""

  model_config = ConfigDict(extra="forbid")

  found: bool
  claim: Optional[ClaimRecord] = None
  error: Optional[str] = None
  available_claim_ids: Optional[list[str]] = None


class PriorAuthLookupResponse(BaseModel):
  """Strict response schema for `check_prior_authorization`."""

  model_config = ConfigDict(extra="forbid")

  found: bool
  count: int = Field(..., ge=0)
  prior_authorizations: list[PriorAuthorizationRecord] = Field(
      default_factory=list
  )
  message: Optional[str] = None


class CostEstimateBreakdown(BaseModel):
  """Detailed financial breakdown of estimated patient responsibility."""

  model_config = ConfigDict(extra="forbid")

  estimated_allowed_amount: float
  remaining_deductible_before_service: float
  applied_to_deductible: float
  coinsurance_rate: float
  applied_coinsurance: float
  remaining_oop_max_before_service: float
  estimated_patient_responsibility: float
  estimated_plan_payment: float


class CostEstimateResponse(BaseModel):
  """Strict response schema for `estimate_patient_responsibility`."""

  model_config = ConfigDict(extra="forbid")

  eligible: bool
  member_id: str
  patient_name: Optional[str] = None
  plan_name: Optional[str] = None
  cpt_code: str
  network_tier: Optional[NetworkStatus] = None
  breakdown: Optional[CostEstimateBreakdown] = None
  error: Optional[str] = None


class ClaimAppealResponse(BaseModel):
  """Strict response schema for `submit_claim_appeal`."""

  model_config = ConfigDict(extra="forbid")

  success: bool
  requires_hitl_confirmation: bool = False
  hitl_prompt: Optional[str] = None
  appeal: Optional[AppealRecord] = None
  error: Optional[str] = None


class PolicySearchResponse(BaseModel):
  """Strict response schema for `search_clinical_policies_and_guidelines`."""

  model_config = ConfigDict(extra="forbid")

  query: str
  count: int = Field(..., ge=0)
  matches: list[PolicyDocumentMatch] = Field(default_factory=list)
