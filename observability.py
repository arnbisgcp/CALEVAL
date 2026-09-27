"""Enterprise Observability: Structured JSON Logging, OpenTelemetry Spans, PII/PHI Redaction, and Intent-vs-Outcome Capture."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import re
import sys
import time
from typing import Any, Optional

from schemas import IntentCategory

try:
  from opentelemetry import trace as otel_trace
except ImportError:  # pragma: no cover
  otel_trace = None  # type: ignore[assignment]


# ============================================================================
# 1. HIPAA PII / PHI Redaction Engine
# ============================================================================


class PiiPhiRedactor:
  """Deterministic HIPAA PII/PHI scrubber for text prompts, tool payloads, and logs."""

  SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
  PHONE_PATTERN = re.compile(
      r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"
  )
  EMAIL_PATTERN = re.compile(
      r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
  )
  CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")
  MBI_PATTERN = re.compile(
      r"\b[1-9][AC-HJKMNP-RT-Yac-hjkmnp-rt-y][AC-HJKMNP-RT-Yac-hjkmnp-rt-y0-9]\d"
      r"-?[AC-HJKMNP-RT-Yac-hjkmnp-rt-y][AC-HJKMNP-RT-Yac-hjkmnp-rt-y0-9]\d"
      r"-?[AC-HJKMNP-RT-Yac-hjkmnp-rt-y]{2}\d{2}\b"
  )
  DOB_FIELD_NAMES = frozenset({"dob", "date_of_birth", "birth_date", "birthdate"})
  SENSITIVE_KEY_NAMES = frozenset(
      {"ssn", "social_security", "password", "api_key", "secret", "token"}
  )

  @classmethod
  def redact_text(cls, text: str) -> str:
    """Scrubs SSNs, phone numbers, emails, credit cards, and Medicare IDs from raw strings."""
    if not text:
      return text
    redacted = cls.SSN_PATTERN.sub("[REDACTED-SSN]", text)
    redacted = cls.CREDIT_CARD_PATTERN.sub("[REDACTED-PAYMENT-CARD]", redacted)
    redacted = cls.PHONE_PATTERN.sub("[REDACTED-PHONE]", redacted)
    redacted = cls.EMAIL_PATTERN.sub("[REDACTED-EMAIL]", redacted)
    redacted = cls.MBI_PATTERN.sub("[REDACTED-MEDICARE-MBI]", redacted)
    return redacted

  @classmethod
  def mask_dob(cls, dob_value: str) -> str:
    """Masks month and day of birth while preserving birth year for age-tier verification."""
    if not dob_value or len(dob_value) < 4:
      return "[REDACTED-DOB]"
    return f"{dob_value[:4]}-**-** [REDACTED-DOB]"

  @classmethod
  def redact_payload(cls, payload: Any) -> Any:
    """Recursively scrubs PII/PHI from dictionaries, lists, Pydantic models, and strings."""
    if payload is None:
      return None
    if hasattr(payload, "model_dump"):
      payload = payload.model_dump(mode="json")
    if isinstance(payload, str):
      return cls.redact_text(payload)
    if isinstance(payload, list):
      return [cls.redact_payload(item) for item in payload]
    if isinstance(payload, dict):
      sanitized: dict[str, Any] = {}
      for k, v in payload.items():
        key_lower = str(k).lower()
        if key_lower in cls.SENSITIVE_KEY_NAMES:
          sanitized[k] = "[REDACTED-SECRET]"
        elif key_lower in cls.DOB_FIELD_NAMES and isinstance(v, str):
          sanitized[k] = cls.mask_dob(v)
        else:
          sanitized[k] = cls.redact_payload(v)
      return sanitized
    return payload


# ============================================================================
# 2. Cloud Logging Compatible Structured JSON Formatter
# ============================================================================


class StructuredJsonFormatter(logging.Formatter):
  """Emits single-line structured JSON records with automatic PII/PHI redaction."""

  def format(self, record: logging.LogRecord) -> str:
    log_entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "severity": record.levelname,
        "logger": record.name,
        "message": PiiPhiRedactor.redact_text(record.getMessage()),
        "component": "healthcare-claims-adk-agent",
    }

    # Attach OpenTelemetry trace/span context when available
    if otel_trace is not None:
      span = otel_trace.get_current_span()
      if span is not None:
        span_ctx = span.get_span_context()
        if span_ctx and span_ctx.is_valid:
          log_entry["logging.googleapis.com/trace"] = format(
              span_ctx.trace_id, "032x"
          )
          log_entry["logging.googleapis.com/spanId"] = format(
              span_ctx.span_id, "016x"
          )

    # Merge structured extra fields after PII redaction
    structured_data = getattr(record, "structured_data", None)
    if isinstance(structured_data, dict):
      log_entry.update(PiiPhiRedactor.redact_payload(structured_data))

    return json.dumps(log_entry, ensure_ascii=False, default=str)


def get_structured_logger(
    name: str = "healthcare_claims_agent",
) -> logging.Logger:
  """Returns a configured structured JSON logger writing to stdout."""
  logger = logging.getLogger(name)
  if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredJsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
  return logger


structured_logger = get_structured_logger()


def log_structured_event(
    event_type: str,
    message: str,
    *,
    severity: str = "INFO",
    **kwargs: Any,
) -> dict[str, Any]:
  """Emits a structured JSON log event with automatic PII/PHI scrubbing."""
  payload = {
      "event_type": event_type,
      **PiiPhiRedactor.redact_payload(kwargs),
  }
  level = getattr(logging, severity.upper(), logging.INFO)
  structured_logger.log(level, message, extra={"structured_data": payload})
  return payload


# ============================================================================
# 3. Intent vs. Outcome Telemetry Classifier & Tracker
# ============================================================================


class IntentOutcomeTracker:
  """Classifies user intent and audits intent-vs-outcome alignment across agent turns."""

  EXPECTED_TOOLS_BY_INTENT: dict[IntentCategory, set[str]] = {
      IntentCategory.ELIGIBILITY_CHECK: {"get_member_eligibility"},
      IntentCategory.CLAIM_STATUS_INQUIRY: {
          "list_member_claims",
          "get_claim_details",
      },
      IntentCategory.DENIAL_ROOT_CAUSE_ANALYSIS: {
          "get_claim_details",
          "check_prior_authorization",
          "search_clinical_policies_and_guidelines",
      },
      IntentCategory.PRIOR_AUTH_VERIFICATION: {"check_prior_authorization"},
      IntentCategory.COST_ESTIMATION: {"estimate_patient_responsibility"},
      IntentCategory.APPEAL_SUBMISSION: {"submit_claim_appeal"},
      IntentCategory.CLINICAL_POLICY_SEARCH: {
          "search_clinical_policies_and_guidelines"
      },
  }

  @classmethod
  def classify_intent(cls, user_query: str) -> list[IntentCategory]:
    """Deterministically classifies one or more healthcare domain intents from user text."""
    q = (user_query or "").lower()
    intents: list[IntentCategory] = []

    if any(
        kw in q
        for kw in (
            "ignore previous",
            "system prompt",
            "dump all",
            "bypass hipaa",
            "drop table",
        )
    ):
      return [IntentCategory.SECURITY_VIOLATION]

    if any(
        kw in q
        for kw in (
            "appeal",
            "dispute",
            "overturn",
            "reconsideration",
            "grievance",
        )
    ):
      intents.append(IntentCategory.APPEAL_SUBMISSION)
    if any(
        kw in q
        for kw in (
            "denied",
            "denial",
            "why was",
            "co-197",
            "co-50",
            "reject",
            "eob",
        )
    ):
      intents.append(IntentCategory.DENIAL_ROOT_CAUSE_ANALYSIS)
    if any(
        kw in q
        for kw in (
            "prior auth",
            "pa-",
            "precertification",
            "authorization",
            "pre-auth",
        )
    ):
      intents.append(IntentCategory.PRIOR_AUTH_VERIFICATION)
    if any(
        kw in q
        for kw in (
            "estimate",
            "how much would",
            "out of pocket",
            "out-of-pocket",
            "cost for",
            "allowed amount",
        )
    ):
      intents.append(IntentCategory.COST_ESTIMATION)
    if any(
        kw in q
        for kw in (
            "policy",
            "policies",
            "guideline",
            "guidelines",
            "medical necessity",
            "clinical criteria",
            "lcd",
            "ncd",
        )
    ):
      intents.append(IntentCategory.CLINICAL_POLICY_SEARCH)
    if any(
        kw in q
        for kw in (
            "eligibility",
            "eligible",
            "coverage",
            "deductible",
            "copay",
            "plan",
        )
    ):
      intents.append(IntentCategory.ELIGIBILITY_CHECK)
    if any(kw in q for kw in ("claim", "clm-", "claims", "billed")):
      intents.append(IntentCategory.CLAIM_STATUS_INQUIRY)

    return intents or [IntentCategory.GENERAL_INQUIRY]

  @classmethod
  def evaluate_and_log_outcome(
      cls,
      *,
      invocation_id: str,
      user_id: str,
      user_query: str,
      classified_intents: list[IntentCategory],
      invoked_tools: list[str],
      tool_errors: list[str],
      final_response_text: str,
      start_time: float,
      blocked_by_guardrail: bool = False,
      awaiting_hitl: bool = False,
      routed_model: str = "gemini-2.5-flash",
  ) -> dict[str, Any]:
    """Computes Intent-vs-Outcome alignment metrics and emits an audit log & OTel attributes."""
    latency_ms = round((time.time() - start_time) * 1000.0, 2)
    invoked_set = set(invoked_tools)

    if blocked_by_guardrail:
      outcome_status = "BLOCKED_BY_GUARDRAIL"
      goal_alignment_score = 1.0 if IntentCategory.SECURITY_VIOLATION in classified_intents else 0.0
    elif awaiting_hitl:
      outcome_status = "AWAITING_HITL_CONFIRMATION"
      goal_alignment_score = 1.0
    elif tool_errors and not final_response_text:
      outcome_status = "ERROR"
      goal_alignment_score = 0.0
    else:
      matched_intents = 0
      for intent in classified_intents:
        expected = cls.EXPECTED_TOOLS_BY_INTENT.get(intent, set())
        if not expected or (invoked_set & expected) or final_response_text:
          matched_intents += 1
      goal_alignment_score = round(
          matched_intents / max(len(classified_intents), 1), 2
      )
      outcome_status = (
          "SUCCESS" if goal_alignment_score >= 0.8 else "PARTIAL_COMPLETION"
      )

    audit_record = log_structured_event(
        "INTENT_VS_OUTCOME_AUDIT",
        f"Intent-vs-Outcome evaluation completed with status={outcome_status}",
        invocation_id=invocation_id,
        user_id=user_id,
        redacted_user_query=PiiPhiRedactor.redact_text(user_query),
        intent_categories=[i.value for i in classified_intents],
        invoked_tools=invoked_tools,
        tool_error_count=len(tool_errors),
        outcome_status=outcome_status,
        goal_alignment_score=goal_alignment_score,
        routed_model=routed_model,
        response_char_length=len(final_response_text or ""),
        latency_ms=latency_ms,
    )

    if otel_trace is not None:
      span = otel_trace.get_current_span()
      if span is not None and span.is_recording():
        span.set_attribute(
            "gen_ai.intent.categories",
            ",".join(i.value for i in classified_intents),
        )
        span.set_attribute("gen_ai.outcome.status", outcome_status)
        span.set_attribute(
            "gen_ai.outcome.goal_alignment_score", goal_alignment_score
        )
        span.set_attribute("gen_ai.tools.invoked", ",".join(invoked_tools))
        span.set_attribute("gen_ai.model.routed", routed_model)

    return audit_record
