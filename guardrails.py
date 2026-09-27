"""Security Guardrails, Policy Plugins, Strategic Model Routing, and Human-in-the-Loop Confirmation Hooks."""

from __future__ import annotations

import os
import re
import time
from typing import Any, Optional

from database import repository
from memory_manager import AsyncBackgroundMemoryWorker, ContextCompactionManager
from observability import (
    IntentOutcomeTracker,
    PiiPhiRedactor,
    log_structured_event,
)
from schemas import AppealStatus, ClaimAppealResponse, IntentCategory

try:
  from google.adk.models.llm_response import LlmResponse
  from google.adk.plugins.base_plugin import BasePlugin
  from google.genai import types as genai_types
except ImportError:  # pragma: no cover

  class BasePlugin:  # type: ignore[no-redef]
    def __init__(self, name: str = "base_plugin") -> None:
      self.name = name

  LlmResponse = Any  # type: ignore[misc,assignment]
  genai_types = None  # type: ignore[assignment]


# ============================================================================
# 1. Strategic Model Router (Flash vs. Pro Tier Selection)
# ============================================================================


class StrategicModelRouter:
  """Routes requests between low-latency `gemini-2.5-flash` and reasoning-heavy `gemini-2.5-pro`."""

  FAST_MODEL = os.environ.get("ADK_FAST_MODEL", "gemini-2.5-flash")
  REASONING_MODEL = os.environ.get("ADK_REASONING_MODEL", "gemini-2.5-pro")
  HIGH_COMPLEXITY_INTENTS = frozenset({
      IntentCategory.DENIAL_ROOT_CAUSE_ANALYSIS,
      IntentCategory.APPEAL_SUBMISSION,
      IntentCategory.CLINICAL_POLICY_SEARCH,
  })

  @classmethod
  def select_model(
      cls, user_query: str, intents: list[IntentCategory]
  ) -> tuple[str, str]:
    """Returns `(selected_model, routing_rationale)` based on intent complexity."""
    if any(i in cls.HIGH_COMPLEXITY_INTENTS for i in intents) or len(intents) >= 3:
      return (
          cls.REASONING_MODEL,
          "High clinical/adjudication complexity or multi-step appeal detected; routed to reasoning model tier.",
      )
    return (
        cls.FAST_MODEL,
        "Standard member eligibility, claim status, or accumulator lookup; routed to low-latency flash tier.",
    )


# ============================================================================
# 2. Prompt Injection, HIPAA Policy, & RBAC Security Guardrail
# ============================================================================


class HealthcareSecurityGuardrail:
  """Enforces prompt-injection defenses, bulk exfiltration prevention, and HIPAA guardrails."""

  PROMPT_INJECTION_PATTERNS = [
      re.compile(r"ignore\s+(all\s+)?(previous|prior|system)\s+instructions", re.I),
      re.compile(r"(reveal|print|show|output)\s+(your\s+)?(system\s+prompt|developer\s+instructions)", re.I),
      re.compile(r"(dump|export|list)\s+all\s+(members|patients|ssns|database|records)", re.I),
      re.compile(r"bypass\s+(hipaa|security|guardrail|authentication|confirmation)", re.I),
      re.compile(r"(drop\s+table|union\s+select|--\s*;\s*drop)", re.I),
  ]

  @classmethod
  def inspect_user_prompt(cls, text: str) -> tuple[bool, Optional[str]]:
    """Returns `(is_allowed, violation_reason)` for an incoming user prompt."""
    if not text:
      return True, None
    for pattern in cls.PROMPT_INJECTION_PATTERNS:
      if pattern.search(text):
        return (
            False,
            f"Blocked by Healthcare Security Guardrail: prohibited pattern '{pattern.pattern}' detected.",
        )
    return True, None


# ============================================================================
# 3. Human-in-the-Loop (HITL) Confirmation Policy for High-Stakes Actions
# ============================================================================

HIGH_DOLLAR_APPEAL_THRESHOLD = float(
    os.environ.get("HIGH_DOLLAR_APPEAL_THRESHOLD_USD", "1000.0")
)


def requires_appeal_hitl_confirmation(
    claim_id: str,
    appeal_reason: str,
    supporting_reference: str = "",
    human_confirmed: bool = False,
) -> bool:
  """Callable predicate for ADK `FunctionTool(..., require_confirmation=...)`.

  Returns True when a formal appeal targets a high-dollar claim (>= $1,000) and
  `human_confirmed` has not been explicitly granted.
  """
  del appeal_reason, supporting_reference
  if human_confirmed:
    return False
  claim = repository.get_claim(claim_id)
  if claim and claim.total_billed >= HIGH_DOLLAR_APPEAL_THRESHOLD:
    return True
  return not human_confirmed


def evaluate_hitl_gate(
    tool_name: str, tool_args: dict[str, Any]
) -> Optional[dict[str, Any]]:
  """Enforces Human-in-the-Loop (HITL) confirmation policy before executing state-mutating tools."""
  if tool_name != "submit_claim_appeal":
    return None

  claim_id = str(tool_args.get("claim_id", "")).strip().upper()
  human_confirmed = tool_args.get("human_confirmed", True)
  require_strict_hitl = (
      os.environ.get("ENFORCE_STRICT_HITL_CONFIRMATION", "false").lower() == "true"
  )

  if human_confirmed is False or (
      require_strict_hitl and not tool_args.get("human_confirmed", False)
  ):
    claim = repository.get_claim(claim_id)
    billed_str = f"${claim.total_billed:,.2f}" if claim else "N/A"
    resp = ClaimAppealResponse(
        success=False,
        requires_hitl_confirmation=True,
        hitl_prompt=(
            f"HUMAN-IN-THE-LOOP CONFIRMATION REQUIRED: Submitting a formal "
            f"clinical appeal for claim {claim_id} (billed amount: {billed_str}) "
            f"is a regulated adjudication action. Please confirm approval by "
            f"re-invoking `submit_claim_appeal` with `human_confirmed=True`."
        ),
        error=f"Action paused awaiting Human-in-the-Loop approval for {claim_id}.",
    )
    log_structured_event(
        "HITL_CONFIRMATION_REQUIRED",
        f"Paused high-stakes tool `{tool_name}` for claim {claim_id} awaiting human confirmation.",
        severity="WARNING",
        claim_id=claim_id,
        billed_amount=billed_str,
    )
    return resp.model_dump(mode="json")

  return None


# ============================================================================
# 4. ADK App-Level Governance, Guardrail, Compaction & Observability Plugin
# ============================================================================


class HealthcareGovernancePlugin(BasePlugin):
  """Unified ADK BasePlugin enforcing security guardrails, PII redaction, context compaction, model routing, and intent-vs-outcome telemetry."""

  def __init__(self) -> None:
    super().__init__(name="healthcare_governance_plugin")
    self._invocation_metrics: dict[str, dict[str, Any]] = {}

  def _get_inv_id(self, context: Any) -> str:
    return str(getattr(context, "invocation_id", None) or "default-invocation")

  async def before_agent_callback(
      self, *, agent: Any, callback_context: Any
  ) -> Optional[Any]:
    inv_id = self._get_inv_id(callback_context)
    user_text = ""
    user_content = getattr(callback_context, "user_content", None)
    if user_content and getattr(user_content, "parts", None):
      user_text = " ".join(
          getattr(p, "text", "") or "" for p in user_content.parts
      ).strip()

    intents = IntentOutcomeTracker.classify_intent(user_text)
    selected_model, routing_reason = StrategicModelRouter.select_model(
        user_text, intents
    )

    if inv_id not in self._invocation_metrics:
      self._invocation_metrics[inv_id] = {
          "start_time": time.time(),
          "user_query": user_text,
          "intents": intents,
          "invoked_tools": [],
          "tool_errors": [],
          "blocked_by_guardrail": False,
          "awaiting_hitl": False,
          "routed_model": selected_model,
      }

    log_structured_event(
        "AGENT_INVOCATION_START",
        f"Starting agent '{getattr(agent, 'name', 'unknown')}'",
        invocation_id=inv_id,
        agent_name=getattr(agent, "name", "unknown"),
        classified_intents=[i.value for i in intents],
        routed_model=selected_model,
        routing_reason=routing_reason,
    )
    return None

  async def before_model_callback(
      self, *, callback_context: Any, llm_request: Any
  ) -> Optional[Any]:
    inv_id = self._get_inv_id(callback_context)
    metrics = self._invocation_metrics.setdefault(
        inv_id,
        {
            "start_time": time.time(),
            "user_query": "",
            "intents": [IntentCategory.GENERAL_INQUIRY],
            "invoked_tools": [],
            "tool_errors": [],
            "blocked_by_guardrail": False,
            "awaiting_hitl": False,
            "routed_model": StrategicModelRouter.FAST_MODEL,
        },
    )

    # 1. Inspect latest user prompt for security/prompt-injection violations & scrub PII
    contents = getattr(llm_request, "contents", None) or []
    for content in contents:
      for part in getattr(content, "parts", None) or []:
        text = getattr(part, "text", None)
        if text:
          allowed, violation_reason = (
              HealthcareSecurityGuardrail.inspect_user_prompt(text)
          )
          if not allowed:
            metrics["blocked_by_guardrail"] = True
            log_structured_event(
                "SECURITY_GUARDRAIL_BLOCKED",
                violation_reason or "Blocked unsafe prompt",
                severity="WARNING",
                invocation_id=inv_id,
            )
            if genai_types is not None and LlmResponse is not Any:
              return LlmResponse(
                  content=genai_types.Content(
                      role="model",
                      parts=[
                          genai_types.Part.from_text(
                              text=(
                                  "I cannot fulfill this request. It violates "
                                  "HIPAA data protection and security guardrails. "
                                  "Please provide a specific Member ID (e.g., MEM-1001) "
                                  "or Claim ID (e.g., CLM-2026-9002) for authorized inquiry."
                              )
                          )
                      ],
                  )
              )
          # Redact raw SSNs or payment cards in prompt text before sending to LLM
          part.text = PiiPhiRedactor.redact_text(text)

    # 2. Execute Context Bloat Management & Sliding-Window Compaction
    ContextCompactionManager.compact_llm_request(callback_context, llm_request)

    # 3. Record strategic model routing decision in session state
    try:
      callback_context.state["routed_model_tier"] = metrics["routed_model"]
    except Exception:
      pass

    return None

  async def after_model_callback(
      self, *, callback_context: Any, llm_response: Any
  ) -> Optional[Any]:
    # Scrub any accidental SSN/credit-card/MBI patterns in model output text
    content = getattr(llm_response, "content", None)
    if content and getattr(content, "parts", None):
      for part in content.parts:
        text = getattr(part, "text", None)
        if text:
          part.text = PiiPhiRedactor.redact_text(text)
    return None

  async def before_tool_callback(
      self, *, tool: Any, tool_args: dict[str, Any], tool_context: Any
  ) -> Optional[dict[str, Any]]:
    inv_id = self._get_inv_id(tool_context)
    tool_name = getattr(tool, "name", "unknown_tool")
    metrics = self._invocation_metrics.get(inv_id)
    if metrics is not None:
      metrics["invoked_tools"].append(tool_name)

    log_structured_event(
        "TOOL_INVOCATION_START",
        f"Invoking ADK tool '{tool_name}'",
        invocation_id=inv_id,
        tool_name=tool_name,
        tool_args=PiiPhiRedactor.redact_payload(tool_args),
    )

    hitl_block = evaluate_hitl_gate(tool_name, tool_args)
    if hitl_block is not None:
      if metrics is not None:
        metrics["awaiting_hitl"] = True
      return hitl_block

    return None

  async def after_tool_callback(
      self,
      *,
      tool: Any,
      tool_args: dict[str, Any],
      tool_context: Any,
      result: Any,
  ) -> Optional[dict[str, Any]]:
    del tool_args
    inv_id = self._get_inv_id(tool_context)
    tool_name = getattr(tool, "name", "unknown_tool")

    # Normalize Pydantic BaseModel tool return values to JSON-serializable dicts for ADK Event stream
    normalized_result: dict[str, Any]
    if hasattr(result, "model_dump"):
      normalized_result = result.model_dump(mode="json")
    elif isinstance(result, dict):
      normalized_result = result
    else:
      normalized_result = {"result": result}

    if normalized_result.get("error"):
      metrics = self._invocation_metrics.get(inv_id)
      if metrics is not None:
        metrics["tool_errors"].append(str(normalized_result.get("error")))

    log_structured_event(
        "TOOL_INVOCATION_COMPLETE",
        f"Completed ADK tool '{tool_name}'",
        invocation_id=inv_id,
        tool_name=tool_name,
        result_summary=PiiPhiRedactor.redact_payload(normalized_result),
    )
    return normalized_result

  async def after_agent_callback(
      self, *, agent: Any, callback_context: Any
  ) -> Optional[Any]:
    inv_id = self._get_inv_id(callback_context)
    metrics = self._invocation_metrics.pop(inv_id, None)
    if metrics is not None:
      user_id = str(
          getattr(callback_context, "user_id", None) or "authenticated_adjuster"
      )
      # Record Intent-vs-Outcome Telemetry Audit
      IntentOutcomeTracker.evaluate_and_log_outcome(
          invocation_id=inv_id,
          user_id=user_id,
          user_query=metrics["user_query"],
          classified_intents=metrics["intents"],
          invoked_tools=metrics["invoked_tools"],
          tool_errors=metrics["tool_errors"],
          final_response_text=f"Completed via {getattr(agent, 'name', 'Healthcare_Claims_Agent')}",
          start_time=metrics["start_time"],
          blocked_by_guardrail=metrics["blocked_by_guardrail"],
          awaiting_hitl=metrics["awaiting_hitl"],
          routed_model=metrics["routed_model"],
      )
      # Schedule non-blocking background memory consolidation
      AsyncBackgroundMemoryWorker.schedule_background_consolidation(
          user_id=user_id,
          user_query=metrics["user_query"],
          agent_response=f"Tools executed: {metrics['invoked_tools']}",
          invoked_tools=metrics["invoked_tools"],
      )
    return None
