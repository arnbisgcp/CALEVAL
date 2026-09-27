"""Context Bloat Management, Sliding-Window Memory Compaction, and Async Background Memory Operations."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import re
from typing import Any, Optional
import uuid

from database import repository
from observability import PiiPhiRedactor, log_structured_event

MAX_HISTORY_CONTENTS = 8
MAX_ESTIMATED_CONTEXT_TOKENS = 6000
MAX_PART_TEXT_CHARS = 2400

_MEMBER_ID_RE = re.compile(r"\bMEM-\d{4}\b", re.IGNORECASE)
_CLAIM_ID_RE = re.compile(r"\bCLM-\d{4}-\d{4}\b", re.IGNORECASE)
_PA_ID_RE = re.compile(r"\bPA-\d{4}-\d{3,4}\b", re.IGNORECASE)


class ContextCompactionManager:
  """Prevents context window bloat via token budgeting, payload truncation, and turn compaction."""

  @staticmethod
  def estimate_tokens(text: str) -> int:
    """Approximates token count (~4 characters per token)."""
    return max(1, len(text or "") // 4)

  @classmethod
  def summarize_evicted_contents(cls, evicted_contents: list[Any]) -> str:
    """Compacts older conversation turns into a dense clinical session summary."""
    mentioned_members: set[str] = set()
    mentioned_claims: set[str] = set()
    mentioned_pas: set[str] = set()
    tools_called: list[str] = []
    key_snippets: list[str] = []

    for content in evicted_contents:
      parts = getattr(content, "parts", None) or []
      for part in parts:
        text = getattr(part, "text", None)
        if text:
          clean = PiiPhiRedactor.redact_text(text)
          mentioned_members.update(m.upper() for m in _MEMBER_ID_RE.findall(clean))
          mentioned_claims.update(c.upper() for c in _CLAIM_ID_RE.findall(clean))
          mentioned_pas.update(p.upper() for p in _PA_ID_RE.findall(clean))
          if len(key_snippets) < 3:
            key_snippets.append(clean[:160].replace("\n", " "))
        fc = getattr(part, "function_call", None)
        if fc and getattr(fc, "name", None):
          tools_called.append(fc.name)

    summary_parts = [
        f"Members referenced: {sorted(mentioned_members) or ['None']}",
        f"Claims referenced: {sorted(mentioned_claims) or ['None']}",
        f"Prior Auths referenced: {sorted(mentioned_pas) or ['None']}",
        f"Prior tools executed: {tools_called or ['None']}",
    ]
    if key_snippets:
      summary_parts.append(f"Prior context notes: {' | '.join(key_snippets)}")
    return " ; ".join(summary_parts)

  @classmethod
  def compact_llm_request(
      cls, callback_context: Any, llm_request: Any
  ) -> dict[str, Any]:
    """Applies sliding-window compaction and oversized payload trimming to `llm_request`."""
    contents = getattr(llm_request, "contents", None)
    if not contents:
      return {"compacted": False, "original_turns": 0, "retained_turns": 0}

    original_count = len(contents)
    total_chars = 0
    truncated_parts = 0

    # 1. Trim oversized individual text parts to prevent single-turn context bloat
    for content in contents:
      parts = getattr(content, "parts", None) or []
      for part in parts:
        text = getattr(part, "text", None)
        if text:
          total_chars += len(text)
          if len(text) > MAX_PART_TEXT_CHARS:
            part.text = (
                text[:MAX_PART_TEXT_CHARS]
                + "\n...[COMPACTED BY CONTEXT BUDGET MANAGER]..."
            )
            truncated_parts += 1

    estimated_tokens = cls.estimate_tokens("x" * total_chars)
    if (
        original_count <= MAX_HISTORY_CONTENTS
        and estimated_tokens <= MAX_ESTIMATED_CONTEXT_TOKENS
    ):
      return {
          "compacted": truncated_parts > 0,
          "original_turns": original_count,
          "retained_turns": original_count,
          "estimated_tokens": estimated_tokens,
      }

    # 2. Split into evicted older turns and retained recent window.
    # Ensure we split on a "user" text boundary so function_call/function_response pairs stay intact.
    split_idx = max(0, original_count - MAX_HISTORY_CONTENTS)
    while split_idx < original_count - 1:
      role = getattr(contents[split_idx], "role", "")
      parts = getattr(contents[split_idx], "parts", None) or []
      has_func_resp = any(
          getattr(p, "function_response", None) is not None for p in parts
      )
      if role == "user" and not has_func_resp:
        break
      split_idx += 1

    if split_idx <= 0 or split_idx >= original_count:
      return {
          "compacted": truncated_parts > 0,
          "original_turns": original_count,
          "retained_turns": original_count,
          "estimated_tokens": estimated_tokens,
      }

    evicted = contents[:split_idx]
    retained = contents[split_idx:]
    compacted_summary = cls.summarize_evicted_contents(evicted)

    # Persist compacted summary in session state and inject into system instructions
    try:
      prior_summary = callback_context.state.get("compacted_session_memory", "")
      merged_summary = (
          f"{prior_summary} || {compacted_summary}"
          if prior_summary
          else compacted_summary
      )
      callback_context.state["compacted_session_memory"] = merged_summary[-1500:]
      if hasattr(llm_request, "append_instructions"):
        llm_request.append_instructions(
            [f"[COMPACTED EARLIER SESSION MEMORY]: {merged_summary[-1500:]}"]
        )
    except Exception:
      pass

    llm_request.contents = retained
    log_structured_event(
        "CONTEXT_COMPACTION_EXECUTED",
        "Compacted older conversation turns to maintain token budget.",
        original_turns=original_count,
        retained_turns=len(retained),
        compacted_summary=compacted_summary,
    )
    return {
        "compacted": True,
        "original_turns": original_count,
        "retained_turns": len(retained),
        "compacted_summary": compacted_summary,
    }


class AsyncBackgroundMemoryWorker:
  """Executes non-blocking background memory extraction and persistence across sessions."""

  _executor = ThreadPoolExecutor(
      max_workers=2, thread_name_prefix="adk-async-memory-worker"
  )
  _pending_tasks: set[asyncio.Task[Any]] = set()

  @classmethod
  def _persist_memory_sync(
      cls,
      *,
      user_id: str,
      member_id: Optional[str],
      fact_summary: str,
  ) -> str:
    memory_id = f"MEMFACT-{uuid.uuid4().hex[:10].upper()}"
    redacted_fact = PiiPhiRedactor.redact_text(fact_summary)
    created_at = datetime.now(timezone.utc).isoformat()
    repository.insert_long_term_memory(
        memory_id=memory_id,
        user_id=user_id,
        member_id=member_id,
        fact_summary=redacted_fact,
        created_at=created_at,
    )
    log_structured_event(
        "ASYNC_MEMORY_CONSOLIDATED",
        f"Persisted long-term memory fact {memory_id} in background worker.",
        memory_id=memory_id,
        user_id=user_id,
        member_id=member_id,
        fact_summary=redacted_fact,
    )
    return memory_id

  @classmethod
  async def consolidate_turn_memory_async(
      cls,
      *,
      user_id: str,
      user_query: str,
      agent_response: str,
      invoked_tools: list[str],
  ) -> Optional[str]:
    """Asynchronously extracts and stores long-term clinical interaction facts in SQLite vector memory."""
    combined = f"{user_query} {agent_response}"
    member_matches = _MEMBER_ID_RE.findall(combined)
    claim_matches = _CLAIM_ID_RE.findall(combined)
    pa_matches = _PA_ID_RE.findall(combined)

    member_id = member_matches[0].upper() if member_matches else None
    fact_summary = (
        f"User '{user_id}' inquired about Members={sorted({m.upper() for m in member_matches})}, "
        f"Claims={sorted({c.upper() for c in claim_matches})}, "
        f"PriorAuths={sorted({p.upper() for p in pa_matches})} "
        f"via tools={invoked_tools}. Summary: {PiiPhiRedactor.redact_text(agent_response)[:240]}"
    )

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        cls._executor,
        lambda: cls._persist_memory_sync(
            user_id=user_id,
            member_id=member_id,
            fact_summary=fact_summary,
        ),
    )

  @classmethod
  def schedule_background_consolidation(
      cls,
      *,
      user_id: str,
      user_query: str,
      agent_response: str,
      invoked_tools: list[str],
  ) -> None:
    """Schedules non-blocking background memory persistence on the running event loop."""
    try:
      loop = asyncio.get_running_loop()
      task = loop.create_task(
          cls.consolidate_turn_memory_async(
              user_id=user_id,
              user_query=user_query,
              agent_response=agent_response,
              invoked_tools=invoked_tools,
          )
      )
      cls._pending_tasks.add(task)
      task.add_done_callback(cls._pending_tasks.discard)
    except RuntimeError:
      # Fallback if called outside an active asyncio loop (e.g., synchronous unit test)
      cls._executor.submit(
          cls._persist_memory_sync,
          user_id=user_id,
          member_id=None,
          fact_summary=f"Query: {user_query[:120]} | Tools: {invoked_tools}",
      )
