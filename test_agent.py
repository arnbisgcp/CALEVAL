"""Live Streaming Verification Client for the Deployed Healthcare Claims ADK Agent."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from observability import log_structured_event
from secrets_manager import secret_manager_service

PROJECT_ID = secret_manager_service.get_secret(
    "gcp-project-id", default=os.environ.get("GOOGLE_CLOUD_PROJECT", "arnbtest")
)
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
ENGINE_ID = secret_manager_service.get_secret(
    "reasoning-engine-id",
    default=os.environ.get("REASONING_ENGINE_ID", "6960836041880109056"),
)
STREAM_URL = (
    f"https://{LOCATION}-aiplatform.googleapis.com/v1beta1/"
    f"projects/{PROJECT_ID}/locations/{LOCATION}/reasoningEngines/{ENGINE_ID}:streamQuery?alt=sse"
)


def main() -> None:
  token = secret_manager_service.get_gcp_access_token()
  payload = {
      "class_method": "stream_query",
      "input": {
          "user_id": "test_adjuster_1",
          "message": (
              "Check Sarah Jenkins's eligibility and list all her claims. "
              "Why was claim CLM-2026-9002 denied, and is there a prior "
              "authorization on file for her?"
          ),
      },
  }
  req = urllib.request.Request(
      STREAM_URL,
      data=json.dumps(payload).encode("utf-8"),
      headers={
          "Authorization": f"Bearer {token}",
          "Content-Type": "application/json",
          "x-goog-user-project": str(PROJECT_ID),
      },
      method="POST",
  )

  log_structured_event(
      "LIVE_AGENT_TEST_START",
      f"Sending streaming verification query to {STREAM_URL}",
      engine_id=ENGINE_ID,
  )
  try:
    with urllib.request.urlopen(req, timeout=60) as resp:
      raw = resp.read().decode("utf-8")
      print("Raw response stream:\n", raw, flush=True)
  except urllib.error.HTTPError as e:
    print(f"HTTP Error {e.code}: {e.read().decode('utf-8')}", flush=True)


if __name__ == "__main__":
  main()
