"""Deploy Healthcare Claims Multi-Agent System to Vertex AI Agent Engine."""

from __future__ import annotations

import base64
import io
import json
import os
import pathlib
import sys
import tarfile
import time
import urllib.error
import urllib.request

from observability import log_structured_event
from secrets_manager import secret_manager_service

PROJECT_ID = secret_manager_service.get_secret(
    "gcp-project-id", default=os.environ.get("GOOGLE_CLOUD_PROJECT", "arnbtest")
)
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
BASE_URL = f"https://{LOCATION}-aiplatform.googleapis.com/v1beta1"
ROOT_DIR = pathlib.Path(__file__).resolve().parent

PACKAGED_FILES = [
    "main.py",
    "schemas.py",
    "database.py",
    "memory_manager.py",
    "observability.py",
    "guardrails.py",
    "secrets_manager.py",
    "requirements.txt",
]


def build_source_archive_b64() -> str:
  """Packages all agent modules and requirements into an in-memory `.tar.gz` archive."""
  buf = io.BytesIO()
  with tarfile.open(fileobj=buf, mode="w:gz") as tar:
    for filename in PACKAGED_FILES:
      file_path = ROOT_DIR / filename
      if file_path.exists():
        tar.add(str(file_path), arcname=filename)
  archive_bytes = buf.getvalue()
  (ROOT_DIR / "source.tar.gz").write_bytes(archive_bytes)
  return base64.b64encode(archive_bytes).decode("ascii")


def main() -> None:
  archive_b64 = build_source_archive_b64()
  payload = {
      "displayName": "Healthcare Claims Agent",
      "description": (
          "Multi-agent Google ADK Healthcare Claims System with Pydantic v2 "
          "schemas, persistent SQLite/vector store, context compaction, "
          "security guardrails, HITL appeals, and structured JSON telemetry."
      ),
      "spec": {
          "agentFramework": "google-adk",
          "sourceCodeSpec": {
              "inlineSource": {
                  "sourceArchive": archive_b64,
              },
              "pythonSpec": {
                  "version": "3.12",
                  "entrypointModule": "main",
                  "entrypointObject": "app",
                  "requirementsFile": "requirements.txt",
              },
          },
          "deploymentSpec": {
              "env": [
                  {
                      "name": "GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY",
                      "value": "true",
                  },
                  {
                      "name": (
                          "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
                      ),
                      "value": "true",
                  },
                  {
                      "name": "VERTEX_SESSION_ENDPOINT",
                      "value": "https://aiplatform.googleapis.com/",
                  },
                  {
                      "name": "GOOGLE_API_USE_MTLS_ENDPOINT",
                      "value": "never",
                  },
                  {
                      "name": (
                          "GOOGLE_API_PREVENT_AGENT_TOKEN_SHARING_FOR_GCP_SERVICES"
                      ),
                      "value": "false",
                  },
              ]
          },
      },
  }

  token = secret_manager_service.get_gcp_access_token()
  existing_engine_id = os.environ.get("REASONING_ENGINE_ID", "")
  if existing_engine_id:
    target_url = (
        f"{BASE_URL}/projects/{PROJECT_ID}/locations/{LOCATION}/"
        f"reasoningEngines/{existing_engine_id}?updateMask=spec,displayName,description"
    )
    http_method = "PATCH"
  else:
    target_url = (
        f"{BASE_URL}/projects/{PROJECT_ID}/locations/{LOCATION}/reasoningEngines"
    )
    http_method = "POST"

  req = urllib.request.Request(
      target_url,
      data=json.dumps(payload).encode("utf-8"),
      headers={
          "Authorization": f"Bearer {token}",
          "Content-Type": "application/json",
          "x-goog-user-project": str(PROJECT_ID),
      },
      method=http_method,
  )

  log_structured_event(
      "DEPLOYMENT_STARTED",
      f"Submitting Reasoning Engine ({http_method}) in {PROJECT_ID}/{LOCATION}",
      project_id=PROJECT_ID,
      location=LOCATION,
      http_method=http_method,
  )
  try:
    with urllib.request.urlopen(req, timeout=60) as resp:
      op_data = json.loads(resp.read().decode("utf-8"))
  except urllib.error.HTTPError as e:
    err_body = e.read().decode("utf-8", errors="replace")
    print(f"HTTP Error {e.code}: {err_body}", file=sys.stderr, flush=True)
    sys.exit(1)

  op_name = op_data.get("name", "")
  print(f"Operation started: {op_name}", flush=True)

  start_time = time.time()
  while not op_data.get("done", False):
    time.sleep(15)
    elapsed = int(time.time() - start_time)
    token = secret_manager_service.get_gcp_access_token()
    op_url = f"{BASE_URL}/{op_name}"
    poll_req = urllib.request.Request(
        op_url,
        headers={
            "Authorization": f"Bearer {token}",
            "x-goog-user-project": str(PROJECT_ID),
        },
    )
    with urllib.request.urlopen(poll_req, timeout=30) as resp:
      op_data = json.loads(resp.read().decode("utf-8"))
    print(
        f"[{elapsed}s] Polling operation... done={op_data.get('done', False)}",
        flush=True,
    )

  if "error" in op_data:
    print(
        f"Deployment failed: {json.dumps(op_data['error'], indent=2)}",
        file=sys.stderr,
        flush=True,
    )
    sys.exit(1)

  log_structured_event(
      "DEPLOYMENT_SUCCEEDED",
      "Vertex AI Reasoning Engine deployment completed.",
      response=op_data.get("response", {}),
  )
  print(json.dumps(op_data.get("response", op_data), indent=2), flush=True)


if __name__ == "__main__":
  main()
