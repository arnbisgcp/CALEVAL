"""Deploy Healthcare Claims ADK Agent to Vertex AI Agent Engine in arnbtest."""

import base64
import json
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request

PROJECT_ID = "arnbtest"
LOCATION = "us-central1"
BASE_URL = f"https://{LOCATION}-aiplatform.googleapis.com/v1beta1"
ARCHIVE_PATH = pathlib.Path(__file__).resolve().parent / "source.tar.gz"


def get_token() -> str:
  return subprocess.check_output(
      [
          "/google/bin/releases/cloud-sdk-build/gcloud.par",
          "auth",
          "print-access-token",
          f"--project={PROJECT_ID}",
          "--quiet",
      ],
      text=True,
  ).strip()


def main() -> None:
  archive_b64 = base64.b64encode(ARCHIVE_PATH.read_bytes()).decode("ascii")
  payload = {
      "displayName": "Healthcare Claims Agent",
      "description": (
          "ADK Healthcare Claims Agent with mock member eligibility, claims"
          " adjudication, prior authorization, cost estimation, and appeal"
          " workflows."
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

  token = get_token()
  create_url = (
      f"{BASE_URL}/projects/{PROJECT_ID}/locations/{LOCATION}/reasoningEngines"
  )
  req = urllib.request.Request(
      create_url,
      data=json.dumps(payload).encode("utf-8"),
      headers={
          "Authorization": f"Bearer {token}",
          "Content-Type": "application/json",
          "x-goog-user-project": PROJECT_ID,
      },
      method="POST",
  )

  print(f"Creating Reasoning Engine in {PROJECT_ID}/{LOCATION}...", flush=True)
  try:
    with urllib.request.urlopen(req, timeout=60) as resp:
      op_data = json.loads(resp.read().decode("utf-8"))
  except urllib.error.HTTPError as e:
    err_body = e.read().decode("utf-8", errors="replace")
    print(f"HTTP Error {e.code}: {err_body}", file=sys.stderr, flush=True)
    sys.exit(1)

  op_name = op_data.get("name", "")
  print(f"Operation started: {op_name}", flush=True)
  print(json.dumps(op_data, indent=2), flush=True)

  # Poll the LRO until completion
  start_time = time.time()
  while not op_data.get("done", False):
    time.sleep(15)
    elapsed = int(time.time() - start_time)
    token = get_token()
    op_url = f"{BASE_URL}/{op_name}"
    poll_req = urllib.request.Request(
        op_url,
        headers={
            "Authorization": f"Bearer {token}",
            "x-goog-user-project": PROJECT_ID,
        },
    )
    try:
      with urllib.request.urlopen(poll_req, timeout=30) as resp:
        op_data = json.loads(resp.read().decode("utf-8"))
      print(
          f"[{elapsed}s] Polling operation... done={op_data.get('done', False)}",
          flush=True,
      )
    except urllib.error.HTTPError as e:
      err_body = e.read().decode("utf-8", errors="replace")
      print(
          f"[{elapsed}s] Poll HTTP Error {e.code}: {err_body}",
          file=sys.stderr,
          flush=True,
      )

  if "error" in op_data:
    print(
        f"Deployment failed: {json.dumps(op_data['error'], indent=2)}",
        file=sys.stderr,
        flush=True,
    )
    sys.exit(1)

  print("Deployment succeeded!", flush=True)
  print(json.dumps(op_data.get("response", op_data), indent=2), flush=True)


if __name__ == "__main__":
  main()
