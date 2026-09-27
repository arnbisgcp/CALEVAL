"""Test the deployed Healthcare Claims ADK Agent on Vertex AI Agent Engine."""

import json
import subprocess
import urllib.error
import urllib.request

PROJECT_ID = "arnbtest"
LOCATION = "us-central1"
ENGINE_ID = "6960836041880109056"
STREAM_URL = (
    f"https://{LOCATION}-aiplatform.googleapis.com/v1beta1/"
    f"projects/{PROJECT_ID}/locations/{LOCATION}/reasoningEngines/{ENGINE_ID}:streamQuery?alt=sse"
)


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
  token = get_token()
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
          "x-goog-user-project": PROJECT_ID,
      },
      method="POST",
  )

  print(f"Sending query to {STREAM_URL}...\n", flush=True)
  try:
    with urllib.request.urlopen(req, timeout=60) as resp:
      raw = resp.read().decode("utf-8")
      print("Raw response stream:\n", raw, flush=True)
  except urllib.error.HTTPError as e:
    print(f"HTTP Error {e.code}: {e.read().decode('utf-8')}", flush=True)


if __name__ == "__main__":
  main()
