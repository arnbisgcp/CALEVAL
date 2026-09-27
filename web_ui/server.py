"""Local Web Chat Server connected to the deployed Healthcare Claims Agent on Vertex AI."""

import ast
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import subprocess
import time
import urllib.error
import urllib.request

PROJECT_ID = "arnbtest"
LOCATION = "us-central1"
ENGINE_ID = "6960836041880109056"
PORT = int(os.environ.get("PORT", "8085"))

WEB_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_PY_PATH = os.path.join(os.path.dirname(WEB_DIR), "main.py")

QUERY_URL = (
    f"https://{LOCATION}-aiplatform.googleapis.com/v1beta1/"
    f"projects/{PROJECT_ID}/locations/{LOCATION}/reasoningEngines/{ENGINE_ID}:query"
)
STREAM_URL = (
    f"https://{LOCATION}-aiplatform.googleapis.com/v1beta1/"
    f"projects/{PROJECT_ID}/locations/{LOCATION}/reasoningEngines/{ENGINE_ID}:streamQuery?alt=sse"
)

_TOKEN_CACHE = {"token": None, "expires_at": 0.0}
_USER_SESSIONS = {}


def get_access_token(force_refresh: bool = False) -> str:
  now = time.time()
  if (
      not force_refresh
      and _TOKEN_CACHE["token"]
      and now < _TOKEN_CACHE["expires_at"]
  ):
    return _TOKEN_CACHE["token"]
  token = subprocess.check_output(
      [
          "/google/bin/releases/cloud-sdk-build/gcloud.par",
          "auth",
          "print-access-token",
          f"--project={PROJECT_ID}",
          "--quiet",
      ],
      text=True,
  ).strip()
  _TOKEN_CACHE["token"] = token
  _TOKEN_CACHE["expires_at"] = now + 2400  # 40 minutes
  return token


def load_mock_data_from_main() -> dict:
  with open(MAIN_PY_PATH, "r", encoding="utf-8") as f:
    source = f.read()
  tree = ast.parse(source)
  extracted = {}
  target_names = {
      "MOCK_MEMBERS": "members",
      "MOCK_CLAIMS": "claims",
      "MOCK_PRIOR_AUTHS": "prior_auths",
  }
  for node in tree.body:
    if isinstance(node, ast.Assign):
      for target in node.targets:
        if isinstance(target, ast.Name) and target.id in target_names:
          extracted[target_names[target.id]] = ast.literal_eval(node.value)
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
      if node.target.id in target_names and node.value is not None:
        extracted[target_names[node.target.id]] = ast.literal_eval(node.value)
  return extracted


MOCK_DATA = load_mock_data_from_main()


def ensure_session(user_id: str, token: str) -> str | None:
  if user_id in _USER_SESSIONS:
    return _USER_SESSIONS[user_id]
  payload = {
      "class_method": "create_session",
      "input": {"user_id": user_id},
  }
  req = urllib.request.Request(
      QUERY_URL,
      data=json.dumps(payload).encode("utf-8"),
      headers={
          "Authorization": f"Bearer {token}",
          "Content-Type": "application/json",
          "x-goog-user-project": PROJECT_ID,
      },
      method="POST",
  )
  try:
    with urllib.request.urlopen(req, timeout=30) as resp:
      body = json.loads(resp.read().decode("utf-8"))
      output = body.get("output", {})
      session_id = output.get("id") if isinstance(output, dict) else None
      if session_id:
        _USER_SESSIONS[user_id] = session_id
      return session_id
  except Exception:
    return None


def query_reasoning_engine(user_id: str, session_id: str | None, message: str) -> dict:
  token = get_access_token()
  if not session_id:
    session_id = ensure_session(user_id, token)

  input_obj = {"user_id": user_id, "message": message}
  if session_id:
    input_obj["session_id"] = session_id

  payload = {
      "class_method": "stream_query",
      "input": input_obj,
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

  with urllib.request.urlopen(req, timeout=90) as resp:
    raw_stream = resp.read().decode("utf-8")

  tool_calls_by_id = {}
  tool_calls_order = []
  text_parts = []

  for line in raw_stream.splitlines():
    line_str = line.strip()
    if not line_str:
      continue
    if line_str.startswith("data:"):
      line_str = line_str[5:].strip()
    try:
      event = json.loads(line_str)
    except json.JSONDecodeError:
      continue

    parts = event.get("content", {}).get("parts", [])
    for part in parts:
      if "function_call" in part:
        fc = part["function_call"]
        fc_id = fc.get("id") or f"call_{len(tool_calls_order)}"
        entry = {
            "id": fc_id,
            "name": fc.get("name", "unknown_tool"),
            "args": fc.get("args", {}),
            "response": None,
        }
        tool_calls_by_id[fc_id] = entry
        tool_calls_order.append(entry)
      elif "function_response" in part:
        fr = part["function_response"]
        fr_id = fr.get("id")
        if fr_id and fr_id in tool_calls_by_id:
          tool_calls_by_id[fr_id]["response"] = fr.get("response")
        else:
          for tc in reversed(tool_calls_order):
            if tc["name"] == fr.get("name") and tc["response"] is None:
              tc["response"] = fr.get("response")
              break
      elif "text" in part and part["text"]:
        text_parts.append(part["text"])

  return {
      "reply": "\n\n".join(text_parts).strip(),
      "tool_calls": tool_calls_order,
      "session_id": session_id,
  }


class ChatServerHandler(BaseHTTPRequestHandler):

  def _send_json(self, status_code: int, data: dict) -> None:
    body = json.dumps(data).encode("utf-8")
    self.send_response(status_code)
    self.send_header("Content-Type", "application/json; charset=utf-8")
    self.send_header("Content-Length", str(len(body)))
    self.end_headers()
    self.wfile.write(body)

  def _send_file(self, filename: str, content_type: str) -> None:
    filepath = os.path.join(WEB_DIR, filename)
    if not os.path.isfile(filepath):
      self.send_error(404, "File not found")
      return
    with open(filepath, "rb") as f:
      content = f.read()
    self.send_response(200)
    self.send_header("Content-Type", content_type)
    self.send_header("Content-Length", str(len(content)))
    self.end_headers()
    self.wfile.write(content)

  def do_GET(self) -> None:
    if self.path in ("/", "/index.html"):
      self._send_file("index.html", "text/html; charset=utf-8")
    elif self.path == "/styles.css":
      self._send_file("styles.css", "text/css; charset=utf-8")
    elif self.path == "/app.js":
      self._send_file("app.js", "application/javascript; charset=utf-8")
    elif self.path == "/api/mock-data":
      self._send_json(200, MOCK_DATA)
    elif self.path == "/api/health":
      self._send_json(
          200,
          {
              "status": "ok",
              "project": PROJECT_ID,
              "location": LOCATION,
              "engine_id": ENGINE_ID,
          },
      )
    else:
      self.send_error(404, "Not Found")

  def do_POST(self) -> None:
    if self.path == "/api/chat":
      content_len = int(self.headers.get("Content-Length", "0"))
      raw_body = self.rfile.read(content_len).decode("utf-8")
      try:
        req_data = json.loads(raw_body)
      except json.JSONDecodeError:
        self._send_json(400, {"error": "Invalid JSON body"})
        return

      user_id = req_data.get("user_id") or "web_user_default"
      session_id = req_data.get("session_id")
      message = (req_data.get("message") or "").strip()
      if not message:
        self._send_json(400, {"error": "Message is required"})
        return

      try:
        result = query_reasoning_engine(user_id, session_id, message)
        self._send_json(200, result)
      except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        self._send_json(e.code, {"error": f"Vertex AI HTTP {e.code}: {err_body}"})
      except Exception as e:
        self._send_json(500, {"error": str(e)})
    else:
      self.send_error(404, "Not Found")


def main() -> None:
  # Pre-warm token cache on startup
  try:
    get_access_token()
    print("OAuth token pre-warmed for project arnbtest.", flush=True)
  except Exception as e:
    print(f"Warning: token pre-warm failed: {e}", flush=True)

  server = ThreadingHTTPServer(("0.0.0.0", PORT), ChatServerHandler)
  print(
      f"Healthcare Claims Agent Web UI listening on http://arnabtest.c.googlers.com:{PORT}",
      flush=True,
  )
  server.serve_forever()


if __name__ == "__main__":
  main()
