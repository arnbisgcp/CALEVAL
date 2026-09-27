"""Enterprise Secret Management via Google Cloud Secret Manager and Application Default Credentials (ADC)."""

from __future__ import annotations

import os
import subprocess
import time
from typing import Optional

from observability import log_structured_event

try:
  import google.auth
  from google.auth.transport.requests import Request as GoogleAuthRequest
except ImportError:  # pragma: no cover
  google = None  # type: ignore[assignment]
  GoogleAuthRequest = None  # type: ignore[assignment]

try:
  from google.cloud import secretmanager
except ImportError:  # pragma: no cover
  secretmanager = None  # type: ignore[assignment]


class SecretManagerService:
  """Retrieves configuration secrets from Google Cloud Secret Manager with TTL caching."""

  _cache: dict[str, tuple[str, float]] = {}
  _DEFAULT_TTL_SECONDS = 1800.0

  def __init__(self, project_id: Optional[str] = None) -> None:
    self.project_id = (
        project_id
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("GCP_PROJECT_ID")
        or "arnbtest"
    )
    self._client = None
    if secretmanager is not None:
      try:
        self._client = secretmanager.SecretManagerServiceClient()
      except Exception:
        self._client = None

  def get_secret(
      self,
      secret_id: str,
      *,
      version: str = "latest",
      default: Optional[str] = None,
  ) -> Optional[str]:
    """Fetches a secret payload from GCP Secret Manager (`projects/*/secrets/*/versions/*`)."""
    cache_key = f"{self.project_id}:{secret_id}:{version}"
    now = time.time()
    if cache_key in self._cache:
      cached_val, expires_at = self._cache[cache_key]
      if now < expires_at:
        return cached_val

    # 1. Attempt Google Cloud Secret Manager API
    if self._client is not None:
      resource_name = (
          f"projects/{self.project_id}/secrets/{secret_id}/versions/{version}"
      )
      try:
        response = self._client.access_secret_version(
            request={"name": resource_name}, timeout=5.0
        )
        payload = response.payload.data.decode("utf-8").strip()
        self._cache[cache_key] = (payload, now + self._DEFAULT_TTL_SECONDS)
        log_structured_event(
            "SECRET_MANAGER_ACCESS_SUCCESS",
            f"Fetched secret '{secret_id}' from Google Cloud Secret Manager.",
            secret_id=secret_id,
            project_id=self.project_id,
        )
        return payload
      except Exception as exc:
        log_structured_event(
            "SECRET_MANAGER_FALLBACK",
            f"Secret Manager lookup for '{secret_id}' fell back to environment config: {type(exc).__name__}",
            severity="DEBUG",
            secret_id=secret_id,
        )

    # 2. Fallback to environment variable or provided default
    env_val = os.environ.get(secret_id.upper().replace("-", "_"), default)
    if env_val is not None:
      self._cache[cache_key] = (env_val, now + self._DEFAULT_TTL_SECONDS)
    return env_val

  def get_gcp_access_token(self) -> str:
    """Acquires an OAuth2 access token via Application Default Credentials (ADC) or Workload Identity."""
    cache_key = f"{self.project_id}:__adc_access_token__"
    now = time.time()
    if cache_key in self._cache:
      token, expires_at = self._cache[cache_key]
      if now < expires_at:
        return token

    # 1. Preferred: Google Auth Application Default Credentials (ADC / Workload Identity)
    if google is not None and GoogleAuthRequest is not None:
      try:
        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        credentials.refresh(GoogleAuthRequest())
        if credentials.token:
          self._cache[cache_key] = (credentials.token, now + 2400.0)
          return credentials.token
      except Exception:
        pass

    # 2. Check if an explicit token was injected via Secret Manager or CI environment
    env_token = self.get_secret("gcp-vertex-access-token") or os.environ.get(
        "GCP_ACCESS_TOKEN"
    )
    if env_token:
      self._cache[cache_key] = (env_token, now + 1800.0)
      return env_token

    # 3. Developer workstation CLI fallback
    gcloud_bin = (
        "/google/bin/releases/cloud-sdk-build/gcloud.par"
        if os.path.exists("/google/bin/releases/cloud-sdk-build/gcloud.par")
        else "gcloud"
    )
    token = subprocess.check_output(
        [
            gcloud_bin,
            "auth",
            "print-access-token",
            f"--project={self.project_id}",
            "--quiet",
        ],
        text=True,
    ).strip()
    self._cache[cache_key] = (token, now + 2400.0)
    return token


secret_manager_service = SecretManagerService()
