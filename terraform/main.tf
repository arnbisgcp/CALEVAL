# ============================================================================
# 1. Enable Required Google Cloud APIs
# ============================================================================

locals {
  required_apis = [
    "aiplatform.googleapis.com",
    "secretmanager.googleapis.com",
    "cloudtrace.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com",
    "storage.googleapis.com",
    "iam.googleapis.com",
  ]
}

resource "google_project_service" "enabled_apis" {
  for_each           = toset(local.required_apis)
  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

# ============================================================================
# 2. Dedicated Least-Privilege Service Account & IAM Bindings
# ============================================================================

resource "google_service_account" "claims_agent_sa" {
  account_id   = "healthcare-claims-adk-${var.environment}"
  display_name = "Healthcare Claims ADK Agent Runtime Service Account"
  description  = "Least-privilege identity for the Vertex AI Healthcare Claims Multi-Agent System"
  depends_on   = [google_project_service.enabled_apis]
}

resource "google_project_iam_member" "claims_agent_roles" {
  for_each = toset([
    "roles/aiplatform.user",
    "roles/secretmanager.secretAccessor",
    "roles/cloudtrace.agent",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
  ])
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.claims_agent_sa.email}"
}

# ============================================================================
# 3. Google Cloud Secret Manager Secrets
# ============================================================================

resource "google_secret_manager_secret" "gcp_project_id_secret" {
  secret_id = "gcp-project-id"
  replication {
    auto {}
  }
  depends_on = [google_project_service.enabled_apis]
}

resource "google_secret_manager_secret_version" "gcp_project_id_version" {
  secret      = google_secret_manager_secret.gcp_project_id_secret.id
  secret_data = var.project_id
}

resource "google_secret_manager_secret" "reasoning_engine_id_secret" {
  secret_id = "reasoning-engine-id"
  replication {
    auto {}
  }
  depends_on = [google_project_service.enabled_apis]
}

resource "google_secret_manager_secret_version" "reasoning_engine_id_version" {
  secret      = google_secret_manager_secret.reasoning_engine_id_secret.id
  secret_data = var.reasoning_engine_id
}

# ============================================================================
# 4. Artifact & Source Archive Staging Bucket
# ============================================================================

resource "google_storage_bucket" "agent_staging_bucket" {
  name                        = "${var.project_id}-healthcare-claims-adk-staging"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = false

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }

  depends_on = [google_project_service.enabled_apis]
}

resource "google_storage_bucket_iam_member" "staging_bucket_writer" {
  bucket = google_storage_bucket.agent_staging_bucket.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.claims_agent_sa.email}"
}
