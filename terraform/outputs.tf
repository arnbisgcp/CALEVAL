output "agent_service_account_email" {
  description = "Email of the least-privilege service account for the Healthcare Claims ADK Agent."
  value       = google_service_account.claims_agent_sa.email
}

output "staging_bucket_url" {
  description = "GCS bucket URL used for Vertex AI Agent Engine source archives and artifacts."
  value       = google_storage_bucket.agent_staging_bucket.url
}

output "reasoning_engine_secret_id" {
  description = "Secret Manager resource ID storing the active Vertex AI Reasoning Engine ID."
  value       = google_secret_manager_secret.reasoning_engine_id_secret.id
}
