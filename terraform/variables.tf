variable "project_id" {
  description = "Google Cloud Project ID hosting the Healthcare Claims ADK Agent."
  type        = string
  default     = "arnbtest"
}

variable "region" {
  description = "Google Cloud region for Vertex AI Agent Engine and Secret Manager."
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Deployment environment tier (dev, staging, prod)."
  type        = string
  default     = "prod"
}

variable "reasoning_engine_id" {
  description = "Deployed Vertex AI Reasoning Engine resource ID."
  type        = string
  default     = "6960836041880109056"
}
