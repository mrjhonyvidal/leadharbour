variable "project_id" {
  type        = string
  description = "Existing Google Cloud project with billing and Service Usage enabled"
}

variable "region" {
  type        = string
  description = "Region for Artifact Registry and BigQuery"
  default     = "europe-west2"
}
