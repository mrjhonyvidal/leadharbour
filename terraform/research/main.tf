terraform {
  required_version = ">= 1.7"
  required_providers {
    google = { source = "hashicorp/google", version = "~> 8.0" }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_project_service" "research" {
  for_each = toset([
    "aiplatform.googleapis.com",
    "compute.googleapis.com",
    "notebooks.googleapis.com",
    "storage.googleapis.com",
  ])
  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

resource "google_storage_bucket" "pipeline_artifacts" {
  name                        = "${var.project_id}-leadharbour-pipelines"
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = false
  depends_on                  = [google_project_service.research]
}

resource "google_service_account" "research" {
  account_id   = "leadharbour-research"
  display_name = "LeadHarbour research and pipeline runs"
  depends_on   = [google_project_service.research]
}

resource "google_storage_bucket_iam_member" "research_artifacts" {
  bucket = google_storage_bucket.pipeline_artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.research.email}"
}

resource "google_project_iam_member" "research_vertex" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.research.email}"
}

resource "google_workbench_instance" "optional" {
  count    = var.create_workbench ? 1 : 0
  name     = "leadharbour-lab"
  location = var.zone
  gce_setup {
    machine_type = "e2-standard-2"
    service_accounts {
      email = google_service_account.research.email
    }
  }
  depends_on = [google_project_service.research]
}

output "pipeline_bucket" { value = google_storage_bucket.pipeline_artifacts.url }
output "research_service_account" { value = google_service_account.research.email }
