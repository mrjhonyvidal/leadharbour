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

locals {
  services = toset([
    "aiplatform.googleapis.com",
    "artifactregistry.googleapis.com",
    "bigquery.googleapis.com",
    "iam.googleapis.com",
    "run.googleapis.com",
  ])
}

resource "google_project_service" "required" {
  for_each           = local.services
  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

resource "google_artifact_registry_repository" "images" {
  repository_id = "leadharbour"
  location      = var.region
  format        = "DOCKER"
  description   = "LeadHarbour private demo images"
  depends_on    = [google_project_service.required]
}

resource "google_bigquery_dataset" "learning" {
  dataset_id                 = "leadharbour_learning"
  location                   = var.region
  description                = "Optional labelled training and evaluation data"
  delete_contents_on_destroy = false
  depends_on                 = [google_project_service.required]
}

resource "google_service_account" "runtime" {
  account_id   = "leadharbour-api"
  display_name = "LeadHarbour private scoring API"
  depends_on   = [google_project_service.required]
}

output "repository" { value = google_artifact_registry_repository.images.id }
output "dataset" { value = google_bigquery_dataset.learning.dataset_id }
output "runtime_service_account" { value = google_service_account.runtime.email }
