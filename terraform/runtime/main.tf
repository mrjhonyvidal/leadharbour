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

resource "google_cloud_run_v2_service" "scoring" {
  name                 = "leadharbour-api"
  location             = var.region
  ingress              = "INGRESS_TRAFFIC_ALL"
  invoker_iam_disabled = false
  deletion_protection  = false

  template {
    service_account                  = "leadharbour-api@${var.project_id}.iam.gserviceaccount.com"
    max_instance_request_concurrency = 20
    scaling { max_instance_count = 3 }
    containers {
      image = var.image
      ports { container_port = 8080 }
      env {
        name  = "LEADHARBOUR_AUTH_MODE"
        value = "cloud_iam"
      }
      resources { limits = { cpu = "1", memory = "1Gi" } }
      startup_probe {
        http_get {
          path = "/healthz"
          port = 8080
        }
        initial_delay_seconds = 5
        period_seconds        = 10
      }
    }
  }
}

# No allUsers binding is created. Grant roles/run.invoker to an approved identity
# after reviewing who may submit feature data.
output "private_url" { value = google_cloud_run_v2_service.scoring.uri }
