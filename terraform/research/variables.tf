variable "project_id" { type = string }
variable "region" {
  type    = string
  default = "europe-west2"
}
variable "zone" {
  type    = string
  default = "europe-west2-a"
}
variable "create_workbench" {
  type        = bool
  default     = false
  description = "Opt in to a billable Vertex AI Workbench instance"
}
