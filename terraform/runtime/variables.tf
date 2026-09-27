variable "project_id" { type = string }
variable "region" {
  type    = string
  default = "europe-west2"
}
variable "image" {
  type        = string
  description = "Immutable image reference from the reviewed local build"
}
