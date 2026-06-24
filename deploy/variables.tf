variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "name" {
  description = "Resource name prefix"
  type        = string
  default     = "creative-rag"
}

variable "image_tag" {
  description = "ECR image tag to deploy"
  type        = string
  default     = "latest"
}

variable "container_port" {
  description = "Port the API listens on"
  type        = number
  default     = 8000
}

variable "cpu" {
  description = "Fargate task CPU units (1024 = 1 vCPU). torch needs headroom."
  type        = number
  default     = 1024
}

variable "memory" {
  description = "Fargate task memory (MiB). torch + models need >= 4GB."
  type        = number
  default     = 4096
}

variable "api_key" {
  description = "Value required in the X-API-Key header on /query (empty = open)"
  type        = string
  default     = ""
  sensitive   = true
}
