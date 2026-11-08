variable "aws_region" {
  description = "AWS region for SageMaker and S3"
  type        = string
  default     = "eu-west-1"
}

variable "project_name" {
  description = "Project identifier"
  type        = string
  default     = "qwen-text-to-sql"
}

variable "wandb_secret_name" {
  description = "Existing Secrets Manager secret containing the W&B API key"
  type        = string
  default     = "wandb-api-key"
}