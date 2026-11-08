output "s3_bucket_name" {
  value = aws_s3_bucket.training.bucket
}

output "sagemaker_role_arn" {
  value = aws_iam_role.sagemaker.arn
}

output "wandb_secret_arn" {
  value = data.aws_secretsmanager_secret.wandb.arn
}

output "aws_region" {
  value = var.aws_region
}