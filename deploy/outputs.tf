output "ecr_repository_url" {
  description = "Push the image here, then terraform apply with -var image_tag"
  value       = aws_ecr_repository.this.repository_url
}

output "cluster_name" {
  value = aws_ecs_cluster.this.name
}

output "service_name" {
  value = aws_ecs_service.this.name
}

output "log_group" {
  value = aws_cloudwatch_log_group.this.name
}

# The Fargate task's public IP is assigned per task. After apply, fetch it:
#   aws ecs list-tasks --cluster creative-rag --query 'taskArns[0]' --output text
#   aws ecs describe-tasks --cluster creative-rag --tasks <arn> \
#     --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' --output text
#   aws ec2 describe-network-interfaces --network-interface-ids <eni> \
#     --query 'NetworkInterfaces[0].Association.PublicIp' --output text
output "get_public_ip_hint" {
  value = "See comment in outputs.tf — fetch the task ENI's public IP after apply."
}
