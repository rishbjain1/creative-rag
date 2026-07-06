#!/usr/bin/env bash
# One-command build → push → apply → verify for the ECS Fargate deployment.
# Prereqs: aws configure done, Docker running, terraform installed.
# Costs money while applied — pair with destroy.sh.
set -euo pipefail
cd "$(dirname "$0")"

REGION="${AWS_REGION:-us-east-1}"
TAG="${1:-latest}"

echo "==> terraform init + create the ECR repo first (it's terraform-managed)"
terraform init -input=false
terraform apply -auto-approve -target=aws_ecr_repository.this -var "image_tag=${TAG}"
ECR_URL=$(terraform output -raw ecr_repository_url)

echo "==> docker build + push ${ECR_URL}:${TAG}"
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "${ECR_URL%%/*}"
docker build -t "creative-rag:${TAG}" ..
docker tag "creative-rag:${TAG}" "${ECR_URL}:${TAG}"
docker push "${ECR_URL}:${TAG}"

echo "==> terraform apply (full stack)"
terraform apply -auto-approve -var "image_tag=${TAG}"

echo "==> resolve the task's public IP (Fargate ENI)"
CLUSTER=$(terraform output -raw cluster_name)
IP=""
for i in $(seq 1 30); do
  TASK=$(aws ecs list-tasks --cluster "$CLUSTER" --query 'taskArns[0]' --output text)
  if [ "$TASK" != "None" ] && [ -n "$TASK" ]; then
    ENI=$(aws ecs describe-tasks --cluster "$CLUSTER" --tasks "$TASK" \
      --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' --output text)
    if [ -n "$ENI" ] && [ "$ENI" != "None" ]; then
      IP=$(aws ec2 describe-network-interfaces --network-interface-ids "$ENI" \
        --query 'NetworkInterfaces[0].Association.PublicIp' --output text)
      [ -n "$IP" ] && [ "$IP" != "None" ] && break
    fi
  fi
  sleep 10
done
[ -z "$IP" ] && { echo "no public IP yet — task still starting; check ECS console" >&2; exit 1; }

echo "==> verify http://${IP}:8000/health"
for i in $(seq 1 30); do
  if curl -sf "http://${IP}:8000/health"; then echo; echo "LIVE: http://${IP}:8000"; exit 0; fi
  sleep 10
done
echo "service did not become healthy — logs: $(terraform output -raw log_group)" >&2
exit 1
