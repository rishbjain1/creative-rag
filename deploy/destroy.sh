#!/usr/bin/env bash
# Tear the ECS deployment down so nothing keeps billing.
set -euo pipefail
cd "$(dirname "$0")"
terraform destroy -auto-approve
echo "==> destroyed. ECR images remain (pennies); delete the repo to zero out:"
echo "    aws ecr delete-repository --repository-name creative-rag --force"
