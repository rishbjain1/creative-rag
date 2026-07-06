# AWS ECS Fargate deploy — runbook

Terraform for an ECS Fargate deployment of the creative-rag FastAPI service is
written and `terraform validate`-clean (`main.tf` / `variables.tf` / `outputs.tf`).
This is the apply → verify → destroy runbook. Nothing incurs cost until `apply`.

## Prereqs (one-time)

- [ ] AWS free-tier account + IAM user with programmatic access keys
- [ ] Docker Desktop installed and running
- [ ] `aws configure` with the IAM keys

## Apply → verify → destroy

- [ ] `docker build -t creative-rag .`
- [ ] Create ECR repo, tag + push the image
- [ ] `terraform apply -var image_tag=latest`
- [ ] Read the task's public IP from `terraform output`
- [ ] `curl http://<ip>:8000/health` → expect `{"status":"ok"}`
- [ ] Hit `/query` with a sample to confirm live retrieval + citation-verify
- [ ] Capture request logs / basic metrics for the observability bullet
- [ ] Screenshot the live `/health` + a query response (portfolio evidence)
- [ ] `terraform destroy` — tear down so there is no ongoing cost

## Status

🚧 In progress — Terraform validated; apply gated on AWS account + Docker Desktop.
