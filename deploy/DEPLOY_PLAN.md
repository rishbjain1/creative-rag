# AWS ECS Fargate deploy — runbook

Terraform for an ECS Fargate deployment of the creative-rag FastAPI service is
written and `terraform validate`-clean (`main.tf` / `variables.tf` / `outputs.tf`),
with one-command `apply.sh` (ECR → docker build+push → apply → resolve task IP →
health poll) and `destroy.sh`. CI (`.github/workflows/ci.yml`) runs pytest, a
full `docker build`, and `terraform fmt+validate` on every push, so the image is
proven to assemble without a local Docker install.
Nothing incurs cost until `apply`.

## Prereqs (one-time)

- [ ] AWS free-tier account + IAM user with programmatic access keys
- [ ] Docker Desktop installed and running
- [ ] `aws configure` with the IAM keys

## Apply → verify → destroy

- [x] Local service smoke test: `uvicorn creative_rag.api:app` →
      `{"status":"ok","indexed":true,"chunks":470}` (verified 2026-07-07)
- [x] `apply.sh` / `destroy.sh` written (bash -n clean); terraform re-validated
- [ ] `./apply.sh` — runs the whole chain and polls `/health` until live
- [ ] Hit `/query` with a sample to confirm live retrieval + citation-verify
- [ ] Capture request logs / basic metrics for the observability bullet
- [ ] Screenshot the live `/health` + a query response (portfolio evidence)
- [ ] `./destroy.sh` — tear down so there is no ongoing cost

## Status

✅ Everything account-independent is done: terraform validated, scripts written,
service smoke-tested locally, CI proves the docker build. Apply/verify/destroy
gated on AWS account + Docker Desktop only.
