# AWS Phase 0 Audit Summary

Audit source: `aws_audit.zip` provided by Aman.

## Account / region

- Account ID: `050752651530`
- Active identity: root account ARN was reported by STS. Avoid using root for day-to-day deployment; use existing console/user workflow if available.
- Region from environment: `us-east-1`

## Existing resources found

| Area | Current state |
|---|---|
| EC2 instances | None |
| EC2 key pairs | None |
| Security groups | Default VPC security group only |
| S3 buckets | None |
| RDS instances | None |
| ECR repositories | None |
| App Runner services | None |
| Secrets Manager secrets | None |
| CloudWatch log groups | Existing AWS Glue log groups only: `/aws-glue/jobs/error`, `/aws-glue/jobs/logs-v2`, `/aws-glue/jobs/output` |

## What must be created for this project

1. EC2 instance for Docker Compose deployment.
2. RDS PostgreSQL instance for auth/RBAC/audit tables.
3. S3 bucket for raw data, policies, model/vector backups, logs/artifacts.
4. Security group rules for SSH, FastAPI, and Streamlit or reverse-proxy ports.
5. Secrets Manager entries for DB URL/password, JWT secret, and OpenAI API key.
6. CloudWatch log group(s) for API/UI logs.
7. Optional ECR repository if we push Docker images instead of building directly on EC2.

## Recommended first AWS deployment approach

Use EC2 + Docker Compose first. Keep RDS for auth. Keep S3 for artifacts/backups. Add ECR only if we choose image-based deployment later.
