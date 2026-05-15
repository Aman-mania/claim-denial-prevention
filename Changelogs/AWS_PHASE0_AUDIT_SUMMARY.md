# AWS Phase 0 Audit Summary

Uploaded audit reviewed from `aws_audit.zip`.

## Account/resource state found

- Region: `us-east-1`
- EC2 instances: none
- EC2 key pairs: none
- S3 buckets: none
- ECR repositories: none
- RDS instances: none
- App Runner services: none
- Secrets Manager secrets: none
- CloudWatch log groups: existing AWS Glue log groups only
- Security groups: only default VPC security group found

## Meaning for this project

The AWS account is effectively clean for this project. Before deployment we will need to create:

1. EC2 instance for Docker Compose deployment.
2. New key pair or SSM Session Manager access path.
3. Security group for SSH/HTTP/API/UI ports.
4. S3 bucket for raw data, policy docs, artifacts, and backups.
5. RDS PostgreSQL instance for users/roles/audit events.
6. Secrets Manager secrets for DB URL/password, JWT secret, and OpenAI API key.
7. Optional ECR repositories if we decide to push Docker images instead of building directly on EC2.
8. CloudWatch log groups/retention policy for app logs.

## Recommended next AWS move

Do not create AWS resources until FastAPI + role-aware Streamlit works locally. Then create AWS resources in one controlled deployment phase.
