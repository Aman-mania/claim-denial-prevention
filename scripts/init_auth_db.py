#!/usr/bin/env python3
"""Initialize auth/RBAC tables and create demo users.

Local example:
    python scripts/init_auth_db.py --database-url sqlite:///data/auth/auth.db \
      --developer-email dev@example.com --developer-password dev123 \
      --analyst-email analyst@example.com --analyst-password analyst123

AWS/RDS example:
    AUTH_DATABASE_URL=postgresql://user:password@host:5432/claim_denial \
    python scripts/init_auth_db.py --developer-email ... --developer-password ...
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.auth.repository import AuthRepository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize auth database and seed demo users.")
    parser.add_argument("--database-url", default=os.getenv("AUTH_DATABASE_URL", "sqlite:///data/auth/auth.db"))
    parser.add_argument("--developer-email", default=os.getenv("DEV_USER_EMAIL"))
    parser.add_argument("--developer-password", default=os.getenv("DEV_USER_PASSWORD"))
    parser.add_argument("--analyst-email", default=os.getenv("ANALYST_USER_EMAIL"))
    parser.add_argument("--analyst-password", default=os.getenv("ANALYST_USER_PASSWORD"))
    parser.add_argument("--overwrite", action="store_true", help="Update password/role if user exists.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = AuthRepository(args.database_url)
    repo.initialize()
    created = []
    if args.developer_email and args.developer_password:
        user = repo.create_user(email=args.developer_email, password=args.developer_password, role="developer", overwrite=args.overwrite)
        created.append(user.to_dict())
    if args.analyst_email and args.analyst_password:
        user = repo.create_user(email=args.analyst_email, password=args.analyst_password, role="analyst", overwrite=args.overwrite)
        created.append(user.to_dict())
    print("Auth database initialized:", args.database_url)
    if not created:
        print("No users created. Provide --developer-email/password and/or --analyst-email/password.")
    else:
        for user in created:
            print(f"  {user['email']} -> {user['role']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
