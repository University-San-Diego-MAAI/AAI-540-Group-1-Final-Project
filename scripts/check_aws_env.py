"""Verify AWS credentials from the repo-root .env before any AWS work.

Loads .env (never committed — see .gitignore), then:
  1. sts.get_caller_identity  -> prints account + ARN
  2. locates the SageMaker default bucket sagemaker-us-east-1-<account>
     (head_bucket, falls back to list_buckets scan)
  3. prints region and a ready/not-ready summary

Exit code 0 = creds good and default bucket found; non-zero = fix .env first
(Learner Labs session tokens expire every few hours — re-export and rerun).

Run from the repo root:  .venv/bin/python scripts/check_aws_env.py
"""

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    sys.exit("python-dotenv not installed: .venv/bin/pip install python-dotenv boto3")

import boto3

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Learner Labs issues temporary credentials: all four values required there.
# Permanent IAM keypairs (personal accounts) omit the session token - boto3
# simply picks AWS_SESSION_TOKEN up from the environment when present.
REQUIRED_KEYS = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"]


def main() -> int:
    missing = [k for k in REQUIRED_KEYS if not os.getenv(k)]
    if not Path(ROOT / ".env").is_file():
        print("FAIL: .env not found at repo root. Create it with: "
              + ", ".join(REQUIRED_KEYS + ["AWS_SESSION_TOKEN (Learner Labs)"]))
        return 1
    if missing:
        print(f"FAIL: .env is missing: {', '.join(missing)}")
        return 2
    if not os.getenv("AWS_SESSION_TOKEN"):
        print("note: no AWS_SESSION_TOKEN - assuming permanent IAM keypair")

    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    sts = boto3.client("sts", region_name=region)
    try:
        ident = sts.get_caller_identity()
    except Exception as exc:  # ExpiredToken, invalid creds, no network...
        print(f"FAIL: get_caller_identity error: {exc}")
        print("Hint: Learner Labs tokens expire — refresh AWS_SESSION_TOKEN in .env.")
        return 3
    account = ident["Account"]
    print(f"identity : {ident['Arn']}")
    print(f"account  : {account}")
    print(f"region   : {region}")

    s3 = boto3.client("s3", region_name=region)
    default_bucket = f"sagemaker-{region}-{account}"
    try:
        s3.head_bucket(Bucket=default_bucket)
        print(f"bucket   : {default_bucket} (exists)")
    except Exception:
        try:
            names = [b["Name"] for b in s3.list_buckets()["Buckets"]]
            print(f"bucket   : default {default_bucket} NOT found; available: {names}")
            return 4
        except Exception as exc:
            print(f"FAIL: cannot list buckets: {exc}")
            return 5

    print("READY — .env credentials verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
