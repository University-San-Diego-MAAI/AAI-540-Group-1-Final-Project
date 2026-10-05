"""Shared AWS session helpers for the project's boto3 scripts.

Personal-account assumptions:
- credentials come from the repo-root .env (AWS_ACCESS_KEY_ID / SECRET [/
  SESSION_TOKEN] / AWS_DEFAULT_REGION); permanent IAM keypairs are fine.
- the SageMaker default bucket ``sagemaker-<region>-<account>`` may not exist
  yet outside Studio/Learner Labs - ``ensure_default_bucket`` creates it
  (private, SSE-S3 is default-on for new buckets) before first use.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

import boto3

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

PREFIX = "aai-540-g1"


def check_credentials() -> None:
    if not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY"):
        raise SystemExit(
            "FAIL: .env credentials missing — run scripts/check_aws_env.py first."
        )


def region() -> str:
    return os.getenv("AWS_DEFAULT_REGION", "us-east-1")


def account_id() -> str:
    return boto3.client("sts", region_name=region()).get_caller_identity()["Account"]


def default_bucket() -> str:
    return f"sagemaker-{region()}-{account_id()}"


def ensure_default_bucket(s3=None) -> str:
    """Return the default bucket name, creating it (private) if missing."""
    s3 = s3 or boto3.client("s3", region_name=region())
    name = default_bucket()
    try:
        s3.head_bucket(Bucket=name)
    except Exception:
        if region() == "us-east-1":
            s3.create_bucket(Bucket=name)
        else:
            s3.create_bucket(
                Bucket=name,
                CreateBucketConfiguration={"LocationConstraint": region()},
            )
        s3.get_waiter("bucket_exists").wait(Bucket=name)
        print(f"created bucket {name}")
    return name
