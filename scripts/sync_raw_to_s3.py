"""Mirror the public DE-SynPUF OMOP tables into the project S3 datalake (Task 3).

Copies ``s3://synpuf-omop/cmsdesynpuf100k/`` (17 gzipped CSV tables, ~881 MB)
into ``s3://<default-bucket>/aai-540-g1/raw/synpuf100k/`` using server-side
CopyObject - nothing transits this machine, so it is fast and safe on home
bandwidth. The source bucket is public: reads succeed with the project's
signed credentials (no --no-sign-request needed when creds exist).

Requires repo-root .env (see scripts/check_aws_env.py). Idempotent: existing
objects with matching size are skipped.

Run from the repo root:  .venv/bin/python scripts/sync_raw_to_s3.py
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import boto3

from aws_common import check_credentials, ensure_default_bucket, region

SOURCE_BUCKET = "synpuf-omop"
SOURCE_PREFIX = "cmsdesynpuf100k/"
DEST_PREFIX = "aai-540-g1/raw/synpuf100k/"
EXPECTED_TABLES = 17


def main() -> int:
    check_credentials()
    s3 = boto3.client("s3", region_name=region())
    dest_bucket = ensure_default_bucket(s3)
    print(f"mirroring {SOURCE_BUCKET}/{SOURCE_PREFIX} -> {dest_bucket}/{DEST_PREFIX}")

    paginator = s3.get_paginator("list_objects_v2")
    copied = skipped = 0
    total_bytes = 0
    for page in paginator.paginate(Bucket=SOURCE_BUCKET, Prefix=SOURCE_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            dest_key = DEST_PREFIX + key[len(SOURCE_PREFIX):]
            try:
                head = s3.head_object(Bucket=dest_bucket, Key=dest_key)
                if head["ContentLength"] == obj["Size"]:
                    skipped += 1
                    total_bytes += obj["Size"]
                    continue
            except Exception:
                pass  # not present yet -> copy below
            s3.copy_object(
                CopySource={"Bucket": SOURCE_BUCKET, "Key": key},
                Bucket=dest_bucket,
                Key=dest_key,
            )
            copied += 1
            total_bytes += obj["Size"]
            print(f"  copied {key} ({obj['Size'] / 1e6:.1f} MB)")

    print(f"\ndone: {copied} copied, {skipped} already present, "
          f"{total_bytes / 1e6:.0f} MB total")
    if copied + skipped != EXPECTED_TABLES:
        print(f"WARN: expected {EXPECTED_TABLES} tables, found {copied + skipped}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
