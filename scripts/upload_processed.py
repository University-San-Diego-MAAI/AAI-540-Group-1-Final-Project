"""Upload the processed training tables to S3 and verify bucket hygiene (Task 4).

Uploads ``data/processed/training_table.csv`` (+ the 1k smoke table) to
``s3://<default-bucket>/aai-540-g1/processed/training/`` and verifies:
  - default encryption is SSE-S3 on the bucket
  - public access is fully blocked
  - uploaded object sizes match the local files

Run from the repo root:  .venv/bin/python scripts/upload_processed.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import boto3

from aws_common import check_credentials, ensure_default_bucket, region

PROCESSED_DIR = ROOT / "data" / "processed"
DEST_PREFIX = "aai-540-g1/processed/training/"
FILES = ["training_table.csv", "training_table_1k.csv"]


def main() -> int:
    check_credentials()
    bucket = ensure_default_bucket()
    s3 = boto3.client("s3", region_name=region())

    errors = []
    # --- bucket hygiene ---
    try:
        enc = s3.get_bucket_encryption(Bucket=bucket)
        rules = enc["ServerSideEncryptionConfiguration"]["Rules"]
        algo = rules[0]["ApplyServerSideEncryptionByDefault"].get("SSEAlgorithm")
        print(f"encryption: {algo} on {bucket}")
        if algo != "AES256":
            errors.append(f"expected SSE-S3 (AES256), found {algo}")
    except Exception as exc:
        errors.append(f"get_bucket_encryption failed: {exc}")
    pab = s3.get_public_access_block(Bucket=bucket)["PublicAccessBlockConfiguration"]
    if not all(pab.values()):
        errors.append(f"public access block incomplete: {pab}")

    # --- uploads ---
    for name in FILES:
        local = PROCESSED_DIR / name
        if not local.is_file():
            errors.append(f"missing local file: {local}")
            continue
        key = DEST_PREFIX + name
        s3.upload_file(str(local), bucket, key)
        remote = s3.head_object(Bucket=bucket, Key=key)
        match = "OK" if remote["ContentLength"] == local.stat().st_size else "SIZE MISMATCH"
        print(f"uploaded {name}: {remote['ContentLength']:,} bytes ({match})")
        if match != "OK":
            errors.append(f"{name} size mismatch")

    print(f"\ns3://{bucket}/{DEST_PREFIX}")
    if errors:
        print("FAILURES:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("READY — processed training tables live in S3.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
