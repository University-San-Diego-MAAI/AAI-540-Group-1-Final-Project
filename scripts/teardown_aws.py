"""Tear down every AWS resource this project created (post-grading cleanup).

Deletes, in order (all idempotent - missing resources are skipped):
  - monitoring schedules (they recur hourly if left alone)
  - the monitoring endpoint + its endpoint config (the only hourly charge)
  - SageMaker models created for scoring/monitoring
  - the Feature Store group (online + offline stores)
  - the project's S3 prefixes, INCLUDING the per-job sourcedir/output
    prefixes the SDK uploaded at the bucket root (logreg-*/rf-*/xgb-*/...)

Only project-specific prefixes are touched - other work in the shared
default bucket (this is a personal account) is left alone, and the bucket
itself is never deleted.

Run from the repo root:  .venv/bin/python scripts/teardown_aws.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import boto3

from aws_common import check_credentials, region

ENDPOINT = "member-utilization-risk-monitor"
MODEL = "member-utilization-risk-champion"
FEATURE_GROUP = "aai-540-g1-member-features"
SCHEDULES = ["aai540-g1-data-quality", "aai540-g1-model-quality", "aai540-g1-model-bias"]
# SDK training/transform runs drop sourcedir.tar.gz at the bucket ROOT under
# the job name; delete only this project's job prefixes.
JOB_PREFIXES = [
    "logreg-c1", "rf-d16-l5", "xgb-e01-d6-w5", "xgb-e005-d6-w5", "xgb-e01-d8-w1",
    "xgb-probe", "score-production", "aai-540-g1",
]


def _ok(label, fn):
    try:
        fn()
        print(f"deleted {label}")
    except Exception as exc:
        print(f"skipped {label}: {str(exc)[:100]}")


def main() -> None:
    check_credentials()
    sm = boto3.client("sagemaker", region_name=region())
    s3 = boto3.client("s3", region_name=region())

    for name in SCHEDULES:
        _ok(f"schedule {name}", lambda n=name: sm.delete_monitoring_schedule(MonitoringScheduleName=n))
    _ok(f"endpoint {ENDPOINT}", lambda: sm.delete_endpoint(EndpointName=ENDPOINT))
    _ok(f"endpoint config {ENDPOINT}", lambda: sm.delete_endpoint_config(EndpointConfigName=ENDPOINT))
    _ok(f"model {MODEL}", lambda: sm.delete_model(ModelName=MODEL))
    _ok(f"feature group {FEATURE_GROUP}",
        lambda: sm.delete_feature_group(FeatureGroupName=FEATURE_GROUP))

    account = boto3.client("sts", region_name=region()).get_caller_identity()["Account"]
    bucket = f"sagemaker-{region()}-{account}"
    paginator = s3.get_paginator("list_objects_v2")
    total = 0
    for prefix in JOB_PREFIXES:
        keys = [o["Key"] for page in paginator.paginate(Bucket=bucket, Prefix=prefix)
                for o in page.get("Contents", [])]
        for i in range(0, len(keys), 1000):
            s3.delete_objects(Bucket=bucket,
                              Delete={"Objects": [{"Key": k} for k in keys[i:i + 1000]]})
        if keys:
            print(f"deleted {len(keys):,} objects under s3://{bucket}/{prefix}")
            total += len(keys)
    print(f"\nteardown complete ({total:,} S3 objects removed). "
          f"Verify: endpoints=0, schedules=0, feature group gone, prefixes empty.")


if __name__ == "__main__":
    main()
