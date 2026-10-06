"""Create (idempotently) the project's dedicated SageMaker execution role.

The account's pre-existing Studio execution roles were rejected by
CreateFeatureGroup with 'trust relationship' validation errors, so this
project uses its own role with an explicit trust policy:

  aai-540-g1-execution-role
    trust   : sagemaker.amazonaws.com (sts:AssumeRole)
    policies: AmazonSageMakerFullAccess + AmazonS3FullAccess
              (S3 full access keeps the feature-store/offline-store/artifact
              paths simple in a personal account; tighten for production)

Run from the repo root:  .venv/bin/python scripts/ensure_execution_role.py
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import boto3

from aws_common import account_id, check_credentials, region

ROLE_NAME = "aai-540-g1-execution-role"
TRUST_POLICY = {
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "sagemaker.amazonaws.com"},
        "Action": "sts:AssumeRole",
    }],
}
MANAGED_POLICIES = [
    "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
    "arn:aws:iam::aws:policy/AmazonS3FullAccess",
]


def main() -> str:
    check_credentials()
    iam = boto3.client("iam", region_name=region())
    try:
        iam.create_role(RoleName=ROLE_NAME, AssumeRolePolicyDocument=json.dumps(TRUST_POLICY))
        print(f"created role {ROLE_NAME}")
    except iam.exceptions.EntityAlreadyExistsException:
        print(f"role {ROLE_NAME} already exists")
    for pol in MANAGED_POLICIES:
        iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn=pol)
    print("managed policies attached")
    # IAM is eventually consistent - give the control plane a moment
    time.sleep(10)
    arn = f"arn:aws:iam::{account_id()}:role/{ROLE_NAME}"
    print(arn)
    return arn


if __name__ == "__main__":
    main()
