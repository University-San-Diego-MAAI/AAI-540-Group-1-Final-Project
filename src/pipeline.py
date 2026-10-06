"""SageMaker Pipeline definition for the utilization-risk model (Task 17).

Chain (design doc, Model CI/CD): validate -> train -> register with an
approval gate; promotion is a human console action on the registered
package, rollback = point the transform job at the previous version.

  sagemaker.workflow.pipeline.Pipeline with a single TrainingStep driving
  src/train.py (the same entry point used by notebook 03) and a
  RegisterModel step writing to the member-utilization-risk group with
  PendingManualApproval.

Usage (creates/updates the pipeline; does NOT execute it):
  .venv-sm/bin/python src/pipeline.py

Executing a run: aws sagemaker start-pipeline-execution --pipeline-name aai-540-g1-utilization-risk
(only when you want to spend the training-job minutes; nothing runs on its own)
"""

import boto3
import sagemaker
from sagemaker import image_uris
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.estimator import Estimator
from sagemaker.session import Session

region = boto3.Session().region_name
boto_session = boto3.Session(region_name=region)
sess = Session(boto_session=boto_session,
               sagemaker_client=boto_session.client("sagemaker", region_name=region))
bucket = sess.default_bucket()
PREFIX = "aai-540-g1"
GROUP = "member-utilization-risk"

try:
    role = sagemaker.get_execution_role()
except Exception:
    iam = boto3.client("iam", region_name=region)
    by_name = {r["RoleName"]: r for r in iam.list_roles()["Roles"]}
    picked = ("aai-540-g1-execution-role" if "aai-540-g1-execution-role" in by_name
              else sorted((n for n in by_name
                           if n.startswith("AmazonSageMaker-ExecutionRole")),
                          key=lambda n: by_name[n]["CreateDate"])[-1])
    account = boto3.client("sts").get_caller_identity()["Account"]
    role = f"arn:aws:iam::{account}:role/{picked}"

xgb_image = image_uris.retrieve("xgboost", region, version="1.7-1")

estimator = Estimator(
    image_uri=xgb_image,
    entry_point="train.py",
    source_dir="src",
    role=role,
    instance_count=1,
    instance_type="ml.m5.2xlarge",
    output_path=f"s3://{bucket}/{PREFIX}/artifacts/pipeline",
    sagemaker_session=sess,
    hyperparameters={"model": "xgb", "eta": "0.05", "max-depth": "6",
                     "min-child-weight": "5"},  # champion config (nb04)
)

train_step = TrainingStep(
    name="train-champion-config",
    estimator=estimator,
    inputs={"train": TrainingInput(
        f"s3://{bucket}/{PREFIX}/processed/training/training_table.csv",
        content_type="text/csv")},
)

register_step = RegisterModel(
    name="register-model",
    estimator=estimator,
    model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["text/csv"],
    response_types=["text/csv"],
    inference_instances=["ml.m5.xlarge"],
    transform_instances=["ml.m5.xlarge"],
    model_package_group_name=GROUP,
    approval_status="PendingManualApproval",  # human gate before production
)

pipeline = Pipeline(
    name="aai-540-g1-utilization-risk",
    steps=[train_step, register_step],
    sagemaker_session=sess,
)

if __name__ == "__main__":
    pipeline.upsert(role_arn=role)
    print("pipeline upserted: aai-540-g1-utilization-risk "
          "(not executed - start manually when needed)")
