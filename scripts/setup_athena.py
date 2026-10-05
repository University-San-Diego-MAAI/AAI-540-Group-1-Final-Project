"""Set up Athena for the project: workgroup, database, training-table DDL (Task 5).

Creates (idempotently):
  - workgroup ``aai540_g1`` with query results in ``aai-540-g1/athena-results/``
    (SSE-S3 encrypted results)
  - database ``aai540_g1``
  - external table ``training_table`` over the uploaded CSV in
    ``aai-540-g1/processed/training/``

Then runs the Update-3.1 validation queries and prints their results:
member count, per-class counts, age-band x class cross-tab, and High-Risk
chronic-flag prevalence (CHF should climb steeply with class, as in the EDA).

Run from the repo root:  .venv/bin/python scripts/setup_athena.py
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

import boto3

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

WORKGROUP = "aai540_g1"
DATABASE = "aai540_g1"

TRAINING_TABLE_DDL = """
CREATE EXTERNAL TABLE IF NOT EXISTS {db}.training_table (
  person_id BIGINT, age INT, age_band STRING, gender STRING, race INT, state STRING,
  observation_months INT, inpatient_visits INT, outpatient_visits INT, total_visits INT,
  distinct_care_sites INT, diabetes INT, chf INT, copd INT, stroke INT, cancer INT,
  renal_disease INT, drug_exposures INT, distinct_drugs INT, drug_eras INT,
  distinct_conditions INT, distinct_procedures INT,
  risk_class STRING, split STRING, label_rule_version STRING
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION 's3://{bucket}/aai-540-g1/processed/training/'
TBLPROPERTIES ('skip.header.line.count'='1')
"""

VALIDATION_QUERIES = {
    "member_count": "SELECT COUNT(*) AS n, COUNT(DISTINCT person_id) AS distinct_members "
                    "FROM {db}.training_table",
    "class_counts": "SELECT split, risk_class, COUNT(*) AS n FROM {db}.training_table "
                    "GROUP BY split, risk_class ORDER BY split, risk_class",
    "age_band_x_class": "SELECT age_band, risk_class, COUNT(*) AS n FROM {db}.training_table "
                        "GROUP BY age_band, risk_class ORDER BY age_band, risk_class",
    "high_risk_chronic": "SELECT risk_class, AVG(CAST(chf AS DOUBLE)) AS chf_rate, "
                         "AVG(CAST(diabetes AS DOUBLE)) AS diabetes_rate, "
                         "AVG(CAST(total_visits AS DOUBLE)) AS mean_visits "
                         "FROM {db}.training_table GROUP BY risk_class ORDER BY risk_class",
}


def _wait(athena, qid: str) -> None:
    while True:
        state = athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["Status"]
        s = state["State"]
        if s == "SUCCEEDED":
            return
        if s in ("FAILED", "CANCELLED"):
            raise SystemExit(f"query {qid} {s}: {state.get('StateChangeReason')}")
        time.sleep(1.5)


def _run(athena, sql: str) -> str:
    qid = athena.start_query_execution(QueryString=sql, WorkGroup=WORKGROUP)["QueryExecutionId"]
    _wait(athena, qid)
    return qid


def main() -> int:
    if not os.getenv("AWS_ACCESS_KEY_ID"):
        sys.exit("FAIL: .env credentials missing — run scripts/check_aws_env.py first.")
    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    account = boto3.client("sts", region_name=region).get_caller_identity()["Account"]
    bucket = f"sagemaker-{region}-{account}"
    athena = boto3.client("athena", region_name=region)

    # workgroup (tolerate AlreadyExists)
    try:
        athena.create_work_group(
            Name=WORKGROUP,
            Configuration={
                "ResultConfiguration": {
                    "OutputLocation": f"s3://{bucket}/aai-540-g1/athena-results/",
                    "EncryptionConfiguration": {"EncryptionOption": "SSE_S3"},
                },
                "EnforceWorkGroupConfiguration": True,
            },
            Description="AAI-540 Group 1 final project queries",
        )
        print(f"created workgroup {WORKGROUP}")
    except athena.exceptions.InvalidRequestException as exc:
        if "AlreadyExists" not in str(exc):
            raise
        print(f"workgroup {WORKGROUP} already exists")

    _run(athena, f"CREATE DATABASE IF NOT EXISTS {DATABASE}")
    print(f"database {DATABASE} ready")
    _run(athena, TRAINING_TABLE_DDL.format(db=DATABASE, bucket=bucket))
    print("external table training_table ready")

    print("\n--- validation queries ---")
    for name, template in VALIDATION_QUERIES.items():
        qid = _run(athena, template.format(db=DATABASE))
        rows = athena.get_query_results(QueryExecutionId=qid)["ResultSet"]["Rows"]
        print(f"\n[{name}]")
        for row in rows:
            print("  " + " | ".join(c.get("VarCharValue", "") for c in row["Data"]))

    print(f"\nREADY — explore at https://console.aws.amazon.com/athena/home?region={region}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
