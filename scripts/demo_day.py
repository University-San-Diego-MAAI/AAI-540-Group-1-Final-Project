"""Keep the serverless demo endpoint warm while you record.

`on`  - sends a keep-alive prediction every 3.5 minutes (serverless endpoints
        sleep after ~5 min idle; a cold start mid-recording looks bad).
        Press Ctrl+C when done - the endpoint returns to $0 at rest.
`off` - just tells you nothing is running (warmth stops by itself).

Usage:  set -a && source .env && set +a && .venv-sm/bin/python scripts/demo_day.py on
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import boto3
import pandas as pd

from aws_common import check_credentials, region
from model_common import encode_features, load_table

ENDPOINT = "aai540-g1-demo-endpoint"


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("on", "off"):
        sys.exit("usage: demo_day.py on|off")
    check_credentials()
    if sys.argv[1] == "off":
        print("warmth is a foreground process - just Ctrl+C the 'on' session. "
              "The endpoint costs $0 when idle.")
        return
    r = region()
    runtime = boto3.client("sagemaker-runtime", region_name=r)
    table = load_table(ROOT / "data" / "processed" / "training_table.csv")
    X_all, _, audit = encode_features(table)
    vec = X_all[(audit["split"] == "test").to_numpy()][0]
    body = ",".join(f"{v:.6g}" for v in vec)
    print("warming endpoint + keep-alive every 3.5 min (Ctrl+C to stop)…", flush=True)
    runtime.invoke_endpoint(EndpointName=ENDPOINT, ContentType="text/csv",
                            Accept="application/json", Body=body)
    print("first response received - endpoint is warm", flush=True)
    while True:
        time.sleep(210)
        resp = runtime.invoke_endpoint(EndpointName=ENDPOINT, ContentType="text/csv",
                                       Accept="application/json", Body=body)
        print(f"keep-alive ok {time.strftime('%H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
