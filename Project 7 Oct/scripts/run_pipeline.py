from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utilization_risk.pipeline import run_pipeline


if __name__ == "__main__":
    run_pipeline()
