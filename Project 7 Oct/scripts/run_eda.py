from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utilization_risk.eda import write_eda_report


if __name__ == "__main__":
    write_eda_report()
