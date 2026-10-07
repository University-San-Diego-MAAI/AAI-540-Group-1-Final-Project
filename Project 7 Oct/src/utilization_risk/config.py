from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHUNK_SIZE = 200_000
MIN_OBSERVATION_DAYS = 365
RANDOM_STATE = 42
TARGET = "Member_Utilization_Risk"
CLASSES = ("Low", "Medium", "High")

EVENT_TABLES = {
    "visits": ("visit_occurrence", "visit_start_date"),
    "conditions": ("condition_occurrence", "condition_start_date"),
    "procedures": ("procedure_occurrence", "procedure_date"),
    "drug_exposures": ("drug_exposure", "drug_exposure_start_date"),
    "measurements": ("measurement", "measurement_date"),
    "observations": ("observation", "observation_date"),
    "devices": ("device_exposure", "device_exposure_start_date"),
}
