"""Load and validate the CMS DE-SynPUF OMOP CDM datasets (1k and 100k samples).

Usage:
    from load_synpuf import load_dataset, validate_dataset

    tables = load_dataset("1k")            # or "100k"
    report = validate_dataset(tables, "1k")
    assert report["passed"].all()

The loader returns a dict mapping table name -> DataFrame with lowercase
column names, so downstream code works uniformly across both samples.

Run as a script for a self-check over both datasets:
    .venv/bin/python src/load_synpuf.py
"""

from pathlib import Path
import re

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"

SAMPLES = {
    "1k": {
        "dir": "synpuf1k",
        "compression": "bz2",
        "glob": "CDM_*.csv.bz2",
        "prefix": "CDM_",
    },
    "100k": {
        "dir": "synpuf100k",
        "compression": "gzip",
        "glob": "*.csv.gz",
        "prefix": "",
    },
}

# The 1k prefix ships versioned duplicates (e.g. CDM_DRUG_EXPOSURE5.2.2.csv.bz2);
# load only the base CDM files.
_VERSIONED_FILE = re.compile(r"\d+\.\d+\.\d+")

# Tables whose person_id must reference person.person_id.
PERSON_REF_TABLES = [
    "visit_occurrence",
    "condition_occurrence",
    "drug_exposure",
    "observation_period",
]

# Rough expected row counts (lower, upper) used as sanity checks against
# loading the wrong sample. Ranges reflect actual DE-SynPUF composition
# (each person carries ~47 visits and ~160 condition occurrences).
EXPECTED_ROWS = {
    "1k": {
        "person": (900, 1100),
        "visit_occurrence": (40_000, 55_000),
        "condition_occurrence": (150_000, 175_000),
    },
    "100k": {
        "person": (90_000, 110_000),
        "visit_occurrence": (4_000_000, 5_500_000),
    },
}


def _table_name(path: Path, prefix: str) -> str:
    # Path("CDM_PERSON.csv.bz2").stem == "CDM_PERSON.csv"; strip the .csv too.
    stem = Path(path.stem).stem
    if prefix and stem.startswith(prefix):
        stem = stem[len(prefix):]
    return stem.lower()


def _data_files(sample: str) -> list[Path]:
    cfg = SAMPLES[sample]
    directory = DATA_DIR / cfg["dir"]
    files = [
        p
        for p in sorted(directory.glob(cfg["glob"]))
        if not _VERSIONED_FILE.search(p.name)
    ]
    return files


def load_dataset(sample: str) -> dict[str, pd.DataFrame]:
    """Load all base CDM tables for a sample into lowercase-keyed DataFrames.

    Parameters
    ----------
    sample:
        "1k" (bz2, CDM_<TABLE>.csv.bz2, uppercase headers) or
        "100k" (gzip, <table>.csv.gz, lowercase headers).

    Returns
    -------
    dict mapping table name (e.g. "person", "visit_occurrence") to DataFrame
    with lowercase column names.
    """
    if sample not in SAMPLES:
        raise ValueError(f"Unknown sample {sample!r}; expected one of {sorted(SAMPLES)}")
    cfg = SAMPLES[sample]
    tables: dict[str, pd.DataFrame] = {}
    for path in _data_files(sample):
        name = _table_name(path, cfg["prefix"])
        df = pd.read_csv(path, compression=cfg["compression"], low_memory=False)
        df.columns = [c.lower() for c in df.columns]
        tables[name] = df
    return tables


def _pk_column(table: str, columns) -> str | None:
    if table == "person":
        return "person_id" if "person_id" in columns else None
    candidate = f"{table}_id"
    return candidate if candidate in columns else None


def validate_dataset(tables: dict[str, pd.DataFrame], sample: str) -> pd.DataFrame:
    """Run sanity / PK / referential-integrity checks over a loaded sample.

    Returns a per-check report DataFrame with columns:
    check, table, detail, passed.
    """
    checks: list[dict] = []

    def add(check: str, table: str, passed: bool, detail: str = "") -> None:
        checks.append(
            {"check": check, "table": table, "passed": bool(passed), "detail": detail}
        )

    for table, df in sorted(tables.items()):
        add("row_count>0", table, len(df) > 0, f"rows={len(df):,}")

    for table, (lo, hi) in EXPECTED_ROWS.get(sample, {}).items():
        if table in tables:
            n = len(tables[table])
            add("expected_row_count", table, lo <= n <= hi, f"rows={n:,}, expected [{lo:,}, {hi:,}]")

    person_ids = set(tables["person"]["person_id"]) if "person" in tables else set()

    for table in PERSON_REF_TABLES:
        if table not in tables:
            continue
        df = tables[table]
        if "person_id" not in df.columns:
            add("person_id_present", table, False, "person_id column missing")
            continue
        missing = int((~df["person_id"].isin(person_ids)).sum())
        add(
            "person_id_referential_integrity",
            table,
            missing == 0,
            f"rows={len(df):,}, orphan_person_id={missing:,}",
        )

    for table, df in sorted(tables.items()):
        pk = _pk_column(table, df.columns)
        if pk is None:
            continue
        dupes = int(df[pk].duplicated().sum())
        add("primary_key_unique", table, dupes == 0, f"pk={pk}, duplicates={dupes:,}")

    return pd.DataFrame(checks, columns=["check", "table", "passed", "detail"])


def main() -> None:
    ok = True
    for sample in ("1k", "100k"):
        tables = load_dataset(sample)
        print(f"\n=== sample={sample} ===")
        for table, df in sorted(tables.items()):
            print(f"  {table:<24} {len(df):>12,} rows x {df.shape[1]} cols")
        report = validate_dataset(tables, sample)
        failed = report[~report["passed"]]
        print(f"  checks: {report['passed'].sum()}/{len(report)} passed")
        if len(failed):
            ok = False
            print("  FAILED CHECKS:")
            for row in failed.itertuples():
                print(f"    [{row.check}] {row.table}: {row.detail}")
    if not ok:
        raise SystemExit("Validation failed")
    print("\nAll checks passed for both datasets.")


if __name__ == "__main__":
    main()
