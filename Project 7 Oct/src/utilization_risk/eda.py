from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import CHUNK_SIZE, DATASET_DIR, REPORTS_DIR


def build_dataset_inventory(
    dataset_dir: Path = DATASET_DIR, chunksize: int = CHUNK_SIZE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inspect every CSV incrementally and summarize row counts and null rates."""
    csv_paths = sorted(dataset_dir.rglob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under {dataset_dir}")

    rows: list[dict[str, object]] = []
    column_rows: list[dict[str, object]] = []
    for path in csv_paths:
        column_names = pd.read_csv(path, nrows=0).columns.tolist()
        missing_counts = pd.Series(0, index=column_names, dtype="int64")
        row_count = 0
        member_ids: set[object] = set()
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            row_count += len(chunk)
            missing_counts = missing_counts.add(
                chunk.isna().sum().astype("int64"), fill_value=0
            ).astype("int64")
            if "person_id" in chunk.columns:
                member_ids.update(chunk["person_id"].dropna().unique())
        row: dict[str, object] = {
            "table": path.parent.name,
            "file": str(path.relative_to(dataset_dir)),
            "rows": row_count,
            "columns": len(column_names),
            "mean_missing_fraction": (
                float(missing_counts.sum() / (row_count * len(column_names)))
                if row_count and column_names
                else 0.0
            ),
        }
        if "person_id" in column_names:
            row["unique_person_ids"] = len(member_ids)
        else:
            row["unique_person_ids"] = ""
        rows.append(row)
        for column in column_names:
            column_rows.append(
                {
                    "table": path.parent.name,
                    "column": column,
                    "missing_values": int(missing_counts[column]),
                    "missing_fraction": (
                        float(missing_counts[column] / row_count) if row_count else 0.0
                    ),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(column_rows)


def write_eda_report(
    dataset_dir: Path = DATASET_DIR, reports_dir: Path = REPORTS_DIR
) -> tuple[pd.DataFrame, Path]:
    inventory, column_missingness = build_dataset_inventory(dataset_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = reports_dir / "dataset_inventory.csv"
    column_missingness_path = reports_dir / "column_missingness.csv"
    report_path = reports_dir / "eda_report.md"
    inventory.to_csv(inventory_path, index=False)
    column_missingness.to_csv(column_missingness_path, index=False)

    total_rows = int(inventory["rows"].sum())
    report_lines = [
        "# Raw dataset EDA",
        "",
        "## Scope",
        "",
        f"Inventoried {len(inventory)} CSV files under `{dataset_dir}` "
        f"({total_rows:,} rows total). Files were scanned in chunks.",
        "",
        "These are OMOP-style tables. `person_id` is the member key in the "
        "person/event tables. There is no provided `Member_Utilization_Risk` "
        "target; the modeling pipeline creates a future-visit-rate proxy "
        "documented in the project README.",
        "",
        "## Dataset inventory",
        "",
        "| Table | Rows | Columns | Unique members | Mean missing fraction |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in inventory.to_dict("records"):
        unique_members = row["unique_person_ids"]
        report_lines.append(
            f"| {row['table']} | {row['rows']:,} | {row['columns']} | "
            f"{unique_members if unique_members != '' else 'n/a'} | "
            f"{row['mean_missing_fraction']:.1%} |"
        )
    report_lines.extend(
        [
            "",
            "## EDA observations and merge recommendations",
            "",
            "- Keep one row per member in the modeling table. Aggregate each "
            "event table by `person_id` before merging; direct event-to-event "
            "joins would create many-to-many row multiplication.",
            "- Preserve members with no rows in a given event table via left "
            "joins and represent their event count as zero.",
            "- Exclude the condition/drug era tables from counts when the "
            "corresponding occurrence/exposure tables are used, to avoid "
            "double counting. Provider/care-site/location/drug-strength are "
            "reference tables, not direct member-level features.",
            "- Avoid dropping whole member rows for optional missing values. "
            "Use training-fitted imputers; remove malformed event dates/IDs "
            "only from the corresponding aggregate.",
            "- Keep only data dated before each prediction index in features. "
            "Use later visits exclusively to build the proxy outcome.",
            "- Baseline duration is measured only up to the prediction index; "
            "total observation-period duration and future follow-up length are "
            "not model features.",
            "",
            "## Most-missing columns",
            "",
            "| Table | Column | Missing values | Missing fraction |",
            "|---|---|---:|---:|",
        ]
    )
    most_missing = column_missingness.sort_values(
        ["missing_fraction", "missing_values"], ascending=False
    ).head(25)
    for row in most_missing.to_dict("records"):
        report_lines.append(
            f"| {row['table']} | {row['column']} | {row['missing_values']:,} | "
            f"{row['missing_fraction']:.1%} |"
        )
    report_lines.extend(
        [
            "",
            "The complete table inventory is saved to `dataset_inventory.csv`; "
            "per-column null counts/rates are saved to `column_missingness.csv`.",
            "Table-level mean missingness is diagnostic, not a reason to drop "
            "whole tables or member rows.",
        ]
    )
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"EDA report: {report_path}")
    print(f"Dataset inventory: {inventory_path}")
    return inventory, report_path
