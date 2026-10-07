from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    CHUNK_SIZE,
    DATASET_DIR,
    EVENT_TABLES,
    MIN_OBSERVATION_DAYS,
)


def _read_csv(path: Path, **kwargs: object) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required input CSV not found: {path}")
    return pd.read_csv(path, **kwargs)


def build_member_windows(dataset_dir: Path = DATASET_DIR) -> pd.DataFrame:
    """Build a one-row-per-member cohort with a within-period prediction date."""
    persons = _read_csv(
        dataset_dir / "person" / "person.csv",
        usecols=[
            "person_id",
            "year_of_birth",
            "gender_concept_id",
            "race_concept_id",
            "ethnicity_concept_id",
        ],
    )
    periods = _read_csv(
        dataset_dir / "observation_period" / "observation_period.csv",
        usecols=[
            "person_id",
            "observation_period_start_date",
            "observation_period_end_date",
        ],
        parse_dates=["observation_period_start_date", "observation_period_end_date"],
    )
    periods = periods.dropna(
        subset=[
            "person_id",
            "observation_period_start_date",
            "observation_period_end_date",
        ]
    )
    if periods["person_id"].duplicated().any():
        raise ValueError("Expected at most one observation period per person_id.")

    periods["observation_days"] = (
        periods["observation_period_end_date"]
        - periods["observation_period_start_date"]
    ).dt.days
    periods = periods.loc[periods["observation_days"] >= MIN_OBSERVATION_DAYS].copy()
    periods["index_date"] = periods["observation_period_start_date"] + pd.to_timedelta(
        periods["observation_days"] * (2 / 3), unit="D"
    )
    periods["baseline_days"] = (
        periods["index_date"] - periods["observation_period_start_date"]
    ).dt.days
    periods["followup_days"] = (
        periods["observation_period_end_date"] - periods["index_date"]
    ).dt.days
    members = periods.merge(persons, on="person_id", how="inner", validate="one_to_one")
    members["age_at_index"] = (
        members["index_date"].dt.year - pd.to_numeric(members["year_of_birth"], errors="coerce")
    )
    members.loc[~members["age_at_index"].between(0, 120), "age_at_index"] = np.nan
    members["baseline_years"] = members["baseline_days"] / 365.25
    members["followup_years"] = members["followup_days"] / 365.25

    for column in ("gender_concept_id", "race_concept_id", "ethnicity_concept_id"):
        members[column] = members[column].astype("string")

    members = members.drop(columns=["year_of_birth"]).set_index("person_id")
    if members.empty:
        raise ValueError(
            "No eligible members with valid person and observation-period records."
        )
    return members


def _aggregate_event_counts(
    table_path: Path,
    date_column: str,
    windows: pd.DataFrame,
    feature_name: str,
    chunksize: int = CHUNK_SIZE,
    include_outcomes: bool = False,
) -> tuple[pd.Series, pd.Series | None]:
    """Aggregate baseline events and optionally future visits with bounded memory."""
    baseline_counts = pd.Series(dtype="int64")
    outcome_counts = pd.Series(dtype="int64") if include_outcomes else None
    usecols = ["person_id", date_column]

    if not table_path.is_file():
        raise FileNotFoundError(f"Required event CSV not found: {table_path}")

    for chunk in pd.read_csv(
        table_path,
        usecols=usecols,
        chunksize=chunksize,
        dtype={"person_id": "Int64"},
    ):
        chunk = chunk.dropna(subset=["person_id"])
        event_dates = pd.to_datetime(chunk[date_column], errors="coerce")
        starts = chunk["person_id"].map(windows["observation_period_start_date"])
        indexes = chunk["person_id"].map(windows["index_date"])
        ends = chunk["person_id"].map(windows["observation_period_end_date"])
        known_member = indexes.notna() & event_dates.notna()

        baseline_mask = (
            known_member
            & event_dates.ge(starts)
            & event_dates.lt(indexes)
        )
        if baseline_mask.any():
            counts = chunk.loc[baseline_mask].groupby("person_id").size()
            baseline_counts = baseline_counts.add(counts, fill_value=0).astype("int64")

        if include_outcomes:
            outcome_mask = (
                known_member
                & event_dates.ge(indexes)
                & event_dates.le(ends)
            )
            if outcome_mask.any():
                counts = chunk.loc[outcome_mask].groupby("person_id").size()
                outcome_counts = outcome_counts.add(counts, fill_value=0).astype("int64")

    baseline_counts.name = f"{feature_name}_baseline_count"
    return baseline_counts, outcome_counts


def _aggregate_baseline_plan_periods(
    dataset_dir: Path, windows: pd.DataFrame
) -> pd.Series:
    path = dataset_dir / "payer_plan_period" / "payer_plan_period.csv"
    periods = _read_csv(
        path,
        usecols=[
            "person_id",
            "payer_plan_period_start_date",
        ],
        parse_dates=["payer_plan_period_start_date"],
        dtype={"person_id": "Int64"},
    )
    person_index = periods["person_id"].map(windows["index_date"])
    started_by_index = (
        person_index.notna()
        & periods["payer_plan_period_start_date"].notna()
        & periods["payer_plan_period_start_date"].lt(person_index)
    )
    counts = periods.loc[started_by_index].groupby("person_id").size()
    counts.name = "baseline_payer_plan_period_count"
    return counts


def build_feature_table(
    dataset_dir: Path = DATASET_DIR, chunksize: int = CHUNK_SIZE
) -> tuple[pd.DataFrame, pd.Series]:
    """Merge member attributes and pre-index event aggregates; return future rate."""
    members = build_member_windows(dataset_dir)
    features = members.drop(
        columns=[
            "observation_period_start_date",
            "observation_period_end_date",
            "index_date",
            "observation_days",
            "baseline_days",
            "followup_days",
            "followup_years",
        ]
    ).copy()
    target_visits: pd.Series | None = None

    for feature_name, (table_name, date_column) in EVENT_TABLES.items():
        baseline, outcomes = _aggregate_event_counts(
            dataset_dir / table_name / f"{table_name}.csv",
            date_column,
            members,
            feature_name,
            chunksize=chunksize,
            include_outcomes=feature_name == "visits",
        )
        features = features.join(baseline, how="left")
        if outcomes is not None:
            target_visits = outcomes

    features["baseline_payer_plan_period_count"] = _aggregate_baseline_plan_periods(
        dataset_dir, members
    )
    count_columns = [
        column
        for column in features.columns
        if column.endswith("_baseline_count")
        or column == "baseline_payer_plan_period_count"
    ]
    features[count_columns] = features[count_columns].fillna(0).astype("int64")

    if target_visits is None:
        raise RuntimeError("Future visit count aggregation did not run.")
    annualized_visits = (
        target_visits.reindex(features.index, fill_value=0)
        / members["followup_years"]
    )
    annualized_visits.name = "future_annualized_visits"
    return features, annualized_visits


def build_feature_table_from_directory(
    dataset_dir: str | Path = DATASET_DIR, chunksize: int = CHUNK_SIZE
) -> tuple[pd.DataFrame, pd.Series]:
    """Path-friendly wrapper used by notebooks and command-line callers."""
    return build_feature_table(Path(dataset_dir), chunksize=chunksize)
