"""Build the temporally-split training table for the utilization-risk model.

Follows Team1_ML_Design_Document.docx (Training Data / Feature Engineering) with
the course-mandated split from Team Update 3.1 (train ~40%, validation ~10%,
test ~10%, production ~40%, stratified by class).

Pipeline:
1. **Eligibility** — members whose observation period overlaps BOTH the 2008
   history window and the 2009-2010 outcome window (``start <= 2008-12-31``
   and ``end >= 2009-01-01``). Empirically 85.3% of the 100k sample (the
   design doc's 83.3% was an estimate; the measured figure and this exact
   rule are recorded in the output and will be reflected in design-doc v2).
2. **Censoring** — members who die inside the outcome window are excluded
   from labeling (truncated utilization is not a fair Low-Risk signal).
3. **History features** — the same modeling contract as
   ``build_features.build_member_features`` but aggregated only over rows
   dated inside 2008 (see FEATURE_COLUMNS).
4. **Labels** (rule-based weak supervision, design doc): outcome score per
   member over 2009-2010 = total_visits + 3*inpatient_visits + drug_eras,
   then terciles -> Low / Medium / High.
5. **Split** — per-class seeded shuffle into 40/10/10/40
   train/validation/test/production; the production slice stays unlabeled
   at scoring time and doubles as the Update-3.1 "production data" reserve
   for Module 5 monitors.

Memory: tables are read one at a time with ``usecols`` (never all tables at
once). Windowing uses ISO-date string prefix comparison (``str[:4]``) instead
of datetime parsing — an order of magnitude faster on 12.7M-row tables.

Usage::

    .venv/bin/python src/build_training_data.py     # builds 100k + 1k tables

Expected 100k output: ~84k eligible rows after censoring x ~25 columns;
integrity checks fail loudly (non-zero exit) on any violation.
"""

from pathlib import Path

import numpy as np
import pandas as pd

try:  # imported as a top-level module (src/ on sys.path)
    from build_features import (
        AGE_BANDS,
        AGE_BIN_EDGES,
        CHRONIC_CONDITIONS,
        GENDER_LABELS,
        REF_YEAR,
        VISIT_TYPE_CONCEPTS,
        _table_path,
    )
    from clinical_codes import icd9_prefixes
    from load_synpuf import SAMPLES
except ImportError:  # imported as a package module
    from src.build_features import (
        AGE_BANDS,
        AGE_BIN_EDGES,
        CHRONIC_CONDITIONS,
        GENDER_LABELS,
        REF_YEAR,
        VISIT_TYPE_CONCEPTS,
        _table_path,
    )
    from src.clinical_codes import icd9_prefixes
    from src.load_synpuf import SAMPLES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Temporal windows (design doc): history = 2008, outcome = 2009-2010.
# Dates are compared as dashless strings ("20081231"): the 100k sample ships
# ISO dates ("2008-12-01") while the 1k sample ships "20080101" - stripping
# dashes normalizes both formats for lexicographic comparison.
HISTORY_YEAR = "2008"
OUTCOME_YEARS = ("2009", "2010")
HISTORY_END = "20081231"            # eligibility: period start <= this
OUTCOME_START = "20090101"          # eligibility: period end >= this
OUTCOME_END = "20101231"            # deaths on/before this date censor the member
ELIGIBILITY_RULE = (
    "observation_period overlaps both windows: "
    "observation_period_start_date <= 2008-12-31 AND "
    "observation_period_end_date >= 2009-01-01"
)
LABEL_RULE_VERSION = (
    "outcome_score_v1: score = total_visits + 3*inpatient_visits + drug_eras "
    "over 2009-2010; rank-based terciles (tie-safe) -> Low/Medium/High"
)
RISK_CLASSES = ["Low", "Medium", "High"]
SEED = 42

# Course-mandated split (Team Update 3.1).
SPLIT_SHARES = {"train": 0.40, "validation": 0.10, "test": 0.10, "production": 0.40}
SPLIT_NAMES = list(SPLIT_SHARES)  # order matters: production takes the remainder

# Output schema: 20 model features + row key + label/split bookkeeping.
# Contract inherited from build_features.py: one-hot the nominal columns
# (age_band, gender, race, state) at train time; never treat them as ordinal.
# ``race`` stays in the table as a fairness-audit slice but is EXCLUDED from
# the primary model configuration per the design doc.
FEATURE_COLUMNS = [
    "person_id",
    "age",
    "age_band",
    "gender",
    "race",
    "state",
    "observation_months",
    "inpatient_visits",
    "outpatient_visits",
    "total_visits",
    "distinct_care_sites",
    "diabetes",
    "chf",
    "copd",
    "stroke",
    "cancer",
    "renal_disease",
    "drug_exposures",
    "distinct_drugs",
    "drug_eras",
    "distinct_conditions",
    "distinct_procedures",
]
META_COLUMNS = ["risk_class", "split", "label_rule_version"]
_COUNT_COLUMNS = [c for c in FEATURE_COLUMNS if c not in
                  ("person_id", "age", "age_band", "gender", "race", "state")]


def _read(sample: str, table: str, usecols: list[str]) -> pd.DataFrame:
    """Memory-safe single-table read (mirrors build_features._read_table)."""
    path = _table_path(sample, table)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {sample} table: {path}")
    cfg = SAMPLES[sample]
    if cfg["prefix"]:  # 1k files use uppercase headers
        usecols = [c.upper() for c in usecols]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    return df


def _dashless(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace("-", "", regex=False)


def _eligible_persons(sample: str) -> pd.Series:
    """person_ids whose observation period overlaps both windows (rule A)."""
    op = _read(sample, "observation_period",
               ["person_id", "observation_period_start_date",
                "observation_period_end_date"])
    mask = ((_dashless(op["observation_period_start_date"]) <= HISTORY_END)
            & (_dashless(op["observation_period_end_date"]) >= OUTCOME_START))
    return op.loc[mask, "person_id"]


def _censor_deaths(sample: str, person_ids: set) -> set:
    """Remove members who die inside the outcome window (truncated labels)."""
    death = _read(sample, "death", ["person_id", "death_date"])
    dead = death.loc[_dashless(death["death_date"]) <= OUTCOME_END, "person_id"]
    censored = person_ids - set(dead)
    print(f"censoring: {len(person_ids):,} eligible -> {len(censored):,} after "
          f"removing {len(person_ids) - len(censored):,} outcome-window deaths")
    return censored


def _demographics(sample: str, person_ids: set) -> pd.DataFrame:
    person = _read(sample, "person",
                   ["person_id", "gender_concept_id", "year_of_birth",
                    "race_source_value", "location_id"])
    location = _read(sample, "location", ["location_id", "state"])
    person = person[person["person_id"].isin(person_ids)]
    state = person["location_id"].map(location.set_index("location_id")["state"])
    demo = pd.DataFrame({"person_id": person["person_id"]})
    demo["age"] = (REF_YEAR - person["year_of_birth"]).astype("int64")
    demo["age_band"] = pd.cut(demo["age"], bins=AGE_BIN_EDGES,
                              labels=AGE_BANDS, right=True).astype(str)
    demo["gender"] = person["gender_concept_id"].map(GENDER_LABELS).fillna("Unknown")
    demo["race"] = person["race_source_value"]
    demo["state"] = state.fillna("Missing")
    return demo.set_index("person_id")


def _utilization(sample: str, person_ids: set, years: tuple[str, ...]) -> pd.DataFrame:
    """Visit counts restricted to rows whose start date falls in ``years``."""
    visits = _read(sample, "visit_occurrence",
                   ["person_id", "visit_concept_id", "care_site_id", "visit_start_date"])
    visits = visits[visits["visit_start_date"].astype(str).str[:4].isin(years)]
    g = visits.groupby("person_id")
    util = pd.DataFrame({"total_visits": g.size(),
                         "distinct_care_sites": g["care_site_id"].nunique()})
    typed = visits[visits["visit_concept_id"].isin(set(VISIT_TYPE_CONCEPTS.values()))]
    counts = typed.groupby(["person_id", "visit_concept_id"]).size().unstack(fill_value=0)
    for col, concept_id in VISIT_TYPE_CONCEPTS.items():
        if col == "er_visits":
            continue  # zero-variance in this export; not carried into the table
        util[col] = counts[concept_id] if concept_id in counts else 0
    return util


def _conditions(sample: str, person_ids: set, years: tuple[str, ...]) -> pd.DataFrame:
    cond = _read(sample, "condition_occurrence",
                 ["person_id", "condition_source_value", "condition_start_date"])
    cond = cond[cond["condition_start_date"].astype(str).str[:4].isin(years)]
    prefix = icd9_prefixes(cond["condition_source_value"])
    feats = pd.DataFrame({
        "distinct_conditions": cond.groupby("person_id")["condition_source_value"].nunique()
    })
    for name, (lo, hi) in CHRONIC_CONDITIONS.items():
        hits = prefix.between(lo, hi)
        feats[name] = hits.groupby(cond["person_id"]).any().astype("int64")
    return feats


def _pharmacy(sample: str, person_ids: set, years: tuple[str, ...]) -> pd.DataFrame:
    exposure = _read(sample, "drug_exposure",
                     ["person_id", "drug_concept_id", "drug_exposure_start_date"])
    exposure = exposure[exposure["drug_exposure_start_date"].astype(str).str[:4].isin(years)]
    era = _read(sample, "drug_era", ["person_id", "drug_era_start_date"])
    era = era[era["drug_era_start_date"].astype(str).str[:4].isin(years)]
    return pd.DataFrame({
        "drug_exposures": exposure.groupby("person_id").size(),
        "distinct_drugs": exposure.groupby("person_id")["drug_concept_id"].nunique(),
        "drug_eras": era.groupby("person_id").size(),
    })


def _procedures(sample: str, person_ids: set, years: tuple[str, ...]) -> pd.Series:
    proc = _read(sample, "procedure_occurrence",
                 ["person_id", "procedure_source_value", "procedure_date"])
    proc = proc[proc["procedure_date"].astype(str).str[:4].isin(years)]
    return proc.groupby("person_id")["procedure_source_value"].nunique().rename(
        "distinct_procedures")


def _history_months(sample: str, person_ids: set) -> pd.Series:
    """Whole calendar months of observation inside the 2008 history window."""
    op = _read(sample, "observation_period",
               ["person_id", "observation_period_start_date",
                "observation_period_end_date"])
    start = pd.to_datetime(op["observation_period_start_date"], errors="coerce")
    end = pd.to_datetime(op["observation_period_end_date"], errors="coerce")
    w_start = pd.Timestamp(f"{HISTORY_YEAR}-01-01")
    w_end = pd.Timestamp(f"{HISTORY_YEAR}-12-31")
    days = (end.clip(upper=w_end) - start.clip(lower=w_start)).dt.days.clip(lower=0)
    months = (days / (365.25 / 12)).round().astype("int64")
    return months.groupby(op["person_id"]).sum().rename("observation_months")


def _assign_labels(outcome: pd.DataFrame) -> pd.Series:
    """Design-doc rule: score -> terciles -> Low/Medium/High.

    Zero outcome utilization is a true 0 (no rows), not missing, so the
    component counts are zero-filled before scoring. Terciles are assigned
    on the score's percentile rank: mass ties at 0 make ``pd.qcut`` bins
    lopsided, while rank-based cut points keep all three classes balanced
    to within one member.
    """
    score = (outcome["total_visits"].fillna(0)
             + 3 * outcome["inpatient_visits"].fillna(0)
             + outcome["drug_eras"].fillna(0))
    pct = score.rank(pct=True)
    labels = pd.Series(np.where(pct <= 1 / 3, "Low",
                                np.where(pct <= 2 / 3, "Medium", "High")),
                       index=score.index, name="risk_class")
    return labels


def _assign_splits(labels: pd.Series, seed: int = SEED) -> pd.Series:
    """Per-class stratified 40/10/10/40; production takes the remainder."""
    rng = np.random.default_rng(seed)
    split = pd.Series(index=labels.index, dtype=object)
    for cls in RISK_CLASSES:
        idx = rng.permutation(labels[labels == cls].index.to_numpy())
        n = len(idx)
        n_train = round(SPLIT_SHARES["train"] * n)
        n_val = round(SPLIT_SHARES["validation"] * n)
        n_test = round(SPLIT_SHARES["test"] * n)
        split.loc[idx[:n_train]] = "train"
        split.loc[idx[n_train:n_train + n_val]] = "validation"
        split.loc[idx[n_train + n_val:n_train + n_val + n_test]] = "test"
        split.loc[idx[n_train + n_val + n_test:]] = "production"
    return split.rename("split")


def build_training_table(sample: str) -> pd.DataFrame:
    if sample not in SAMPLES:
        raise ValueError(f"Unknown sample {sample!r}; expected one of {sorted(SAMPLES)}")

    eligible = set(_eligible_persons(sample))
    n_population = len(_read(sample, "person", ["person_id"]))
    print(f"[{sample}] eligible (dual-window): {len(eligible):,} "
          f"({100 * len(eligible) / n_population:.1f}% of the {n_population:,} members)")
    person_ids = _censor_deaths(sample, eligible)

    features = _demographics(sample, person_ids)
    outcome = (
        _utilization(sample, person_ids, OUTCOME_YEARS)[["total_visits", "inpatient_visits"]]
        .join(_pharmacy(sample, person_ids, OUTCOME_YEARS)[["drug_eras"]])
        # members with NO outcome-window rows are absent from every outcome
        # frame — reindex so they get a true score of 0 (genuinely Low
        # utilization), not a missing label.
        .reindex(sorted(person_ids)).fillna(0)
    )
    labels = _assign_labels(outcome)
    splits = _assign_splits(labels)

    table = features.join([
        _history_months(sample, person_ids),
        _utilization(sample, person_ids, (HISTORY_YEAR,)),
        _conditions(sample, person_ids, (HISTORY_YEAR,)),
        _pharmacy(sample, person_ids, (HISTORY_YEAR,)),
        _procedures(sample, person_ids, (HISTORY_YEAR,)),
        labels,
        splits,
    ])
    table = table.fillna({c: 0 for c in _COUNT_COLUMNS})
    for c in _COUNT_COLUMNS:
        table[c] = table[c].astype("int64")
    table["label_rule_version"] = LABEL_RULE_VERSION
    table = table.reset_index()[FEATURE_COLUMNS + META_COLUMNS]
    return table


def _integrity_checks(table: pd.DataFrame, sample: str) -> None:
    errors = []
    nulls = int(table.isna().sum().sum())
    if nulls:
        errors.append(f"null cells: {nulls}")
    shares = table["split"].value_counts(normalize=True)
    for name, share in SPLIT_SHARES.items():
        actual = shares.get(name, 0.0)
        if abs(actual - share) > 0.01:  # "~10%"-style rounding at small n
            errors.append(f"split '{name}' = {actual:.3f}, expected {share:.2f}")
    class_dist = table["risk_class"].value_counts(normalize=True)
    if len(class_dist) != 3 or (class_dist.max() - class_dist.min()) > 0.02:
        errors.append(f"class balance off: {class_dist.round(3).to_dict()}")
    if table["person_id"].nunique() != len(table):
        errors.append("duplicate person_id rows")
    print(f"\n--- [{sample}] integrity ---")
    print(f"rows={len(table):,} cols={table.shape[1]} nulls={nulls}")
    print("split shares :", (shares * 100).round(1).to_dict())
    print("class balance:", (class_dist * 100).round(1).to_dict())
    if errors:
        raise SystemExit("INTEGRITY FAIL: " + "; ".join(errors))


def main() -> None:
    for sample, name in [("100k", "training_table.csv"),
                         ("1k", "training_table_1k.csv")]:
        table = build_training_table(sample)
        _integrity_checks(table, sample)
        out = PROCESSED_DIR / name
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False)
        print(f"[{sample}] wrote {len(table):,} rows -> {out}\n")


if __name__ == "__main__":
    main()
