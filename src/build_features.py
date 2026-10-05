"""Build the member-level feature table for the CMS DE-SynPUF OMOP CDM dataset.

One row per ``person_id``; the column set maps onto the feature list in
``Team1_RFC.docx`` (Task 5 of the approved plan):

- Demographics: ``age``, ``age_band``, ``gender``, ``race``, ``ethnicity``,
  ``state``.
- Utilization: ``inpatient_visits``, ``outpatient_visits``, ``er_visits``
  (OMOP visit concepts 9201/9202/9203), ``total_visits`` (all visit rows —
  the robust signal, see caveats), ``distinct_care_sites``,
  ``observation_months``.
- Chronic conditions: binary 0/1 ANY-match flags on the 3-digit ICD-9
  prefix of ``condition_source_value`` — ``diabetes``, ``chf``, ``copd``,
  ``stroke``, ``cancer``, ``renal_disease`` (definitions in
  ``clinical_codes.CHRONIC_CONDITIONS``, shared with the EDA notebook).
- Pharmacy: ``drug_exposures``, ``distinct_drugs`` (distinct
  ``drug_concept_id`` — the reliable grouping key; ``drug_source_value``
  mixes NDC and HCPCS codes), ``drug_eras``.
- Interaction breadth: ``distinct_conditions`` (distinct
  ``condition_source_value``), ``distinct_procedures`` (distinct
  ``procedure_source_value``).

Usage::

    from build_features import build_member_features
    features = build_member_features("100k")

    .venv/bin/python src/build_features.py     # builds 100k artifact + 1k check

Like the loader, this module expects to be imported with ``src/`` on
``sys.path`` (see ``load_synpuf`` docstring).

Caveats (also reflected in the module-level constants):
- **No spending/cost columns.** The OMOP conversion of DE-SynPUF carries no
  charge/payment fields, so the RFC's "previous spending" feature cannot be
  built directly; utilization intensity (visit/drug/procedure counts) is
  used as its proxy.
- **Synthetic data.** DE-SynPUF is fully synthetic; per CMS it has limited
  inferential value, so all downstream associations are methodological
  exercises, not clinical findings.
- **Sparse visit typing.** ~85% of visit rows have ``visit_concept_id = 0``
  (unmapped; typed rows are only ~15% of the table), and this export
  contains no ER (9203) rows. The typed counts (inpatient + outpatient +
  er) therefore capture only ~15% of a member's full visit activity, while
  ``total_visits`` (all rows) and the zero-visit share are the robust
  utilization signals.
- **Claim-line duplication.** DE-SynPUF repeats diagnoses across claim
  lines (~160 condition rows/member at 1k, ~127 at 100k), which inflates
  row-level prevalence; the member-level ANY-match flags are immune to
  duplication (a member either has or has not a code in range), but they
  should be read as utilization-flavored signals, not clinical truth.
- **Demographics encoding.** ``race``/``ethnicity`` carry the raw
  DE-SynPUF source codes (1=White, 2=Black, 3=Other, 4=Asian,
  5=Hispanic, 6=North American Native); there is no vocabulary table to
  resolve them further. In this export ``ethnicity`` is byte-for-byte
  identical to ``race`` for every member, so it is redundant — consider
  dropping it for modeling (kept in the CSV for artifact stability).
- **State encoding.** ``state`` mixes USPS abbreviations with one raw
  numeric code ("54", 1,456 members) that maps to no state in the
  location table.

Modeling contract (for the later modeling phase):
- ``person_id`` is the row key, not a feature — exclude it before fitting.
- ``er_visits`` is zero-variance in this export (no 9203 rows) — drop it
  before fitting.
- ``gender``, ``age_band``, ``race``, ``ethnicity``, and ``state`` are
  nominal — one-hot encode them; ``age_band`` is ordered but its labels
  should not be treated as ordinal distances.
- All count/flag columns are numeric and usable as-is.
- **Observation months** are summed over ``observation_period`` rows as
  rounded calendar months; members without a period get 0.
- **Exposure-time confounding:** count features scale with observation time
  (``observation_months`` correlates ~0.47 with ``total_visits``, and
  members with no observation period get all-zero counts) — use rate
  features (per observation month) or include ``observation_months`` as a
  control.

Memory: big tables are read with ``pd.read_csv(..., usecols=[...])`` one
table at a time (never all 18 tables at once), which keeps the 100k build
at ~1 GB peak.
"""

from pathlib import Path

import numpy as np
import pandas as pd

try:  # imported as a top-level module (src/ on sys.path)
    from clinical_codes import CHRONIC_CONDITIONS, icd9_prefixes
    from load_synpuf import DATA_DIR, SAMPLES
except ImportError:  # imported as a package module (src/ on sys.path)
    from src.clinical_codes import CHRONIC_CONDITIONS, icd9_prefixes
    from src.load_synpuf import DATA_DIR, SAMPLES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT = PROCESSED_DIR / "member_features.csv"

# OMOP concept IDs (no vocabulary table ships with SynPUF).
GENDER_LABELS = {8507: "Male", 8532: "Female"}
VISIT_TYPE_CONCEPTS = {"inpatient_visits": 9201, "outpatient_visits": 9202, "er_visits": 9203}

# Anchor age to the start of the 2008-2010 observation window (as in the EDA).
REF_YEAR = 2008
AGE_BANDS = ["<65", "65-74", "75-84", "85+"]
AGE_BIN_EDGES = [-np.inf, 64, 74, 84, np.inf]

# Columns that are counts/flags and must default to 0 for members with no rows.
_ZERO_FILL_COLUMNS = [
    "inpatient_visits",
    "outpatient_visits",
    "er_visits",
    "total_visits",
    "distinct_care_sites",
    "observation_months",
    *CHRONIC_CONDITIONS,
    "drug_exposures",
    "distinct_drugs",
    "drug_eras",
    "distinct_conditions",
    "distinct_procedures",
]

FEATURE_COLUMNS = [
    "person_id",
    "age",
    "age_band",
    "gender",
    "race",
    "ethnicity",
    "state",
    "inpatient_visits",
    "outpatient_visits",
    "er_visits",
    "total_visits",
    "distinct_care_sites",
    "observation_months",
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

# usecols per table: person_id plus exactly the columns the features need.
_TABLE_USECOLS = {
    "person": ["person_id", "gender_concept_id", "year_of_birth",
               "race_source_value", "ethnicity_source_value", "location_id"],
    "location": ["location_id", "state"],
    "observation_period": ["person_id", "observation_period_start_date",
                           "observation_period_end_date"],
    "visit_occurrence": ["person_id", "visit_concept_id", "care_site_id"],
    "condition_occurrence": ["person_id", "condition_source_value"],
    "drug_exposure": ["person_id", "drug_concept_id"],
    "drug_era": ["person_id"],
    "procedure_occurrence": ["person_id", "procedure_source_value"],
}


def _table_path(sample: str, table: str) -> Path:
    cfg = SAMPLES[sample]
    stem = f"{cfg['prefix']}{table.upper()}" if cfg["prefix"] else table
    ext = "gz" if cfg["compression"] == "gzip" else cfg["compression"]
    return DATA_DIR / cfg["dir"] / f"{stem}.csv.{ext}"


def _read_table(sample: str, table: str) -> pd.DataFrame:
    """Read one table with only the columns the features need."""
    path = _table_path(sample, table)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {sample} table: {path}")
    usecols = _TABLE_USECOLS[table]
    cfg = SAMPLES[sample]
    if cfg["prefix"]:  # 1k files use uppercase headers
        usecols = [c.upper() for c in usecols]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    return df


def _demographics(sample: str) -> pd.DataFrame:
    person = _read_table(sample, "person")
    location = _read_table(sample, "location")
    state = person["location_id"].map(location.set_index("location_id")["state"])

    demo = pd.DataFrame({"person_id": person["person_id"]})
    demo["age"] = (REF_YEAR - person["year_of_birth"]).astype("int64")
    demo["age_band"] = pd.cut(
        demo["age"], bins=AGE_BIN_EDGES, labels=AGE_BANDS, right=True
    ).astype(str)
    demo["gender"] = person["gender_concept_id"].map(GENDER_LABELS).fillna("Unknown")
    demo["race"] = person["race_source_value"]
    demo["ethnicity"] = person["ethnicity_source_value"]
    demo["state"] = state.fillna("Missing")
    return demo.set_index("person_id")


def _utilization(sample: str) -> pd.DataFrame:
    visits = _read_table(sample, "visit_occurrence")
    g = visits.groupby("person_id")
    util = pd.DataFrame({
        "total_visits": g.size(),
        "distinct_care_sites": g["care_site_id"].nunique(),
    })
    typed = visits[visits["visit_concept_id"].isin(set(VISIT_TYPE_CONCEPTS.values()))]
    counts = (
        typed.groupby(["person_id", "visit_concept_id"]).size().unstack(fill_value=0)
    )
    for col, concept_id in VISIT_TYPE_CONCEPTS.items():
        util[col] = counts[concept_id] if concept_id in counts else 0
    return util


def _observation_months(sample: str) -> pd.Series:
    period = _read_table(sample, "observation_period")
    start = pd.to_datetime(period["observation_period_start_date"], errors="coerce")
    end = pd.to_datetime(period["observation_period_end_date"], errors="coerce")
    days = (end - start).dt.days.clip(lower=0)
    months = (days / (365.25 / 12)).round().astype("int64")
    return months.groupby(period["person_id"]).sum().rename("observation_months")


def _condition_features(cond: pd.DataFrame) -> pd.DataFrame:
    """Chronic flags + distinct-condition count from one condition read."""
    prefix = icd9_prefixes(cond["condition_source_value"])
    feats = pd.DataFrame({
        "distinct_conditions": cond.groupby("person_id")["condition_source_value"].nunique()
    })
    for name, (lo, hi) in CHRONIC_CONDITIONS.items():
        hits = prefix.between(lo, hi)
        feats[name] = hits.groupby(cond["person_id"]).any().astype("int64")
    return feats


def _interaction_breadth(sample: str) -> pd.Series:
    proc = _read_table(sample, "procedure_occurrence")
    return (
        proc.groupby("person_id")["procedure_source_value"].nunique()
        .rename("distinct_procedures")
    )


def _pharmacy(sample: str) -> pd.DataFrame:
    exposure = _read_table(sample, "drug_exposure")
    era = _read_table(sample, "drug_era")
    pharm = pd.DataFrame({
        "drug_exposures": exposure.groupby("person_id").size(),
        "distinct_drugs": exposure.groupby("person_id")["drug_concept_id"].nunique(),
        "drug_eras": era.groupby("person_id").size(),
    })
    return pharm


def build_member_features(sample: str) -> pd.DataFrame:
    """Build the full member-level feature table for ``sample`` ("1k"/"100k").

    Returns one row per person in the person table, every column populated
    (0 for counts/flags where the member has no such records).
    """
    if sample not in SAMPLES:
        raise ValueError(f"Unknown sample {sample!r}; expected one of {sorted(SAMPLES)}")
    features = _demographics(sample)
    cond = _read_table(sample, "condition_occurrence")
    features = features.join(
        [_utilization(sample), _observation_months(sample), _condition_features(cond),
         _pharmacy(sample), _interaction_breadth(sample)]
    )
    features = features.fillna({c: 0 for c in _ZERO_FILL_COLUMNS})
    for c in _ZERO_FILL_COLUMNS:
        features[c] = features[c].astype("int64")
    features = features.reset_index()[FEATURE_COLUMNS]
    return features


def _prevalence_report(features: pd.DataFrame) -> pd.Series:
    any_chronic = features[list(CHRONIC_CONDITIONS)].any(axis=1)
    return pd.Series(
        {**{f: features[f].mean() for f in CHRONIC_CONDITIONS},
         "any_chronic": any_chronic.mean()}
    ).rename("prevalence")


def build_and_write(sample: str, output_path: Path) -> pd.DataFrame:
    features = build_member_features(sample)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    print(f"[{sample}] wrote {len(features):,} rows x {features.shape[1]} cols -> {output_path}")
    return features


def main() -> None:
    features_100k = build_and_write("100k", DEFAULT_OUTPUT)
    features_1k = build_and_write("1k", PROCESSED_DIR / "member_features_1k.csv")

    print("\n--- 100k integrity checks ---")
    n_persons = len(_read_table("100k", "person"))
    print(f"rows={len(features_100k):,} distinct_person_id={features_100k['person_id'].nunique():,} "
          f"person_table_rows={n_persons:,}")
    nulls = features_100k.isna().sum()
    print(f"null cells total: {int(nulls.sum())}")
    if nulls.sum():
        print(nulls[nulls > 0])

    print("\n--- chronic-flag prevalence (1k vs 100k) ---")
    prev = pd.concat([_prevalence_report(features_1k), _prevalence_report(features_100k)],
                     axis=1)
    prev.columns = ["synpuf_1k", "synpuf_100k"]
    print((prev * 100).round(1).to_string())


if __name__ == "__main__":
    main()
