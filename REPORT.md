# Analysis Summary — CMS DE-SynPUF (OMOP CDM): EDA & Member Feature Table

**Project:** member-level utilization-risk classification (Low / Medium / High) — see `Team1_RFC.docx`. This report covers everything up to and including the modeling-ready member feature table: where the data came from, what it contains, what the EDA found, what the feature table's columns mean, and the caveats that constrain how far the OMOP export can satisfy the RFC. Modeling itself (target construction, training) is a later phase.

## 1. Data source and how it was obtained

We use the **CMS DE-SynPUF** (Designated Entity Synthetic Public Use File) in its **OMOP CDM v5.x-conformant release** prepared by OHDSI, served as plain CSV exports from a public AWS S3 bucket (region **us-east-1**, anonymous access — no AWS account or credentials needed):

```bash
aws s3 sync --no-sign-request s3://synpuf-omop/cmsdesynpuf1k/   data/raw/synpuf1k/
aws s3 sync --no-sign-request s3://synpuf-omop/cmsdesynpuf100k/ data/raw/synpuf100k/
```

(Bucket ARN: `arn:aws:s3:::synpuf-omop`. Total download ~1 GB; full setup instructions, including the venv and Jupyter kernel, are in `README.md`.)

Two samples ship in the bucket:

- **1k sample** (`data/raw/synpuf1k/`) — 1,000 persons, files `CDM_<TABLE>.csv.bz2` (bz2-compressed, uppercase headers), plus versioned duplicate files (`...5.2.2...`) that the loader skips.
- **100k sample** (`data/raw/synpuf100k/`) — 100,000 persons, files `<table>.csv.gz` (gzip-compressed, lowercase headers). This is the full working set.

**Synthetic-data caveat.** DE-SynPUF is fully synthetic: CMS built it by sampling from real Medicare fee-for-service claims and stripping all beneficiary identity, preserving structural/statistical properties but not individual-level truth. Per CMS it has limited inferential value. Every association reported below is therefore a methodological exercise on synthetic records, not a clinical finding.

## 2. Schema and lineage

The export follows the **OMOP Common Data Model v5.x**: each person's records are spread across normalized clinical tables joined on `person_id`. Tables used in this project (per-capita densities at 100k):

| Table | 100k rows | Role in the project |
|---|---:|---|
| `person` | 100,000 | Demographics: birth year, gender concept, race/ethnicity source codes, location |
| `location` | 3,185 | `state` per `location_id` |
| `observation_period` | 90,217 | Observation window start/end per member (2008–2010) |
| `visit_occurrence` | 4,785,453 | Utilization backbone: visit rows, `visit_concept_id` typing, care sites |
| `condition_occurrence` | 12,695,221 | Diagnosis layer: raw ICD-9 `condition_source_value` |
| `drug_exposure` / `drug_era` | 5,423,469 / 5,365,324 | Pharmacy layer (`drug_concept_id` is the reliable grouping key) |
| `procedure_occurrence` | 11,897,410 | Procedure breadth |
| `death` | 4,635 (4.6%) | Mortality (under-reported, weak signal) |

**Lineage:** raw CSVs → `src/load_synpuf.py` (loads + validates both samples) → `notebooks/01_eda_synpuf.ipynb` (EDA, 1k primary with streaming 100k headline checks) → `src/build_features.py` → `data/processed/member_features.csv` (100k) and `member_features_1k.csv` (1k) → `notebooks/02_member_feature_profile.ipynb` (this profile). `src/clinical_codes.py` holds the canonical ICD-9 prefix logic shared by the EDA and the feature builder.

**Structural data-quality notes:** no concept/vocabulary table ships with the export, so all categorical work uses `*_source_value` columns and hard-coded concept IDs (gender 8507/8532, visits 9201/9202/9203); many operational OMOP columns (`stop_reason`, `lot_number`, `sig`, address fields) are always empty and were excluded from features; the analytic columns are ~fully populated.

## 3. Headline EDA findings (notebook 01)

- **Population.** Elderly Medicare-like cohort: mean age ~72 at the start of the 2008–2010 window, balanced gender (100k: 55.6% female / 44.4% male), 82.8% White. Observation coverage is nearly uniform (~36 months) with a short-tail minority entering late or exiting early.
- **Utilization is count-heavy, right-skewed, and polarized.** ~47–48 visit rows/member (median 41) with a long high-utilization tail and ~15% of members with **zero** visit rows; condition (~127 rows/member at 100k), drug (~54 exposures), and procedure (~119) layers show the same skew. This polarization maps directly onto the RFC's Low/Medium/High tiers.
- **Visit typing is sparse.** ~85% of visit rows carry `visit_concept_id = 0` (unmapped); among typed rows outpatient dwarfs inpatient, and this export contains **zero ER (9203) rows**. Typed counts therefore capture only ~15% of a member's visit activity — *all-row counts* are the robust signal.
- **Chronic disease dominates the diagnosis layer.** Top ICD-9 codes are the classic elderly cluster (hypertension 4019/4011, diabetes 25000, hyperlipidemia 2724, atrial fibrillation 42731, ischemic heart disease 41401). Member-level prevalence of the RFC chronic families is high (diabetes ~70%, cancer ~51%, CHF/COPD/stroke/renal ~45–48% each; ~77% carry at least one). **Claim-line duplication caveat:** DE-SynPUF repeats diagnoses across claim lines, inflating row-level prevalence; member-level ANY-match flags are immune to duplication but remain utilization-flavored signals, not clinical truth.
- **Drug utilization is broad:** ~54 exposure events/member spanning ~48 distinct drug concepts, with drug eras tracking exposures closely.
- **1k vs 100k representativeness.** The 1k sample matches 100k almost exactly on visit density (47.5 vs 47.9 rows/member), visit-type mix, top ICD-9 codes, and chronic prevalence. The one deviation: 1k carries ~160 condition rows/member vs ~127 at 100k (COPD/cancer ~6–12 points higher) — the 1k sample slightly overstates diagnosis-layer density. Final modeling/validation should run at 100k scale.

## 4. Feature-table dictionary (`data/processed/member_features.csv`)

100,000 rows × 24 columns, one row per `person_id`, zero null cells. Built by `src/build_features.py`; the definitions below are the *Modeling contract* the modeling phase must follow.

**Row key**

| Column | Type | Meaning |
|---|---|---|
| `person_id` | int | Row key — **exclude before fitting** |

**Demographics**

| Column | Type | Meaning |
|---|---|---|
| `age` | int | Age at start of the 2008 observation window (`2008 − year_of_birth`) |
| `age_band` | str | `age` binned: `<65`, `65-74`, `75-84`, `85+` (ordered labels, not ordinal distances — one-hot) |
| `gender` | str | `Male`/`Female` from `gender_concept_id` 8507/8532 |
| `race` | str | Raw DE-SynPUF code (1=White, 2=Black, 3=Other, 4=Asian, 5=Hispanic, 6=Native American); 4/6 never occur in this export; no vocabulary table to resolve further |
| `ethnicity` | str | Raw code — **byte-identical to `race` for every member in this export**; redundant, consider dropping |
| `state` | str | USPS abbreviation from the location table; includes raw code `"54"` (1,456 members) that maps to no state; `"Missing"` if no location |

**Utilization** (OMOP `visit_occurrence`)

| Column | Type | Meaning |
|---|---|---|
| `inpatient_visits` | int | Rows with `visit_concept_id = 9201` |
| `outpatient_visits` | int | Rows with `visit_concept_id = 9202` |
| `er_visits` | int | Rows with `visit_concept_id = 9203` — **zero-variance (all 0); drop before fitting** |
| `total_visits` | int | **All** visit rows (typed + untyped) — the robust utilization signal |
| `distinct_care_sites` | int | Distinct `care_site_id` across visit rows |
| `observation_months` | int | Sum of observation-period lengths, rounded calendar months; 0 if no period |

**Chronic-condition flags** (member-level ANY-match on the 3-digit ICD-9 prefix of `condition_source_value`; definitions in `src/clinical_codes.py`, 0/1)

| Column | ICD-9 range | Meaning |
|---|---|---|
| `diabetes` | 250.x | Diabetes mellitus |
| `chf` | 428.x | Congestive heart failure |
| `copd` | 491–496 | Chronic obstructive pulmonary disease |
| `stroke` | 430–438 | Cerebrovascular disease |
| `cancer` | 140–209 | Malignant neoplasms |
| `renal_disease` | 580–589 | Chronic kidney disease / nephritis |

**Pharmacy** (OMOP `drug_exposure` / `drug_era`)

| Column | Type | Meaning |
|---|---|---|
| `drug_exposures` | int | Drug-exposure event rows |
| `distinct_drugs` | int | Distinct `drug_concept_id` (reliable grouping key; `drug_source_value` mixes NDC and HCPCS) |
| `drug_eras` | int | `drug_era` rows |

**Interaction breadth**

| Column | Type | Meaning |
|---|---|---|
| `distinct_conditions` | int | Distinct `condition_source_value` codes (immune to claim-line duplication) |
| `distinct_procedures` | int | Distinct `procedure_source_value` codes |

All count/flag columns are numeric and usable as-is; the five nominal columns (`gender`, `age_band`, `race`, `ethnicity`, `state`) need one-hot encoding. Members with no records in a domain get 0 for that domain's counts/flags.

## 5. OMOP-export vs RFC caveats

The RFC assumes claims-style features; the OMOP conversion supports most, but not all, of them:

1. **No spending/cost columns.** The OMOP export carries no charge/payment fields, so the RFC's "previous spending" feature cannot be built directly. **Utilization intensity** (visit/drug/procedure counts) is its proxy — this is the deepest departure from the RFC and should be acknowledged in any modeling write-up.
2. **Chronic conditions are derived, not pre-computed.** We built 0/1 flags from ICD-9 ranges on `condition_source_value` (the EDA's five RFC families plus renal disease) rather than using any pre-computed condition flags. Claim-line duplication inflates the underlying rows; the flags are utilization-flavored.
3. **Sparse/zero visit typing.** ~85% of visit rows are untyped and there are no ER rows, so `er_visits` is zero-variance and typed visit counts undercount activity; `total_visits` is the dependable feature.
4. **Demographic encoding quirks.** `ethnicity` duplicates `race` exactly; `race` uses raw numeric codes (no vocabulary); `state` mixes USPS codes with an unresolvable `"54"` code (1,456 members) and a `Missing` bucket.
5. **Synthetic data.** All findings are methodological exercises on synthetic records with limited inferential value.

## 6. Reproducibility

Environment setup and data download are documented in `README.md` (venv → `requirements.txt`, kernel `synpuf-venv`, the two `aws s3 sync` commands). The pipeline order is:

```bash
.venv/bin/python src/load_synpuf.py       # 1. load + validate raw tables (1k & 100k)
.venv/bin/python src/build_features.py    # 2. build member_features.csv (100k + 1k check)
# 3. notebooks/01_eda_synpuf.ipynb        # EDA (execute in place)
# 4. notebooks/02_member_feature_profile.ipynb  # feature profile (execute in place)
```

### End-to-end verification evidence (run 2026-09-14, branch `synpuf-omop-analysis`)

1. **Byte-identical rebuild.** `shasum -a 256 data/processed/member_features.csv` before and after deleting the CSV and re-running `.venv/bin/python src/build_features.py`:
   - before: `28de31da8e9381c69a2b2f463ed7f3fe127305d1b96e5b2af31836fbce397bc6`
   - after:  `28de31da8e9381c69a2b2f463ed7f3fe127305d1b96e5b2af31836fbce397bc6`
   - `cmp` confirms the regenerated file is byte-identical; git shows no CSV change.
   - The builder's own integrity checks passed: `rows=100,000 distinct_person_id=100,000 person_table_rows=100,000`, `null cells total: 0`.
2. **Loader validation.** `.venv/bin/python src/load_synpuf.py` exited 0: *"All checks passed for both datasets."*
3. **Notebook 02 re-execution.** `.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace notebooks/02_member_feature_profile.ipynb` completed successfully (~7 s), so the profile is reproducible end-to-end from raw data.
4. **Artifact invariants** (also asserted in notebook 02, §1): feature-table row count (100,000) == distinct `person_id` count == `person` table rows; every column fully populated (zero null cells).
