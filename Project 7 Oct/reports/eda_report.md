# Raw dataset EDA

## Scope

Inventoried 17 CSV files under `C:\Users\riyaz\OneDrive\Desktop\USD\Module9_MLOps\Week7 Project\Project 6 Oct\dataset` (58,911,691 rows total). Files were scanned in chunks.

These are OMOP-style tables. `person_id` is the member key in the person/event tables. There is no provided `Member_Utilization_Risk` target; the modeling pipeline creates a future-visit-rate proxy documented in the project README.

## Dataset inventory

| Table | Rows | Columns | Unique members | Mean missing fraction |
|---|---:|---:|---:|---:|
| care_site | 320,545 | 6 | n/a | 33.3% |
| condition_era | 9,788,262 | 6 | 84608 | 0.0% |
| condition_occurrence | 12,695,221 | 15 | 84608 | 13.4% |
| death | 4,635 | 7 | 4635 | 42.9% |
| device_exposure | 202,805 | 14 | 36431 | 21.5% |
| drug_era | 5,365,324 | 7 | 88243 | 14.3% |
| drug_exposure | 5,423,469 | 22 | 88706 | 46.0% |
| drug_strength | 2,430,381 | 12 | n/a | 41.0% |
| location | 3,185 | 8 | n/a | 50.0% |
| measurement | 2,909,229 | 18 | 79106 | 44.5% |
| observation | 1,655,312 | 17 | 79010 | 41.2% |
| observation_period | 90,217 | 5 | 90217 | 0.0% |
| payer_plan_period | 334,751 | 7 | 99931 | 28.6% |
| person | 100,000 | 18 | 100000 | 33.3% |
| procedure_occurrence | 11,897,410 | 13 | 84971 | 23.1% |
| provider | 905,492 | 13 | n/a | 38.5% |
| visit_occurrence | 4,785,453 | 17 | 85196 | 29.5% |

## EDA observations and merge recommendations

- Keep one row per member in the modeling table. Aggregate each event table by `person_id` before merging; direct event-to-event joins would create many-to-many row multiplication.
- Preserve members with no rows in a given event table via left joins and represent their event count as zero.
- Exclude the condition/drug era tables from counts when the corresponding occurrence/exposure tables are used, to avoid double counting. Provider/care-site/location/drug-strength are reference tables, not direct member-level features.
- Avoid dropping whole member rows for optional missing values. Use training-fitted imputers; remove malformed event dates/IDs only from the corresponding aggregate.
- Keep only data dated before each prediction index in features. Use later visits exclusively to build the proxy outcome.
- Baseline duration is measured only up to the prediction index; total observation-period duration and future follow-up length are not model features.

## Most-missing columns

| Table | Column | Missing values | Missing fraction |
|---|---|---:|---:|
| condition_occurrence | condition_end_datetime | 12,695,221 | 100.0% |
| condition_occurrence | stop_reason | 12,695,221 | 100.0% |
| procedure_occurrence | modifier_concept_id | 11,897,410 | 100.0% |
| procedure_occurrence | quantity | 11,897,410 | 100.0% |
| procedure_occurrence | qualifier_source_value | 11,897,410 | 100.0% |
| drug_exposure | verbatim_end_date | 5,423,469 | 100.0% |
| drug_exposure | stop_reason | 5,423,469 | 100.0% |
| drug_exposure | refills | 5,423,469 | 100.0% |
| drug_exposure | sig | 5,423,469 | 100.0% |
| drug_exposure | route_concept_id | 5,423,469 | 100.0% |
| drug_exposure | lot_number | 5,423,469 | 100.0% |
| drug_exposure | route_source_value | 5,423,469 | 100.0% |
| drug_exposure | dose_unit_source_value | 5,423,469 | 100.0% |
| drug_era | gap_days | 5,365,324 | 100.0% |
| visit_occurrence | visit_start_datetime | 4,785,453 | 100.0% |
| visit_occurrence | visit_end_datetime | 4,785,453 | 100.0% |
| visit_occurrence | visit_source_concept_id | 4,785,453 | 100.0% |
| visit_occurrence | admitting_source_value | 4,785,453 | 100.0% |
| visit_occurrence | discharge_to_source_value | 4,785,453 | 100.0% |
| measurement | measurement_datetime | 2,909,229 | 100.0% |
| measurement | operator_concept_id | 2,909,229 | 100.0% |
| measurement | value_as_number | 2,909,229 | 100.0% |
| measurement | unit_concept_id | 2,909,229 | 100.0% |
| measurement | range_low | 2,909,229 | 100.0% |
| measurement | range_high | 2,909,229 | 100.0% |

The complete table inventory is saved to `dataset_inventory.csv`; per-column null counts/rates are saved to `column_missingness.csv`.
Table-level mean missingness is diagnostic, not a reason to drop whole tables or member rows.
