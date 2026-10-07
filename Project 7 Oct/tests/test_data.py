import tempfile
import unittest
from pathlib import Path

import pandas as pd

from utilization_risk.config import EVENT_TABLES
from utilization_risk.data import build_feature_table


class FeatureAggregationTests(unittest.TestCase):
    def test_uses_only_pre_index_events_as_features(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "person").mkdir()
            (root / "observation_period").mkdir()
            pd.DataFrame(
                {
                    "person_id": [1, 2],
                    "year_of_birth": [1980, 1990],
                    "gender_concept_id": [1, 2],
                    "race_concept_id": [3, 4],
                    "ethnicity_concept_id": [5, 6],
                }
            ).to_csv(root / "person" / "person.csv", index=False)
            pd.DataFrame(
                {
                    "person_id": [1, 2],
                    "observation_period_start_date": ["2020-01-01", "2020-01-01"],
                    "observation_period_end_date": ["2022-01-01", "2022-01-01"],
                }
            ).to_csv(
                root / "observation_period" / "observation_period.csv", index=False
            )

            for _, (table_name, date_column) in EVENT_TABLES.items():
                (root / table_name).mkdir()
                rows = [
                    {"person_id": 1, date_column: "2020-03-01"},
                    {"person_id": 1, date_column: "2021-11-01"},
                ]
                if table_name == "visit_occurrence":
                    rows.append({"person_id": 1, date_column: "2021-12-01"})
                pd.DataFrame(rows).to_csv(
                    root / table_name / f"{table_name}.csv", index=False
                )

            (root / "payer_plan_period").mkdir()
            pd.DataFrame(
                {
                    "person_id": [1],
                    "payer_plan_period_start_date": ["2020-01-01"],
                    "payer_plan_period_end_date": ["2020-12-31"],
                }
            ).to_csv(root / "payer_plan_period" / "payer_plan_period.csv", index=False)

            features, annualized_visits = build_feature_table(root, chunksize=1)

        self.assertEqual(features.loc[1, "visits_baseline_count"], 1)
        self.assertEqual(features.loc[1, "conditions_baseline_count"], 1)
        self.assertEqual(features.loc[2, "visits_baseline_count"], 0)
        self.assertIn("baseline_years", features.columns)
        self.assertNotIn("observation_days", features.columns)
        self.assertNotIn("followup_years", features.columns)
        self.assertGreater(annualized_visits.loc[1], 0)
        self.assertEqual(annualized_visits.loc[2], 0)
        self.assertNotIn("future_annualized_visits", features.columns)


if __name__ == "__main__":
    unittest.main()
