import unittest

import numpy as np
import pandas as pd

from utilization_risk.config import TARGET
from utilization_risk.modeling import (
    LabelEncodingClassifier,
    candidate_models,
    make_partitions,
    make_preprocessor,
)
from xgboost import XGBClassifier


class ModelingTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.Index(range(1, 101), name="person_id")
        self.features = pd.DataFrame(
            {
                "age_at_index": np.arange(100, dtype=float),
                "visit_baseline_count": np.arange(100),
                "gender_concept_id": pd.Series(
                    ["1", "2"] * 50, index=self.index, dtype="string"
                ),
            },
            index=self.index,
        )
        self.rates = pd.Series(np.arange(100, dtype=float), index=self.index)

    def test_splits_are_disjoint_and_match_requested_proportions(self):
        parts = make_partitions(self.features, self.rates, random_state=7)
        ids = [
            set(parts.train.index),
            set(parts.validation.index),
            set(parts.test.index),
            set(parts.production.index),
        ]
        self.assertEqual([len(part) for part in ids], [40, 10, 10, 40])
        self.assertFalse(ids[0] & ids[1])
        self.assertFalse(ids[0] & ids[2])
        self.assertFalse(ids[0] & ids[3])
        self.assertFalse(ids[1] & ids[2])
        self.assertFalse(ids[1] & ids[3])
        self.assertFalse(ids[2] & ids[3])
        self.assertNotIn(TARGET, parts.production.columns)
        self.assertEqual(set(parts.train[TARGET]), {"Low", "Medium", "High"})

    def test_preprocessor_standardizes_numeric_and_encodes_categorical(self):
        transformer = make_preprocessor(self.features)
        transformed = transformer.fit_transform(self.features)
        self.assertEqual(transformed.shape[0], len(self.features))
        self.assertGreater(transformed.shape[1], self.features.shape[1])

    def test_candidate_models_include_requested_estimators(self):
        models = candidate_models()
        self.assertIn("decision_tree", models)
        self.assertIn("xgboost", models)

    def test_xgboost_wrapper_encodes_and_restores_string_classes(self):
        classifier = LabelEncodingClassifier(
            XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                n_estimators=3,
                n_jobs=1,
            )
        )
        features = pd.DataFrame({"value": [0, 1, 2, 3, 4, 5]})
        target = pd.Series(["Low", "Medium", "High", "Low", "Medium", "High"])
        classifier.fit(features, target)
        predictions = classifier.predict(features)
        self.assertEqual(set(classifier.classes_), {"Low", "Medium", "High"})
        self.assertLessEqual(set(predictions), {"Low", "Medium", "High"})
        self.assertEqual(classifier.predict_proba(features).shape, (6, 3))


if __name__ == "__main__":
    unittest.main()
