import unittest

from incontext_collapse.evaluation import mean_t_ci, summarize_records
from incontext_collapse.scoring import (parse_answer, shortcut_diagnostics,
                                        subtract_contextual_bias)


class ScoringTests(unittest.TestCase):
    def test_exact_and_verbose_single_labels_parse(self):
        labels = ["cat", "bird", "car"]
        self.assertEqual(parse_answer("cat", labels).status, "exact")
        result = parse_answer("The answer is bird.", labels)
        self.assertEqual(result.prediction, 1)
        self.assertEqual(result.status, "single_label_verbose")

    def test_option_echo_and_ambiguity_are_not_class_zero(self):
        labels = ["cat", "bird", "car"]
        echo = parse_answer("cat, bird, car", labels)
        self.assertIsNone(echo.prediction)
        self.assertEqual(echo.status, "option_echo")
        ambiguous = parse_answer("cat or bird", labels)
        self.assertIsNone(ambiguous.prediction)
        self.assertEqual(ambiguous.status, "ambiguous_labels")

    def test_label_matching_uses_word_boundaries(self):
        result = parse_answer("concatenate", ["cat", "dog"])
        self.assertIsNone(result.prediction)

    def test_shortcuts_are_reported_separately(self):
        result = shortcut_diagnostics(2, [1, 2, 2, 0])
        self.assertFalse(result["copies_first"])
        self.assertFalse(result["copies_last"])
        self.assertTrue(result["copies_any_demo"])
        self.assertTrue(result["copies_majority"])

    def test_contextual_calibration_subtracts_log_bias(self):
        calibrated = subtract_contextual_bias([-2.0, -1.5, -3.0], [-4.0, -1.0, -2.0])
        self.assertEqual(calibrated, [2.0, -0.5, -1.0])
        with self.assertRaises(ValueError):
            subtract_contextual_bias([1.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            subtract_contextual_bias([], [])

    def test_student_t_interval_uses_small_sample_critical_value(self):
        interval = mean_t_ci([0.2, 0.4, 0.6])
        self.assertAlmostEqual(interval["mean"], 0.4)
        self.assertGreater(interval["ci95_high"] - interval["mean"], 0.4)

    def test_summary_counts_unparseable_outputs(self):
        records = [{
            "correct": False, "unparseable": True, "copies_first": False,
            "copies_last": False, "copies_any_demo": False,
            "copies_majority": False, "parse_status": "option_echo",
            "gold": 1, "generation_prediction": None,
            "likelihood_prediction": 1, "likelihood_sum_prediction": 0,
            "label_space": ["a", "b"],
            "raw_output": "a, b", "likelihood_scores": [-2.0, -1.0],
            "follows_corrupted_mapping": False,
            "generation_follows_corrupted_mapping": False,
            "likelihood_follows_corrupted_mapping": True,
        }]
        summary = summarize_records(records)
        self.assertEqual(summary["unparseable_rate"], 1.0)
        self.assertEqual(summary["parse_status_counts"], {"option_echo": 1})
        self.assertEqual(summary["generation_accuracy"], 0.0)
        self.assertEqual(summary["likelihood_accuracy"], 1.0)
        self.assertEqual(summary["likelihood_sum_accuracy"], 0.0)
        self.assertEqual(summary["likelihood_follows_corrupted_mapping_rate"], 1.0)
        self.assertEqual(summary["likelihood_confusion"], [[0, 0], [0, 1]])


if __name__ == "__main__":
    unittest.main()
