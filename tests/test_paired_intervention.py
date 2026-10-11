import random
import unittest

from incontext_collapse.paired_intervention import (
    make_paired_interventions,
    summarize_paired_records,
)


class PairedInterventionTests(unittest.TestCase):
    def setUp(self):
        self.demos = [("a", 0), ("b", 1), ("c", 2), ("d", 0)]

    def test_variants_preserve_images_and_label_multiset(self):
        variants, metadata = make_paired_interventions(self.demos, random.Random(7))
        original_images = sorted(image for image, _ in self.demos)
        original_labels = sorted(label for _, label in self.demos)
        for demos in variants.values():
            self.assertEqual(sorted(image for image, _ in demos), original_images)
            self.assertEqual(sorted(label for _, label in demos), original_labels)
        self.assertNotEqual(metadata["old_last_class"], metadata["new_last_class"])
        self.assertEqual(variants["order_swap"][-1][1], metadata["new_last_class"])
        self.assertEqual(variants["label_swap"][-1][1], metadata["new_last_class"])

    def test_order_swap_preserves_pairs_and_label_swap_preserves_image_order(self):
        variants, _ = make_paired_interventions(self.demos, random.Random(3))
        self.assertEqual(sorted(variants["order_swap"]), sorted(self.demos))
        self.assertEqual(
            [image for image, _ in variants["label_swap"]],
            [image for image, _ in self.demos],
        )

    def test_rejects_prompts_without_two_classes(self):
        with self.assertRaises(ValueError):
            make_paired_interventions([("a", 0), ("b", 0)], random.Random(0))

    def test_summary_uses_paired_denominators(self):
        records = [
            {
                "gold": 0,
                "new_last_class": 1,
                "predictions": {"original": 0, "order_swap": 1, "label_swap": 1},
                "last_classes": {"original": 2, "order_swap": 1, "label_swap": 1},
            },
            {
                "gold": 2,
                "new_last_class": 1,
                "predictions": {"original": 2, "order_swap": 2, "label_swap": None},
                "last_classes": {"original": 0, "order_swap": 1, "label_swap": 1},
            },
        ]
        summary = summarize_paired_records(records)
        order = summary["effects"]["order_swap"]
        labels = summary["effects"]["label_swap"]
        self.assertEqual(order["n_comparable"], 2)
        self.assertEqual(order["prediction_change_rate"], 0.5)
        self.assertEqual(order["moves_to_new_last_given_change"], 1.0)
        self.assertEqual(order["gold_to_new_last_flip_rate"], 0.5)
        self.assertEqual(labels["n_comparable"], 1)
        self.assertEqual(summary["variants"]["label_swap"]["unparseable_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
