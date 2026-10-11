import unittest

from PIL import Image

from incontext_collapse.protocol import image_id
from incontext_collapse.vqa_diverse import (
    paired_image_interval, select_image_disjoint, validate_training_partition)


class DiverseVQATests(unittest.TestCase):
    def setUp(self):
        self.images = [Image.new("RGB", (4, 4), (index, 0, 0)) for index in range(20)]
        self.items = [(image, f"question {question}", ["yes"] * 10)
                      for image in self.images for question in range(3)]

    def test_image_disjoint_sampling_is_deterministic(self):
        pool, test, metadata = select_image_disjoint(self.items, 6, 12, 8, 11)
        _, repeated, _ = select_image_disjoint(self.items, 6, 12, 8, 11)
        keys = [image_id(item[0]) for item in test]
        self.assertEqual(len(set(keys)), 8)
        self.assertEqual(keys, [image_id(item[0]) for item in repeated])
        self.assertFalse(set(keys).intersection(metadata["excluded_image_ids"]))
        self.assertEqual(len(pool), 6)
        self.assertEqual(metadata["n_training_images"], 2)
        self.assertEqual(metadata["n_eligible_images"], 16)

    def test_later_repeated_training_image_is_excluded(self):
        repeated = [*self.items, (self.images[0], "different question", ["yes"])]
        _, test, metadata = select_image_disjoint(repeated, 6, 12, 16, 7)
        self.assertEqual(metadata["n_eligible_images"], 16)
        self.assertNotIn(image_id(self.images[0]), [image_id(item[0]) for item in test])

    def test_insufficient_unique_images_fails(self):
        with self.assertRaises(ValueError):
            select_image_disjoint(self.items, 6, 12, 17, 7)

    def test_actual_training_prefix_cannot_be_shortened(self):
        training = {"data_partition": "prefix_train_then_prefix_test_v2", "n_train": 96,
                    "n": 40, "dataset": "vqav2", "dataset_revision": "pinned"}
        data = {"n_pool": 96, "exclude_prefix": 136, "dataset": "vqav2",
                "dataset_revision": "pinned"}
        validate_training_partition(training, data)
        with self.assertRaises(ValueError):
            validate_training_partition(training, {**data, "n_pool": 32})
        with self.assertRaises(ValueError):
            validate_training_partition(training, {**data, "exclude_prefix": 96})

    def test_paired_bootstrap_and_pairing_checks(self):
        def cell(scores, demos=None):
            return {"records": [{"query_id": f"image{i}:question", "score": score,
                                 "demo_ids": demos or []} for i, score in enumerate(scores)]}
        result = paired_image_interval([cell([0, 0])], [cell([1, 1])], draws=100)
        self.assertEqual(result["mean"], 1)
        self.assertEqual(result["ci95_low"], 1)
        self.assertEqual(result["n_images"], 2)
        with self.assertRaises(ValueError):
            paired_image_interval([cell([0, 0])], [cell([1, 1], ["other"])])
        with self.assertRaises(ValueError):
            paired_image_interval([cell([0])], [cell([1, 1])])


if __name__ == "__main__":
    unittest.main()
