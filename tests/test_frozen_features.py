import unittest

import torch

from incontext_collapse.frozen_features import nearest_demo_label


class FrozenFeatureTests(unittest.TestCase):
    def test_nearest_demo_uses_cosine_similarity(self):
        prediction = nearest_demo_label(
            torch.tensor([2.0, 0.1]),
            [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 4.0])],
            [3, 7],
        )
        self.assertEqual(prediction, 3)

    def test_nearest_demo_validates_inputs(self):
        with self.assertRaises(ValueError):
            nearest_demo_label(torch.ones(2), [], [])
        with self.assertRaises(ValueError):
            nearest_demo_label(torch.ones(2), [torch.ones(2)], [])


if __name__ == "__main__":
    unittest.main()
