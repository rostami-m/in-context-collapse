import random
import unittest
from collections import Counter

from incontext_collapse.protocol import (
    ProtocolConfig,
    corrupt_demo_labels,
    sample_balanced_demos,
    sample_label_space,
    sample_balanced_queries,
    seed_training,
    sample_training_episode,
)
from incontext_collapse.core import task_shapes4


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.pool = {c: [f"{c}-{i}" for i in range(8)] for c in range(4)}

    def test_k_below_class_count_is_not_prefix_biased(self):
        seen = Counter()
        for seed in range(200):
            demos = sample_balanced_demos(self.pool, 1, random.Random(seed))
            seen[demos[0][1]] += 1
        self.assertEqual(set(seen), set(self.pool))
        self.assertLess(max(seen.values()) - min(seen.values()), 30)

    def test_counts_are_balanced_and_images_do_not_repeat(self):
        demos = sample_balanced_demos(self.pool, 7, random.Random(4))
        counts = Counter(label for _, label in demos)
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        self.assertEqual(len({image for image, _ in demos}), len(demos))

    def test_legacy_policy_reproduces_prefix_classes(self):
        demos = sample_balanced_demos(
            self.pool, 2, random.Random(3), class_policy="legacy_cycle")
        self.assertEqual({label for _, label in demos}, {0, 1})

    def test_episode_remap_changes_and_train_eval_tokens_are_disjoint(self):
        names = ["a", "b", "c", "d"]
        first = sample_label_space(names, "remap", random.Random(1), "episode", "eval")
        second = sample_label_space(names, "remap", random.Random(2), "episode", "eval")
        train = sample_label_space(names, "remap", random.Random(1), "fixed", "train")
        self.assertNotEqual(first, second)
        self.assertTrue(set(first).isdisjoint(train))

    def test_shuffled_control_is_wrong_at_k_one(self):
        demos, mapping = corrupt_demo_labels([("image", 0)], range(4), random.Random(0))
        self.assertNotEqual(demos[0][1], 0)
        self.assertTrue(all(source != target for source, target in mapping.items()))

    def test_protocol_metadata_is_versioned(self):
        metadata = ProtocolConfig().metadata()
        self.assertIn("protocol_version", metadata)

    def test_shapes_test_set_does_not_depend_on_training_pool_size(self):
        _, _, small_train_test = task_shapes4(4, 8)
        _, _, large_train_test = task_shapes4(64, 8)
        small = [(image.tobytes(), label) for image, label in small_train_test]
        large = [(image.tobytes(), label) for image, label in large_train_test]
        self.assertEqual(small, large)

    def test_training_query_is_excluded_from_demonstrations(self):
        query, _, demos = sample_training_episode(self.pool, 12, random.Random(8))
        self.assertTrue(all(image is not query for image, _ in demos))

    def test_query_sample_is_exactly_balanced(self):
        items = [(f"{label}-{index}", label) for label in range(4) for index in range(20)]
        selected = sample_balanced_queries(items, 40, random.Random(5))
        self.assertEqual(Counter(label for _, label in selected), Counter({0: 10, 1: 10, 2: 10, 3: 10}))
        self.assertEqual(len({item for item, _ in selected}), 40)

    def test_training_seed_controls_python_and_torch(self):
        import torch

        seed_training(17)
        first = (random.random(), torch.rand(3))
        seed_training(17)
        second = (random.random(), torch.rand(3))
        self.assertEqual(first[0], second[0])
        self.assertTrue(torch.equal(first[1], second[1]))


if __name__ == "__main__":
    unittest.main()
