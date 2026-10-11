import unittest

from incontext_collapse.vqa import partition_task_items, vqa_norm, vqa_score
from PIL import Image

from incontext_collapse.vqa_repair import vqa_query_id
from incontext_collapse.vqa_train import representative_answer


class VQATests(unittest.TestCase):
    def test_query_id_distinguishes_questions_on_same_image(self):
        image = Image.new("RGB", (2, 2), "white")
        self.assertNotEqual(
            vqa_query_id(image, "What is shown?"),
            vqa_query_id(image, "What color is it?"),
        )

    def test_answer_normalization(self):
        self.assertEqual(vqa_norm("The two!"), "2")
        self.assertEqual(vqa_norm("DON'T"), "don't")
        self.assertEqual(vqa_norm("1,000."), "1000")
        self.assertEqual(vqa_norm("3.14"), "3.14")

    def test_soft_and_single_answer_scores(self):
        self.assertEqual(vqa_score("cat", ["cat"]), 1.0)
        self.assertEqual(vqa_score("dog", ["cat"]), 0.0)
        self.assertEqual(vqa_score("cat", ["cat", "cat", "dog", "bird"]), 0.5)
        answers = ["cat"] * 3 + ["dog"] * 7
        self.assertAlmostEqual(vqa_score("cat", answers), 0.9)
        self.assertEqual(vqa_score("cat", ["cat"] * 4 + ["dog"] * 6), 1.0)

    def test_representative_answer_uses_normalized_majority(self):
        self.assertEqual(
            representative_answer(["Two", "2", "the two", "three"]), "2")
        self.assertEqual(representative_answer(["red", "blue"]), "red")
        with self.assertRaises(ValueError):
            representative_answer([])

    def test_training_partition_does_not_depend_on_test_size(self):
        items = list(range(30))
        small_pool, small_test = partition_task_items(items, 10, 5, seed=3)
        large_pool, large_test = partition_task_items(items, 10, 12, seed=3)
        self.assertEqual(small_pool, large_pool)
        self.assertTrue(set(small_pool).isdisjoint(large_test))
        self.assertEqual(set(small_test), set(items[10:15]))


if __name__ == "__main__":
    unittest.main()
