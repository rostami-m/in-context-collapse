import unittest

import torch

from incontext_collapse.adapters import (
    select_targets_by_budget,
    select_targets_by_depth_profile,
    target_parameter_budget,
    target_parameter_costs,
)


class AdapterBudgetTests(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.ModuleDict({
            "a": torch.nn.Linear(4, 6),
            "b": torch.nn.Linear(4, 6),
            "c": torch.nn.Linear(8, 6),
        })
        self.targets = ["a", "b", "c"]

    def test_cost_is_rank_times_input_plus_output(self):
        self.assertEqual(target_parameter_costs(self.model, self.targets, 2),
                         {"a": 20, "b": 20, "c": 28})
        self.assertEqual(target_parameter_budget(self.model, self.targets, 2), 68)

    def test_selection_never_exceeds_budget(self):
        selected, used = select_targets_by_budget(self.model, self.targets, 2, 40)
        self.assertLessEqual(used, 40)
        self.assertEqual(used, target_parameter_budget(self.model, selected, 2))
        self.assertTrue(selected)

    def test_too_small_budget_is_explicitly_empty(self):
        self.assertEqual(select_targets_by_budget(self.model, self.targets, 2, 10), ([], 0))

    def test_depth_profile_matches_components_across_band(self):
        targets = [
            f"model.layers.{layer}.{suffix}"
            for layer in range(9)
            for suffix in ("q_proj", "up_proj", "v_proj")
        ]
        self.assertEqual(
            select_targets_by_depth_profile(targets, ("q_proj", "up_proj", "v_proj")),
            ["model.layers.0.q_proj", "model.layers.4.up_proj", "model.layers.8.v_proj"],
        )


if __name__ == "__main__":
    unittest.main()
