import unittest

import torch

from incontext_collapse.core import region_targets


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.connector = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(10)])
        self.layers = torch.nn.ModuleList([
            torch.nn.ModuleDict({
                "q_proj": torch.nn.Linear(2, 2),
                "gate_proj": torch.nn.Linear(2, 2),
            }) for _ in range(28)
        ])


class RegionTests(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()

    def test_connector_targets_are_not_silently_truncated(self):
        self.assertEqual(len(region_targets(self.model, "connector")), 10)

    def test_actual_layers_are_partitioned_once(self):
        early = region_targets(self.model, "early")
        mid = region_targets(self.model, "mid")
        late = region_targets(self.model, "late")
        all_targets = region_targets(self.model, "all")
        self.assertEqual((len(early), len(mid), len(late)), (18, 18, 20))
        self.assertEqual(set(early) | set(mid) | set(late), set(all_targets))
        self.assertFalse(set(early) & set(mid))
        self.assertFalse(set(mid) & set(late))


if __name__ == "__main__":
    unittest.main()
