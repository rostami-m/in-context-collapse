import unittest

import torch

from incontext_collapse.frozen_patch import query_residual_patch


class FrozenPatchTests(unittest.TestCase):
    def test_patch_changes_only_last_prefill_position(self):
        original = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
        vector = torch.full((1, 1, 8), -1.0)
        patched = query_residual_patch(vector)(None, (), original)
        self.assertTrue(torch.equal(patched[:, :2], original[:, :2]))
        self.assertTrue(torch.equal(patched[:, -1:], vector))
        self.assertTrue(torch.equal(original, torch.arange(24).reshape(1, 3, 8)))

    def test_patch_preserves_tuple_outputs(self):
        hidden = torch.zeros(1, 2, 4)
        auxiliary = torch.tensor([7])
        output = query_residual_patch(torch.ones(1, 1, 4))(
            None, (), (hidden, auxiliary))
        self.assertIsInstance(output, tuple)
        self.assertTrue(torch.equal(output[0][:, -1:], torch.ones(1, 1, 4)))
        self.assertIs(output[1], auxiliary)

    def test_patch_does_not_change_single_token_decode(self):
        decode = torch.zeros(1, 1, 4)
        output = query_residual_patch(torch.ones(1, 1, 4))(None, (), decode)
        self.assertIs(output, decode)

    def test_patch_rejects_incompatible_width(self):
        with self.assertRaises(ValueError):
            query_residual_patch(torch.ones(1, 1, 3))(
                None, (), torch.zeros(1, 2, 4))


if __name__ == "__main__":
    unittest.main()
