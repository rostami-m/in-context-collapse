import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLAttention

from incontext_collapse.dara import (
    _dara_attention_forward,
    amplify_attention_logits,
    image_token_ranges,
    explicit_causal_mask,
    load_factor_source,
    main,
    restore_factors,
)


class DaraCompatibilityTests(unittest.TestCase):
    def factor_source(self):
        return {
            "run_config": {"method": "DARA compatibility port", "implementation_version": 2,
                           "model": "tiny", "model_revision": "pinned", "task": "shapes4",
                           "protocol": {"protocol_version": "2.4-audit"},
                           "objective": "natural", "steps": 2, "train_k": 8,
                           "learning_rate": .001},
            "training_runs": [{"torch_seed": 0, "trainable_parameters": 16,
                               "factors": [[1.1] * 8] * 2,
                               "final_factors": [[1.2] * 8] * 2,
                               "steps": 2, "losses": [1., 2.], "best_step": 1}],
        }

    def test_saved_factor_selection_is_explicit_and_preserves_source(self):
        original = json.dumps(self.factor_source()).encode()
        with patch("incontext_collapse.dara.Path.read_bytes", return_value=original), \
                patch("incontext_collapse.dara.model_revision", return_value="pinned"):
            source, final, digest = load_factor_source("source.json", model_id="tiny",
                                                       task="shapes4", seeds=[0])
            _, selected, _ = load_factor_source("source.json", model_id="tiny", task="shapes4",
                                                seeds=[0], snapshot="selected")
            self.assertEqual(final[0]["evaluation_snapshot"], "final")
            self.assertAlmostEqual(final[0]["factors"][0][0], 1.2)
            self.assertAlmostEqual(selected[0]["factors"][0][0], 1.1)
            self.assertEqual(source["training_runs"][0]["factors"][0][0], 1.1)
            self.assertEqual(len(digest), 64)
            with self.assertRaisesRegex(ValueError, "missing requested"):
                load_factor_source("source.json", model_id="tiny", task="shapes4", seeds=[1])
            with self.assertRaisesRegex(ValueError, "mismatch: task"):
                load_factor_source("source.json", model_id="tiny", task="fashion4", seeds=[0])

    def test_saved_factors_reject_nonfinite_and_old_causality_sources(self):
        for invalid in ("nonfinite", "old", "invalidated"):
            source = self.factor_source()
            if invalid == "nonfinite":
                source["training_runs"][0]["final_factors"][0][0] = float("nan")
            elif invalid == "old":
                source["run_config"]["implementation_version"] = 1
            else:
                source["invalidated"] = "test invalidation"
            with patch("incontext_collapse.dara.Path.read_bytes", return_value=json.dumps(source).encode()), \
                    patch("incontext_collapse.dara.model_revision", return_value="pinned"), \
                    self.assertRaises(ValueError):
                load_factor_source("source.json", model_id="tiny", task="shapes4", seeds=[0])

    def test_replay_cli_evaluates_final_factors_without_training(self):
        attention = SimpleNamespace(dara_factors=torch.nn.Parameter(torch.ones(2, 8)))
        model = Mock()
        result = {"seeds_done": [], "seed_curves": [], "training_runs": [], "evidence": []}
        evaluated = []

        def evaluate(*args, **kwargs):
            evaluated.append(float(attention.dara_factors[0, 0].detach()))
            return {4: .5}, {4: {"records": [], "summary": {}}}

        argv = ["dara", "--model", "tiny", "--ks", "4", "--seeds", "0",
                "--eval-factors-from", "/tmp/source.json", "--option-order", "random",
                "--out", "/tmp/replay.json"]
        with patch("sys.argv", argv), \
                patch("incontext_collapse.dara.Path.read_bytes", return_value=json.dumps(self.factor_source()).encode()), \
                patch("incontext_collapse.dara.model_revision", return_value="pinned"), \
                patch("incontext_collapse.dara.load_result", return_value=result), \
                patch("incontext_collapse.dara.write_json_atomic"), \
                patch("incontext_collapse.dara.os.makedirs"), \
                patch("incontext_collapse.dara.load_model", return_value=(model, Mock())), \
                patch("incontext_collapse.dara.install_dara", return_value=(attention, Mock())), \
                patch.dict("incontext_collapse.dara.TASKS", {"shapes4": lambda *args: (["a", "b"], {}, [])}), \
                patch("incontext_collapse.dara.eval_kcurve", side_effect=evaluate), \
                patch("incontext_collapse.dara.train_dara") as train:
            main()
        train.assert_not_called()
        self.assertEqual(evaluated[:2], [1., 1.])
        self.assertTrue(all(abs(value - 1.2) < 1e-6 for value in evaluated[2:]))
        self.assertEqual(result["seeds_done"], [0])
        self.assertEqual(result["training_runs"][0]["evaluation_snapshot"], "final")
        with self.assertRaisesRegex(ValueError, "incompatible"):
            restore_factors(attention, [[1.] * 8])

    def test_image_token_ranges_groups_contiguous_runs(self):
        ids = torch.tensor([7, 9, 9, 9, 4, 9, 9, 3])
        self.assertEqual(image_token_ranges(ids, 9), [(1, 3), (5, 6)])

    def test_attention_amplification_is_head_and_slot_specific(self):
        logits = torch.ones(1, 2, 1, 7)
        factors = torch.tensor([[2.0, 3.0], [5.0, 7.0]])
        result = amplify_attention_logits(logits, factors, [(1, 2), (5, 5)])
        self.assertTrue(torch.equal(result[0, 0, 0], torch.tensor([1., 2., 2., 1., 1., 3., 1.])))
        self.assertTrue(torch.equal(result[0, 1, 0], torch.tensor([1., 5., 5., 1., 1., 7., 1.])))

    def test_unit_factors_leave_logits_unchanged(self):
        logits = torch.randn(1, 3, 2, 8)
        result = amplify_attention_logits(logits, torch.ones(3, 4), [(1, 2), (5, 7)])
        self.assertTrue(torch.equal(result, logits))

    def test_unit_factors_match_native_qwen_attention(self):
        torch.manual_seed(4)
        config = Qwen2VLConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            rope_scaling={"type": "mrope", "mrope_section": [1, 1, 0]},
            attention_dropout=0.0,
        )
        attention = Qwen2VLAttention(config, layer_idx=0).eval()
        hidden = torch.randn(1, 5, 16)
        position_embeddings = (
            torch.ones(3, 1, 5, attention.head_dim),
            torch.zeros(3, 1, 5, attention.head_dim),
        )
        mask = explicit_causal_mask(torch.zeros(1, 4, 5, 5), None)
        native, _, _ = attention(hidden, position_embeddings=position_embeddings, attention_mask=mask)
        attention.dara_factors = torch.nn.Parameter(torch.ones(attention.num_heads, 8))
        attention.dara_image_ranges = [(1, 2)]
        patched, _, _ = _dara_attention_forward(
            attention, hidden, position_embeddings=position_embeddings)
        self.assertTrue(torch.equal(patched, native))

    def test_implicit_sdpa_causality_is_restored_including_cache(self):
        mask = explicit_causal_mask(torch.zeros(1, 2, 3, 3), None)[0, 0]
        self.assertTrue(torch.equal(mask == 0, torch.ones(3, 3, dtype=torch.bool).tril()))
        cached = explicit_causal_mask(torch.zeros(1, 2, 2, 5), None,
                                     torch.tensor([3, 4]))[0, 0]
        self.assertEqual(float(cached[0, 4]), torch.finfo(torch.float32).min)
        self.assertTrue(bool((cached[1] == 0).all()))

    def test_image_slots_wrap_like_upstream_release(self):
        logits = torch.ones(1, 1, 1, 4)
        factors = torch.tensor([[2.0, 3.0]])
        result = amplify_attention_logits(logits, factors, [(0, 0), (1, 1), (2, 2)])
        self.assertTrue(torch.equal(result.flatten(), torch.tensor([2., 3., 2., 1.])))

    def test_invalid_batch_is_rejected(self):
        with self.assertRaises(ValueError):
            amplify_attention_logits(torch.ones(2, 1, 1, 2), torch.ones(1, 2), [(0, 0)])


if __name__ == "__main__":
    unittest.main()
