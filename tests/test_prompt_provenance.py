import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from scripts.export_prompt_provenance import (
    cached_snapshot, cached_task_data, count_tokens, decoder_config, first_episode,
    panel_design, saved_target_profiles)


class PromptProvenanceTest(unittest.TestCase):
    def test_visual_and_non_image_counts_partition_actual_sequence(self):
        processor = Mock()
        result = count_tokens([1, 9, 9, 2, 9, 9, 3],
                              {"model_type": "qwen2_vl", "image_token_id": 9}, processor, 2)
        self.assertEqual(result["visual_tokens_in_lm_sequence"], 4)
        self.assertEqual(result["non_image_tokens_in_lm_sequence"], 3)
        self.assertEqual(result["input_tokens_total"], 7)
        self.assertFalse(result["model_forward_performed"])

    def test_cross_attention_markers_are_not_visual_features(self):
        result = count_tokens([1, 9, 2, 9],
                              {"model_type": "mllama", "image_token_index": 9}, Mock(), 2)
        self.assertEqual(result["image_placeholder_tokens"], 2)
        self.assertIsNone(result["visual_tokens_in_lm_sequence"])

    def test_unexpanded_marker_does_not_claim_feature_count(self):
        result = count_tokens([1, 9, 2],
                              {"model_type": "llava", "image_token_index": 9}, Mock(), 1)
        self.assertIsNone(result["visual_tokens_in_lm_sequence"])

    def test_explicit_cached_pin_is_required_not_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "models--test--model" / "snapshots"
            (root / "one").mkdir(parents=True)
            self.assertEqual(cached_snapshot(temp, "test/model")[0].name, "one")
            with self.assertRaises(FileNotFoundError):
                cached_snapshot(temp, "test/model", "missing")
            (root / "two").mkdir()
            with self.assertRaises(FileNotFoundError):
                cached_snapshot(temp, "test/model")

    def test_decoder_indices_do_not_invent_region_thirds(self):
        result = decoder_config({"text_config": {"num_hidden_layers": 28}})
        self.assertEqual(result["zero_based_block_range"], [0, 27])
        self.assertEqual(result["config_field"], "text_config.num_hidden_layers")
        self.assertNotIn("early", result)

    def test_saved_targets_preserve_sparse_indices_and_source_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "source.json"
            path.write_text(json.dumps({"model": "example", "early:both": {"training_runs": [{
                "torch_seed": 0, "targets": ["model.layers.0.self_attn.q_proj",
                                               "model.layers.4.mlp.up_proj",
                                               "model.layers.8.self_attn.v_proj"]}]}}))
            result = saved_target_profiles([path])[0]
            self.assertEqual(result["zero_based_target_layer_indices"], [0, 4, 8])
            self.assertEqual(result["target_span"], [0, 8])
            self.assertEqual(len(result["source_sha256"]), 64)

    def test_panel_grid_is_read_not_replaced_by_requested_grid(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cell.json"
            path.write_text(json.dumps({"model": "test/model", "task": "shapes4",
                                        "cond": "remap", "K": 64, "seed": 2, "n": 25}))
            groups, sources = panel_design(temp)
            self.assertEqual(list(groups), [("test/model", "shapes4", "remap", 64)])
            self.assertIsNone(next(iter(groups.values()))[0]["recorded_protocol"])
            self.assertEqual(len(sources), 1)

    def test_missing_dataset_snapshot_never_falls_back_to_newer_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):
                cached_task_data("cifar4", temp, {"uoft-cs/cifar10": "not-cached"})

    def test_representative_episode_matches_current_evaluator(self):
        from incontext_collapse.core import build_messages, task_shapes4
        from incontext_collapse.evaluation import evaluate_classification_cell
        data = task_shapes4(4, 8)
        for condition in ("standard", "shuffled", "remap", "remap_shuffled"):
            with self.subTest(condition=condition):
                _, episode = first_episode(data, condition, 4, 0, 4)
                result = evaluate_classification_cell(
                    None, None, *data, condition, 4, 4, 112, 0,
                    build_messages, lambda *args: "circle")
                record = result["records"][0]
                for ours, theirs in (("query_image_id", "query_id"),
                                    ("demo_image_ids", "demo_source_ids"),
                                    ("demo_source_classes", "demo_source_classes"),
                                    ("demo_shown_classes", "demo_label_classes"),
                                    ("label_space", "label_space"),
                                    ("option_order", "option_order"),
                                    ("shuffle_map", "shuffle_map")):
                    self.assertEqual(episode[ours], record[theirs])


if __name__ == "__main__":
    unittest.main()
