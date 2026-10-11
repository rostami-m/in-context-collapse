"""CPU-only regression probes for the maintained runtime audit, 2026-10-11.

These tests use tiny tensors and mock drivers; no model weights or datasets are
loaded. Frozen historical results and their original regression tests are unchanged.
"""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
import random
import statistics
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from incontext_collapse import (adapter_eval, adapters, circuit, core, evaluation,
                                frozen_patch, option_order_eval, paired_intervention,
                                protocol, queue_check, results, scoring)


class Batch(dict):
    def to(self, device):
        return Batch({key: value.to(device) for key, value in self.items()})


class Processor:
    def apply_chat_template(self, messages, **kwargs):
        if kwargs.get("tokenize") is False:
            return "prompt"
        return Batch(input_ids=torch.tensor([[1, 2, 3]]))

    def batch_decode(self, tokens, **kwargs):
        assert tokens.tolist() == [[4]]
        return ["label"]


class DecodeModel:
    device = "cpu"
    def generate(self, **kwargs):
        self.kwargs = kwargs
        return torch.tensor([[1, 2, 3, 4]])


class GenerationAndRegionsTests(unittest.TestCase):
    def test_all_shared_generation_paths_force_single_greedy_output(self):
        for fn, budget in ((core.predict, 8), (circuit._generate, 6),
                           (frozen_patch.generate_with_patch, 6)):
            with self.subTest(fn=fn.__name__):
                model = DecodeModel()
                self.assertEqual(fn(model, Processor(), []), "label")
                self.assertIs(model.kwargs["do_sample"], False)
                self.assertEqual(model.kwargs["num_beams"], 1)
                self.assertEqual(model.kwargs["num_return_sequences"], 1)
                self.assertEqual(model.kwargs["max_new_tokens"], budget)

    def test_multimodal_layers_do_not_mix_vision_with_text_targets(self):
        def blocks(count):
            return torch.nn.ModuleList([torch.nn.ModuleDict({
                "q_proj": torch.nn.Linear(2, 2),
                "up_proj": torch.nn.Linear(2, 3),
                "v_proj": torch.nn.Identity(),
            }) for _ in range(count)])
        model = torch.nn.Module()
        model.vision_model = torch.nn.ModuleDict({"layers": blocks(40)})
        model.language_model = torch.nn.ModuleDict({"layers": blocks(28)})
        model.visual = torch.nn.ModuleDict({"merger": torch.nn.Linear(4, 2)})
        bands = [core.region_targets(model, region) for region in ("early", "mid", "late")]
        self.assertEqual([len(band) for band in bands], [18, 18, 20])
        self.assertTrue(all(name.startswith("language_model.") for band in bands for name in band))
        self.assertEqual(len(set(sum(bands, []))), 56)
        self.assertEqual(core.region_targets(model, "connector"), ["visual.merger"])
        self.assertFalse(any(name.endswith("v_proj") for name in core.region_targets(model, "all")))

    def test_ambiguous_decoder_namespaces_fail_closed(self):
        model = torch.nn.Module()
        for name in ("first_decoder", "second_decoder"):
            setattr(model, name, torch.nn.ModuleDict({"layers": torch.nn.ModuleList([
                torch.nn.ModuleDict({"q_proj": torch.nn.Linear(2, 2)})])}))
        with self.assertRaisesRegex(ValueError, "ambiguous language-layer"):
            core.region_targets(model, "all")

    def test_prefill_patch_is_not_reapplied_when_decoding_recomputes_prefix(self):
        vector = torch.ones(1, 1, 3)
        hook = frozen_patch.query_residual_patch(vector)
        initial = torch.zeros(1, 4, 3)
        patched, sentinel = hook(None, (), (initial, "cache"))
        self.assertEqual(sentinel, "cache")
        self.assertTrue(torch.equal(patched[:, -1:], vector))
        self.assertTrue(torch.equal(patched[:, :-1], initial[:, :-1]))
        self.assertTrue(torch.equal(initial, torch.zeros_like(initial)))
        recomputed = torch.zeros(1, 5, 3)
        self.assertIs(hook(None, (), recomputed), recomputed)

    def test_patch_skip_decode_only_and_shape_error_do_not_consume_prefill(self):
        hook = frozen_patch.query_residual_patch(torch.ones(1, 1, 3))
        one = torch.zeros(1, 1, 3)
        self.assertIs(hook(None, (), one), one)
        with self.assertRaises(ValueError):
            hook(None, (), torch.zeros(1, 4, 2))
        self.assertTrue(torch.equal(hook(None, (), torch.zeros(1, 4, 3))[:, -1:], torch.ones(1, 1, 3)))


class PersistenceAndStatisticsTests(unittest.TestCase):
    def test_nonfinite_json_does_not_replace_existing_result(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/"result.json"
            results.write_json_atomic(target, {"valid": 1})
            original = target.read_bytes()
            for invalid in (float("nan"), float("inf"), -float("inf")):
                with self.assertRaises(ValueError):
                    results.write_json_atomic(target, {"invalid": invalid})
                self.assertEqual(target.read_bytes(), original)
                self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_failed_atomic_replace_cleans_unique_temp_and_keeps_old_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/"result.json"
            results.write_json_atomic(target, {"old": 1})
            original = target.read_bytes()
            paths = []
            def fail(source, destination):
                paths.append(Path(source))
                raise OSError("injected replace failure")
            with patch.object(results.os, "replace", side_effect=fail):
                for _ in range(2):
                    with self.assertRaises(OSError):
                        results.write_json_atomic(target, {"new": 2})
            self.assertNotEqual(paths[0], paths[1])
            self.assertEqual(target.read_bytes(), original)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_seed_intervals_keep_three_seed_values_but_use_t_for_larger_df(self):
        for n, critical in ((3, 4.303), (32, 2.0395134463964077), (101, 1.9839715184496334)):
            values = list(range(n))
            interval = evaluation.mean_t_ci(values)
            observed = (interval["ci95_high"] - interval["mean"]) * math.sqrt(n) / statistics.stdev(values)
            self.assertAlmostEqual(observed, critical, delta=1e-7)
        for values in ([float("nan")], [0, float("inf")]):
            with self.assertRaises(ValueError):
                evaluation.mean_t_ci(values)

    def test_queue_requires_complete_integer_query_index_range(self):
        queue_check._check_records({"records": [{"query_index": 1}, {"query_index": 0}]}, 2, "fixture")
        for records in ([{}], [{"query_index": None}], [{"query_index": True}],
                        [{"query_index": 7}], [{"query_index": 0}, {"query_index": 0}],
                        [{"query_index": 1}, {"query_index": 2}]):
            with self.subTest(records=records), self.assertRaises(ValueError):
                queue_check._check_records({"records": records}, len(records), "fixture")
        with self.assertRaises(ValueError):
            queue_check._check_records({"records": []}, 0, "fixture")

    def test_grid_rejects_duplicate_empty_or_invalid_axes(self):
        protocol.validate_evaluation_grid([0, 1, 2], [0, 8, 16], 40)
        for seeds, ks, n in (([], [0], 1), ([0, 0], [8], 4), ([0], [8, 8], 4),
                             ([0], [], 4), ([0], [-1], 4), ([0], [0], 0),
                             ([0], [0], 121), ([False], [8], 4)):
            with self.subTest(seeds=seeds, ks=ks, n=n), self.assertRaises(ValueError):
                protocol.validate_evaluation_grid(seeds, ks, n)
        for values in ([], ["remap", "remap"], ["unknown"]):
            with self.assertRaises(ValueError):
                protocol.validate_named_axis(values, "conditions", {"remap", "standard"})


class SupervisionAndSamplingTests(unittest.TestCase):
    class TokenProcessor:
        def apply_chat_template(self, messages, **kwargs):
            return "prompt"
        def __call__(self, text, images, **kwargs):
            ids = {"prompt": [1, 2, 3], "prompt long": [1, 2, 3, 4, 5],
                   "prompt short": [1, 2, 3, 6], "prompt retokenized": [1, 2, 7, 8]}[text[0]]
            return Batch(input_ids=torch.tensor([ids]))

    def test_answer_mask_includes_all_target_tokens_and_boundary_retokenization(self):
        processor = self.TokenProcessor()
        for label, boundary, expected in (("long", 3, [-100, -100, -100, 4, 5]),
                                         ("retokenized", 2, [-100, -100, 7, 8])):
            inputs, labels, got = scoring.encode_supervised_answer(processor, [], label, "cpu")
            self.assertEqual(got, boundary)
            self.assertEqual(labels.tolist(), [expected])
            self.assertEqual(inputs["input_ids"].shape, labels.shape)

    def test_restricted_score_uses_causally_shifted_complete_answer(self):
        class Model:
            device = "cpu"
            def __call__(self, input_ids):
                logits = torch.zeros(1, input_ids.shape[1], 9)
                for index in range(input_ids.shape[1] - 1):
                    logits[0, index, int(input_ids[0, index + 1])] = float(index + 1)
                return SimpleNamespace(logits=logits)
        output = scoring.restricted_label_scores(Model(), self.TokenProcessor(), [], ["long", "short"])
        token3 = 3 - math.log(math.exp(3) + 8)
        token4 = 4 - math.log(math.exp(4) + 8)
        self.assertEqual(output["token_counts"], [2, 1])
        self.assertAlmostEqual(output["sum"][0], token3 + token4, places=6)
        self.assertAlmostEqual(output["mean"][0], (token3 + token4)/2, places=6)
        self.assertAlmostEqual(output["sum"][1], token3, places=6)

    def test_actual_shapes_pixels_are_unique_and_train_test_disjoint(self):
        names, pool, test = core.task_shapes4(96, 120)
        train_ids = [protocol.image_id(image) for images in pool.values() for image in images]
        test_ids = [protocol.image_id(image) for image, _ in test]
        self.assertEqual(len(set(train_ids)), 384)
        self.assertEqual(len(set(test_ids)), 120)
        self.assertFalse(set(train_ids) & set(test_ids))
        self.assertFalse(set(protocol.REMAP_TOKEN_SETS["train"]) & set(protocol.REMAP_TOKEN_SETS["eval"]))
        for seed in range(3):
            rng = random.Random(seed)
            for _ in range(40):
                query, _, demos = protocol.sample_training_episode(pool, 8, rng)
                self.assertNotIn(protocol.image_id(query), [protocol.image_id(image) for image, _ in demos])
                self.assertNotIn(protocol.image_id(query), set(test_ids))

    def test_invalid_training_sizes_fail_before_adapter_attachment(self):
        for kwargs in ({"steps": 0}, {"rank": 0}, {"steps": -1}, {"rank": -2}):
            with self.subTest(kwargs=kwargs), patch.object(circuit, "get_peft_model") as attach:
                with self.assertRaises(ValueError):
                    circuit.train_region(object(), object(), [], {}, [], ["q_proj"], "remove", **kwargs)
                attach.assert_not_called()
        with self.assertRaises(ValueError):
            adapters.target_parameter_costs(torch.nn.Linear(2, 2), [], 0)
        with self.assertRaises(ValueError):
            adapters.select_targets_by_depth_profile([], [])


class TinyTrainingTests(unittest.TestCase):
    def test_real_peft_training_freezes_base_and_repeats_adapter_fit_at_same_seed(self):
        class TinyLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(10, 4)
                self.proj = torch.nn.Linear(4, 10)
                self.config = {"use_cache": True}
            def gradient_checkpointing_enable(self):
                pass
            def enable_input_require_grads(self):
                pass
            @property
            def device(self):
                return self.embedding.weight.device
            def forward(self, input_ids, labels):
                logits = self.proj(self.embedding(input_ids))
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, 10), labels[:, 1:].reshape(-1), ignore_index=-100)
                return SimpleNamespace(loss=loss)
        class TextProcessor:
            def apply_chat_template(self, messages, **kwargs):
                return "prompt"
            def __call__(self, text, images, **kwargs):
                ids = [1, 2, 3] if text == ["prompt"] else [1, 2, 3, 4, 5]
                return Batch(input_ids=torch.tensor([ids]))
        final, traces = [], []
        for repeat in range(2):
            torch.manual_seed(99)
            base = TinyLM()
            before = {key: value.clone() for key,value in base.state_dict().items()}
            with patch.object(circuit, "sample_training_episode", return_value=(object(), 1, [])), \
                 patch.object(circuit, "BUILD", return_value=[]):
                adapted = circuit.train_region(base, TextProcessor(), ["first", "second"], {}, [],
                                               ["proj"], "remove", steps=3, rank=2,
                                               objective="episode_remap", seed=7)
            trainables = [(key, value) for key,value in adapted.named_parameters() if value.requires_grad]
            self.assertTrue(trainables)
            self.assertTrue(all("lora_" in key for key,_ in trainables))
            self.assertTrue(any(torch.count_nonzero(value).item() for key,value in trainables if "lora_B" in key))
            self.assertEqual(adapted.revision_training_metadata["parameters_with_grad_last_step"], 2)
            self.assertFalse(adapted.training)
            final.append({key: value.clone() for key,value in trainables})
            traces.append(adapted.revision_training_metadata["losses"])
            restored = adapted.unload()
            self.assertEqual(set(restored.state_dict()), set(before))
            for key,value in restored.state_dict().items():
                self.assertTrue(torch.equal(value, before[key]), key)
        self.assertEqual(traces[0], traces[1])
        for key in final[0]:
            self.assertTrue(torch.equal(final[0][key], final[1][key]), key)


class CheckpointResumeTests(unittest.TestCase):
    def checkpoint(self, directory, base="fixture"):
        seed = Path(directory)/"connector_both"/"seed_0"
        seed.mkdir(parents=True)
        (seed/"adapter_config.json").write_text(json.dumps({"base_model_name_or_path": base}))
        (seed/"adapter_model.safetensors").write_bytes(b"fixture weights v1")
        return seed

    def test_checkpoint_base_identity_missing_or_different_is_rejected(self):
        for base in (None, "wrong"):
            with self.assertRaises(ValueError):
                option_order_eval.validate_checkpoint_models({"arm": {"0": {"base_model_name_or_path": base}}}, "fixture")

    def test_saved_adapter_and_paired_drivers_bind_weights_before_resume(self):
        for module in (adapter_eval, paired_intervention):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                seed = self.checkpoint(directory)
                out = Path(directory)/"result.json"
                argv = [module.__name__, "--model", "fixture", "--adapters-dir", directory,
                        "--conditions", "remap", "--ks", "4", "--n", "4", "--seeds", "0", "--out", str(out)]
                captured = {}
                def capture(path, config, initial):
                    captured.update(deepcopy(config))
                    raise RuntimeError("capture before model load")
                stack.enter_context(patch("sys.argv", argv))
                stack.enter_context(patch.object(module, "software_versions", return_value={}))
                model_load = stack.enter_context(patch.object(module, "load_model"))
                with patch.object(module, "load_result", side_effect=capture):
                    with self.assertRaisesRegex(RuntimeError, "capture"):
                        module.main()
                self.assertEqual(captured["generation"]["max_new_tokens"], 8)
                self.assertEqual(captured["checkpoints"]["connector_both"]["0"]["sha256"]["adapter_model.safetensors"], hashlib.sha256(b"fixture weights v1").hexdigest())
                results.write_json_atomic(out, {"run_config": captured})
                (seed/"adapter_model.safetensors").write_bytes(b"fixture weights v2")
                with self.assertRaisesRegex(RuntimeError, "configuration mismatch"):
                    module.main()
                model_load.assert_not_called()


class CircuitCompletionTests(unittest.TestCase):
    def run_fixture(self, initial_cell, *, training_error=False, profile_budget=None):
        result = {"baseline": {}, "baseline_evidence": []}
        if initial_cell is not None:
            result["early:both"] = deepcopy(initial_cell)
        events = []
        class Adapter:
            revision_training_metadata = {"torch_seed": 0, "losses": [0.1]}
            def save_pretrained(self, path):
                if training_error:
                    raise OSError("checkpoint failure")
            def unload(self):
                events.append("unload")
        def load(path, config, defaults):
            return result
        def evaluate(*args, **kwargs):
            return {4: .5}, {4: {"summary": {"accuracy": .5}, "records": []}}
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            argv = ["circuit", "--model", "fixture", "--task", "fixture", "--regions", "early",
                    "--comps", "both", "--ks", "4", "--n", "4", "--seeds", "0", "--out", str(Path(directory)/"result.json")]
            if profile_budget is not None:
                argv += ["--target-profile", "q_up_v", "--parameter-budget", str(profile_budget)]
            stack.enter_context(patch("sys.argv", argv))
            stack.enter_context(patch.object(circuit, "load_model", return_value=(object(), object())))
            stack.enter_context(patch.dict(circuit.TASKS, {"fixture": lambda *_: ([], {}, [])}))
            stack.enter_context(patch.object(circuit, "load_result", side_effect=load))
            stack.enter_context(patch.object(circuit, "write_json_atomic"))
            stack.enter_context(patch.object(circuit, "software_versions", return_value={}))
            stack.enter_context(patch.object(circuit, "region_targets", return_value=["layers.0.q_proj"]))
            stack.enter_context(patch.object(circuit, "target_parameter_budget", return_value=64))
            stack.enter_context(patch.object(circuit, "select_targets_by_depth_profile", return_value=["layers.0.q_proj"]))
            trained = stack.enter_context(patch.object(circuit, "train_region", side_effect=lambda *a, **kw: (events.append("train") or Adapter())))
            stack.enter_context(patch.object(circuit, "eval_kcurve", side_effect=evaluate))
            stack.enter_context(redirect_stdout(io.StringIO()))
            status = circuit.main()
        return status, result, events

    def test_error_only_cell_is_retried_and_completed(self):
        status, result, events = self.run_fixture({"error": "transient previous failure"})
        self.assertEqual(status, 0)
        self.assertEqual(events, ["train", "unload"])
        self.assertEqual(result["early:both"]["seeds_done"], [0])
        self.assertNotIn("error", result["early:both"])

    def test_recoverable_failed_cell_returns_nonzero_status_after_cleanup(self):
        status, result, events = self.run_fixture(None, training_error=True)
        self.assertEqual(status, 1)
        self.assertEqual(events, ["train", "unload"])
        self.assertEqual(result["early:both"]["seeds_done"], [])
        self.assertIn("0", result["early:both"]["seed_errors"])

    def test_unsupported_legacy_completion_is_not_trusted(self):
        with self.assertRaisesRegex(ValueError, "completion lacks seed-level evidence"):
            self.run_fixture({"standard": {4: 1.0}})

    def test_target_profile_cannot_silently_exceed_requested_budget(self):
        status, result, events = self.run_fixture(None, profile_budget=16)
        self.assertEqual(status, 1)
        self.assertEqual(events, [])
        self.assertIn("exceeds", result["early:both"]["error"])


if __name__ == "__main__":
    unittest.main()
