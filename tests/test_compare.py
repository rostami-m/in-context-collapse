import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image
import torch

from incontext_collapse import compare
from incontext_collapse.core import build_messages
from incontext_collapse.evaluation import evaluate_classification_cell
from incontext_collapse.protocol import ProtocolConfig
from incontext_collapse.provenance import model_revision


class MatchedCompareTests(unittest.TestCase):
    def setUp(self):
        self.names = ["circle", "square", "triangle", "cross"]
        self.pool = {c: [Image.new("RGB", (4, 4), (40 * c, i, 0))
                         for i in range(20)] for c in range(4)}
        self.test = [(Image.new("RGB", (4, 4), (40 * c, 100 + i, 0)), c)
                     for c in range(4) for i in range(30)]
        self.protocol = ProtocolConfig(mapping_mode="fixed", remap_token_set="legacy")
        self.episodes = compare.comparison_episodes(
            self.names, self.pool, self.test, k=4, n=8, seed=0, protocol=self.protocol)

    def test_plan_matches_shared_evaluator_and_is_balanced(self):
        cell = evaluate_classification_cell(
            None, None, self.names, self.pool, self.test, "remap", 4, 8, 4, 0,
            build_messages, lambda *_: "alpha", protocol=self.protocol)
        self.assertEqual([episode["evidence"]["gold"] for episode in self.episodes].count(0), 2)
        for episode, record in zip(self.episodes, cell["records"]):
            info = episode["evidence"]
            for field in ("query_id", "gold", "label_space", "option_order", "displayed_options"):
                self.assertEqual(info[field], record[field])
            self.assertEqual(info["support_ids"], record["demo_source_ids"])
            self.assertEqual(info["support_classes"], record["demo_source_classes"])
            self.assertNotIn(info["query_id"], info["support_ids"])
        with self.assertRaisesRegex(ValueError, "fixed mapping"):
            compare.comparison_episodes(self.names, self.pool, self.test, k=4, n=8,
                                        seed=0, protocol=ProtocolConfig())

    def test_all_arms_share_queries_options_size_budget_and_support(self):
        calls = []

        def generate(model, proc, messages, **kwargs):
            calls.append((model, messages, kwargs))
            return "alpha and beta"  # ambiguous outputs must not get substring credit

        with patch.object(compare, "_generate", side_effect=generate):
            cells = {method: compare.evaluate_method(
                method, None, self.episodes, method, size=12, max_new_tokens=6,
                layer="layer", vector=torch.ones(3), gate=2) for method in compare.METHODS}
        identity = ("query_id", "gold", "label_space", "option_order", "displayed_options",
                    "support_ids", "support_classes", "img_size", "max_new_tokens", "prompt_format")
        for method, cell in cells.items():
            for i, record in enumerate(cell["records"]):
                full = cells["manyshot"]["records"][i]
                for field in identity:
                    self.assertEqual(record[field], full[field])
                expected = 4 if method in {"manyshot", "vaccine"} else 2 if method == "gate" else 0
                self.assertEqual(record["k"], expected)
                self.assertEqual(record["demo_source_ids"], full["demo_source_ids"][:expected])
                self.assertTrue(record["unparseable"])
                self.assertIsNone(record["prediction"])
        for method, messages, kwargs in calls:
            self.assertEqual(kwargs["max_new_tokens"], 6)
            self.assertEqual(kwargs["hook_layer"], "layer" if method in {"inject", "gate"} else None)
            for message in messages:
                for block in message["content"]:
                    if block["type"] == "image":
                        self.assertEqual(block["image"].size, (12, 12))

    def test_vector_uses_exact_full_prompts_and_no_query_labels(self):
        with patch.object(compare, "_generate", return_value=("alpha", torch.ones(1, 3))) as generate:
            vector, evidence = compare.extract_matched_vector(
                None, None, self.episodes, "layer", size=12, max_new_tokens=6, n=2)
        self.assertEqual(vector.tolist(), [1, 1, 1])
        self.assertEqual(len(evidence["sources"]), 2)
        self.assertFalse(evidence["query_labels_used"])
        for call, episode, source in zip(generate.call_args_list, self.episodes, evidence["sources"]):
            self.assertEqual(call.args[2], compare.episode_messages(episode, episode["demos"], 12))
            self.assertEqual(source["support_ids"], episode["evidence"]["support_ids"])
            self.assertEqual(call.kwargs["max_new_tokens"], 6)
            self.assertTrue(call.kwargs["grab"])

    def test_missing_or_nonfinite_vector_cannot_be_reported_as_inject(self):
        for vector in (None, torch.tensor([[float("nan")]])):
            with patch.object(compare, "_generate", return_value=("alpha", vector)):
                with self.assertRaisesRegex(RuntimeError, "finite prefill"):
                    compare.extract_matched_vector(None, None, self.episodes, "layer",
                                                   size=4, max_new_tokens=6)
        with self.assertRaisesRegex(ValueError, "require a layer and vector"):
            compare.evaluate_method(None, None, self.episodes, "inject", size=4,
                                    max_new_tokens=6)

    def make_checkpoint(self, temporary):
        root = Path(temporary) / "source_adapters" / "connector_both"
        directory = root / "seed_0"
        directory.mkdir(parents=True)
        model = "Qwen/Qwen2-VL-2B-Instruct"
        (directory / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": model}))
        (directory / "adapter_model.safetensors").write_bytes(b"fake adapter")
        source = Path(temporary) / "source.json"
        source.write_text(json.dumps({"run_config": {
            "model": model, "model_revision": model_revision(model), "task": "shapes4",
            "format": "single", "steps": 123, "rank": 7, "objective": "episode_remap",
        }}))
        return root, source

    def test_saved_checkpoint_preserves_actual_training_and_validates_transfer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, source = self.make_checkpoint(temporary)
            model = "Qwen/Qwen2-VL-2B-Instruct"
            config = compare.saved_adapter_config(root, [0], model, "shapes4", "cifar4")
            checkpoint = config["checkpoints"]["0"]
            self.assertEqual(checkpoint["training_run_config"]["steps"], 123)
            self.assertEqual(checkpoint["training_run_config"]["rank"], 7)
            with self.assertRaisesRegex(ValueError, "training task"):
                compare.saved_adapter_config(root, [0], model, "fashion4", "cifar4")
            with self.assertRaisesRegex(ValueError, "training task"):
                compare.saved_adapter_config(root, [0], model, "shapes4", "shapes4")
            before = copy.deepcopy(config)
            (root / "seed_0" / "adapter_model.safetensors").write_bytes(b"changed")
            after = compare.saved_adapter_config(root, [0], model, "shapes4", "cifar4")
            self.assertNotEqual(before, after)
            source.unlink()
            with self.assertRaisesRegex(ValueError, "training provenance"):
                compare.saved_adapter_config(root, [0], model, "shapes4", "cifar4")

    def test_missing_saved_format_or_revision_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, source = self.make_checkpoint(temporary)
            original = json.loads(source.read_text())
            for field, message in (("format", "explicit single-turn"),
                                   ("model_revision", "model/revision")):
                with self.subTest(field=field):
                    changed = copy.deepcopy(original)
                    del changed["run_config"][field]
                    source.write_text(json.dumps(changed))
                    with self.assertRaisesRegex(ValueError, message):
                        compare.saved_adapter_config(
                            root, [0], "Qwen/Qwen2-VL-2B-Instruct", "shapes4", "cifar4")

    def test_unpinned_models_are_rejected_before_loading_or_claiming_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, source = self.make_checkpoint(temporary)
            changed = json.loads(source.read_text())
            changed["run_config"]["model"] = "unknown/unpinned"
            del changed["run_config"]["model_revision"]
            source.write_text(json.dumps(changed))
            config = root / "seed_0" / "adapter_config.json"
            config.write_text(json.dumps({"base_model_name_or_path": "unknown/unpinned"}))
            with self.assertRaisesRegex(ValueError, "pinned model revision"):
                compare.saved_adapter_config(root, [0], "unknown/unpinned", "shapes4", "cifar4")
            argv = ["compare", "--model", "unknown/unpinned", "--preflight"]
            with patch("sys.argv", argv), patch.object(compare, "load_model") as load:
                with self.assertRaises(SystemExit) as error:
                    compare.main()
                self.assertEqual(error.exception.code, 2)
                load.assert_not_called()

    def test_runner_checkpoints_completed_cells_and_fails_without_false_vaccine_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result.json"
            argv = ["compare", "--k", "4", "--n", "4", "--seeds", "0", "--size", "8",
                    "--out", str(output)]
            model = MagicMock()
            generation = lambda *args, **kwargs: ("alpha", torch.ones(1, 3)) if kwargs.get("grab") else "alpha"
            mocks = [patch("sys.argv", argv),
                     patch.object(compare, "load_model", return_value=(model, None)),
                     patch.dict(compare.TASKS, {task: lambda *_: (self.names, self.pool, self.test)
                                               for task in ("shapes4", "cifar4")}),
                     patch.object(compare, "get_layers", return_value=["layer"] * 12),
                     patch.object(compare, "_generate", side_effect=generation),
                     patch.object(compare, "region_targets", return_value=["target"]),
                     patch.object(compare, "train_region", side_effect=RuntimeError("failed training"))]
            from contextlib import ExitStack
            with ExitStack() as stack:
                handles = [stack.enter_context(mock) for mock in mocks]
                with self.assertRaisesRegex(RuntimeError, "failed training"):
                    compare.main()
                train = handles[-1]
                self.assertEqual(train.call_args.kwargs["objective"], "episode_remap")
                self.assertEqual(train.call_args.kwargs["rank"], 8)
                self.assertEqual(train.call_args.kwargs["steps"], 150)
                self.assertEqual(train.call_args.kwargs["size"], 8)
            partial = json.loads(output.read_text())
            self.assertEqual(set(partial["cells"]), {f"0:{method}" for method in compare.METHODS[:-1]})
            self.assertEqual(partial["seeds_done"], [])
            adapted = MagicMock()
            adapted.revision_training_metadata = {"objective": "episode_remap"}
            mocks[-1] = patch.object(compare, "train_region", return_value=adapted)
            with ExitStack() as stack:
                handles = [stack.enter_context(mock) for mock in mocks]
                compare.main()
                self.assertEqual(handles[4].call_count, 4)  # only missing Vaccine cell
            complete = json.loads(output.read_text())
            self.assertEqual(complete["seeds_done"], [0])
            with patch("sys.argv", argv), patch.object(compare, "load_model") as load:
                compare.main()
                load.assert_not_called()
            for change in (["--max-new-tokens", "7"], ["--size", "9"]):
                with patch("sys.argv", argv + change), patch.object(compare, "load_model") as load:
                    with self.assertRaisesRegex(RuntimeError, "configuration mismatch"):
                        compare.main()
                    load.assert_not_called()
            complete["run_config"]["schema_version"] = 2
            output.write_text(json.dumps(complete))
            with patch("sys.argv", argv), patch.object(compare, "load_model") as load:
                with self.assertRaisesRegex(RuntimeError, "configuration mismatch"):
                    compare.main()
                load.assert_not_called()

    def test_partial_resume_requires_identical_vector_and_extraction_evidence(self):
        from contextlib import ExitStack

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result.json"
            argv = ["compare", "--k", "4", "--n", "4", "--seeds", "0", "--out", str(output)]
            model = MagicMock()
            base_evaluate = compare.evaluate_method

            def stop_at_gate(*args, **kwargs):
                if args[3] == "gate":
                    raise RuntimeError("interrupted at gate")
                return base_evaluate(*args, **kwargs)

            generation = lambda *args, **kwargs: ("alpha", torch.ones(1, 3)) if kwargs.get("grab") else "alpha"
            mocks = [patch("sys.argv", argv),
                     patch.object(compare, "load_model", return_value=(model, None)),
                     patch.dict(compare.TASKS, {task: lambda *_: (self.names, self.pool, self.test)
                                               for task in ("shapes4", "cifar4")}),
                     patch.object(compare, "get_layers", return_value=["layer"] * 12),
                     patch.object(compare, "_generate", side_effect=generation),
                     patch.object(compare, "evaluate_method", side_effect=stop_at_gate)]
            with ExitStack() as stack:
                for mock in mocks:
                    stack.enter_context(mock)
                with self.assertRaisesRegex(RuntimeError, "interrupted at gate"):
                    compare.main()
            persisted = json.loads(output.read_text())
            self.assertIn("0:inject", persisted["cells"])
            self.assertNotIn("0:gate", persisted["cells"])
            unchanged_bytes = output.read_bytes()
            vector_evidence = persisted["vectors"]["0"]
            # Identical reconstruction is allowed through to the pending arm.
            with ExitStack() as stack:
                for mock in mocks:
                    stack.enter_context(mock)
                with self.assertRaisesRegex(RuntimeError, "interrupted at gate"):
                    compare.main()
            self.assertEqual(output.read_bytes(), unchanged_bytes)
            for field in ("vector_float32", "sources"):
                changed = copy.deepcopy(vector_evidence)
                if field == "vector_float32":
                    changed[field][0] += 1
                else:
                    changed[field][0]["query_id"] = "different-query"
                with self.subTest(field=field), ExitStack() as stack:
                    for mock in mocks[:-1]:
                        stack.enter_context(mock)
                    stack.enter_context(patch.object(
                        compare, "extract_matched_vector", return_value=(torch.ones(3), changed)))
                    evaluate = stack.enter_context(patch.object(compare, "evaluate_method"))
                    with self.assertRaisesRegex(RuntimeError, "differs from saved evidence"):
                        compare.main()
                    evaluate.assert_not_called()
                    self.assertEqual(output.read_bytes(), unchanged_bytes)


if __name__ == "__main__":
    unittest.main()
