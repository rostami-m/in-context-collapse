"""CPU fault injection for adapter isolation in the region sweep."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from peft import LoraConfig, get_peft_model
from peft.tuners.tuners_utils import BaseTunerLayer

from incontext_collapse import circuit


class TinyBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)

    def forward(self, inputs):
        return self.proj(inputs)


class CircuitIsolationTests(unittest.TestCase):
    def run_sweep(self, failure=None, seeds="0"):
        self.events, self.snapshots, self.results = [], [], {}
        self.base = SimpleNamespace(attached=False)
        self.config = {}
        test = self

        class Adapter:
            def __init__(self, region, seed):
                self.region = region
                self.seed = seed
                self.revision_training_metadata = {"torch_seed": seed, "losses": [0.5]}

            def save_pretrained(self, path):
                test.events.append(("save", self.region, self.seed))
                if self.region == "early" and failure == "save":
                    raise OSError("injected checkpoint failure")

            def unload(self):
                test.events.append(("unload", self.region, self.seed))
                if self.region == "early" and failure == "cleanup":
                    raise RuntimeError("injected cleanup failure")
                test.base.attached = False

        def train(model, proc, names, pool, queries, targets, arm, **kwargs):
            self.assertIs(model, self.base)
            self.assertFalse(model.attached, "the next training run received a dirty base")
            region = targets[0].split(".")[0]
            self.events.append(("train", region, kwargs["seed"]))
            model.attached = True
            if region == "early" and failure == "training":
                raise RuntimeError("injected failure after attachment")
            return Adapter(region, kwargs["seed"])

        def evaluate(adapter, proc, names, pool, queries, ks, remap, n, **kwargs):
            self.assertTrue(self.base.attached)
            self.events.append(("remap" if remap else "standard", adapter.region, kwargs["seed"]))
            if adapter.region == "early" and failure == "evaluation" and remap:
                raise RuntimeError("injected remap evaluation failure")
            curve = {k: 0.75 if remap else 0.5 for k in ks}
            return curve, {k: {"records": [], "summary": {"accuracy": curve[k]}} for k in ks}

        def load(path, config, defaults):
            self.config.update(deepcopy(config))
            self.results.update(defaults)
            self.results.update(baseline={}, baseline_evidence=[])
            return self.results

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            argv = ["circuit", "--model", "fixture", "--task", "fixture",
                    "--regions", "early,late", "--comps", "both", "--ks", "0,4",
                    "--seeds", seeds, "--out", str(Path(directory) / "result.json")]
            stack.enter_context(patch("sys.argv", argv))
            stack.enter_context(patch.object(circuit, "load_model", return_value=(self.base, object())))
            stack.enter_context(patch.dict(circuit.TASKS, {"fixture": lambda *_: (["a", "b"], {}, [])}))
            stack.enter_context(patch.object(circuit, "model_revision", return_value="fixture-revision"))
            stack.enter_context(patch.object(circuit, "software_versions", return_value={}))
            stack.enter_context(patch.object(circuit, "load_result", side_effect=load))
            stack.enter_context(patch.object(circuit, "write_json_atomic",
                                             side_effect=lambda path, data: self.snapshots.append(deepcopy(data))))
            stack.enter_context(patch.object(circuit, "region_targets",
                                             side_effect=lambda model, region, comp: [f"{region}.q_proj"]))
            stack.enter_context(patch.object(circuit, "target_parameter_budget", return_value=64))
            stack.enter_context(patch.object(circuit, "train_region", side_effect=train))
            stack.enter_context(patch.object(circuit, "eval_kcurve", side_effect=evaluate))
            stack.enter_context(redirect_stdout(io.StringIO()))
            circuit.main()

    def assert_failed_seed_not_committed(self):
        early = self.results["early:both"]
        self.assertEqual(early["seeds_done"], [])
        self.assertEqual(early["training_runs"], [])
        self.assertEqual(early["seed_curves"], [])
        self.assertEqual(early["evidence"], [])
        self.assertIn("0", early["seed_errors"])
        self.assertEqual(self.snapshots[-1]["early:both"], early)

    def test_success_preserves_call_order_and_curves_across_regions_and_seeds(self):
        self.run_sweep(seeds="0,1")
        expected = [(operation, region, seed) for region in ("early", "late")
                    for seed in (0, 1)
                    for operation in ("train", "save", "standard", "remap", "unload")]
        self.assertEqual(self.events, expected)
        self.assertFalse(self.base.attached)
        self.assertEqual(self.config["adapter_isolation_policy"], "fail_closed_v1")
        for region in ("early", "late"):
            cell = self.results[f"{region}:both"]
            self.assertEqual(cell["seeds_done"], [0, 1])
            self.assertEqual(cell["seed_errors"], {})
            self.assertEqual(cell["standard"], {0: 0.5, 4: 0.5})
            self.assertEqual(cell["remap"], {0: 0.75, 4: 0.75})
            self.assertEqual([run["torch_seed"] for run in cell["training_runs"]], [0, 1])

    def test_patched_run_rejects_resume_into_legacy_configuration(self):
        self.run_sweep()
        legacy = dict(self.config)
        legacy.pop("adapter_isolation_policy")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            circuit.write_json_atomic(path, {"run_config": legacy, "early:both": {"seeds_done": []}})
            original = path.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "configuration mismatch"):
                circuit.load_result(path, self.config, {})
            self.assertEqual(path.read_bytes(), original)

    def test_evaluation_failure_unloads_before_next_region(self):
        self.run_sweep(failure="evaluation")
        self.assert_failed_seed_not_committed()
        self.assertLess(self.events.index(("unload", "early", 0)),
                        self.events.index(("train", "late", 0)))
        self.assertEqual(self.results["late:both"]["seeds_done"], [0])
        self.assertFalse(self.base.attached)

    def test_checkpoint_failure_unloads_before_next_region(self):
        self.run_sweep(failure="save")
        self.assert_failed_seed_not_committed()
        self.assertNotIn(("standard", "early", 0), self.events)
        self.assertEqual(self.results["late:both"]["seeds_done"], [0])
        self.assertFalse(self.base.attached)

    def test_training_failure_before_wrapper_return_fails_closed(self):
        with self.assertRaisesRegex(circuit.AdapterIsolationError, "recoverable wrapper") as caught:
            self.run_sweep(failure="training")
        self.assertIn("after attachment", str(caught.exception.__cause__))
        self.assert_failed_seed_not_committed()
        self.assertEqual(self.events, [("train", "early", 0)])
        self.assertNotIn("late:both", self.results)

    def test_cleanup_failure_fails_closed_without_marking_seed_done(self):
        with self.assertRaisesRegex(circuit.AdapterIsolationError, "cleanup failed") as caught:
            self.run_sweep(failure="cleanup")
        self.assertIn("injected cleanup failure", str(caught.exception.__cause__))
        self.assert_failed_seed_not_committed()
        self.assertNotIn(("train", "late", 0), self.events)
        self.assertNotIn("late:both", self.results)

    def test_missing_unload_fails_closed(self):
        with patch.object(circuit, "train_region", return_value=object()):
            with self.assertRaisesRegex(circuit.AdapterIsolationError, "cleanup failed"):
                with circuit.isolated_trained_adapter(object()):
                    pass

    def test_interrupt_still_unloads_returned_adapter(self):
        calls = []
        adapter = SimpleNamespace(unload=lambda: calls.append("unload"))
        with patch.object(circuit, "train_region", return_value=adapter):
            with self.assertRaises(KeyboardInterrupt):
                with circuit.isolated_trained_adapter(object()):
                    raise KeyboardInterrupt()
        self.assertEqual(calls, ["unload"])

    def test_real_peft_unload_restores_original_weights_after_evaluation_failure(self):
        base = TinyBase()
        original = {key: value.clone() for key, value in base.state_dict().items()}

        def attach(model):
            self.assertFalse(any(isinstance(layer, BaseTunerLayer) for layer in model.modules()))
            adapter = get_peft_model(model, LoraConfig(r=2, target_modules=["proj"]))
            with torch.no_grad():
                model.proj.lora_B["default"].weight.fill_(1.0)
            return adapter

        with patch.object(circuit, "train_region", side_effect=attach):
            with self.assertRaisesRegex(ValueError, "evaluation failed"):
                with circuit.isolated_trained_adapter(base):
                    self.assertTrue(any(isinstance(layer, BaseTunerLayer) for layer in base.modules()))
                    raise ValueError("evaluation failed")
            with circuit.isolated_trained_adapter(base):
                pass
        self.assertEqual(set(base.state_dict()), set(original))
        for key, value in base.state_dict().items():
            self.assertTrue(torch.equal(value, original[key]), key)
        self.assertFalse(any(isinstance(layer, BaseTunerLayer) for layer in base.modules()))


if __name__ == "__main__":
    unittest.main()
