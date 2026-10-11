from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from incontext_collapse import circuit
from incontext_collapse.circuit import load_compatible_baseline
from incontext_collapse.protocol import ProtocolConfig


class CircuitBaselineImportTests(unittest.TestCase):
    def _arguments(self):
        return {
            "model": "model", "task": "shapes4", "ks": [4, 8], "n": 40,
            "seeds": [0, 1], "score_mode": "generation", "protocol": ProtocolConfig(),
            "prompt_format": "single", "model_revision": "fixture-pinned-revision",
            "dtype": "bfloat16", "software_versions": {"transformers": "4.49.0"},
        }

    def _load(self, result, **overrides):
        arguments = self._arguments()
        arguments.update(overrides)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(result))
            return load_compatible_baseline(path, **arguments)

    def _result(self):
        protocol = ProtocolConfig()
        cells = {str(k): {"summary": {"n": 40}} for k in (0, 4, 8)}
        return {
            "run_config": {
                "model": "model", "task": "shapes4", "ks": [0, 4, 8],
                "n": 40, "seeds": [0, 1], "score_mode": "generation",
                "protocol": protocol.metadata(),
                "format": "single", "model_revision": "fixture-pinned-revision",
                "dtype": "bfloat16", "software_versions": {"transformers": "4.49.0"},
            },
            "baseline": {
                "standard": {"0": 1.0, "4": 0.5, "8": 0.4},
                "standard_ci95": {"0": {}, "4": {}, "8": {}},
                "remap": {"0": 0.2, "4": 0.3, "8": 0.25},
                "remap_ci95": {"0": {}, "4": {}, "8": {}},
            },
            "baseline_evidence": [
                {"seed": seed, "standard": cells, "remap": cells}
                for seed in (0, 1)
            ],
        }

    def test_compatible_baseline_is_subset_to_requested_k(self):
        baseline, evidence = self._load(self._result())
        self.assertEqual(set(baseline["standard"]), {"4", "8"})
        self.assertEqual(set(evidence[0]["remap"]), {"4", "8"})
        self.assertEqual(baseline["standard"], {"4": 0.5, "8": 0.4})

    def test_incompatible_protocol_is_rejected(self):
        result = self._result()
        result["run_config"]["n"] = 20
        with self.assertRaisesRegex(ValueError, "incompatible imported baseline"):
            self._load(result)

    def test_different_inference_configuration_is_rejected(self):
        alternatives = {
            "format": "multi", "model_revision": "different-model-weights",
            "dtype": "float32", "software_versions": {"transformers": "4.48.0"},
        }
        for key, value in alternatives.items():
            with self.subTest(key=key):
                result = self._result()
                result["run_config"][key] = value
                with self.assertRaisesRegex(ValueError, key):
                    self._load(result)

    def test_missing_or_unknown_source_metadata_fails_closed(self):
        for key in self._result()["run_config"]:
            for missing in (True, False):
                with self.subTest(key=key, missing=missing):
                    result = self._result()
                    if missing:
                        del result["run_config"][key]
                    else:
                        result["run_config"][key] = None
                    with self.assertRaises(ValueError):
                        self._load(result)

    def test_unknown_runtime_inference_configuration_fails_closed(self):
        for overrides in ({"model_revision": None}, {"model_revision": ""},
                          {"prompt_format": None}, {"dtype": None},
                          {"software_versions": {}},
                          {"software_versions": {"transformers": None}}):
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, "baseline import requires"):
                    self._load(self._result(), **overrides)

    def test_missing_requested_curve_or_evidence_is_rejected(self):
        for evidence in (True, False):
            with self.subTest(evidence=evidence):
                result = self._result()
                curves = result["baseline_evidence"][0] if evidence else result["baseline"]
                del curves["standard"]["4"]
                with self.assertRaisesRegex(ValueError, "missing requested K"):
                    self._load(result)

    def test_missing_seed_evidence_is_rejected(self):
        result = self._result()
        result["baseline_evidence"].pop()
        with self.assertRaisesRegex(ValueError, "requested seeds"):
            self._load(result)

    def test_main_passes_runtime_configuration_to_import(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            source = Path(directory) / "source.json"
            source.write_text(json.dumps(self._result()))
            out = Path(directory) / "import.json"
            argv = ["circuit", "--model", "model", "--task", "shapes4",
                    "--ks", "4,8", "--seeds", "0,1", "--score-mode", "generation",
                    "--baseline-from", str(source), "--baseline-only", "--out", str(out)]
            stack.enter_context(patch("sys.argv", argv))
            stack.enter_context(patch.object(circuit, "load_model", return_value=(object(), object())))
            stack.enter_context(patch.dict(circuit.TASKS, {"shapes4": lambda *_: ([], {}, [])}))
            stack.enter_context(patch.object(circuit, "model_revision", return_value="fixture-pinned-revision"))
            stack.enter_context(patch.object(circuit, "software_versions",
                                             return_value={"transformers": "4.49.0"}))
            imported = stack.enter_context(patch.object(circuit, "load_compatible_baseline",
                                                        wraps=load_compatible_baseline))
            evaluated = stack.enter_context(patch.object(circuit, "eval_kcurve"))
            stack.enter_context(redirect_stdout(io.StringIO()))
            circuit.main()
            imported.assert_called_once_with(str(source), **self._arguments())
            evaluated.assert_not_called()
            result = json.loads(out.read_text())
            self.assertEqual(result["baseline"]["standard"], {"4": 0.5, "8": 0.4})
            self.assertEqual(result["run_config"]["baseline_import_policy"], "strict_runtime_v1")

    def test_unchecked_baseline_cannot_be_resumed_under_new_policy(self):
        config = deepcopy(self._result()["run_config"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            circuit.write_json_atomic(path, self._result())
            original = path.read_bytes()
            config["baseline_import_policy"] = "strict_runtime_v1"
            with self.assertRaisesRegex(RuntimeError, "configuration mismatch"):
                circuit.load_result(path, config, {})
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
