import json
import tempfile
import unittest
from pathlib import Path

from incontext_collapse.circuit import aggregate_stored_curves
from incontext_collapse.results import ensure_run_config, load_result, write_json_atomic


class ResultPersistenceTests(unittest.TestCase):
    def test_manifest_rejects_mixed_protocols(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            ensure_run_config(path, {"protocol": "v2"})
            ensure_run_config(path, {"protocol": "v2"})
            with self.assertRaises(RuntimeError):
                ensure_run_config(path, {"protocol": "legacy"})

    def test_result_requires_same_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            result = load_result(path, {"seed": 1}, {"cells": []})
            write_json_atomic(path, result)
            self.assertEqual(load_result(path, {"seed": 1}, {})["cells"], [])
            with self.assertRaises(RuntimeError):
                load_result(path, {"seed": 2}, {})
            self.assertEqual(json.loads(path.read_text())["run_config"], {"seed": 1})

    def test_aggregate_stored_curves_accepts_json_string_keys(self):
        curves = [
            {"seed": 0, "standard": {"4": 0.5}, "remap": {"4": 0.25}},
            {"seed": 1, "standard": {"4": 1.0}, "remap": {"4": 0.75}},
        ]
        result = aggregate_stored_curves(curves, [4])
        self.assertEqual(result["standard"][4], 0.75)
        self.assertEqual(result["standard_ci95"][4]["seed_values"], [0.5, 1.0])
        self.assertEqual(result["remap"][4], 0.5)


if __name__ == "__main__":
    unittest.main()
