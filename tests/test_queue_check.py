import unittest
import json
import tempfile
from pathlib import Path

from incontext_collapse.queue_check import (
    validate_frozen_features,
    validate_paired,
    validate_panel,
    validate_vqa_train,
)


class QueueCheckTests(unittest.TestCase):
    def _paired(self):
        record = {"query_index": 0}
        cell = {"records": [record]}
        return {
            "run_config": {
                "seeds": [0], "conditions": ["standard"], "ks": [4], "n": 1,
            },
            "cells_done": ["0:frozen:standard:4", "0:repaired:standard:4"],
            "seeds_done": [0],
            "evidence": {"0": {
                "frozen": {"standard": {"4": cell}},
                "repaired": {"standard": {"4": cell}},
            }},
        }

    def test_complete_paired_result_passes(self):
        validate_paired(self._paired())

    def test_incomplete_paired_result_fails(self):
        result = self._paired()
        result["cells_done"].pop()
        with self.assertRaises(ValueError):
            validate_paired(result)

    def test_complete_panel_result_passes(self):
        result = {"run_config": {
            "models": ["org/model"], "tasks": ["shapes4"],
            "conditions": ["standard"], "k_grid": [0],
            "seeds": 1, "n_test": 1,
        }, "cells": {"model|shapes4|standard|K0": {}}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model__shapes4__standard__K0__s0.json"
            path.write_text(json.dumps({"records": [{"query_index": 0}]}))
            validate_panel(result, Path(tmp))

    def test_incomplete_panel_result_fails(self):
        result = {"run_config": {
            "models": ["org/model"], "tasks": ["shapes4"],
            "conditions": ["standard"], "k_grid": [0],
            "seeds": 1, "n_test": 1,
        }, "cells": {"model|shapes4|standard|K0": {}}}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                validate_panel(result, Path(tmp))

    def test_panel_validator_honors_recorded_heavy_model_cap(self):
        result = {"run_config": {
            "models": ["Qwen/Qwen2-VL-7B-Instruct"], "tasks": ["shapes4"],
            "conditions": ["standard"], "k_grid": [0, 8, 16],
            "seeds": 1, "n_test": 1, "disable_heavy_k_cap": False,
        }, "cells": {
            "Qwen2-VL-7B-Instruct|shapes4|standard|K0": {},
            "Qwen2-VL-7B-Instruct|shapes4|standard|K8": {},
        }}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for k in (0, 8):
                path = root / f"Qwen2-VL-7B-Instruct__shapes4__standard__K{k}__s0.json"
                path.write_text(json.dumps({"records": [{"query_index": 0}]}))
            validate_panel(result, root)

    def test_complete_frozen_feature_result_passes(self):
        cell = {"records": [{"query_index": 0}]}
        result = {
            "run_config": {"seeds": [0], "conditions": ["remap"],
                           "ks": [4], "n": 1},
            "seeds_done": [0],
            "evidence": {"0": {"remap": {"4": cell}}},
        }
        validate_frozen_features(result)

    def test_complete_vqa_training_result_passes(self):
        cell = {"records": [{"query_index": 0}]}
        with tempfile.TemporaryDirectory() as tmp:
            adapter = Path(tmp) / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}")
            result = {
                "run_config": {"seeds": [0], "ks": [0], "n": 1,
                               "steps": 2},
                "seeds_done": [0],
                "training_runs": {"0": {
                    "torch_seed": 0, "losses": [1.0, 0.5],
                    "adapter_path": str(adapter),
                }},
                "evidence": {"0": {
                    "baseline": {"0": cell}, "repaired": {"0": cell},
                }},
            }
            validate_vqa_train(result)


if __name__ == "__main__":
    unittest.main()
