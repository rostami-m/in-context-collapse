import copy
import unittest

from scripts.summarize_cama import summarize, verify_reference


def artifact():
    cells = []
    for condition in ("standard", "remap"):
        for arm in ("frozen", "cama"):
            record = {"query_index": 0, "query_id": "held-out", "gold": 0,
                      "label_space": ["amber", "birch"], "demo_source_ids": [],
                      "demo_source_classes": [], "demo_label_classes": [],
                      "correct": arm == "cama", "raw_output": "amber" if arm == "cama" else "birch",
                      "unparseable": False}
            cells.append({"seed": 0, "k": 0, "condition": condition, "arm": arm,
                          "records": [record], "cama_diagnostics": [{"seconds": 1.}]})
    return {"run_config": {"ks": [0], "seeds": [0], "n": 1}, "cells": cells}


class CamaSummaryTests(unittest.TestCase):
    def test_reference_episode_identity(self):
        payload = artifact()
        payload["run_config"].update(model="tiny", model_revision="pinned", task="test", protocol={})
        reference = {"run_config": copy.deepcopy(payload["run_config"]),
                     "evidence": {"0": {}}}
        for cell in payload["cells"]:
            if cell["arm"] == "frozen":
                reference["evidence"]["0"][cell["condition"]] = {"0": copy.deepcopy(cell)}
        self.assertEqual(verify_reference(payload, reference), 2)
        reference["evidence"]["0"]["remap"]["0"]["records"][0]["demo_source_ids"] = ["other"]
        with self.assertRaisesRegex(ValueError, "reference episode mismatch"):
            verify_reference(payload, reference)

    def test_paired_gain_counts(self):
        report = summarize(artifact())
        for row in report["comparisons"]:
            self.assertEqual(row["wins"], 1)
            self.assertEqual(row["losses"], 0)
            self.assertEqual(row["paired_difference"]["mean"], 1.)

    def test_unpaired_queries_are_rejected(self):
        payload = artifact()
        payload["cells"][1]["records"][0]["query_id"] = "different"
        with self.assertRaisesRegex(ValueError, "unpaired query_id"):
            summarize(payload)

    def test_duplicate_cells_are_rejected(self):
        payload = artifact()
        payload["cells"].append(copy.deepcopy(payload["cells"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate cell"):
            summarize(payload)

    def test_random_display_order_is_required_and_paired(self):
        payload = artifact()
        payload["run_config"]["protocol"] = {"option_order": "random"}
        with self.assertRaisesRegex(ValueError, "unpaired option_order"):
            summarize(payload)
        for cell in payload["cells"]:
            cell["records"][0].update(option_order=[1, 0], displayed_options=["birch", "amber"])
        summarize(payload)
        payload["cells"][1]["records"][0]["displayed_options"].reverse()
        with self.assertRaisesRegex(ValueError, "unpaired displayed_options"):
            summarize(payload)
