import copy
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.summarize_revision_followup import InvalidArtifact, seed_interval, summarize_file


def option_payload(seeds=(0, 1, 2)):
    protocols = {"class": {"protocol_version": "2.4-audit"},
                 "random": {"protocol_version": "2.5-options", "option_order": "random"}}
    config = {"experiment": "paired_option_order_saved_adapters", "seeds": list(seeds),
              "tasks": ["shapes4"], "arms": ["frozen", "connector"], "ks": [0],
              "conditions": ["remap"], "option_orders": ["class", "random"],
              "protocols": protocols, "n": 2}
    cells = {}
    for seed in seeds:
        for arm in config["arms"]:
            for order in config["option_orders"]:
                n_correct = ({"frozen": [0, 1, 2], "connector": [1, 2, 2]}[arm][seed]
                             if order == "class" else {"frozen": [0, 0, 1], "connector": [1, 2, 2]}[arm][seed])
                records = []
                for index in range(2):
                    displayed = [0, 1]
                    if order == "random":
                        key = f"option-order-v1:evaluation:{seed}:0:{index}"
                        random.Random(int.from_bytes(hashlib.sha256(key.encode()).digest(), "big")).shuffle(displayed)
                    prediction = index if index < n_correct else 1 - index
                    labels = ["mip", "zorp"]
                    records.append({"query_index": index, "query_id": f"query{index}", "gold": index,
                                    "prediction": prediction, "generation_prediction": prediction,
                                    "correct": prediction == index, "raw_output": labels[prediction],
                                    "parse_status": "exact", "label_space": labels, "k": 0,
                                    "option_order": displayed, "displayed_options": [labels[j] for j in displayed],
                                    "demo_source_ids": [], "demo_source_classes": [],
                                    "demo_label_classes": [], "shuffle_map": None})
                cells[f"{seed}:shapes4:{arm}:remap:0:{order}"] = {
                    "seed": seed, "k": 0, "condition": "remap", "protocol": protocols[order],
                    "summary": {"n": 2, "accuracy": n_correct / 2}, "records": records}
    return {"run_config": config, "cells": cells}


def vqa_payload():
    evidence = {}
    for seed in (0, 1, 2):
        evidence[str(seed)] = {}
        for arm in ("frozen", "vqa_connector"):
            records = [{"query_index": index, "query_id": f"image{index}:question",
                        "question": "Red?", "answers": ["yes"] * 10,
                        "prediction": "yes" if arm == "vqa_connector" and index == 0 else "no",
                        "score": float(arm == "vqa_connector" and index == 0), "demo_ids": []}
                       for index in range(2)]
            evidence[str(seed)][arm] = {"0": {"seed": seed, "k": 0, "records": records,
                                              "summary": {"n": 2, "accuracy": sum(r["score"] for r in records) / 2}}}
    return {"run_config": {"method": "image-disjoint VQA checkpoint evaluation", "dataset": "vqav2",
                           "seeds": [0, 1, 2], "ks": [0], "arms": ["frozen", "vqa_connector"], "n": 2},
            "diversity": {"excluded_image_ids": ["training"]}, "evidence": evidence}


def attention_payload(kind):
    options = option_payload((0,))
    config = {"method": "CAMA independent implementation" if kind == "cama" else "DARA compatibility port",
              "implementation_version": 2, "task": "shapes4", "seeds": [0], "ks": [0], "n": 2,
              "protocol": options["run_config"]["protocols"]["class"]}
    cells = []
    for condition in ("standard", "remap"):
        for arm in ("frozen", kind):
            source_arm = "frozen" if arm == "frozen" else "connector"
            cell = copy.deepcopy(options["cells"][f"0:shapes4:{source_arm}:remap:0:class"])
            cell.update(condition=condition, arm=arm)
            cells.append(cell)
    if kind == "cama":
        return {"run_config": config, "cells": cells}
    config["factor_evaluation"] = {"training_performed": False,
                                   "source_training_protocol": config["protocol"], "snapshot": "final"}
    return {"run_config": config, "seeds_done": [0],
            "training_runs": [{"torch_seed": 0, "steps": 1, "losses": [1.0]}],
            "baseline_evidence": [{"seed": 0, "standard": {"0": cells[0]}, "remap": {"0": cells[2]}}],
            "evidence": [{"seed": 0, "standard": {"0": cells[1]}, "remap": {"0": cells[3]}}]}


def calibration_payload():
    record = {"query_index": 0, "query_id": "query", "gold": 0, "label_space": ["mip", "zorp"],
              "k": 0, "shuffle_map": None, "query_mean_scores": [-2.0, -1.0],
              "content_free_mean_scores": [-3.0, -1.5], "calibrated_scores": [1.0, .5],
              "raw_prediction": 1, "prediction": 0, "raw_correct": False, "correct": True}
    cell = {"seed": 0, "k": 0, "condition": "standard", "records": [record],
            "summary": {"n": 1, "raw_accuracy": 0.0, "calibrated_accuracy": 1.0,
                        "follows_corrupted_mapping_rate": 0.0}}
    return {"run_config": {"baseline": "contextual_calibration_neutral_query", "task": "shapes4",
                           "seeds": [0], "ks": [0], "conditions": ["standard"], "n": 1},
            "evidence": {"0": {"standard": {"0": cell}}}}


def feature_payload():
    cell = copy.deepcopy(next(iter(option_payload((0,))["cells"].values())))
    cell["k"] = 1
    for record in cell["records"]:
        record.update(k=1, condition="remap", demo_source_classes=[record["prediction"]],
                      demo_label_classes=[record["prediction"]], demo_source_ids=["training-image"])
        for field in ("raw_output", "generation_prediction", "parse_status", "option_order", "displayed_options"):
            del record[field]
    return {"run_config": {"baseline": "frozen_connector_feature_1nn", "task": "shapes4",
                           "seeds": [0], "ks": [1], "conditions": ["remap"], "n": 2,
                           "protocol": cell["protocol"]}, "evidence": {"0": {"remap": {"1": cell}}}}


class FollowupSummaryTests(unittest.TestCase):
    def summarize(self, payload, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            content = json.dumps(payload).encode()
            path.write_bytes(content)
            result = summarize_file(path, bootstrap_draws=200, **kwargs)
            self.assertEqual(path.read_bytes(), content)
            self.assertEqual(result["sha256"], hashlib.sha256(content).hexdigest())
            return result

    def test_complete_means_and_paired_intervals_use_seed_units(self):
        result = self.summarize(option_payload())
        self.assertEqual(result["status"], "complete")
        self.assertFalse(result["publication_ready"])
        frozen = next(row for row in result["estimates"] if row["arm"] == "frozen" and row["option_order"] == "class")
        self.assertEqual(frozen["metrics"]["accuracy"]["mean"], .5)
        self.assertEqual(frozen["metrics"]["accuracy"]["n_seeds"], 3)
        self.assertAlmostEqual(frozen["metrics"]["accuracy"]["ci95_high"], .5 + 4.303 * .5 / 3 ** .5)
        effect = next(row for row in result["paired_comparisons"]
                      if row["kind"] == "arm" and row["left"]["option_order"] == "class")
        self.assertEqual(effect["right"]["arm"], "connector")
        self.assertEqual(effect["metrics"]["accuracy"]["seed_differences"], {"0": .5, "1": .5, "2": 0.0})
        self.assertAlmostEqual(effect["metrics"]["accuracy"]["mean"], 1 / 3)
        self.assertEqual(effect["expected_seeds"], [0, 1, 2])
        order = next(row for row in result["paired_comparisons"]
                     if row["kind"] == "option_order" and row["left"]["arm"] == "frozen")
        self.assertEqual(order["right"]["option_order"], "random")
        self.assertAlmostEqual(order["metrics"]["accuracy"]["mean"], -1 / 3)

    def test_single_seed_has_no_ci_not_zero_width(self):
        result = self.summarize(option_payload((0,)))
        for estimate in result["estimates"]:
            self.assertIsNone(estimate["metrics"]["accuracy"]["ci95_low"])
        self.assertEqual(seed_interval([1, 1, 1])["ci95_low"], 1)

    def test_partial_withholds_all_scientific_aggregates(self):
        payload = option_payload()
        payload["cells"].pop("2:shapes4:connector:remap:0:random")
        result = self.summarize(payload)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["estimates"], [])
        self.assertEqual(result["paired_comparisons"], [])
        self.assertEqual(len(result["coverage"]["missing_cells"]), 1)
        group = next(g for g in result["coverage"]["groups"] if not g["complete"])
        self.assertEqual(group["expected_seeds"], [0, 1, 2])
        self.assertEqual(group["observed_seeds"], [0, 1])

    def test_corrupt_summary_raw_output_and_partial_records_are_rejected(self):
        for field, value in (("summary", {"n": 2, "accuracy": 1}), ("raw_output", "mip zorp"),
                             ("query_index", 9), ("query_id", "")):
            payload = option_payload()
            cell = next(iter(payload["cells"].values()))
            if field == "summary":
                cell[field] = value
            else:
                cell["records"][0][field] = value
            with self.assertRaises(InvalidArtifact):
                self.summarize(payload)
        payload = option_payload()
        next(iter(payload["cells"].values()))["records"].pop()
        with self.assertRaisesRegex(InvalidArtifact, "incomplete query"):
            self.summarize(payload)

    def test_pairing_is_required_even_when_accuracies_still_match(self):
        payload = option_payload()
        payload["cells"]["0:shapes4:connector:remap:0:random"]["records"][0]["query_id"] = "other"
        with self.assertRaisesRegex(InvalidArtifact, "identities differ"):
            self.summarize(payload)

    def test_unexpected_grid_is_invalid_not_partial(self):
        payload = option_payload()
        payload["cells"]["0:shapes4:unexpected:remap:0:class"] = copy.deepcopy(next(iter(payload["cells"].values())))
        with self.assertRaisesRegex(InvalidArtifact, "unexpected grid"):
            self.summarize(payload)

    def test_vqa_bootstrap_uses_images_not_seed_times_questions(self):
        result = self.summarize(vqa_payload())
        effect = result["paired_comparisons"][0]
        interval = effect["paired_image_bootstrap"]
        self.assertEqual(interval["n_images"], 2)
        self.assertEqual(interval["n_fixed_seeds"], 3)
        self.assertEqual(interval["mean"], .5)
        self.assertEqual(effect["metrics"]["accuracy"]["ci95_low"], .5)
        self.assertLess(interval["ci95_low"], .5)
        self.assertGreater(interval["ci95_high"], .5)
        self.assertEqual(interval, self.summarize(vqa_payload())["paired_comparisons"][0]["paired_image_bootstrap"])

    def test_vqa_partial_and_repeated_images(self):
        payload = vqa_payload()
        del payload["evidence"]["2"]["vqa_connector"]
        self.assertEqual(self.summarize(payload)["status"], "partial")
        payload = vqa_payload()
        for by_arm in payload["evidence"].values():
            for shots in by_arm.values():
                shots["0"]["records"][1]["query_id"] = "image0:other-question"
        with self.assertRaisesRegex(InvalidArtifact, "repeated VQA"):
            self.summarize(payload)

    def test_no_cross_condition_comparisons(self):
        payload = option_payload()
        payload["run_config"]["conditions"].append("standard")
        for key, cell in list(payload["cells"].items()):
            other = copy.deepcopy(cell)
            other["condition"] = "standard"
            other["records"][0]["query_id"] = "different-condition-query"
            payload["cells"][key.replace(":remap:", ":standard:")] = other
        result = self.summarize(payload)
        for comparison in result["paired_comparisons"]:
            self.assertEqual(comparison["left"]["condition"], comparison["right"]["condition"])

    def test_input_change_during_audit_is_rejected(self):
        payload = option_payload()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(payload))
            from scripts import summarize_revision_followup as reducer
            original = reducer.audit_file

            def mutate(p):
                report = original(p)
                p.write_text(json.dumps(payload) + " ")
                return report

            with patch.object(reducer, "audit_file", side_effect=mutate):
                with self.assertRaisesRegex(InvalidArtifact, "changed during audit"):
                    summarize_file(path)

    def test_cli_status_and_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            payload = option_payload((0,))
            path.write_text(json.dumps(payload))
            command = [sys.executable, "scripts/summarize_revision_followup.py", str(path)]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(completed.stdout)["artifacts"][0]["status"], "complete")
            payload["cells"].pop(next(iter(payload["cells"])))
            path.write_text(json.dumps(payload))
            self.assertEqual(subprocess.run(command, capture_output=True, check=False).returncode, 2)
            path.write_text("{}")
            rejected = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(rejected.returncode, 1)
            self.assertIn("sha256", json.loads(rejected.stdout)["artifacts"][0])

    def test_attention_effects_use_own_frozen_and_replay_is_not_new_training(self):
        for kind in ("cama", "dara"):
            result = self.summarize(attention_payload(kind))
            self.assertEqual(result["status"], "complete")
            self.assertEqual(len(result["paired_comparisons"]), 2)
            for comparison in result["paired_comparisons"]:
                self.assertEqual(comparison["left"]["arm"], "frozen")
                self.assertEqual(comparison["right"]["arm"], kind)
                self.assertEqual(comparison["metrics"]["accuracy"]["mean"], .5)
            if kind == "dara":
                self.assertEqual(result["audit"]["new_training_runs"], 0)
                self.assertEqual(result["audit"]["reused_training_runs"], 1)

    def test_partial_attention_and_corrupt_replay_training_are_distinguished(self):
        for kind in ("cama", "dara"):
            payload = attention_payload(kind)
            if kind == "cama":
                payload["cells"].pop()
            else:
                payload["evidence"][0]["remap"].pop("0")
            self.assertEqual(self.summarize(payload)["status"], "partial")
        payload = attention_payload("dara")
        payload["training_runs"].append(copy.deepcopy(payload["training_runs"][0]))
        with self.assertRaisesRegex(InvalidArtifact, "training seeds"):
            self.summarize(payload)

    def test_calibration_reduces_raw_and_calibrated_score_predictions(self):
        result = self.summarize(calibration_payload())
        effects = result["paired_comparisons"]
        self.assertEqual(len(effects), 1)
        self.assertEqual(effects[0]["left"]["arm"], "raw")
        self.assertEqual(effects[0]["right"]["arm"], "calibrated")
        self.assertEqual(effects[0]["metrics"]["accuracy"]["mean"], 1.0)
        payload = calibration_payload()
        payload["evidence"]["0"]["standard"]["0"]["records"][0]["calibrated_scores"][0] = .9
        with self.assertRaisesRegex(InvalidArtifact, "query minus"):
            self.summarize(payload)
        payload = calibration_payload()
        payload["run_config"]["ks"].append(4)
        self.assertEqual(self.summarize(payload)["status"], "partial")

    def test_parse_failure_and_corrupted_mapping_metrics_are_recomputed(self):
        payload = option_payload((0,))
        payload["run_config"]["conditions"] = ["remap_shuffled"]
        cells = {}
        for key, cell in payload["cells"].items():
            cell["condition"] = "remap_shuffled"
            for record in cell["records"]:
                record["shuffle_map"] = {"0": 1, "1": 0}
            cells[key.replace(":remap:", ":remap_shuffled:")] = cell
        payload["cells"] = cells
        result = self.summarize(payload)
        frozen = next(row for row in result["estimates"] if row["arm"] == "frozen")
        self.assertEqual(frozen["metrics"]["follows_corrupted_mapping_rate"]["mean"], 1)
        cell = next(iter(payload["cells"].values()))
        cell["records"][0].update(prediction=None, generation_prediction=None,
                                   raw_output="unclear", parse_status="no_label", correct=False)
        parsed = self.summarize(payload)
        frozen = next(row for row in parsed["estimates"] if row["arm"] == "frozen" and row["option_order"] == "class")
        self.assertEqual(frozen["metrics"]["unparseable_rate"]["mean"], .5)

    def test_malformed_partial_generation_is_not_accepted_as_only_incomplete(self):
        payload = option_payload()
        payload["cells"].pop(next(iter(payload["cells"])))
        next(iter(payload["cells"].values()))["records"][0]["raw_output"] = None
        with self.assertRaisesRegex(InvalidArtifact, "raw generation"):
            self.summarize(payload)

    def test_incomplete_or_nonbijective_corrupted_mapping_is_rejected(self):
        for mapping in ({}, {"1": 0}, {"0": 1, "1": 1}, {"0": 1, "1": None},
                        {"0": "1", "1": "0"}, {"0": 2, "1": 0},
                        {"0": 1, "1": 0, "2": 3}):
            with self.subTest(mapping=mapping):
                payload = option_payload((0,))
                record = next(iter(payload["cells"].values()))["records"][0]
                record.update(prediction=None, generation_prediction=None, raw_output="unclear",
                              parse_status="no_label", correct=False, shuffle_map=mapping)
                with self.assertRaisesRegex(InvalidArtifact, "complete class-index derangement"):
                    self.summarize(payload)

    def test_unparseable_prediction_never_follows_valid_corrupted_mapping(self):
        payload = option_payload((0,))
        for cell in payload["cells"].values():
            for record in cell["records"]:
                record.update(prediction=None, generation_prediction=None, raw_output="unclear",
                              parse_status="no_label", correct=False, shuffle_map={"0": 1, "1": 0})
            cell["summary"]["accuracy"] = 0.0
        for estimate in self.summarize(payload)["estimates"]:
            self.assertEqual(estimate["metrics"]["follows_corrupted_mapping_rate"]["mean"], 0.0)

    def test_feature_baseline_exact_grid_and_metadata_without_generation(self):
        payload = feature_payload()
        result = self.summarize(payload)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["estimates"][0]["option_order"], "not_applicable")
        self.assertEqual(set(result["estimates"][0]["metrics"]), {"accuracy"})
        self.assertEqual(result["paired_comparisons"], [])
        payload["run_config"]["ks"].append(4)
        self.assertEqual(self.summarize(payload)["status"], "partial")
        payload = feature_payload()
        payload["evidence"]["0"]["remap"]["1"]["seed"] = 99
        with self.assertRaisesRegex(InvalidArtifact, "metadata disagrees"):
            self.summarize(payload)
        payload = feature_payload()
        payload["evidence"]["0"]["remap"]["9"] = payload["evidence"]["0"]["remap"].pop("1")
        with self.assertRaisesRegex(InvalidArtifact, "unexpected grid"):
            self.summarize(payload)

    def test_out_report_is_atomic_hashed_and_cannot_overwrite_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path, output = Path(directory) / "input.json", Path(directory) / "report.json"
            path.write_text(json.dumps(option_payload((0,))))
            command = [sys.executable, "scripts/summarize_revision_followup.py", str(path), "--out"]
            completed = subprocess.run(command + [str(output)], capture_output=True, text=True, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            digest = json.loads(completed.stdout)
            self.assertEqual(digest["sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
            self.assertEqual(json.loads(output.read_text())["artifacts"][0]["status"], "complete")
            self.assertEqual(len(list(Path(directory).iterdir())), 2)
            before = path.read_bytes()
            refused = subprocess.run(command + [str(path)], capture_output=True, text=True, check=False)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("must not overwrite", refused.stderr)
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
