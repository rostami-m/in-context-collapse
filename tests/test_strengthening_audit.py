import copy
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from incontext_collapse import rule_transfer as runtime
from scripts.audit_strengthening_artifacts import (
    ROOT, audit_file, audit_manifest, bootstrap_cases, compatible_semantic_rules,
    file_digest, independent_shuffle, metric_intervals, summarize_objective_transfer,
)


class StrengtheningAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.development = runtime.build_manifest(self.directory, "development", 4, 64)
        self.manifest_path = runtime.build_manifest(self.directory, "confirmatory", 4, 64)
        self.manifest = json.loads(self.manifest_path.read_text())
        self.path = self.directory / "artifact.json"

    def checkpoint(self, seeds):
        source = self.directory / "training.json"
        training = {"protocol": {"option_order": "class"}, "objective": "episode_remap", "steps": 150, "rank": 8}
        source.write_text(json.dumps({"run_config": training}))
        manifest = {}
        for seed in seeds:
            directory = self.directory / "adapter" / f"seed_{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "test-model"}))
            (directory / "adapter_model.safetensors").write_bytes(b"not loaded: byte hashing only")
            manifest[str(seed)] = {
                "path": str(directory), "base_model_name_or_path": "test-model",
                "sha256": {name: file_digest(directory / name) for name in
                           ("adapter_config.json", "adapter_model.safetensors")},
                "training_source": {"path": str(source), "sha256": file_digest(source), **training},
            }
        return manifest

    def payload(self, *, seeds=(0, 1, 2), ks=(0, 8, 16), partition="confirmatory", negative=False):
        path = self.manifest_path if partition == "confirmatory" else self.development
        manifest = json.loads(path.read_text())
        config = {
            "schema_version": 1, "experiment": "two_object_rule_transfer",
            "family": "two_object_color_selection", "model": "test-model",
            "model_revision": "0" * 40, "dtype": "bfloat16", "software_versions": {},
            "arms": ["frozen", "connector"], "seeds": list(seeds), "n": 4, "img_size": 64,
            "rules": list(runtime.RULES), "conditions": list(runtime.CONDITIONS),
            "condition_ks": {"explicit_natural": [0], "implicit_remap": list(ks)},
            "partition": partition, "manifest": str(path), "manifest_sha256": file_digest(path),
            "development_manifest": ({"path": str(self.development), "sha256": file_digest(self.development)}
                                     if partition == "confirmatory" else None),
            "checkpoints": {"connector": self.checkpoint(seeds)}, "max_new_tokens": 6,
            "score_mode": "generation", "option_order": "random",
            "protocol": {"protocol_version": "2.5-rule-transfer-v1", "option_order": "random",
                         "mapping_mode": "episode", "remap_token_set": "eval"},
            "mapping_policy": "per_seed_query_eval_nonce_bijection_shared_across_rules_and_K",
            "identifiable_implicit_ks": [8, 16], "source_sha256": file_digest(Path(runtime.__file__)),
        }
        payload = {"run_config": config, "cells": {}, "paired_rules": {}}
        for seed, arm, condition, rule in itertools.product(seeds, config["arms"], runtime.CONDITIONS, runtime.RULES):
            for k in config["condition_ks"][condition]:
                records = []
                for index in range(4):
                    base = runtime.episode(manifest, seed, condition, rule, k, index)
                    prediction = (base["gold"] if condition == "explicit_natural" else 0)
                    if arm == "connector" and condition == "implicit_remap" and k:
                        prediction = (base["gold"] if index <= seed and not negative else base["other_rule_gold"])
                    raw = base["label_space"][prediction]
                    if negative and arm == "connector" and condition == "implicit_remap" and k:
                        raw = "unrecognized output"
                    records.append(runtime.record_output(base, raw))
                key = f"{seed}:{arm}:{condition}:{rule}:{k}"
                payload["cells"][key] = {"seed": seed, "arm": arm, "condition": condition, "rule": rule,
                                         "k": k, "score_mode": "generation", "records": records,
                                         "summary": runtime.summarize_records(records)}
        for seed, arm, k in itertools.product(seeds, config["arms"], ks):
            payload["paired_rules"][f"{seed}:{arm}:implicit_remap:{k}"] = runtime.paired_rule_metrics(
                payload["cells"][f"{seed}:{arm}:implicit_remap:left_color:{k}"],
                payload["cells"][f"{seed}:{arm}:implicit_remap:right_color:{k}"])
        return payload

    def audit(self, payload, **kwargs):
        self.path.write_text(json.dumps(payload))
        before = self.path.read_bytes()
        report = audit_file(self.path, bootstrap_draws=30, **kwargs)
        self.assertEqual(self.path.read_bytes(), before)
        return report

    def assert_rejected(self, payload, fragment):
        report = self.audit(payload)
        self.assertNotEqual(report["status"], "complete", report)
        self.assertTrue(any(fragment in error for error in report["errors"]), report["errors"])
        self.assertEqual(report["estimates"], [])
        self.assertEqual(report["paired_comparisons"], [])

    def test_complete_independent_semantics_and_case_reduction(self):
        report = self.audit(self.payload())
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["coverage"], {"expected_cells": 48, "observed_cells": 48, "records": 192})
        primary = next(row for row in report["paired_comparisons"]
                       if row["metric"] == "implicit_remap_mean_k8_k16")
        self.assertAlmostEqual(primary["seed_interval"]["mean"], .25)
        self.assertEqual(primary["seed_interval"]["n_seeds"], 3)
        self.assertEqual(primary["conditional_case_interval"]["n_cases"], 4)
        self.assertEqual(primary["conditional_case_interval"]["n_fixed_seeds"], 3)
        self.assertTrue(any(row["metric"] == "explicit_natural_mean" for row in report["paired_comparisons"]))
        self.assertTrue(any(row["metric"] == "both_rules_correct_mean_k8_k16" for row in report["paired_comparisons"]))

    def test_negative_unparseable_outcomes_are_valid_not_filtered(self):
        report = self.audit(self.payload(negative=True))
        self.assertEqual(report["errors"], [])
        primary = next(row for row in report["paired_comparisons"]
                       if row["metric"] == "implicit_remap_mean_k8_k16")
        self.assertEqual(primary["seed_interval"]["mean"], -.25)
        self.assertIn("not collapse", report["interpretation_scope"])

    def test_development_subset_is_complete_but_not_confirmatory(self):
        report = self.audit(self.payload(seeds=(0,), ks=(0, 8), partition="development"))
        self.assertEqual(report["errors"], [])
        self.assertFalse(any(row["metric"] == "implicit_remap_mean_k8_k16" for row in report["estimates"]))
        self.assertTrue(all(row["seed_interval"]["ci95_low"] is None for row in report["estimates"]))

    def test_missing_cell_and_missing_query_fail_closed(self):
        payload = self.payload()
        payload["cells"].pop(next(iter(payload["cells"])))
        self.assert_rejected(payload, "incomplete rule grid")
        payload = self.payload()
        next(iter(payload["cells"].values()))["records"].pop()
        self.assert_rejected(payload, "incomplete rule query")

    def test_frozen_expected_config_catches_joint_grid_redefinition(self):
        payload = self.payload(seeds=(0,))
        report = self.audit(payload, expected_config={"seeds": [0, 1, 2]})
        self.assertIn("frozen stage config", report["errors"][0])
        self.assertEqual(report["estimates"], [])

    def test_wrong_semantic_gold_and_other_rule_score_rejected(self):
        for field, value, fragment in (("gold", 3, "semantic query gold"),
                                        ("answers_other_rule", True, "other-rule score")):
            payload = self.payload()
            record = payload["cells"]["0:frozen:explicit_natural:left_color:0"]["records"][0]
            record[field] = value
            self.assert_rejected(payload, fragment)

    def test_generation_and_parse_fields_are_independently_checked(self):
        for field, value, fragment in (("raw_output", "red green", "parsed generation"),
                                        ("prediction", 1, "parsed generation"),
                                        ("unparseable", True, "unparseable flag"),
                                        ("parse_status", "no_label", "parse status"),
                                        ("parse_matches", [], "parse matches")):
            payload = self.payload()
            next(iter(payload["cells"].values()))["records"][0][field] = value
            self.assert_rejected(payload, fragment)

    def test_demo_order_labels_and_option_rng_are_checked(self):
        for field, action, fragment in (
            ("demo_scene_ids", lambda value: value[::-1], "demonstration identities"),
            ("demo_label_classes", lambda value: [(x + 1) % 4 for x in value], "semantic labels"),
            ("option_order", lambda value: value[::-1], "displayed options"),
            ("compatible_rules", lambda value: list(runtime.RULES), "compatible-rule"),
        ):
            payload = self.payload()
            record = payload["cells"]["0:connector:implicit_remap:left_color:8"]["records"][0]
            record[field] = action(record[field])
            self.assert_rejected(payload, fragment)

    def test_implicit_prompt_rule_leak_and_stored_pair_metrics_rejected(self):
        payload = self.payload()
        record = payload["cells"]["0:connector:implicit_remap:left_color:8"]["records"][0]
        record["prompt_transcript"][-1]["text"] += " Use the left object."
        self.assert_rejected(payload, "prompt transcript")
        payload = self.payload()
        next(iter(payload["paired_rules"].values()))["semantic_switch_rate"] = .9
        self.assert_rejected(payload, "paired-rule metrics")

    def test_manifest_hash_pixels_and_semantic_positions_are_checked(self):
        payload = self.payload()
        scene = next(iter(self.manifest["scenes"].values()))
        png = self.directory / scene["png_path"]
        png.write_bytes(png.read_bytes() + b"tampered")
        self.assert_rejected(payload, "PNG hash mismatch")

    def test_consistently_rehashed_false_scene_metadata_fails_pixel_reconstruction(self):
        payload = self.payload()
        scene = next(iter(self.manifest["scenes"].values()))
        scene["left_color"] = 2
        scene["objects"][0]["color"] = 2
        self.manifest_path.write_text(json.dumps(self.manifest))
        payload["run_config"]["manifest_sha256"] = file_digest(self.manifest_path)
        self.assert_rejected(payload, "semantic rendering")

    def test_missing_development_provenance_and_changed_checkpoint_rejected(self):
        payload = self.payload()
        payload["run_config"]["development_manifest"] = None
        self.assert_rejected(payload, "development exclusion")
        payload = self.payload()
        checkpoint = payload["run_config"]["checkpoints"]["connector"]["0"]
        (Path(checkpoint["path"]) / "adapter_model.safetensors").write_bytes(b"changed")
        self.assert_rejected(payload, "checkpoint hash mismatch")

    def test_cross_partition_pixel_reuse_is_detected_after_consistent_rehashing(self):
        payload = self.payload()
        development = json.loads(self.development.read_text())
        old = development["scenes"][development["queries"][0]["scene_id"]]
        new = self.manifest["scenes"][self.manifest["queries"][0]["scene_id"]]
        for field in ("objects", "background", "image_id", "png_sha256"):
            new[field] = copy.deepcopy(old[field])
        (self.directory / new["png_path"]).write_bytes((self.directory / old["png_path"]).read_bytes())
        self.manifest_path.write_text(json.dumps(self.manifest))
        payload["run_config"]["manifest_sha256"] = file_digest(self.manifest_path)
        self.assert_rejected(payload, "development/confirmatory pixels overlap")

    def test_source_fingerprint_and_cell_metadata_are_checked(self):
        payload = self.payload()
        payload["run_config"]["source_sha256"] = "0" * 64
        self.assert_rejected(payload, "source fingerprint")
        payload = self.payload()
        next(iter(payload["cells"].values()))["seed"] = 9
        self.assert_rejected(payload, "cell metadata")

    def test_identical_k0_prompt_cannot_have_rule_dependent_raw_output(self):
        payload = self.payload()
        cell = payload["cells"]["0:frozen:implicit_remap:right_color:0"]
        row = cell["records"][0]
        cell["records"][0] = runtime.record_output(row, row["raw_output"] + ".")
        cell["summary"] = runtime.summarize_records(cell["records"])
        self.assert_rejected(payload, "identical implicit K0")

    def test_nonidentifiability_checks_unknown_nonce_bijection(self):
        scenes = [self.manifest["scenes"][key] for key in self.manifest["queries"][0]["demo_scene_ids"]]
        labels = [scene["left_color"] for scene in scenes]
        self.assertEqual(compatible_semantic_rules(scenes[:4], labels[:4]), list(runtime.RULES))
        self.assertEqual(compatible_semantic_rules(scenes[:8], labels[:8]), ["left_color"])

    def test_cli_nonzero_and_json_for_partial_evidence(self):
        payload = self.payload(seeds=(0,), ks=(0, 8), partition="development")
        payload["cells"].pop(next(iter(payload["cells"])))
        self.path.write_text(json.dumps(payload))
        run = subprocess.run([sys.executable, str(ROOT / "scripts/audit_strengthening_artifacts.py"),
                              str(self.path), "--json", "--bootstrap-draws", "10"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 1, run.stderr)
        self.assertEqual(json.loads(run.stdout)["status"], "partial")

    def test_case_bootstrap_is_deterministic_and_uses_case_not_seed_count(self):
        values = {seed: {"a": 1.0, "b": 0.0} for seed in range(3)}
        result = metric_intervals(values, draws=100)
        self.assertEqual(result["seed_interval"]["ci95_low"], .5)
        self.assertEqual(result["conditional_case_interval"]["n_cases"], 2)
        self.assertLess(result["conditional_case_interval"]["ci95_low"], .5)
        self.assertEqual(bootstrap_cases([0, 1], draws=100), bootstrap_cases([0, 1], draws=100))

    def option_payload(self, ks=(0, 4, 8, 16)):
        arms = ["frozen", "natural_connector", "episode_connector", "all_layer"]
        seeds = [0, 1, 2]
        protocol = {"protocol_version": "2.5-options", "option_order": "random",
                    "mapping_mode": "episode", "demo_class_policy": "random_balanced", "remap_token_set": "eval"}
        config = {"schema_version": 2, "experiment": "paired_option_order_saved_adapters",
                  "model": "test-model", "model_revision": "0" * 40, "dtype": "bfloat16", "software_versions": {},
                  "seeds": seeds, "arms": arms, "tasks": ["cifar4", "fashion4"], "conditions": ["standard", "remap"],
                  "ks": list(ks), "option_orders": ["random"], "protocols": {"random": protocol},
                  "score_mode": "generation", "n": 4,
                  "checkpoints": {arm: self.checkpoint(seeds) for arm in arms[1:]}}
        cells = {}
        for seed, task, arm, condition, k in itertools.product(seeds, config["tasks"], arms, config["conditions"], ks):
            records = []
            for index in range(4):
                labels = ["red", "green", "blue", "yellow"] if condition == "standard" else ["mip", "zorp", "kiki", "lurn"]
                order = independent_shuffle(range(4), f"option-order-v1:evaluation:{seed}:{k}:{index}")
                prediction = index if index <= arms.index(arm) else (index + 1) % 4
                records.append({"query_index": index, "query_id": f"{task}:test:{index}",
                                "gold": index, "condition": condition, "k": k,
                                "prediction": prediction, "generation_prediction": prediction,
                                "raw_output": labels[prediction], "parse_status": "exact",
                                "correct": prediction == index, "unparseable": False,
                                "label_space": labels, "option_order": order,
                                "displayed_options": [labels[value] for value in order],
                                "demo_source_ids": [f"{task}:train:{j}" for j in range(k)],
                                "demo_source_classes": [j % 4 for j in range(k)],
                                "demo_label_classes": [j % 4 for j in range(k)], "shuffle_map": None})
            cells[f"{seed}:{task}:{arm}:{condition}:{k}:random"] = {
                "seed": seed, "k": k, "condition": condition, "score_mode": "generation", "protocol": protocol,
                "records": records, "summary": {"n": 4, "accuracy": (arms.index(arm) + 1) / 4, "unparseable_rate": 0.0}}
        return {"run_config": config, "cells": cells}

    def test_target_transfer_existing_audit_and_prespecified_objective_effect(self):
        report = self.audit(self.option_payload())
        self.assertEqual(report["errors"], [])
        objective = report["objective_transfer"]
        self.assertEqual(objective["primary_status"], "complete")
        self.assertEqual(len(objective["primary"]), 2)
        for result in objective["primary"]:
            self.assertEqual(result["left_arm"], "natural_connector")
            self.assertEqual(result["right_arm"], "episode_connector")
            self.assertEqual(result["mean"], .25)
            self.assertEqual(result["ci95_low"], .25)
            self.assertEqual(result["ks"], [4, 8, 16])
        self.assertEqual(len(objective["standard_diagnostics"]), 8)

    def test_target_smoke_withholds_primary_without_rejecting_complete_grid(self):
        report = self.audit(self.option_payload((0, 4)))
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["objective_transfer"]["primary"], [])
        self.assertIn("withheld", report["objective_transfer"]["primary_status"])

    def test_target_missing_cell_and_split_leak_rejected(self):
        payload = self.option_payload()
        payload["cells"].pop(next(iter(payload["cells"])))
        self.assert_rejected(payload, "incomplete")
        payload = self.option_payload()
        for key, cell in payload["cells"].items():
            if ":cifar4:" in key and cell["k"]:
                cell["records"][0]["demo_source_ids"][0] = "cifar4:test:1"
        self.assert_rejected(payload, "image overlap")

    def test_pure_objective_comparison_rejects_unpaired_arm(self):
        payload = self.option_payload()
        payload["cells"]["0:cifar4:episode_connector:remap:8:random"]["records"][0]["query_id"] = "changed"
        with self.assertRaisesRegex(ValueError, "unpaired"):
            summarize_objective_transfer(payload)


if __name__ == "__main__":
    unittest.main()
