"""CPU-only invariants for the prospective rule-factorial study; no VLM loads."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from incontext_collapse import rule_factorial as rf
from incontext_collapse import rule_transfer as rt
from incontext_collapse.protocol import REMAP_TOKEN_SETS, image_id
from incontext_collapse.results import load_result


class RuleFactorialScenes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temporary.name)
        cls.path = rf.build_manifest(cls.directory, "development", 4)
        cls.manifest, cls.images = rf.load_manifest(cls.path)
        cls.confirm_path = rf.build_manifest(cls.directory, "confirmatory", 4)
        cls.confirm, _ = rf.load_manifest(cls.confirm_path)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def episode(self, condition="implicit_nonce", rule="left_color", k=8, index=0, seed=0):
        return rf.episode(self.manifest, seed, condition, rule, k, index)

    def cell(self, condition="implicit_nonce", rule="left_color", k=8,
             arm="frozen", seed=0, outputs=None):
        records = []
        for index in range(4):
            base = rf.episode(self.manifest, seed, condition, rule, k, index)
            raw = base["label_space"][base["gold"]] if outputs is None else outputs[index]
            records.append(rt.record_output(base, raw))
        return {"seed": seed, "arm": arm, "condition": condition, "rule": rule, "k": k,
                "score_mode": "generation", "summary": rt.summarize_records(records), "records": records}

    def test_new_namespace_and_partitions_disjoint_from_legacy(self):
        rt.validate_disjoint(self.manifest, self.confirm)
        old = rt._blueprint("development", 4, 112)
        old_ids = {image_id(rt.render_scene(scene, 112)) for scene in old["scenes"].values()}
        rf.validate_exclusions(self.manifest, old_ids)
        self.assertTrue(all(rf.NAMESPACE in key for key in self.manifest["scenes"]))
        self.assertEqual(len({s["image_id"] for s in self.manifest["scenes"].values()}), 68)

    def test_balanced_sides_and_unique_support_query_pixels(self):
        for side in rf.RULES:
            self.assertEqual(sorted(self.manifest["scenes"][q["scene_id"]][side]
                                    for q in self.manifest["queries"]), list(range(4)))
        for query in self.manifest["queries"]:
            ids = [self.manifest["scenes"][key]["image_id"] for key in query["demo_scene_ids"]]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertNotIn(self.manifest["scenes"][query["scene_id"]]["image_id"], ids)

    def test_exclusions_are_fail_closed_before_writing(self):
        with tempfile.TemporaryDirectory() as target:
            digest = next(iter(self.manifest["scenes"].values()))["image_id"]
            with self.assertRaisesRegex(ValueError, "overlap prior"):
                rf.build_manifest(target, "development", 4, excluded_identifiers=[digest])
            self.assertEqual(list(Path(target).iterdir()), [])

    def test_existing_manifest_cannot_change_configuration(self):
        self.assertEqual(rf.build_manifest(self.directory, "development", 4), self.path)
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            rf.build_manifest(self.directory, "development", 8)
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            rf.build_manifest(self.directory, "development", 4, inventory_sha256="a" * 64)

    def test_declared_scene_metadata_cannot_override_renderer(self):
        with tempfile.TemporaryDirectory() as target:
            path = rf.build_manifest(target, "development", 4)
            data = json.loads(path.read_text())
            first = next(iter(data["scenes"]))
            data["scenes"][first]["left_color"] = 3
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "scene metadata"):
                rf.load_manifest(path)

    def test_png_corruption_is_detected(self):
        with tempfile.TemporaryDirectory() as target:
            path = rf.build_manifest(target, "development", 4)
            data = json.loads(path.read_text())
            png = Path(target) / next(iter(data["scenes"].values()))["png_path"]
            Image.new("RGB", (112, 112), "red").save(png)
            with self.assertRaisesRegex(ValueError, "PNG checksum"):
                rf.load_manifest(path)

    def test_all_four_conditions_use_same_images_options_and_class_oracle(self):
        for k in (8, 16):
            for rule in rf.RULES:
                cells = [self.episode(condition, rule, k) for condition in rf.CONDITIONS]
                for other in cells[1:]:
                    for field in rf.FACTOR_IMAGE_FIELDS:
                        self.assertEqual(cells[0][field], other[field], field)
                for row in cells:
                    self.assertEqual(row["k"], k)
                    self.assertEqual(len(row["demo_source_ids"]), k)
                    self.assertEqual(row["compatible_rules"], [rule])
                    self.assertTrue(row["rule_identifiable"])

    def test_explicitness_changes_only_final_instruction_text(self):
        for labels in ("natural", "nonce"):
            explicit = self.episode(f"explicit_{labels}")
            implicit = self.episode(f"implicit_{labels}")
            self.assertEqual(explicit["prompt_transcript"][:-1], implicit["prompt_transcript"][:-1])
            e = explicit["prompt_transcript"][-1]["text"]
            i = implicit["prompt_transcript"][-1]["text"]
            self.assertEqual(e.split(" Answer with", 1)[1], i.split(" Answer with", 1)[1])
            self.assertIn("left object", e)
            self.assertNotIn("left", i)
            self.assertNotIn("right", i)

    def test_verbalizer_factor_changes_only_labels(self):
        for instruction in ("explicit", "implicit"):
            natural = self.episode(f"{instruction}_natural")
            nonce = self.episode(f"{instruction}_nonce")
            converted = deepcopy(nonce["prompt_transcript"])
            for block in converted:
                if block["type"] == "text":
                    for color, label in zip(rt.COLORS, nonce["label_space"]):
                        block["text"] = block["text"].replace(label, color)
            self.assertEqual(converted, natural["prompt_transcript"])
            self.assertEqual(set(nonce["label_space"]), set(REMAP_TOKEN_SETS["eval"][:4]))
            self.assertFalse(set(nonce["label_space"]) & set(REMAP_TOKEN_SETS["train"]))

    def test_counterfactual_rules_share_images_options_and_mapping(self):
        for condition in rf.CONDITIONS:
            left, right = self.episode(condition, "left_color"), self.episode(condition, "right_color")
            for field in rf.PAIR_FIELDS:
                self.assertEqual(left[field], right[field], field)
            self.assertNotEqual(left["gold"], right["gold"])
            self.assertNotEqual(left["demo_label_classes"], right["demo_label_classes"])
            if condition.startswith("implicit"):
                self.assertEqual(left["prompt_transcript"][-1], right["prompt_transcript"][-1])

    def test_seed_varies_nonce_mapping_but_rules_and_K_do_not(self):
        observed = set()
        for seed in range(8):
            row = self.episode(seed=seed)
            observed.add(tuple(row["label_space"]))
            for k in (8, 16):
                for rule in rf.RULES:
                    other = self.episode("explicit_nonce", rule, k, seed=seed)
                    self.assertEqual(row["label_space"], other["label_space"])
                    self.assertEqual(row["option_order"], other["option_order"])
        self.assertGreater(len(observed), 1)

    def test_identifiability_and_held_out_color_pair(self):
        for index in range(4):
            for rule in rf.RULES:
                for k in (8, 16):
                    row = self.episode(rule=rule, k=k, index=index)
                    self.assertEqual(rt.compatible_rules(row["demo_colors"], row["demo_label_classes"]), [rule])
                    self.assertNotIn(row["query_colors"], row["demo_colors"])
                    for side in ("left", "right"):
                        self.assertEqual([sum(item[side] == color for item in row["demo_colors"])
                                          for color in range(4)], [k // 4] * 4)

    def test_invalid_K_rejected_for_all_conditions(self):
        for condition in rf.CONDITIONS:
            for k in (0, 4, 12):
                with self.assertRaisesRegex(ValueError, "K outside"):
                    self.episode(condition, k=k)

    def test_oracle_metadata_does_not_enter_model_messages(self):
        row = self.episode()
        messages = rt.build_messages(row, self.images)
        self.assertEqual(list(messages[0]), ["role", "content"])
        for block in messages[0]["content"]:
            self.assertEqual(set(block), {"type", "image"} if block["type"] == "image" else {"type", "text"})
            if block["type"] == "text":
                self.assertNotIn("left", block["text"])
                self.assertNotIn("right", block["text"])
                self.assertNotIn(rf.NAMESPACE, block["text"])

    def test_mock_evaluation_records_raw_outputs_and_invalids(self):
        outputs = iter(["mip zorp kiki lurn", "unknown", "mip", "zorp"])
        def generate(model, processor, messages):
            self.assertIsNone(model)
            self.assertEqual(sum(b["type"] == "image" for b in messages[0]["content"]), 9)
            return next(outputs)
        cell = rf.evaluate_cell(None, None, self.manifest, self.images, 0, "frozen",
                                "explicit_nonce", "left_color", 8, generate)
        rf.validate_cell(cell, self.manifest)
        self.assertEqual(cell["summary"]["unparseable_rate"], .5)
        self.assertEqual(cell["records"][0]["parse_status"], "option_echo")
        self.assertFalse(cell["records"][0]["correct"])
        self.assertEqual(cell["records"][1]["raw_output"], "unknown")

    def test_saved_raw_score_or_prompt_corruption_rejected(self):
        for field, replacement in (("correct", False), ("raw_output", "invalid"),
                                    ("prompt_sha256", "a" * 64), ("query_id", "b" * 64)):
            cell = self.cell()
            cell["records"][0][field] = replacement
            with self.assertRaisesRegex(ValueError, "differs from scene"):
                rf.validate_cell(cell, self.manifest)

    def test_paired_metrics_invalid_outputs_and_joint_success(self):
        left = self.cell()
        right = self.cell(rule="right_color")
        metrics = rf.paired_rule_metrics(left, right)
        self.assertEqual(metrics["both_rules_correct_rate"], 1)
        self.assertEqual(metrics["semantic_switch_rate"], 1)
        right["records"][0] = rt.record_output(self.episode(rule="right_color"), "invalid")
        metrics = rf.paired_rule_metrics(left, right)
        self.assertEqual(metrics["both_rules_correct_rate"], .75)
        self.assertEqual(metrics["semantic_switch_rate"], .75)
        self.assertEqual(metrics["either_unparseable_rate"], .25)
        right["records"][0]["option_order"] = [3, 2, 1, 0]
        if right["records"][0]["option_order"] == left["records"][0]["option_order"]:
            right["records"][0]["option_order"] = [0, 1, 2, 3]
        with self.assertRaisesRegex(ValueError, "counterfactual"):
            rf.paired_rule_metrics(left, right)

    def test_complete_grid_sizes_and_missing_arm_rejection(self):
        self.assertEqual(len(rf.make_plan([0], [8])) * 16, 512)
        self.assertEqual(len(rf.make_plan([0, 1, 2], [8, 16])) * 80, 15360)
        with self.assertRaises(ValueError):
            rf.make_plan([0], [8], arms=("frozen",))
        with self.assertRaises(ValueError):
            rf.make_plan([0, 0], [8])

    def test_resume_rejects_unplanned_or_wrong_derived_cells(self):
        plan = rf.make_plan([0], [8])
        cells = {rf.cell_key((0, "frozen", "implicit_nonce", rule, 8)): self.cell(rule=rule)
                 for rule in rf.RULES}
        result = {"cells": cells, "paired_rules": rf.derived_pairs(cells, plan)}
        rf.validate_result(result, self.manifest, plan)
        wrong = deepcopy(result)
        wrong["paired_rules"] = {}
        with self.assertRaisesRegex(ValueError, "paired-rule"):
            rf.validate_result(wrong, self.manifest, plan)
        wrong = deepcopy(result)
        wrong["cells"]["unexpected"] = self.cell()
        with self.assertRaisesRegex(ValueError, "outside requested"):
            rf.validate_result(wrong, self.manifest, plan)

    def test_exact_resume_config_rejects_hash_changes(self):
        with tempfile.TemporaryDirectory() as target:
            path = Path(target) / "result.json"
            path.write_text(json.dumps({"run_config": {"source_sha256": "old"}, "cells": {}}))
            with self.assertRaises((RuntimeError, ValueError)):
                load_result(path, {"source_sha256": "new"}, {"cells": {}})

    def test_interaction_uses_paired_seed_effects_without_pseudoreplication(self):
        cells = {}
        for seed in (0, 1, 2):
            for k in (8, 16):
                for rule in rf.RULES:
                    for condition in ("explicit_nonce", "implicit_nonce"):
                        for arm in ("frozen", "episode_connector"):
                            # All frozen and implicit cells wrong. Adapter explicit
                            # gets 1/4, 2/4, 3/4 right in the three fitting seeds.
                            cell = self.cell(condition, rule, k, arm, seed, outputs=["invalid"] * 4)
                            if arm == "episode_connector" and condition == "explicit_nonce":
                                for index in range(seed + 1):
                                    base = rf.episode(self.manifest, seed, condition, rule, k, index)
                                    cell["records"][index] = rt.record_output(base, base["label_space"][base["gold"]])
                                cell["summary"] = rt.summarize_records(cell["records"])
                            cells[rf.cell_key((seed, arm, condition, rule, k))] = cell
        result = rf.nonce_instruction_interaction(cells, [0, 1, 2], [8, 16])
        self.assertEqual([x["effect"] for x in result["seed_effects"]], [.25, .5, .75])
        self.assertEqual(result["seed_interval"]["mean"], .5)
        self.assertEqual(result["seed_interval"]["n_seeds"], 3)
        expected_half = 4.303 * .25 / 3 ** .5
        self.assertAlmostEqual(result["seed_interval"]["ci95_low"], .5 - expected_half)
        self.assertEqual(len(result["pair_ids"]), 4)
        one = rf.nonce_instruction_interaction(cells, [0], [8])
        self.assertIsNone(one["seed_interval"]["ci95_low"])
        # Coherently changing both model arms must still fail cross-cell pairing.
        for arm in ("frozen", "episode_connector"):
            cell = cells[rf.cell_key((0, arm, "implicit_nonce", "left_color", 8))]
            cell["records"][0]["query_id"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "different query frames"):
            rf.nonce_instruction_interaction(cells, [0, 1, 2], [8, 16])


class RuleFactorialPriorInventory(unittest.TestCase):
    def test_snapshot_covers_json_composite_ids_cached_pixels_and_source_pool(self):
        with tempfile.TemporaryDirectory() as target:
            root = Path(target)
            old = root / "old"
            old.mkdir()
            campaign = old / "new_campaign"
            campaign.mkdir()
            (old / "prior.json").write_text(json.dumps({"query_id": "a" * 64 + ":42", "nested": ["b" * 64]}))
            (old / "prior.jsonl").write_text(json.dumps({"image_id": "e" * 64}) + "\n")
            image = Image.new("RGB", (16, 16), "blue")
            image.save(old / "cached.png")
            (campaign / "ongoing.json").write_text(json.dumps({"query_id": "c" * 64}))
            with patch.object(rf, "_source_shapes_hashes", return_value=["d" * 64]):
                path = rf.build_exclusion_inventory(campaign / "inventory.json", old, [campaign])
                inventory = rf.load_exclusion_inventory(path)
                self.assertEqual(set(inventory["excluded_identifiers"]),
                                 {"a" * 64, "b" * 64, "d" * 64, "e" * 64, image_id(image)})
                # Excluded campaign files can evolve without invalidating prior evidence.
                (campaign / "ongoing.json").write_text("{}")
                self.assertEqual(rf.build_exclusion_inventory(path, old, [campaign]), path)
                (old / "prior.json").write_text("{}")
                with self.assertRaisesRegex(ValueError, "prior source checksum"):
                    rf.load_exclusion_inventory(path)

    def test_exclusion_list_cannot_be_edited_without_reconstruction_failure(self):
        with tempfile.TemporaryDirectory() as target:
            root = Path(target)
            (root / "prior.json").write_text(json.dumps({"image_id": "a" * 64}))
            with patch.object(rf, "_source_shapes_hashes", return_value=["d" * 64]):
                path = rf.build_exclusion_inventory(root / "inventory.json", root)
                value = json.loads(path.read_text())
                value["excluded_identifiers"] = ["d" * 64]
                path.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError, "identifiers differ"):
                    rf.load_exclusion_inventory(path)


class RuleFactorialCheckpoints(unittest.TestCase):
    def test_strict_provenance_requires_all_three_arms(self):
        with self.assertRaisesRegex(ValueError, "all three canonical"):
            rf.strict_checkpoint_manifest({}, [0])

    def test_strict_provenance_cross_checks_objective_seed_path_and_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapters, entries = {}, {}
            for arm in rf.ARMS[1:]:
                region = "all" if arm == "all_layer" else "connector"
                objective = "natural" if arm == "natural_connector" else "episode_remap"
                arm_root = root / arm
                seed_root = arm_root / "seed_0"
                seed_root.mkdir(parents=True)
                adapters[arm] = arm_root
                targets = (["model.layers.0.self_attn.q_proj", "model.layers.1.self_attn.q_proj"]
                           if arm == "all_layer" else ["visual.merger.mlp.0"])
                config = {"r": 8, "peft_type": "LORA", "target_modules":
                          ["q_proj"] if arm == "all_layer" else targets}
                (seed_root / "adapter_config.json").write_text(json.dumps(config))
                source = {"run_config": {"model": rf.MODEL, "model_revision": rf.MODEL_REVISION,
                           "task": "shapes4", "regions": region, "components": "both", "steps": 150,
                           "rank": 8, "objective": objective, "format": "single", "dtype": "bfloat16", "seeds": [0]},
                          f"{region}:both": {"seeds_done": [0], "training_runs": [{"torch_seed": 0,
                              "objective": objective, "steps": 150, "rank": 8, "targets": targets,
                              "adapter_path": str(seed_root)}]}}
                source_path = root / f"{arm}.json"
                source_path.write_text(json.dumps(source))
                entries[arm] = {"0": {"base_model_name_or_path": rf.MODEL,
                                      "training_source": {"path": str(source_path)}}}
            with patch("incontext_collapse.option_order_eval.checkpoint_manifest", return_value=entries):
                actual = rf.strict_checkpoint_manifest(adapters, [0])
                self.assertIn("source_training_run", actual["episode_connector"]["0"])
                source_path = root / "episode_connector.json"
                value = json.loads(source_path.read_text())
                value["run_config"]["objective"] = "natural"
                source_path.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError, "configuration differs"):
                    rf.strict_checkpoint_manifest(adapters, [0])


if __name__ == "__main__":
    unittest.main()
