import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from incontext_collapse import rule_transfer as rules


class RuleTransferTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = rules.build_manifest(self.root, "development", 8)
        self.manifest, self.images = rules.load_manifest(self.path)

    def cell(self, rule="left_color", k=8, output="correct"):
        records = []
        for index in range(8):
            base = rules.episode(self.manifest, 0, "implicit_remap", rule, k, index)
            raw = base["label_space"][base["gold"]] if output == "correct" else output
            records.append(rules.record_output(base, raw))
        return {"seed": 0, "arm": "frozen", "condition": "implicit_remap", "rule": rule,
                "k": k, "score_mode": "generation", "records": records,
                "summary": rules.summarize_records(records)}

    def test_manifest_is_reproducible_and_configuration_immutable(self):
        original = self.path.read_bytes()
        self.assertEqual(rules.build_manifest(self.root, "development", 8), self.path)
        self.assertEqual(self.path.read_bytes(), original)
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            rules.build_manifest(self.root, "development", 12)
        self.assertEqual(len(self.manifest["scenes"]), 8 * 17)
        self.assertFalse(self.manifest["config"]["training_used"])

    def test_development_and_confirmatory_are_disjoint_in_pixels_and_scene_ids(self):
        path = rules.build_manifest(self.root, "confirmatory", 8)
        confirmatory, _ = rules.load_manifest(path)
        rules.validate_disjoint(self.manifest, confirmatory)
        bad = copy.deepcopy(confirmatory)
        next(iter(bad["scenes"].values()))["image_id"] = next(iter(self.manifest["scenes"].values()))["image_id"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            rules.validate_disjoint(self.manifest, bad)

    def test_queries_balanced_for_both_rules_and_gold_always_differs(self):
        for side in rules.RULES:
            values = [self.manifest["scenes"][query["scene_id"]][side]
                      for query in self.manifest["queries"]]
            self.assertEqual([values.count(color) for color in range(4)], [2] * 4)
        for query in self.manifest["queries"]:
            scene = self.manifest["scenes"][query["scene_id"]]
            self.assertNotEqual(scene["left_color"], scene["right_color"])

    def test_k4_balanced_unknown_mapping_does_not_identify_side(self):
        query = self.manifest["queries"][0]
        scenes = [self.manifest["scenes"][key] for key in query["demo_scene_ids"][:4]]
        colors = [{"left": scene["left_color"], "right": scene["right_color"]} for scene in scenes]
        for side in ("left", "right"):
            self.assertEqual(rules.compatible_rules(colors, [item[side] for item in colors]), list(rules.RULES))
        with self.assertRaises(ValueError):
            rules.episode(self.manifest, 0, "implicit_remap", "left_color", 4, 0)

    def test_k8_k16_uniquely_identify_rule_without_query_color_combination(self):
        for k in (8, 16):
            for rule in rules.RULES:
                for index in range(8):
                    record = rules.episode(self.manifest, 2, "implicit_remap", rule, k, index)
                    self.assertEqual(record["compatible_rules"], [rule])
                    self.assertTrue(record["rule_identifiable"])
                    self.assertNotIn(record["query_colors"], record["demo_colors"])
                    for side, other in (("left", "right"), ("right", "left")):
                        for color in range(4):
                            entries = [item for item in record["demo_colors"] if item[side] == color]
                            self.assertEqual(len(entries), k // 4)
                            self.assertEqual(len({item[other] for item in entries}), 2)
                    self.assertNotIn(record["query_id"], record["demo_source_ids"])
                    self.assertEqual(len(set(record["demo_source_ids"])), k)

    def test_rule_pair_changes_only_demo_labels_in_actual_prompt(self):
        for k in (0, 8, 16):
            left = rules.episode(self.manifest, 0, "implicit_remap", "left_color", k, 1)
            right = rules.episode(self.manifest, 0, "implicit_remap", "right_color", k, 1)
            for field in rules.PAIR_FIELDS:
                self.assertEqual(left[field], right[field])
            self.assertNotEqual(left["gold"], right["gold"])
            a, b = left["prompt_transcript"], right["prompt_transcript"]
            self.assertEqual(a[-2:], b[-2:])
            for i in range(0, 2 * k, 2):
                self.assertEqual(a[i], b[i])
                self.assertNotEqual(a[i + 1], b[i + 1])
            for record in (left, right):
                texts = " ".join(block["text"] for block in record["prompt_transcript"] if block["type"] == "text")
                self.assertNotIn("left", texts.lower())
                self.assertNotIn("right", texts.lower())
                self.assertTrue(all(color not in texts.lower() for color in rules.COLORS))
                actual = rules.build_messages(record, self.images)[0]["content"]
                self.assertEqual(actual[-1], record["prompt_transcript"][-1])
                self.assertIs(actual[-2]["image"], self.images[record["scene_id"]])
            if not k:
                self.assertEqual(a, b)
                self.assertEqual(left["compatible_rules"], list(rules.RULES))
                self.assertFalse(left["rule_identifiable"])

    def test_mapping_options_and_queries_are_shared_across_k_not_fixed_across_seed(self):
        base = rules.episode(self.manifest, 0, "implicit_remap", "left_color", 0, 1)
        for k in (8, 16):
            other = rules.episode(self.manifest, 0, "implicit_remap", "right_color", k, 1)
            for field in ("query_id", "label_space", "option_order", "displayed_options"):
                self.assertEqual(base[field], other[field])
        mappings, options = set(), set()
        for seed in range(12):
            row = rules.episode(self.manifest, seed, "implicit_remap", "left_color", 8, 1)
            mappings.add(tuple(row["label_space"]))
            options.add(tuple(row["option_order"]))
            self.assertEqual(row["displayed_options"], [row["label_space"][i] for i in row["option_order"]])
        self.assertGreater(len(mappings), 4)
        self.assertGreater(len(options), 4)

    def test_explicit_rule_natural_answerability_is_separate(self):
        for rule in rules.RULES:
            row = rules.episode(self.manifest, 0, "explicit_natural", rule, 0, 0)
            self.assertEqual(row["label_space"], list(rules.COLORS))
            self.assertIn(f"the {rule.removesuffix('_color')} object", row["prompt_transcript"][-1]["text"])
            self.assertTrue(row["rule_identifiable"])
            with self.assertRaises(ValueError):
                rules.episode(self.manifest, 0, "explicit_natural", rule, 8, 0)

    def test_parser_and_other_rule_metrics_are_recomputed(self):
        good = self.cell()
        rules.validate_cell(good, self.manifest)
        self.assertEqual(good["summary"]["accuracy"], 1)
        self.assertEqual(good["summary"]["answers_other_rule_rate"], 0)
        bad = self.cell(output="mip zorp kiki lurn")
        self.assertEqual(bad["summary"]["unparseable_rate"], 1)
        self.assertEqual(bad["summary"]["answers_other_rule_rate"], 0)
        row = good["records"][0]
        with self.assertRaisesRegex(ValueError, "string"):
            rules.record_output(row, None)
        flipped = rules.record_output(row, row["label_space"][row["other_rule_gold"]])
        self.assertFalse(flipped["correct"])
        self.assertTrue(flipped["answers_other_rule"])
        tampered = copy.deepcopy(good)
        tampered["records"][0]["correct"] = False
        with self.assertRaises(ValueError):
            rules.validate_cell(tampered, self.manifest)
        tampered = copy.deepcopy(good)
        tampered["summary"]["accuracy"] = 0
        with self.assertRaises(ValueError):
            rules.validate_cell(tampered, self.manifest)

    def test_pair_metrics_require_pairing_and_count_invalid_as_failures(self):
        left, right = self.cell(), self.cell("right_color")
        paired = rules.paired_rule_metrics(left, right)
        self.assertEqual(paired["both_rules_correct_rate"], 1)
        self.assertEqual(paired["semantic_switch_rate"], 1)
        invalid = self.cell("right_color", output="no answer")
        paired = rules.paired_rule_metrics(left, invalid)
        self.assertEqual(paired["both_rules_correct_rate"], 0)
        self.assertEqual(paired["semantic_switch_rate"], 0)
        self.assertEqual(paired["either_unparseable_rate"], 1)
        right["records"][0]["query_id"] = "changed"
        with self.assertRaises(ValueError):
            rules.paired_rule_metrics(left, right)

    def test_k0_exact_same_prompt_determinism_is_checked(self):
        left = self.cell(k=0, output="mip")
        right = self.cell("right_color", k=0, output="mip")
        self.assertEqual(rules.paired_rule_metrics(left, right)["both_rules_correct_rate"], 0)
        right = self.cell("right_color", k=0, output="zorp")
        with self.assertRaisesRegex(ValueError, "identical K0"):
            rules.paired_rule_metrics(left, right)

    def test_manifest_pixel_and_semantic_corruption_rejected(self):
        changed = copy.deepcopy(self.manifest)
        next(iter(changed["scenes"].values()))["left_color"] = 3
        self.path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "metadata"):
            rules.load_manifest(self.path)
        self.path.write_text(json.dumps(self.manifest))
        scene = next(iter(self.manifest["scenes"].values()))
        (self.root / scene["png_path"]).write_bytes(b"not an image")
        with self.assertRaisesRegex(ValueError, "checksum"):
            rules.load_manifest(self.path)

    def test_image_path_traversal_and_symlink_escape_rejected(self):
        for path in ("../outside.png", "/tmp/outside.png"):
            with self.assertRaises(ValueError):
                rules._safe_image_path(self.root, path)
        (self.root / "escape").symlink_to(self.root.parent, target_is_directory=True)
        with self.assertRaises(ValueError):
            rules._safe_image_path(self.root, "escape/outside.png")

    def test_cli_preflight_never_loads_model_or_writes_results(self):
        output = self.root / "result.json"
        argv = ["rule_transfer", "--data-dir", str(self.root), "--partition", "development",
                "--n", "8", "--seeds", "0", "--ks", "0,8", "--out", str(output), "--preflight"]
        with patch("sys.argv", argv), patch("incontext_collapse.models.load_model") as loader:
            rules.main()
        loader.assert_not_called()
        self.assertFalse(output.exists())

    def test_cli_mock_run_resume_and_tampered_pairs_fail_closed(self):
        output = self.root / "result.json"
        argv = ["rule_transfer", "--data-dir", str(self.root), "--partition", "development",
                "--n", "8", "--seeds", "0", "--ks", "0,8", "--out", str(output)]
        model = Mock()
        model.eval.return_value = model
        with (patch("sys.argv", argv), patch("incontext_collapse.core.pick_device", return_value="cpu"),
              patch("incontext_collapse.models.load_model", return_value=(model, None)) as loader,
              patch("incontext_collapse.core.predict", return_value="mip") as predict):
            rules.main()
            self.assertEqual(predict.call_count, 6 * 8)
            self.assertEqual(loader.call_count, 1)
            original = output.read_bytes()
            rules.main()
            self.assertEqual(loader.call_count, 1)
            self.assertEqual(output.read_bytes(), original)
        with (patch("sys.argv", [*argv, "--max-new-tokens", "5", "--preflight"]),
              self.assertRaisesRegex(RuntimeError, "configuration mismatch")):
            rules.main()
        result = json.loads(output.read_text())
        result["paired_rules"]["0:frozen:implicit_remap:8"]["semantic_switch_rate"] = 1
        output.write_text(json.dumps(result))
        with patch("sys.argv", [*argv, "--preflight"]), self.assertRaisesRegex(ValueError, "paired-rule"):
            rules.main()


if __name__ == "__main__":
    unittest.main()
