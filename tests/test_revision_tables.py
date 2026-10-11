import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.export_revision_tables import (DEPTH, SHAPES, Evidence, cross_file_comparisons,
                                           estimate_text, interval_text, number, resolve_artifact_path)
from scripts.summarize_revision_followup import summarize_file
from tests.test_followup_summary import option_payload


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data/revision_v25/completed_revision_summary.json"


class RevisionTableTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.artifact_path = self.directory / "data/revision_v25" / SHAPES
        self.artifact_path.parent.mkdir(parents=True)
        self.artifact_path.write_text(json.dumps(option_payload()))
        self.artifact = summarize_file(self.artifact_path)
        self.report_path = self.directory / "report.json"
        self.write_report()

    def write_report(self):
        self.report_path.write_text(json.dumps({"artifacts": [self.artifact]}))

    def test_rounding_and_tex_intervals(self):
        self.assertEqual(number(.6325), "0.633")
        self.assertEqual(number(-.6325), "-0.633")
        estimate = {"mean": .5, "ci95_low": -.1, "ci95_high": 1.1}
        self.assertEqual(estimate_text(estimate, True), r"$0.500\pm0.600$")
        self.assertEqual(interval_text(estimate), r"$0.500\,[-0.100, 1.100]$")

    def test_estimates_recomputed_from_raw_seed_records(self):
        evidence = Evidence(self.report_path, root=self.directory)
        estimate = evidence.estimate(SHAPES, "frozen", "remap", 0, order="class")
        self.assertEqual(estimate["mean"], .5)
        self.assertEqual(estimate["n_seeds"], 3)
        effect, wins, losses = evidence.paired(SHAPES, "shapes4", "remap", 0, right="connector")
        self.assertEqual((wins, losses), (4, 0))
        self.assertAlmostEqual(effect["mean"], 2 / 3)

    def test_modified_raw_artifact_is_rejected(self):
        self.artifact_path.write_text(self.artifact_path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "artifact bytes changed"):
            Evidence(self.report_path, root=self.directory)

    def test_incomplete_report_is_rejected(self):
        self.artifact["status"] = "partial"
        self.write_report()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            Evidence(self.report_path, root=self.directory)

    def test_tampered_report_mean_or_seed_values_is_rejected(self):
        for field in ("mean", "seed_values"):
            original = copy.deepcopy(self.artifact)
            row = next(row for row in self.artifact["estimates"]
                       if row["arm"] == "connector" and row["option_order"] == "random")
            if field == "mean":
                row["metrics"]["accuracy"][field] = .123
            else:
                row["metrics"]["accuracy"][field]["0"] = .123
            self.write_report()
            with self.assertRaisesRegex(ValueError, "disagree"):
                Evidence(self.report_path, root=self.directory).estimate(SHAPES, "connector", "remap", 0)
            self.artifact = original

    def test_relocated_absolute_report_uses_local_artifact_and_keeps_report_bytes(self):
        external = self.directory / "old-workspace/data/revision_v25" / SHAPES
        external.parent.mkdir(parents=True)
        external.write_text("external path must not be trusted")
        self.artifact["path"] = str(external)
        self.write_report()
        before = self.report_path.read_bytes()
        evidence = Evidence(self.report_path, root=self.directory)
        self.assertEqual(evidence.estimate(SHAPES, "frozen", "remap", 0, order="class")["mean"], .5)
        self.assertEqual(self.report_path.read_bytes(), before)
        self.artifact_path.write_text(self.artifact_path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "artifact bytes changed"):
            Evidence(self.report_path, root=self.directory)

    def test_artifact_resolver_rejects_traversal_unrelated_paths_and_symlink_escape(self):
        for path in ("/outside/unrelated.json", "../data/revision_v25/result.json",
                     "data/revision_v25/../../outside.json",
                     "/old/data/revision_v25/../result.json",
                     "/old/data/revision_v25/repo/data/revision_v25/result.json"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                resolve_artifact_path(path, self.directory)
        outside = self.directory / "outside.json"
        outside.write_text("not an artifact")
        link = self.artifact_path.parent / "escape.json"
        link.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "escapes"):
            resolve_artifact_path("data/revision_v25/escape.json", self.directory)
        self.assertEqual(resolve_artifact_path(f"data/revision_v25/{SHAPES}", self.directory),
                         self.artifact_path.resolve())

    @unittest.skipUnless(REPORT.exists(), "complete revision artifacts are not present")
    def test_real_report_fragments_and_stale_check(self):
        output = self.directory / "fragments"
        command = [sys.executable, str(ROOT / "scripts/export_revision_tables.py"),
                   "--report", str(REPORT), "--out-dir", str(output)]
        generated = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(generated.returncode, 0, generated.stderr)
        checked = subprocess.run(command + ["--check"], capture_output=True, text=True, check=False)
        self.assertEqual(checked.returncode, 0, checked.stderr)
        self.assertIn("Middle &", (output / "repair_confirmatory.tex").read_text())
        self.assertIn("0.325/0.308/0.350", (output / "source_controls.tex").read_text())
        replication = (output / "connector_replication.tex").read_text()
        self.assertEqual(replication.count("Qwen2-VL-2B &"), 4)
        self.assertEqual(replication.count("Qwen2.5-VL-3B &"), 4)
        self.assertNotIn("Frozen &", replication)
        dara = (output / "dara_details.tex").read_text()
        self.assertIn("Local frozen & -- & 0.617/0.500/0.467 & 0.325/0.317/0.350 & 0.000/0.000/0.000", dara)
        self.assertNotIn("Episode connector", (output / "transfer_retention.tex").read_text())
        self.assertNotIn("likelihood", (output / "target_baselines.tex").read_text())
        cross = json.loads((output / "cross_file_comparisons.json").read_text())
        self.assertEqual(cross["matched_frozen_records"], 1440)
        self.assertEqual(cross["frozen_prediction_differences"], 0)
        self.assertEqual(len(cross["contrasts"]), 12)
        self.assertEqual([target["matched_feature_semantic_episodes"] for target in cross["target_baselines"]], [720, 720])
        self.assertEqual([target["matched_calibration_prompt_episodes"] for target in cross["target_baselines"]], [960, 960])
        (output / "source_controls.tex").write_text("stale rows\n")
        before = (output / "source_controls.tex").read_text()
        stale = subprocess.run(command + ["--check"], capture_output=True, text=True, check=False)
        self.assertEqual(stale.returncode, 1)
        self.assertIn("source_controls.tex", stale.stderr)
        self.assertEqual((output / "source_controls.tex").read_text(), before)

    @unittest.skipUnless(REPORT.exists(), "complete revision artifacts are not present")
    def test_crossfile_join_rejects_config_or_episode_mismatch(self):
        evidence = Evidence(REPORT)
        config = evidence.raw[DEPTH]["run_config"]
        original = config["img_size"]
        config["img_size"] = original + 1
        with self.assertRaisesRegex(ValueError, "configurations are not matched"):
            cross_file_comparisons(evidence)
        config["img_size"] = original
        record = evidence.cell(DEPTH, "shapes4", "remap", "frozen", 4, 0)["records"][0]
        record["query_id"] = "unpaired-query"
        with self.assertRaisesRegex(ValueError, "not paired"):
            cross_file_comparisons(evidence)


if __name__ == "__main__":
    unittest.main()
