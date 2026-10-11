import copy
import hashlib
import json
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest.mock import patch

from scripts import export_strengthening_tables as exporter


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/strengthening_v1"


class StrengtheningTableGuards(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directory = self.root / "data/strengthening_v1"
        self.directory.mkdir(parents=True)
        self.path = self.directory / "natural_transfer_cifar.json"
        self.path.write_text(json.dumps({"run_config": {}, "cells": {}}))
        self.report_path = self.directory / "reports/natural_transfer_cifar.json"
        self.report_path.parent.mkdir()
        self.saved = {"path": str(self.path), "sha256": exporter.sha256(self.path),
                      "status": "complete", "errors": []}
        self.report_path.write_text(json.dumps(self.saved))

    def test_changed_artifact_hash_is_rejected_before_audit(self):
        self.path.write_text(self.path.read_text() + " ")
        with patch.object(exporter, "audit_file") as audit:
            with self.assertRaisesRegex(ValueError, "artifact bytes changed"):
                exporter.Evidence(self.root)
        audit.assert_not_called()

    def test_failed_or_partial_fresh_audit_is_never_exported(self):
        for status in ("partial", "invalid"):
            with self.subTest(status=status), patch.object(exporter, "audit_file", return_value={
                    "status": status, "errors": ["incomplete grid"]}) as audit:
                with self.assertRaisesRegex(ValueError, "independent audit rejected"):
                    exporter.Evidence(self.root)
                self.assertEqual(audit.call_args.kwargs["expected_config"]["seeds"], [0, 1, 2])
                self.assertEqual(audit.call_args.kwargs["bootstrap_draws"], 10000)

    def test_report_must_refer_to_the_expected_local_artifact(self):
        self.saved["path"] = "/old/workspace/data/strengthening_v1/unrelated.json"
        self.report_path.write_text(json.dumps(self.saved))
        with self.assertRaisesRegex(ValueError, "another artifact"):
            exporter.Evidence(self.root)

    def test_relocated_report_paths_are_local_and_do_not_read_external_decoys(self):
        saved = {**self.saved, "path": "/old/workspace/data/strengthening_v1/natural_transfer_cifar.json"}
        exporter.validate_saved_report(saved, self.saved, self.root)
        self.assertEqual(exporter.canonical_data_path(saved["path"], self.root),
                         "data/strengthening_v1/natural_transfer_cifar.json")
        self.assertEqual(exporter.canonical_data_path(str(self.path), self.root),
                         "data/strengthening_v1/natural_transfer_cifar.json")

    def test_unrelated_paths_traversal_and_symlinks_rejected(self):
        for path in ("/external/unrelated.json", "../data/result.json", "data/../outside.json",
                     "/old/data/repo/data/result.json", "elsewhere/result.json"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                exporter.canonical_data_path(path, self.root)
        (self.directory / "escape.json").symlink_to(self.root / "outside.json")
        with self.assertRaisesRegex(ValueError, "escapes"):
            exporter.canonical_data_path("data/strengthening_v1/escape.json", self.root)

    def test_saved_report_status_and_numerical_tampering_rejected(self):
        fresh = {**self.saved, "estimates": [{"mean": .5, "ci95_low": .1,
                    "conditional_case_interval": {"ci95_high": .9}}]}
        for field in ("mean", "bootstrap", "status", "sha256"):
            bad = copy.deepcopy(fresh)
            if field == "mean":
                bad["estimates"][0]["mean"] = .9
            elif field == "bootstrap":
                bad["estimates"][0]["conditional_case_interval"]["ci95_high"] = .95
            elif field == "status":
                bad["status"] = "partial"
            else:
                bad["sha256"] = "0" * 64
            with self.subTest(field=field), self.assertRaises(ValueError):
                exporter.validate_saved_report(bad, fresh, self.root)
        exporter.validate_saved_report({**fresh, "publication_ready": False,
                                        "queue_purpose": "fixed evaluation"}, fresh, self.root)

    def test_nonfinite_or_boolean_numeric_report_fields_rejected(self):
        for value in (float("nan"), float("inf"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                exporter.require_equal(value, 1.0)


@unittest.skipUnless(all((DATA / f"{name}.json").exists() for name in exporter.INPUTS),
                     "completed strengthening evidence is not present")
class CompletedStrengtheningTables(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = exporter.Evidence(ROOT)
        cls.fragments = exporter.build_fragments(cls.evidence)

    def test_real_inputs_are_freshly_audited_and_reports_verified(self):
        self.assertEqual(set(self.evidence.raw), set(exporter.INPUTS))
        self.assertTrue(all(report["status"] == "complete" and not report["errors"]
                            for report in self.evidence.reports.values()))
        provenance = json.loads(self.fragments["strengthening_provenance.json"])
        self.assertEqual(provenance["audit"]["cells"], 480)
        self.assertEqual(provenance["audit"]["records"], 19200)
        self.assertEqual(provenance["audit"]["bootstrap_draws_recomputed"], 10000)
        for name, digest in provenance["fragment_sha256"].items():
            self.assertEqual(hashlib.sha256(self.fragments[name].encode()).hexdigest(), digest)

    def test_primary_arithmetic_and_standard_tradeoff_from_raw_records(self):
        expected = {"cifar": (.13333333333333333, .07855882440168062, -.044444444444444446),
                    "fashion": (.06111111111111111, -.022558333333333354, -.06944444444444445)}
        for domain, (mean, lower, standard) in expected.items():
            name, task = f"natural_transfer_{domain}", domain + "4"
            effect = self.evidence.objective_effect(name, task, "remap")
            self.assertAlmostEqual(effect["mean"], mean)
            self.assertAlmostEqual(effect["ci95_low"], lower)
            self.assertEqual(effect["n_seeds"], 3)
            self.assertEqual(set(effect["seed_differences"]), {"0", "1", "2"})
            self.assertAlmostEqual(self.evidence.objective_effect(name, task, "standard")["mean"], standard)

    def test_estimators_use_raw_records_not_cell_summaries(self):
        name = "natural_transfer_cifar"
        cell = self.evidence.raw[name]["cells"]["0:cifar4:episode_connector:remap:4:random"]
        before = self.evidence.target_estimate(name, "cifar4", "episode_connector", "remap", [4])
        original = cell["summary"]
        try:
            cell["summary"] = {"accuracy": 1.0, "n": 999}
            after = self.evidence.target_estimate(name, "cifar4", "episode_connector", "remap", [4])
        finally:
            cell["summary"] = original
        self.assertEqual(before, after)

    def test_rule_aggregation_joint_denominator_and_invalid_definition(self):
        evidence = self.evidence
        self.assertAlmostEqual(evidence.rule_estimate("frozen", "implicit_remap", [8, 16])["mean"], .24583333333333335)
        self.assertAlmostEqual(evidence.rule_estimate("episode_connector", "implicit_remap", [8, 16])["mean"], .25)
        self.assertAlmostEqual(evidence.rule_estimate("frozen", "implicit_remap", [8, 16], metric="both_correct")["mean"], 1 / 240)
        self.assertAlmostEqual(evidence.rule_estimate("natural_connector", "implicit_remap", [8, 16], metric="both_correct")["mean"], 14 / 240)
        self.assertEqual(evidence.rule_estimate("all_layer", "implicit_remap", [0], metric="both_correct")["mean"], 0)
        self.assertAlmostEqual(evidence.rule_estimate("all_layer", "implicit_remap", [0], metric="invalid")["mean"], .05)
        for arm in exporter.ARMS:
            self.assertEqual(evidence.rule_estimate(arm, "explicit_natural", [0])["mean"], 1)
        manual = []
        for seed in (0, 1, 2):
            manual.append(statistics.mean(row["prediction"] is None
                for rule in exporter.RULES
                for row in evidence.rule_rows(seed, "all_layer", "implicit_remap", rule, 8)))
        self.assertAlmostEqual(evidence.rule_estimate("all_layer", "implicit_remap", [8], metric="invalid")["mean"], statistics.mean(manual))

    def test_fragment_layout_rounding_and_full_curves(self):
        expected_rows = {"objective_summary": 4, "objective_effects": 2,
                         "objective_curves": 8, "rule_summary": 4, "rule_curves": 12}
        for name, count in expected_rows.items():
            rows = [line for line in self.fragments[f"strengthening_{name}.tex"].splitlines() if not line.startswith("%")]
            self.assertEqual(len(rows), count)
            self.assertTrue(all(line.endswith(r"\\") for line in rows))
        self.assertIn(r"CIFAR-4 & $0.133\,[0.079, 0.188]$", self.fragments["strengthening_objective_effects.tex"])
        self.assertIn("Episodic connector & 1.000 & 0.250 & 0.042", self.fragments["strengthening_rule_summary.tex"])
        self.assertIn("0.892/0.558/0.600/0.592", self.fragments["strengthening_objective_curves.tex"])
        self.assertEqual(self.fragments, exporter.build_fragments(self.evidence))

    def test_check_detects_stale_output_without_rewriting_it(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(exporter, "Evidence", return_value=self.evidence):
                self.assertEqual(exporter.main(["--out-dir", str(output)]), 0)
                self.assertEqual(exporter.main(["--out-dir", str(output), "--check"]), 0)
                path = output / "strengthening_rule_summary.tex"
                path.write_text("stale rows\n")
                with self.assertRaises(SystemExit) as stopped:
                    exporter.main(["--out-dir", str(output), "--check"])
                self.assertEqual(stopped.exception.code, 1)
                self.assertEqual(path.read_text(), "stale rows\n")


if __name__ == "__main__":
    unittest.main()
