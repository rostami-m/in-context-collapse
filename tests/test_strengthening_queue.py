import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_strengthening_queue as queue


class StrengtheningQueueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def test_fixed_budget_and_sequence(self):
        jobs = queue.stages()
        self.assertEqual([job["name"] for job in jobs], [
            "target_smoke", "rule_development", "natural_transfer_cifar",
            "natural_transfer_fashion", "rule_confirmatory"])
        self.assertEqual([job["planned_generations"] for job in jobs], [256, 192, 7680, 7680, 3840])
        self.assertEqual(sum(job["planned_generations"] for job in jobs), 19648)
        self.assertEqual(len({job["output"] for job in jobs}), 5)
        for job in jobs:
            self.assertTrue(job["output"].startswith(queue.OUT + "/"))
            self.assertEqual(job["command"][-2:], ["--out", job["output"]])
            self.assertEqual(job["expected_config"]["arms"],
                             ["frozen", "natural_connector", "episode_connector", "all_layer"])
            self.assertNotIn("--steps", job["command"])
        self.assertEqual(jobs[-1]["expected_config"]["condition_ks"],
                         {"explicit_natural": [0], "implicit_remap": [0, 8, 16]})
        self.assertEqual(jobs[-1]["expected_config"]["partition"], "confirmatory")
        self.assertEqual(jobs[1]["expected_config"]["partition"], "development")

    def checkpoints(self, path="old"):
        return {arm: {str(seed): {"path": path, "base_model_name_or_path": queue.MODEL,
                "sha256": {"adapter_model.safetensors": f"{arm}{seed}"},
                "training_source": {"path": path + "/training.json", "sha256": "source",
                                    "objective": "natural" if arm == "natural_connector" else "episode_remap"}}
                      for seed in (0, 1, 2)} for arm in queue.ADAPTERS}

    def test_checkpoint_identity_ignores_paths_but_not_bytes_or_objective(self):
        one, two = self.checkpoints(), self.checkpoints("relocated")
        self.assertEqual(queue.checkpoint_identity(one), queue.checkpoint_identity(two))
        two["natural_connector"]["0"]["training_source"]["objective"] = "episode_remap"
        self.assertNotEqual(queue.checkpoint_identity(one), queue.checkpoint_identity(two))

    def test_validation_cannot_shrink_axes_or_swap_checkpoints(self):
        job = queue.stages()[0]
        checkpoints = self.checkpoints()
        selected = {arm: {"0": by_seed["0"]} for arm, by_seed in checkpoints.items()}
        config = {**job["expected_config"], "checkpoints": selected}
        inputs = {"checkpoints": queue.checkpoint_identity(checkpoints)}
        queue.validate_stage_config(job, {"run_config": config}, inputs)
        for key, value in (("n", 0), ("arms", ["frozen"]), ("conditions", ["standard"]),
                           ("ks", [0]), ("seeds", [])):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "configuration mismatch"):
                queue.validate_stage_config(job, {"run_config": {**config, key: value}}, inputs)
        with self.assertRaisesRegex(ValueError, "checkpoint provenance"):
            queue.validate_stage_config(job, {"run_config": {**config, "checkpoints": {}}}, inputs)

    def test_rule_artifact_is_bound_to_both_preflight_manifests(self):
        job = queue.stages()[-1]
        checkpoints = self.checkpoints()
        inputs = {"checkpoints": queue.checkpoint_identity(checkpoints), "manifests": {
            "development": {"sha256": "development-hash"},
            "confirmatory": {"sha256": "confirmatory-hash"}}}
        config = {**job["expected_config"], "checkpoints": checkpoints,
                  "manifest_sha256": "confirmatory-hash",
                  "development_manifest": {"sha256": "development-hash"}}
        queue.validate_stage_config(job, {"run_config": config}, inputs)
        for key, replacement in (("manifest_sha256", "other"), ("development_manifest", None)):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "manifest differs"):
                queue.validate_stage_config(job, {"run_config": {**config, key: replacement}}, inputs)

    def patch_execution(self):
        for name, value in (("source_fingerprint", {"fixed": "hash"}), ("preflight", {"inputs": "fixed"})):
            patcher = patch.object(queue, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def output_report(self, job, *_args):
        output = self.root / job["output"]
        self.write(output, {"valid": True})
        return {"path": "report.json", "sha256": "report", "artifact_sha256": queue.digest(output)}

    def test_tests_model_audit_order_then_safe_resume(self):
        self.patch_execution()
        jobs = queue.stages(self.root)[:1]
        with patch.object(queue.subprocess, "run") as run, \
                patch.object(queue, "audit_stage", side_effect=self.output_report) as audit:
            queue.execute(jobs, self.root)
            commands = [call.args[0] for call in run.call_args_list]
            self.assertIn("unittest", commands[1])
            self.assertEqual(commands[2], jobs[0]["command"])
            self.assertEqual(audit.call_count, 1)
            environment = run.call_args_list[2].kwargs["env"]
            self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
            self.assertEqual(environment["HF_DATASETS_OFFLINE"], "1")
        state_path = self.root / queue.CONTROL / "status.json"
        self.assertEqual(json.loads(state_path.read_text())["status"], "complete")
        with patch.object(queue.subprocess, "run") as run, \
                patch.object(queue, "audit_stage", side_effect=self.output_report):
            queue.execute(jobs, self.root)
            self.assertNotIn(jobs[0]["command"], [call.args[0] for call in run.call_args_list])

    def test_audit_failure_prevents_next_job(self):
        self.patch_execution()
        jobs = queue.stages(self.root)[:2]
        with patch.object(queue.subprocess, "run") as run, \
                patch.object(queue, "audit_stage", side_effect=ValueError("invalid predictions")):
            with self.assertRaisesRegex(ValueError, "invalid predictions"):
                queue.execute(jobs, self.root)
            self.assertNotIn(jobs[1]["command"], [call.args[0] for call in run.call_args_list])
        self.assertEqual(json.loads((self.root / queue.CONTROL / "status.json").read_text())["status"], "failed")

    def test_failed_unit_tests_prevent_inference(self):
        self.patch_execution()
        jobs = queue.stages(self.root)[:1]
        def run(command, **_kwargs):
            if "unittest" in command:
                raise subprocess.CalledProcessError(1, command)
        with patch.object(queue.subprocess, "run", side_effect=run) as process:
            with self.assertRaises(subprocess.CalledProcessError):
                queue.execute(jobs, self.root)
            self.assertNotIn(jobs[0]["command"], [call.args[0] for call in process.call_args_list])

    def test_source_change_during_inference_stops_before_audit(self):
        self.patch_execution()
        job = queue.stages(self.root)[0]
        changed = False
        def run(command, **_kwargs):
            nonlocal changed
            if command == job["command"]:
                changed = True
        with patch.object(queue, "source_fingerprint", side_effect=lambda _root: {"fixed": "changed" if changed else "hash"}), \
                patch.object(queue.subprocess, "run", side_effect=run), patch.object(queue, "audit_stage") as audit:
            with self.assertRaisesRegex(RuntimeError, "sources changed"):
                queue.execute([job], self.root)
            audit.assert_not_called()

    def test_changed_plan_sources_and_inputs_refuse_resume(self):
        self.patch_execution()
        jobs = queue.stages(self.root)[:1]
        base = {"source_sha256": {"fixed": "hash"}, "fixed_plan": jobs, "inputs": {"inputs": "fixed"}}
        for field, replacement in (("source_sha256", {}), ("fixed_plan", []), ("inputs", {})):
            with self.subTest(field=field):
                self.write(self.root / queue.CONTROL / "status.json", {**base, field: replacement})
                with patch.object(queue.subprocess, "run") as run, self.assertRaises(RuntimeError):
                    queue.execute(jobs, self.root)
                run.assert_not_called()

    def test_changed_completed_output_refuses_resume(self):
        self.patch_execution()
        jobs = queue.stages(self.root)[:1]
        with patch.object(queue.subprocess, "run"), patch.object(queue, "audit_stage", side_effect=self.output_report):
            queue.execute(jobs, self.root)
        self.write(self.root / jobs[0]["output"], {"valid": "changed"})
        with self.assertRaisesRegex(RuntimeError, "artifact changed"):
            queue.execute(jobs, self.root)

    def test_pilot_boundary_keeps_full_plan_and_resumes(self):
        self.patch_execution()
        jobs = queue.stages(self.root)[:2]
        with patch.object(queue.subprocess, "run") as run, patch.object(queue, "audit_stage", side_effect=self.output_report):
            queue.execute(jobs, self.root, stop_after=jobs[0]["name"])
            self.assertNotIn(jobs[1]["command"], [call.args[0] for call in run.call_args_list])
        state = json.loads((self.root / queue.CONTROL / "status.json").read_text())
        self.assertEqual(state["status"], "pilot_complete")
        self.assertEqual(len(state["fixed_plan"]), 2)
        with patch.object(queue.subprocess, "run") as run, patch.object(queue, "audit_stage", side_effect=self.output_report):
            queue.execute(jobs, self.root)
            self.assertNotIn(jobs[0]["command"], [call.args[0] for call in run.call_args_list])
            self.assertIn(jobs[1]["command"], [call.args[0] for call in run.call_args_list])

    def test_status_shows_partial_cells_without_declaring_completion(self):
        self.assertEqual(queue.status(self.root)["status"], "not_started")
        job = queue.stages()[0]
        self.write(self.root / job["output"], {"cells": {"one": {"records": [{}, {}, {}, {}]}}})
        self.write(self.root / queue.CONTROL / "status.json", {
            "status": "running", "pid": 123, "current_stage": job["name"], "stages": [job]})
        result = queue.status(self.root)
        self.assertEqual(result["stages"][0]["saved_records"], 4)
        self.assertFalse(result["publication_ready"])


if __name__ == "__main__":
    unittest.main()
