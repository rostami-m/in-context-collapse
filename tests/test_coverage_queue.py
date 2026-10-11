import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts import run_revision_coverage_queue as queue


class CoverageQueueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def first_manifest(self, status="complete"):
        return {"status": status, "source_sha256": {"core.py": "abc"},
                "stages": [{"name": "first", "status": "complete"}]}

    def test_stage_dimensions_and_commands(self):
        jobs = queue.stages()
        self.assertEqual(len(jobs), 6)
        self.assertEqual([job["planned_generations"] for job in jobs], [7200, 2880, 0, 0, 0, 0])
        self.assertEqual([job["planned_scoring_queries"] for job in jobs], [0, 0, 960, 960, 0, 0])
        self.assertEqual([job["planned_feature_records"] for job in jobs], [0, 0, 0, 0, 720, 720])
        self.assertEqual(len({job["output"] for job in jobs}), 6)
        for job in jobs:
            self.assertTrue(job["output"].startswith(queue.OUT + "/"))
            self.assertEqual(job["command"][-2:], ["--out", job["output"]])
            self.assertEqual(job["expected_config"]["seeds"], [0, 1, 2])
            self.assertEqual(job["expected_config"]["n"], 40)
        self.assertEqual(jobs[0]["expected_config"]["arms"],
                         ["frozen", "early", "mid", "late", "natural_connector"])
        self.assertEqual(jobs[1]["expected_config"]["ks"], [0, 1, 2, 4, 8, 16])
        self.assertEqual(jobs[1]["expected_config"]["model"], queue.QWEN7)
        for job in jobs[:2]:
            self.assertEqual(job["expected_config"]["max_new_tokens"], 6)
            self.assertEqual(job["expected_config"]["option_orders"], ["random"])

    def test_wait_requires_complete_first_grid(self):
        path = self.root / "first.json"
        self.write(path, self.first_manifest("running"))
        sleeper = Mock(side_effect=lambda _: self.write(path, self.first_manifest()))
        check = Mock()
        queue.wait_for_first_success(path, {"core.py": "abc"}, ["first"], check,
                                     sleep=sleeper, poll_seconds=1)
        self.assertEqual(sleeper.call_count, 1)
        self.assertEqual(check.call_count, 2)
        partial = self.first_manifest()
        partial["stages"] = []
        self.write(path, partial)
        with self.assertRaisesRegex(RuntimeError, "entire required stage grid"):
            queue.first_queue_complete(path, {"core.py": "abc"}, ["first"])

    def test_wait_rejects_failed_or_changed_first_queue(self):
        path = self.root / "first.json"
        self.write(path, self.first_manifest("failed"))
        with self.assertRaisesRegex(RuntimeError, "first queue failed"):
            queue.wait_for_first_success(path, {"core.py": "abc"}, ["first"], Mock(), sleep=Mock())
        with self.assertRaisesRegex(RuntimeError, "fingerprint"):
            queue.first_queue_complete(path, {"core.py": "changed"}, ["first"])

    def test_wait_rechecks_coverage_sources(self):
        check = Mock(side_effect=[None, RuntimeError("changed")])
        with self.assertRaisesRegex(RuntimeError, "changed"):
            queue.wait_for_first_success(self.root / "absent.json", {}, [], check,
                                         sleep=Mock(), poll_seconds=1)
        self.assertEqual(check.call_count, 2)

    def test_fingerprint_includes_first_queue_and_new_sources(self):
        with patch.object(queue.first_queue, "source_fingerprint", return_value={"core.py": "abc"}):
            fingerprint = queue.source_fingerprint()
        self.assertEqual(fingerprint["first_queue"], {"core.py": "abc"})
        self.assertEqual(set(fingerprint["coverage"]), {
            "scripts/run_revision_coverage_queue.py", "tests/test_coverage_queue.py",
            "incontext_collapse/frozen_features.py"})
        self.assertTrue(all(len(digest) == 64 for digest in fingerprint["coverage"].values()))

    def test_shared_lock_excludes_other_model_runner(self):
        path = self.root / queue.FIRST_CONTROL / "run.lock"
        with queue.execution_lock(path):
            with self.assertRaisesRegex(RuntimeError, "another process"):
                with queue.execution_lock(path):
                    self.fail("second runner acquired occupied lock")
        with queue.execution_lock(path):
            pass

    def test_lock_wait_checks_status_before_retry(self):
        calls = [BlockingIOError(), None, None]
        with patch.object(queue.fcntl, "flock", side_effect=calls), patch.object(queue.time, "sleep"):
            check, sleeper = Mock(), Mock()
            with queue.execution_lock(self.root / "lock", wait=True, check=check, sleep=sleeper):
                pass
        self.assertEqual(check.call_count, 2)
        sleeper.assert_called_once_with(10)

    def feature_result(self):
        job = queue.stages()[4]
        config = job["expected_config"]
        result = {"run_config": config, "seeds_done": config["seeds"], "evidence": {},
                  "summary": {condition: {str(k): {} for k in config["ks"]}
                              for condition in config["conditions"]}}
        for seed in config["seeds"]:
            result["evidence"][str(seed)] = {
                condition: {str(k): {"seed": seed, "condition": condition, "k": k,
                                    "summary": {"n": 40},
                                    "records": [{"query_index": index} for index in range(40)]}
                            for k in config["ks"]} for condition in config["conditions"]}
        return job, result

    def test_frozen_feature_complete_grid(self):
        job, result = self.feature_result()
        self.write(self.root / job["output"], result)
        queue.validate_output(job, self.root)

    def test_frozen_feature_missing_extra_and_misidentified_cells(self):
        for corruption in ("seed", "condition", "k", "extra", "identity", "records", "index", "summary"):
            with self.subTest(corruption=corruption):
                job, result = self.feature_result()
                shots = result["evidence"]["0"]["remap"]
                if corruption == "seed":
                    result["seeds_done"] = [0, 1]
                elif corruption == "condition":
                    del result["evidence"]["0"]["remap_shuffled"]
                elif corruption == "k":
                    del shots["8"]
                elif corruption == "extra":
                    shots["32"] = shots["16"]
                elif corruption == "identity":
                    shots["4"]["seed"] = 1
                elif corruption == "records":
                    shots["4"]["records"].pop()
                elif corruption == "index":
                    shots["4"]["records"][0]["query_index"] = 1
                else:
                    del result["summary"]["remap"]["4"]
                self.write(self.root / job["output"], result)
                with self.assertRaises(ValueError):
                    queue.validate_output(job, self.root)

    def test_output_config_cannot_shrink_expected_grid(self):
        job, result = self.feature_result()
        result["run_config"] = {**result["run_config"], "ks": [4]}
        self.write(self.root / job["output"], result)
        with self.assertRaisesRegex(ValueError, "configuration mismatch: ks"):
            queue.validate_output(job, self.root)

    def test_cached_model_requires_all_pinned_shards(self):
        snapshot = (self.root / ("models--" + queue.QWEN7.replace("/", "--")) / "snapshots" /
                    queue.model_revision(queue.QWEN7))
        snapshot.mkdir(parents=True)
        for name in ("config.json", "preprocessor_config.json", "tokenizer.json", "vocab.json",
                     "merges.txt", "generation_config.json"):
            (snapshot / name).write_text("{}")
        self.write(snapshot / "tokenizer_config.json", {"chat_template": "template"})
        self.write(snapshot / "model.safetensors.index.json", {"weight_map": {"a": "part1.safetensors",
                                                                              "b": "part2.safetensors"}})
        (snapshot / "part1.safetensors").write_bytes(b"one")
        with self.assertRaisesRegex(FileNotFoundError, "weight shard missing"):
            queue.cached_model_manifest(queue.QWEN7, self.root)
        (snapshot / "part2.safetensors").write_bytes(b"two")
        report = queue.cached_model_manifest(queue.QWEN7, self.root)
        self.assertEqual(len(report["shard_bytes"]), 2)

    def execute_patches(self):
        fingerprint = {"first_queue": {"core.py": "abc"}, "coverage": {"queue.py": "def"}}
        self.write(self.root / queue.FIRST_CONTROL / "status.json", self.first_manifest())
        for target, kwargs in (
            ("source_fingerprint", {"return_value": fingerprint}),
            ("preflight", {"return_value": {"cache": "present"}}),
        ):
            patcher = patch.object(queue, target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(queue.first_queue, "stages", return_value=[{"name": "first"}])
        patcher.start()
        self.addCleanup(patcher.stop)
        return fingerprint

    def test_execution_tests_then_model_then_audit_and_resumes_complete(self):
        self.execute_patches()
        job = queue.stages()[0]
        self.write(self.root / job["output"], {"run_config": job["expected_config"]})
        with patch.object(queue.subprocess, "run") as run:
            queue.execute([job], self.root)
            commands = [call.args[0] for call in run.call_args_list]
        self.assertIn("unittest", commands[1])
        self.assertEqual(commands[2], job["command"])
        self.assertIn("scripts/audit_revision_artifacts.py", commands[3])
        status = json.loads((self.root / queue.CONTROL / "status.json").read_text())
        self.assertEqual(status["status"], "complete")
        with patch.object(queue.subprocess, "run") as run:
            queue.execute([job], self.root)
            self.assertNotIn(job["command"], [call.args[0] for call in run.call_args_list])

    def test_failed_test_or_audit_prevents_later_jobs(self):
        self.execute_patches()
        jobs = queue.stages()[:2]
        for failure_index in (1, 3):
            with self.subTest(failure_index=failure_index):
                status_path = self.root / queue.CONTROL / "status.json"
                status_path.unlink(missing_ok=True)
                calls = []

                def run(command, **kwargs):
                    calls.append(command)
                    if len(calls) - 1 == failure_index:
                        raise subprocess.CalledProcessError(1, command)

                with patch.object(queue.subprocess, "run", side_effect=run):
                    with self.assertRaises(subprocess.CalledProcessError):
                        queue.execute(jobs, self.root)
                self.assertNotIn(jobs[1]["command"], calls)
                self.assertEqual(json.loads(status_path.read_text())["status"], "failed")
                if failure_index == 1:
                    self.assertNotIn(jobs[0]["command"], calls)

    def test_changed_source_rejects_resume_before_preflight_or_model(self):
        self.execute_patches()
        self.write(self.root / queue.CONTROL / "status.json", {"source_sha256": {"old": "hash"}})
        with patch.object(queue.subprocess, "run") as run, patch.object(queue, "preflight") as preflight:
            with self.assertRaisesRegex(RuntimeError, "source changed"):
                queue.execute(queue.stages(), self.root)
        run.assert_not_called()
        preflight.assert_not_called()

    def test_first_queue_failure_never_starts_model_subprocess(self):
        self.execute_patches()
        self.write(self.root / queue.FIRST_CONTROL / "status.json", self.first_manifest("failed"))
        with patch.object(queue.subprocess, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "first queue failed"):
                queue.execute(queue.stages(), self.root)
        run.assert_not_called()
        status = json.loads((self.root / queue.CONTROL / "status.json").read_text())
        self.assertEqual(status["status"], "failed")

    def test_source_change_during_tests_prevents_model_job(self):
        fingerprint = self.execute_patches()
        changed = {**fingerprint, "coverage": {"queue.py": "changed"}}
        job = queue.stages()[0]
        with patch.object(queue, "source_fingerprint", side_effect=[fingerprint] * 5 + [changed]), \
                patch.object(queue.subprocess, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "sources changed"):
                queue.execute([job], self.root)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(len(commands), 2)
        self.assertIn("unittest", commands[1])
        self.assertNotIn(job["command"], commands)


if __name__ == "__main__":
    unittest.main()
