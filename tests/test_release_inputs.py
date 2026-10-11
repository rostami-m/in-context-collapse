import tempfile
import hashlib
import json
from pathlib import Path
import unittest

from scripts.check_revision_release import manuscript_sources, runtime_amendments, verify_recorded_sources


class RuntimeProvenanceTests(unittest.TestCase):
    def test_unchanged_nested_fingerprints_pass_without_amendments(self):
        recorded = {"previous": {"module.py": "a"}, "new": {"test.py": "b"}}
        verify_recorded_sources(recorded, recorded, {})

    def test_unapproved_change_is_rejected(self):
        with self.assertRaisesRegex(SystemExit, "unrecorded runtime"):
            verify_recorded_sources({"module.py": "a"}, {"module.py": "b"}, {})

    def test_exact_amendment_does_not_allow_a_different_change(self):
        entries = {"module.py": {"recorded_sha256": "a", "current_sha256": "b"}}
        verify_recorded_sources({"previous": {"module.py": "a"}},
                                {"previous": {"module.py": "b"}}, entries)
        with self.assertRaisesRegex(SystemExit, "unrecorded runtime"):
            verify_recorded_sources({"module.py": "a"}, {"module.py": "c"}, entries)
        with self.assertRaisesRegex(SystemExit, "keys changed"):
            verify_recorded_sources({"module.py": "a"}, {}, entries)

    def fixture(self, root):
        files = {"module.py": b"current", "archive/module.py": b"recorded", "tests/test.py": b"test"}
        for name, data in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        sha = lambda name: hashlib.sha256(files[name]).hexdigest()
        entry = {"path": "module.py", "recorded_sha256": sha("archive/module.py"),
                 "current_sha256": sha("module.py"), "archived_source": "archive/module.py",
                 "reason": "tested cleanup change", "regression_tests": {"tests/test.py": sha("tests/test.py")}}
        path = root / "reproducibility/runtime_amendments.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"schema_version": 1, "amendments": [entry]}))
        return entry, path

    def test_snapshot_current_runtime_and_tests_are_all_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry, _ = self.fixture(root)
            self.assertEqual(runtime_amendments(root), {"module.py": entry})
            for name in ("module.py", "archive/module.py", "tests/test.py"):
                original = (root / name).read_bytes()
                (root / name).write_bytes(b"changed")
                with self.assertRaisesRegex(SystemExit, "hash mismatch"):
                    runtime_amendments(root)
                (root / name).write_bytes(original)

    def test_missing_snapshot_and_external_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry, path = self.fixture(root)
            (root / entry["archived_source"]).unlink()
            with self.assertRaisesRegex(SystemExit, "missing runtime"):
                runtime_amendments(root)
            entry["archived_source"] = "../outside.py"
            path.write_text(json.dumps({"schema_version": 1, "amendments": [entry]}))
            with self.assertRaisesRegex(SystemExit, "outside repository"):
                runtime_amendments(root)

    def test_test_manifest_cannot_override_source_or_archive_checks(self):
        for collision in ("module.py", "archive/module.py", "archive/../module.py"):
            with self.subTest(collision=collision), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                entry, path = self.fixture(root)
                entry["regression_tests"][collision] = hashlib.sha256((root / collision).read_bytes()).hexdigest()
                path.write_text(json.dumps({"schema_version": 1, "amendments": [entry]}))
                with self.assertRaisesRegex(SystemExit, "colliding runtime"):
                    runtime_amendments(root)

    def test_wrong_archive_hash_cannot_be_replaced_by_a_test_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry, path = self.fixture(root)
            (root / entry["archived_source"]).write_bytes(b"not recorded execution code")
            entry["regression_tests"][entry["archived_source"]] = hashlib.sha256(
                (root / entry["archived_source"]).read_bytes()).hexdigest()
            path.write_text(json.dumps({"schema_version": 1, "amendments": [entry]}))
            with self.assertRaisesRegex(SystemExit, "hash mismatch"):
                runtime_amendments(root)


class ManuscriptInputTests(unittest.TestCase):
    def test_nested_tables_are_counted_and_included(self):
        with tempfile.TemporaryDirectory() as directory:
            paper = Path(directory)
            (paper / "Main.tex").write_text(r"\input{appendix}")
            (paper / "appendix.tex").write_text("\\label{app:test}\n\\inputrows rows.tex\n")
            (paper / "rows.tex").write_text("Frozen & 0.250 \\\\\n")
            source, count = manuscript_sources(paper, paper / "Main.tex")
            self.assertEqual(count, 1)
            self.assertIn("Frozen & 0.250", source)
            self.assertIn(r"\label{app:test}", source)

    def test_missing_nested_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            paper = Path(directory)
            (paper / "Main.tex").write_text(r"\input{missing}")
            with self.assertRaisesRegex(SystemExit, "missing manuscript input"):
                manuscript_sources(paper, paper / "Main.tex")

    def test_recursive_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            paper = Path(directory)
            (paper / "Main.tex").write_text(r"\input{Main}")
            with self.assertRaisesRegex(SystemExit, "recursive manuscript input"):
                manuscript_sources(paper, paper / "Main.tex")

    def test_path_outside_paper_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            paper = Path(directory) / "paper"
            paper.mkdir()
            (paper / "Main.tex").write_text(r"\input{../outside}")
            with self.assertRaisesRegex(SystemExit, "outside paper directory"):
                manuscript_sources(paper, paper / "Main.tex")


if __name__ == "__main__":
    unittest.main()
