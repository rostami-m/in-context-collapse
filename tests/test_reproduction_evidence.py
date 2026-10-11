"""Failure-oriented tests for the saved-evidence boundary, without model loads."""
import hashlib
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import reproduce as r


class EvidenceTests(unittest.TestCase):
    def test_unknown_model_cannot_load_a_moving_revision(self):
        from incontext_collapse import models
        with patch.object(models.AutoProcessor, 'from_pretrained') as download:
            with self.assertRaisesRegex(ValueError, 'verified model commit'):
                models.load_model('unregistered/model', device='cpu')
            download.assert_not_called()

    def test_detects_changed_and_missing_raw_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'evidence.json').write_text('original')
            expected = {'evidence.json': {'sha256': hashlib.sha256(b'original').hexdigest()}}
            self.assertEqual(r.check_hashes(root, expected), 1)
            (root / 'evidence.json').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'changed:'):
                r.check_hashes(root, expected)
            (root / 'evidence.json').unlink()
            with self.assertRaisesRegex(ValueError, 'missing:'):
                r.check_hashes(root, expected)

    def test_manifest_cannot_escape_through_path_or_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'root'
            root.mkdir()
            outside = Path(tmp) / 'outside'
            outside.write_text('preserved')
            (root / 'link').symlink_to(outside)
            for path in ('../outside', str(outside), 'link'):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    r.contained(root, path)
            self.assertEqual(outside.read_text(), 'preserved')

    def test_output_collision_never_replaces_previous_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / 'result'
            destination.mkdir()
            (destination / 'proof').write_text('keep')
            with self.assertRaises(ValueError):
                r.fresh(destination)
            self.assertEqual((destination / 'proof').read_text(), 'keep')

    def test_training_dependencies_are_optional_but_development_is_required(self):
        manifest = {'studies': [
            {'id': 'train', 'kind': 'training', 'dependencies': []},
            {'id': 'precursor', 'kind': 'evaluation', 'dependencies': ['train']},
            {'id': 'development', 'kind': 'evaluation', 'dependencies': ['train'],
             'retrain_dependencies': ['precursor']},
            {'id': 'confirmation', 'kind': 'evaluation', 'dependencies': ['development']},
        ]}
        self.assertEqual([s['id'] for s in r.select_studies(manifest, 'confirmation')],
                         ['development', 'confirmation'])
        self.assertEqual([s['id'] for s in r.select_studies(manifest, 'confirmation', True)],
                         ['train', 'precursor', 'development', 'confirmation'])

    def test_dependency_cycle_cannot_launch(self):
        manifest = {'studies': [
            {'id': 'a', 'kind': 'evaluation', 'dependencies': ['b']},
            {'id': 'b', 'kind': 'evaluation', 'dependencies': ['a']},
        ]}
        with self.assertRaisesRegex(ValueError, 'cyclic'):
            r.select_studies(manifest, 'a')


if __name__ == '__main__':
    unittest.main()
