"""Bounded HF fallback guards; no downloads or model execution."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from datasets.packaged_modules.cache.cache import Cache
from incontext_collapse import provenance


class DatasetRevisionGuardTests(unittest.TestCase):
    dataset_id = 'uoft-cs/cifar10'

    def builder(self, *, cache=True, hash_value=None):
        result = Mock(spec=Cache) if cache else Mock()
        result.hash = hash_value or provenance.DATASET_REVISIONS[self.dataset_id]
        result.info = SimpleNamespace(dataset_size=123)
        result.as_dataset.return_value = object()
        result.as_streaming_dataset.return_value = object()
        return result

    def test_matching_cache_uses_validated_builder_once_and_defaults(self):
        builder = self.builder()
        with patch('datasets.load_dataset_builder', return_value=builder) as loader:
            result = provenance.load_pinned_dataset(self.dataset_id, 'plain_text',
                                                    split='test', cache_dir='/tmp/cache')
        loader.assert_called_once_with(self.dataset_id, 'plain_text',
            revision=provenance.DATASET_REVISIONS[self.dataset_id], cache_dir='/tmp/cache')
        self.assertIs(result, builder.as_dataset.return_value)
        builder.download_and_prepare.assert_called_once()
        self.assertEqual(str(builder.download_and_prepare.call_args.kwargs['verification_mode']),
                         'VerificationMode.BASIC_CHECKS')
        self.assertEqual(builder.as_dataset.call_args.kwargs['split'], 'test')
        builder.as_streaming_dataset.assert_not_called()

    def test_mismatched_cache_rejected_before_preparation_or_rows(self):
        builder = self.builder(hash_value='a'*40)
        with patch('datasets.load_dataset_builder', return_value=builder):
            with self.assertRaisesRegex(ValueError, 'refusing offline fallback'):
                provenance.load_pinned_dataset(self.dataset_id)
        builder.download_and_prepare.assert_not_called()
        builder.as_dataset.assert_not_called()
        builder.as_streaming_dataset.assert_not_called()

    def test_missing_or_auto_cache_identity_is_not_proof(self):
        for value in (None, 'auto', ''):
            with self.subTest(value=value):
                builder = self.builder();builder.hash=value
                with patch('datasets.load_dataset_builder', return_value=builder):
                    with self.assertRaisesRegex(ValueError, 'cached dataset revision'):
                        provenance.load_pinned_dataset(self.dataset_id, streaming=True)
                builder.as_streaming_dataset.assert_not_called()

    def test_nonfallback_builder_hash_is_not_a_repository_revision(self):
        builder = self.builder(cache=False, hash_value='loading-code-fingerprint')
        with patch('datasets.load_dataset_builder', return_value=builder):
            result = provenance.load_pinned_dataset(self.dataset_id)
        self.assertIs(result, builder.as_dataset.return_value)

    def test_streaming_preserves_subset_split_and_cache_without_preparing(self):
        dataset_id='lmms-lab/VQAv2';builder=self.builder()
        builder.hash=provenance.DATASET_REVISIONS[dataset_id]
        with patch('datasets.load_dataset_builder', return_value=builder) as loader:
            result=provenance.load_pinned_dataset(dataset_id, None, split='validation',
                                                  streaming=True, cache_dir='/tmp/frame')
        loader.assert_called_once_with(dataset_id,None,
            revision=provenance.DATASET_REVISIONS[dataset_id],cache_dir='/tmp/frame')
        self.assertIs(result,builder.as_streaming_dataset.return_value)
        builder.as_streaming_dataset.assert_called_once_with(split='validation')
        builder.download_and_prepare.assert_not_called();builder.as_dataset.assert_not_called()

    def test_unregistered_or_malformed_pin_rejected_before_access(self):
        for value in (None, 'main', '1234'):
            with self.subTest(value=value),patch.dict(provenance.DATASET_REVISIONS,{'new/dataset':value}):
                with patch('datasets.load_dataset_builder') as loader:
                    with self.assertRaisesRegex(ValueError,'verified full dataset revision'):
                        provenance.load_pinned_dataset('new/dataset')
                loader.assert_not_called()

    def test_repository_callers_do_not_bypass_the_guard(self):
        root=Path(provenance.__file__).parent
        # Dataset.from_parquet is a separately hash-checked local transport.
        for name in ('core.py','vqa.py','vqa_diverse.py','vqa_confirmatory.py','continual.py'):
            tree=ast.parse((root/name).read_text())
            calls=[node.func.id for node in ast.walk(tree)
                   if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)]
            self.assertNotIn('load_dataset',calls,name)
            self.assertIn('load_pinned_dataset',calls,name)


if __name__=='__main__':unittest.main()
