import copy
import json
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from PIL import Image

from incontext_collapse.shortcut_controls import (
    NAMES, VARIANTS, digest_json, evaluate_episode, image_interval, load_manifest,
    nearest_evidence, prepare_data, prior_image_ids, score_outputs, summarize, validate_episode,
    validate_nearest, validate_result, validate_payload, variants_for,
)


class ShortcutControlTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = prepare_data(self.temporary.name, 'development', 4, [0, 1], prior_ids=set())
        self.manifest, self.images = load_manifest(self.path)
        self.episode = self.manifest['episodes']['0:0']

    def nearest(self, episode=None, index=None):
        episode = episode or self.episode
        if index is None:
            index = episode['support_classes'].index(episode['gold'])
        embeddings = [torch.tensor([0., 1.]) for _ in range(8)]
        embeddings[index] = torch.tensor([1., 0.])
        return nearest_evidence(episode, torch.tensor([1., 0.]), embeddings)

    def test_three_variants_preserve_pairs_and_actual_manipulation(self):
        e = self.episode
        pairs = list(zip(e['support_ids'], e['support_classes']))
        for variant in VARIANTS:
            self.assertCountEqual([pairs[i] for i in e['variants'][variant]], pairs)
        self.assertEqual(e['support_classes'][e['variants']['endpoint'][-1]], e['roles']['majority'])
        self.assertEqual(e['variants']['sham'][0], 0)
        self.assertEqual(e['variants']['sham'][-1], 7)
        self.assertEqual([e['support_classes'][j] for j in e['variants']['sham']], e['support_classes'])
        self.assertNotEqual(e['support_ids'][e['source_index']], e['support_ids'][e['sham_index']])

    def test_query_frame_shared_roles_balanced_and_partitions_disjoint(self):
        for i in range(4):
            self.assertEqual(self.manifest['episodes'][f'0:{i}']['query_id'],
                             self.manifest['episodes'][f'1:{i}']['query_id'])
        for seed in (0, 1):
            for role in ('gold', 'first', 'last', 'majority'):
                self.assertEqual(sorted(self.manifest['episodes'][f'{seed}:{i}']['roles'][role]
                                        for i in range(4)), list(range(4)))
        other = prepare_data(self.temporary.name, 'confirmatory', 4, [0, 1], prior_ids=set())
        confirm, _ = load_manifest(other)
        self.assertFalse(set(self.manifest['images']) & set(confirm['images']))
        self.assertEqual(len(self.images), 4 + 2 * 4 * 8)

    def test_prior_pixels_are_excluded_before_outputs_exist(self):
        prior = set(self.manifest['images'])
        with tempfile.TemporaryDirectory() as root:
            path = prepare_data(root, 'development', 4, [0, 1], prior_ids=prior)
            payload, _ = load_manifest(path)
            self.assertFalse(prior & set(payload['images']))
            self.assertGreater(sum(x['collision_retries'] for x in payload['images'].values()), 0)

    def test_prior_scan_hashes_pixels_and_ignores_all_followup_artifacts(self):
        from incontext_collapse.protocol import image_id
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            old = Image.new('RGB', (8, 8), 'red')
            fresh = Image.new('RGB', (8, 8), 'blue')
            old.save(root / 'old.png')
            (root / 'old.json').write_text(json.dumps({'image_id': 'a' * 64}))
            followup = root / 'followup_20261010'
            followup.mkdir()
            fresh.save(followup / 'fresh.png')
            (followup / 'new.json').write_text(json.dumps({'image_id': 'b' * 64}))
            self.assertEqual(prior_image_ids(root), {image_id(old), 'a' * 64})

    def test_tied_nearest_uses_order_independent_pixel_identity(self):
        e = self.episode
        vectors = [torch.tensor([1., 0.]) for _ in range(8)]
        first = nearest_evidence(e, torch.tensor([1., 0.]), vectors)
        selected = e['support_ids'][first['nearest_index']]
        for order in e['variants'].values():
            permuted = copy.deepcopy(e)
            permuted['support_ids'] = [e['support_ids'][j] for j in order]
            permuted['support_classes'] = [e['support_classes'][j] for j in order]
            other = nearest_evidence(permuted, torch.tensor([1., 0.]), [vectors[j] for j in order])
            self.assertEqual(permuted['support_ids'][other['nearest_index']], selected)
            self.assertEqual(other['nearest_class'], first['nearest_class'])

    def test_preflight_validates_plan_without_model_or_feature_calls(self):
        import incontext_collapse.shortcut_controls as module
        root = Path(self.temporary.name) / 'preflight'
        source = Path(self.temporary.name) / 'training.json'
        source.write_text(json.dumps({'run_config': {'model_revision': 'pinned',
            'objective': 'episode_remap', 'task': 'shapes4', 'format': 'single',
            'model': 'Qwen/Qwen2-VL-2B-Instruct', 'regions': 'connector', 'components': 'both',
            'rank': 8, 'steps': 150, 'dtype': 'bfloat16', 'seeds': [0]}}))
        checkpoint = {'base_model_name_or_path': 'Qwen/Qwen2-VL-2B-Instruct',
                      'training_source': {'path': str(source)}}
        out = root / 'results.json'
        with patch('sys.argv', ['shortcut_controls', '--data-dir', str(root),
                    '--partition', 'development', '--adapter', 'unused', '--out', str(out), '--preflight']), \
             patch.object(module, 'prior_image_ids', return_value=set()), \
             patch.object(module, 'checkpoint_manifest', return_value={'connector': {'0': checkpoint}}), \
             patch.object(module, 'model_revision', return_value='pinned'), \
             patch.object(module, 'source_hashes', return_value={}), \
             patch.object(module, 'software_versions', return_value={}), \
             patch.object(module, 'load_model', side_effect=AssertionError('must not load model')), \
             patch.object(module, 'image_embedding', side_effect=AssertionError('must not embed')), \
             patch.object(module.PeftModel, 'from_pretrained', side_effect=AssertionError('must not load adapter')):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                module.main()
        plan = json.loads(output.getvalue().split('\n', 1)[1])
        self.assertEqual(plan['plan']['generations'], 192)
        self.assertEqual(plan['plan']['paired_records'], 64)
        self.assertEqual(plan['plan']['unique_images'], 144)
        self.assertFalse(out.exists())
        manifest, _ = load_manifest(root / 'development' / 'manifest.json')
        for kind in range(4):
            labels = [manifest['episodes'][f'0:{i}']['remap_labels'][kind] for i in range(16)]
            self.assertEqual(sorted(labels), sorted(manifest['episodes']['0:0']['remap_labels'] * 4))

    def test_prepare_resume_config_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, 'differs'):
            prepare_data(self.temporary.name, 'development', 8, [0, 1], prior_ids=set())

    def test_changed_cache_pixels_fail(self):
        record = next(iter(self.manifest['images'].values()))
        Image.new('RGB', (112, 112), 'red').save(self.path.parent / record['path'])
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            load_manifest(self.path)

    def test_mutated_constraints_and_bad_permutations_fail(self):
        e = copy.deepcopy(self.episode)
        e['roles']['first'] = e['roles']['last']
        with self.assertRaises(ValueError):
            validate_episode(e)
        with self.assertRaises(ValueError):
            variants_for([0, 1, 2, 3] * 2, 2, 4)
        e = copy.deepcopy(self.episode)
        e['source_index'] = 0
        with self.assertRaises(ValueError):
            validate_episode(e)

    def test_nearest_collisions_retained_and_derived_fields_checked(self):
        collision = self.nearest(index=self.episode['source_index'])
        self.assertEqual(collision['collides_with'], ['majority'])
        self.assertFalse(collision['four_distinct_roles'])
        validate_nearest(self.episode, collision)
        raw = dict.fromkeys(VARIANTS, NAMES[self.episode['gold']])
        scored = score_outputs(self.episode, 'standard', raw, collision)
        self.assertEqual(scored['nearest'], collision)
        collision['nearest_class'] = self.episode['gold']
        with self.assertRaises(ValueError):
            validate_nearest(self.episode, collision)

    def test_primary_contrast_keeps_invalid_outputs_in_denominator(self):
        e = self.episode
        majority = NAMES[e['roles']['majority']]
        invalid = ', '.join(NAMES)
        record = score_outputs(e, 'standard', {'base': majority, 'endpoint': majority, 'sham': invalid}, self.nearest())
        self.assertEqual(record['primary_contrast'], 1)
        self.assertEqual(record['sham_minus_base'], -1)
        self.assertEqual(record['metrics']['sham']['invalid'], 1)
        self.assertEqual(record['metrics']['sham']['accuracy'], 0)
        self.assertIsNone(record['predictions']['sham'])
        self.assertEqual(record['metrics']['endpoint']['accuracy'], 0)

    def test_generation_uses_identical_pixels_labels_options_and_budget(self):
        seen = []
        def generate(model, processor, messages, max_new_tokens):
            seen.append((messages, max_new_tokens))
            return self.episode['remap_labels'][self.episode['gold']]
        record = evaluate_episode(None, None, self.episode, self.images, 'remap',
                                  self.nearest(), generator=generate)
        self.assertEqual(len(seen), 3)
        self.assertTrue(all(n == 6 for _, n in seen))
        for messages, _ in seen:
            self.assertEqual(messages[0]['content'][-1]['text'], seen[0][0][0]['content'][-1]['text'])
            self.assertCountEqual([b['image'].tobytes() for b in messages[0]['content'] if b['type'] == 'image'],
                                  [b['image'].tobytes() for b in seen[0][0][0]['content'] if b['type'] == 'image'])
        self.assertTrue(all(record['metrics'][v]['accuracy'] == 1 for v in VARIANTS))

    def test_partial_resume_validates_prefix_raw_outputs_and_nearest(self):
        e = self.episode
        nearest = self.nearest()
        record = score_outputs(e, 'standard', dict.fromkeys(VARIANTS, NAMES[e['gold']]), nearest)
        result = {'nearest': {e['episode_id']: nearest}, 'cells': {'0:frozen:standard': [record]}}
        validate_result(result, self.manifest)
        bad = copy.deepcopy(result)
        bad['cells']['0:frozen:standard'][0]['predictions']['base'] = 3
        with self.assertRaises(ValueError):
            validate_result(bad, self.manifest)
        bad = copy.deepcopy(result)
        bad['cells']['0:frozen:standard'].append(record)
        with self.assertRaises((ValueError, KeyError)):
            validate_result(bad, self.manifest)

    def test_exact_ten_percent_threshold_is_not_lost_averaging_thirds(self):
        manifest = {'config': {'n': 80, 'seeds': [0, 1, 2], 'partition': 'confirmatory'},
                    'query_ids': list(range(80))}
        result = {'cells': {}}
        for seed in range(3):
            result['cells'][f'{seed}:frozen:standard'] = [
                {'query_id': i, 'primary_contrast': int(seed == 0 and i < 24),
                 'metrics': {v: {'accuracy': 0} for v in VARIANTS},
                 'nearest': {'collides_with': []}} for i in range(80)]
        result = summarize(result, manifest, draws=1000)
        self.assertEqual(result['frozen:standard']['primary_contrast']['mean'], .10)
        self.assertGreater(result['frozen:standard']['primary_contrast']['ci95_low'], 0)
        self.assertTrue(result['prespecified_primary']['meets_effect_and_interval_criteria'])

    def test_summary_averages_fixed_seeds_within_shared_image(self):
        result = {'nearest': {}, 'cells': {}}
        for seed in [0, 1]:
            for arm in ['frozen', 'connector']:
                for condition in ['standard', 'remap']:
                    rows = []
                    for i in range(4):
                        e = self.manifest['episodes'][f'{seed}:{i}']
                        nearest = self.nearest(e)
                        result['nearest'][e['episode_id']] = nearest
                        labels = NAMES if condition == 'standard' else e['remap_labels']
                        raw = dict.fromkeys(VARIANTS, labels[e['gold']])
                        if seed == 0 and arm == 'frozen':
                            raw['endpoint'] = labels[e['roles']['majority']]
                        rows.append(score_outputs(e, condition, raw, nearest))
                    result['cells'][f'{seed}:{arm}:{condition}'] = rows
        validate_result(result, self.manifest)
        summary = summarize(result, self.manifest, draws=100)
        ci = summary['frozen:standard']['primary_contrast']
        self.assertEqual(ci['n_images'], 4)
        self.assertEqual(ci['mean'], .5)
        self.assertEqual(ci['ci95_low'], .5)
        self.assertEqual(summary['moderation:standard']['mean'], -.5)
        self.assertFalse(summary['prespecified_primary']['confirmatory'])
        self.assertEqual(summary['frozen:standard']['n_query_seed_records'], 8)
        result['summary'] = summarize(result, self.manifest)
        self.assertEqual(validate_payload(result, self.manifest), result['summary'])
        damaged = copy.deepcopy(result)
        damaged['summary']['frozen:standard']['primary_contrast']['mean'] = .75
        with self.assertRaisesRegex(ValueError, 'stored summary'):
            validate_payload(damaged, self.manifest)
        incomplete = copy.deepcopy(result)
        incomplete['cells']['1:frozen:standard'].pop()
        self.assertNotIn('frozen:standard', summarize(incomplete, self.manifest, draws=20))


if __name__ == '__main__':
    unittest.main()
