"""Prespecified, paired position intervention; saved adapters, no optimization.

Primary: frozen STANDARD probability of selecting the incorrect majority label
when one majority pair moves to the last position, minus an interior-swap sham.
Image/label pairs and counts are identical in all three variants. Other shortcut
predictions are descriptive; no episode is selected by nearest-feature identity
or model outputs. Development: 16 images/seed 0. Confirmatory: 80 shared images,
seeds 0/1/2. Image-bootstrap uncertainty conditions on these fitted checkpoints.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
from fractions import Fraction
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
import re
import statistics

from PIL import Image
import torch
from peft import PeftModel

from .core import _draw_shape, build_messages, task_shapes4
from .paired_bootstrap import exact_estimate, integer_image_interval
from .evaluation import mean_t_ci
from .frozen_features import image_embedding
from .models import load_model
from .option_order_eval import checkpoint_manifest
from .protocol import image_id, REMAP_TOKEN_SETS
from .provenance import model_revision, software_versions
from .results import load_result, write_json_atomic
from .scoring import parse_answer

NAMES = ['circle', 'square', 'triangle', 'cross']
VARIANTS = ('base', 'endpoint', 'sham')
CONDITIONS = ('standard', 'remap')
NAMESPACE = 'shortcut-position-20261010-v1'
PARTITIONS = {'development': (16, [0]), 'confirmatory': (80, [0, 1, 2])}


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def rng_for(*parts):
    return random.Random(int.from_bytes(hashlib.sha256(
        ':'.join(map(str, (NAMESPACE, *parts))).encode()).digest(), 'big'))


def variants_for(classes, source, sham):
    if len(classes) != 8 or source not in range(1, 7) or sham not in range(1, 7) or source == sham:
        raise ValueError('invalid K8 intervention indices')
    counts = Counter(classes)
    if sorted(counts.values()) != [1, 1, 1, 5]:
        raise ValueError('support must have counts 5/1/1/1')
    majority = max(counts, key=counts.get)
    if classes[source] != majority or classes[sham] != majority or classes[-1] == majority:
        raise ValueError('source and sham must be interior majority pairs; last a minority')
    variants = {name: list(range(8)) for name in VARIANTS}
    variants['endpoint'][source], variants['endpoint'][7] = 7, source
    variants['sham'][source], variants['sham'][sham] = sham, source
    return variants


def validate_episode(episode):
    classes = episode['support_classes']
    roles = episode['roles']
    if set(roles.values()) != set(range(4)) or roles['gold'] != episode['gold']:
        raise ValueError('four class roles must be distinct')
    if classes[0] != roles['first'] or classes[-1] != roles['last']:
        raise ValueError('endpoint class mismatch')
    if Counter(classes) != Counter({roles['majority']: 5, roles['gold']: 1,
                                   roles['first']: 1, roles['last']: 1}):
        raise ValueError('support role counts mismatch')
    if len(set(episode['support_ids'])) != 8 or episode['query_id'] in episode['support_ids']:
        raise ValueError('repeated support pixels or support/query overlap')
    expected = variants_for(classes, episode['source_index'], episode['sham_index'])
    if episode['variants'] != expected:
        raise ValueError('variant permutation changed')
    if sorted(episode['option_order']) != list(range(4)):
        raise ValueError('invalid displayed-option permutation')
    if sorted(episode['remap_labels']) != sorted(REMAP_TOKEN_SETS['eval'][:4]):
        raise ValueError('invalid remap verbalizers')
    check = {k: v for k, v in episode.items() if k != 'constraint_sha256'}
    if episode['constraint_sha256'] != digest_json(check):
        raise ValueError('episode constraint checksum mismatch')


def prior_image_ids(root=None):
    """Exclude historical cached pixels and saved identities before generation.

    All data/followup_20261010 artifacts are omitted so this historical frame
    stays independent of new experiment preparation order. Sibling A partitions
    are excluded explicitly by prepare_data. Historical 64-hex JSON identities
    conservatively include non-image hashes; no saved outcome is interpreted.
    """
    root = Path(root) if root is not None else Path(_draw_shape.__code__.co_filename).resolve().parent.parent / 'data'
    identities = set()
    for path in sorted(root.rglob('*')):
        if 'followup_20261010' in path.relative_to(root).parts or not path.is_file():
            continue
        if path.suffix.lower() == '.json':
            identities.update(x.decode() for x in re.findall(rb'(?<![a-f0-9])[a-f0-9]{64}(?![a-f0-9])', path.read_bytes()))
        elif path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}:
            with Image.open(path) as image:
                identities.add(image_id(image))
    return identities


def prepare_data(data_dir, partition, n, seeds, prior_ids=None):
    if partition not in PARTITIONS or n < 4 or n % 4 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('invalid partition, balanced query count or seeds')
    directory = Path(data_dir) / partition
    config = {'namespace': NAMESPACE, 'partition': partition, 'n': n, 'seeds': seeds,
              'size': 112, 'k': 8, 'query_frame_shared_across_seeds': True,
              'mapping_counterbalance': 'each gold verbalizer per four-query block; each class/verbalizer per 16-query superblock',
              'selection': 'none: all frozen-nearest role collisions and outputs retained'}
    path = directory / 'manifest.json'
    if path.exists():
        if json.loads(path.read_text())['config'] != config:
            raise ValueError('existing data manifest differs')
        load_manifest(path)
        return path
    _, pool, test = task_shapes4(96, 120)
    legacy = {image_id(im) for images in pool.values() for im in images}
    legacy.update(image_id(im) for im, _ in test)
    excluded = set(legacy) | (prior_image_ids() if prior_ids is None else set(prior_ids))
    for sibling in Path(data_dir).glob('*/manifest.json'):
        old = json.loads(sibling.read_text())
        for record in old['images'].values():
            excluded.add(record['image_id'])
    used, images = set(excluded), {}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'images').mkdir(exist_ok=True)

    def draw(kind, key):
        rng = rng_for(partition, *key)
        for attempt in range(100):
            image = _draw_shape(kind, rng)
            identity = image_id(image)
            if identity not in used:
                used.add(identity)
                image_path = f'images/{identity}.png'
                image.save(directory / image_path, format='PNG')
                images[identity] = {'image_id': identity, 'path': image_path,
                                    'class': kind, 'draw_key': list(key), 'collision_retries': attempt}
                return identity
        raise RuntimeError('cannot create unique procedural pixels')

    queries = [draw(i % 4, ('query', i)) for i in range(n)]
    episodes = {}
    for seed in seeds:
        offsets = {}
        for i in range(n):
            block = i // 4
            if block not in offsets:
                offsets[block] = rng_for(partition, 'roles', seed, block).sample([1, 2, 3], 3)
            gold = i % 4
            first, last, majority = [(gold + offset) % 4 for offset in offsets[block]]
            rng = rng_for(partition, 'positions', seed, i)
            gold_slot = rng.choice(list(range(1, 7)))
            classes = [first] + [majority] * 6 + [last]
            classes[gold_slot] = gold
            source, sham = rng.sample([j for j in range(1, 7) if classes[j] == majority], 2)
            support = [draw(c, ('support', seed, i, j)) for j, c in enumerate(classes)]
            mapping_base = rng_for(partition, 'mapping', seed, i // 16).sample(list(REMAP_TOKEN_SETS['eval'][:4]), 4)
            labels = [mapping_base[(c + block % 4) % 4] for c in range(4)]
            order = rng_for(partition, 'options', seed, i).sample(list(range(4)), 4)
            record = {'episode_id': f'{seed}:{i}', 'seed': seed, 'query_index': i,
                      'query_id': queries[i], 'gold': gold, 'support_ids': support,
                      'support_classes': classes, 'roles': {'gold': gold, 'first': first,
                      'last': last, 'majority': majority}, 'source_index': source,
                      'sham_index': sham, 'variants': variants_for(classes, source, sham),
                      'remap_labels': labels, 'option_order': order}
            record['constraint_sha256'] = digest_json(record)
            validate_episode(record)
            episodes[record['episode_id']] = record
    write_json_atomic(path, {'config': config, 'legacy_exclusion_sha256': digest_json(sorted(legacy)),
                             'prior_excluded_ids': sorted(excluded), 'prior_exclusion_sha256': digest_json(sorted(excluded)),
                             'query_ids': queries, 'images': images, 'episodes': episodes})
    load_manifest(path)
    return path


def load_manifest(path):
    path = Path(path)
    payload = json.loads(path.read_text())
    images = {}
    for identity, record in payload['images'].items():
        with Image.open(path.parent / record['path']) as im:
            image = im.copy()
        if image_id(image) != identity or record['image_id'] != identity:
            raise ValueError('cached image hash mismatch')
        images[identity] = image
    n, seeds = payload['config']['n'], payload['config']['seeds']
    excluded = payload['prior_excluded_ids']
    if payload['prior_exclusion_sha256'] != digest_json(excluded) or set(excluded) & set(images):
        raise ValueError('prior-exclusion checksum or pixel overlap')
    if len(payload['query_ids']) != n or len(set(payload['query_ids'])) != n:
        raise ValueError('invalid common query frame')
    if set(payload['episodes']) != {f'{s}:{i}' for s in seeds for i in range(n)}:
        raise ValueError('incomplete or unexpected episodes')
    support_ids = []
    for key, episode in payload['episodes'].items():
        validate_episode(episode)
        if episode['episode_id'] != key or episode['query_id'] != payload['query_ids'][episode['query_index']]:
            raise ValueError('episode does not match shared query frame')
        for identity, kind in [(episode['query_id'], episode['gold']),
                               *zip(episode['support_ids'], episode['support_classes'])]:
            if identity not in images or payload['images'][identity]['class'] != kind:
                raise ValueError('image class/identity manifest mismatch')
        support_ids.extend(episode['support_ids'])
    if len(set(support_ids)) != len(support_ids) or set(support_ids) & set(payload['query_ids']):
        raise ValueError('global support/query image overlap')
    if set(images) != set(support_ids) | set(payload['query_ids']):
        raise ValueError('unexpected or missing cached images')
    for seed in seeds:
        for role in ('gold', 'first', 'last', 'majority'):
            if Counter(payload['episodes'][f'{seed}:{i}']['roles'][role] for i in range(n)) != Counter({c: n // 4 for c in range(4)}):
                raise ValueError('class roles are not counterbalanced')
    for seed in seeds:
        if Counter(payload['episodes'][f'{seed}:{i}']['remap_labels'][i % 4] for i in range(n)) != Counter({label: n // 4 for label in REMAP_TOKEN_SETS['eval'][:4]}):
            raise ValueError('gold remap labels are not counterbalanced')
        if n % 16 == 0:
            for kind in range(4):
                if Counter(payload['episodes'][f'{seed}:{i}']['remap_labels'][kind] for i in range(n)) != Counter({label: n // 4 for label in REMAP_TOKEN_SETS['eval'][:4]}):
                    raise ValueError('class-to-remap mappings are not counterbalanced')
    for sibling in path.parent.parent.glob('*/manifest.json'):
        if sibling != path and set(payload['images']) & set(json.loads(sibling.read_text())['images']):
            raise ValueError('development/confirmatory pixel overlap')
    return payload, images


def nearest_evidence(episode, query_embedding, support_embeddings):
    query = torch.nn.functional.normalize(query_embedding.float(), dim=-1)
    support = torch.nn.functional.normalize(torch.stack(support_embeddings).float(), dim=-1)
    scores = (support @ query).tolist()
    if not all(math.isfinite(v) for v in scores):
        raise ValueError('nonfinite frozen-feature scores')
    index = max(range(8), key=lambda j: (scores[j], episode['support_ids'][j]))
    label = episode['support_classes'][index]
    ordered = sorted(scores, reverse=True)
    return {'constraint_sha256': episode['constraint_sha256'], 'cosines': scores,
            'nearest_index': index, 'nearest_class': label, 'top_two_margin': ordered[0] - ordered[1],
            'four_distinct_roles': len({episode['roles'][r] for r in ('first', 'last', 'majority')} | {label}) == 4,
            'collides_with': [r for r in ('first', 'last', 'majority') if episode['roles'][r] == label],
            'tie_break': 'maximum pixel SHA256 among exact cosine ties',
            'selection': 'retained regardless of nearest role; invariant under pair permutation'}


def validate_nearest(episode, evidence):
    if evidence['constraint_sha256'] != episode['constraint_sha256']:
        raise ValueError('nearest feature episode changed')
    scores = evidence['cosines']
    if len(scores) != 8 or not all(math.isfinite(x) for x in scores):
        raise ValueError('invalid frozen-nearest evidence')
    index = max(range(8), key=lambda j: (scores[j], episode['support_ids'][j]))
    label = episode['support_classes'][index]
    collides = [r for r in ('first', 'last', 'majority') if episode['roles'][r] == label]
    margin = sorted(scores, reverse=True)[0] - sorted(scores, reverse=True)[1]
    if (evidence['nearest_index'] != index or evidence['nearest_class'] != label or
            evidence['collides_with'] != collides or evidence['four_distinct_roles'] != (not collides)
            or evidence['top_two_margin'] != margin
            or evidence['tie_break'] != 'maximum pixel SHA256 among exact cosine ties'):
        raise ValueError('frozen-nearest derived fields disagree')


def score_outputs(episode, condition, raw_outputs, nearest):
    validate_nearest(episode, nearest)
    labels = NAMES if condition == 'standard' else episode['remap_labels']
    if condition not in CONDITIONS or set(raw_outputs) != set(VARIANTS):
        raise ValueError('unexpected condition or incomplete variants')
    parsed = {v: parse_answer(raw_outputs[v], labels) for v in VARIANTS}
    majority = episode['roles']['majority']
    metrics = {}
    for variant in VARIANTS:
        prediction = parsed[variant].prediction
        order = episode['variants'][variant]
        metrics[variant] = {'accuracy': int(prediction == episode['gold']),
                            'invalid': int(prediction is None),
                            'first_match': int(prediction == episode['support_classes'][order[0]]),
                            'last_match': int(prediction == episode['support_classes'][order[-1]]),
                            'majority_match': int(prediction == majority),
                            'nearest_match': int(prediction == nearest['nearest_class'])}
    endpoint_change = metrics['endpoint']['majority_match'] - metrics['base']['majority_match']
    sham_change = metrics['sham']['majority_match'] - metrics['base']['majority_match']
    return {'episode_id': episode['episode_id'], 'query_id': episode['query_id'],
            'constraint_sha256': episode['constraint_sha256'], 'gold': episode['gold'],
            'label_space': list(labels), 'option_order': episode['option_order'],
            'displayed_options': [labels[i] for i in episode['option_order']],
            'raw_outputs': raw_outputs, 'predictions': {v: parsed[v].prediction for v in VARIANTS},
            'parse_status': {v: parsed[v].status for v in VARIANTS}, 'metrics': metrics,
            'nearest': nearest, 'endpoint_minus_base': endpoint_change,
            'sham_minus_base': sham_change, 'primary_contrast': endpoint_change - sham_change}


@torch.no_grad()
def generate(model, processor, messages, max_new_tokens=6):
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors='pt').to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                            num_beams=1, num_return_sequences=1)
    return processor.batch_decode(output[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]


def evaluate_episode(model, processor, episode, images, condition, nearest, max_new_tokens=6,
                     generator=generate):
    labels = NAMES if condition == 'standard' else episode['remap_labels']
    raw = {}
    for variant in VARIANTS:
        demos = [(images[episode['support_ids'][j]], episode['support_classes'][j])
                 for j in episode['variants'][variant]]
        messages = build_messages(demos, images[episode['query_id']], labels, 112,
                                  option_order=episode['option_order'])
        raw[variant] = generator(model, processor, messages, max_new_tokens=max_new_tokens)
    return score_outputs(episode, condition, raw, nearest)


def image_interval(values, draws=10000, seed=90210):
    if not values or draws < 2:
        raise ValueError('nonempty image values and >=2 draws required')
    rng = random.Random(seed)
    sampled = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(draws))
    def quantile(p):
        position = p * (draws - 1)
        low = int(position)
        return sampled[low] + (sampled[min(low + 1, draws - 1)] - sampled[low]) * (position - low)
    return {'mean': statistics.mean(values), 'ci95_low': quantile(.025), 'ci95_high': quantile(.975),
            'n_images': len(values), 'draws': draws, 'seed': seed,
            'scope': 'query-image bootstrap, averaging fixed support/checkpoint seeds within image'}


def summarize(result, manifest, draws=10000):
    n, seeds = manifest['config']['n'], manifest['config']['seeds']
    summary = {}
    for arm in ('frozen', 'connector'):
        for condition in CONDITIONS:
            cells = [result['cells'].get(f'{s}:{arm}:{condition}', []) for s in seeds]
            if not all(len(cell) == n for cell in cells):
                continue
            for i in range(n):
                if any(cell[i]['query_id'] != manifest['query_ids'][i] for cell in cells):
                    raise ValueError('summary query pairing mismatch')
            estimates = [statistics.mean(cell[i]['primary_contrast'] for cell in cells) for i in range(n)]
            seed_effects = {str(seed): statistics.mean(r['primary_contrast'] for r in cell)
                            for seed, cell in zip(seeds, cells)}
            variants = {variant: {metric: statistics.mean(r['metrics'][variant][metric]
                         for cell in cells for r in cell) for metric in cells[0][0]['metrics'][variant]}
                        for variant in VARIANTS}
            exact = integer_image_interval([sum(cell[i]['primary_contrast'] for cell in cells) for i in range(n)], len(seeds), draws=draws)
            interval = {'mean': exact['mean'], 'ci95_low': exact['low'], 'ci95_high': exact['high'],
                        'exact_rationals': exact['exact_rationals'], 'n_images': n, 'draws': draws, 'seed': 90210,
                        'scope': 'query-image bootstrap, averaging fixed support/checkpoint seeds within image'}
            # Integer contrasts retain an exact rational point estimate at the .10
            # decision boundary; averaging thirds first can round .10 downward.
            interval['mean'] = statistics.mean(r['primary_contrast'] for cell in cells for r in cell)
            summary[f'{arm}:{condition}'] = {'primary_contrast': interval,
                'checkpoint_support_seed_effects': seed_effects,
                'seed_t_interval_secondary': mean_t_ci(list(seed_effects.values())),
                'variants': variants, 'n_query_seed_records': n * len(seeds),
                'nearest_role_collision_rate': statistics.mean(bool(r['nearest']['collides_with'])
                                                              for cell in cells for r in cell)}
    for condition in CONDITIONS:
        if all(f'{arm}:{condition}' in summary for arm in ('frozen', 'connector')):
            values = [statistics.mean(
                result['cells'][f'{s}:connector:{condition}'][i]['primary_contrast'] -
                result['cells'][f'{s}:frozen:{condition}'][i]['primary_contrast'] for s in seeds)
                      for i in range(n)]
            interval = image_interval(values, draws)
            interval['mean'] = statistics.mean(
                result['cells'][f'{s}:connector:{condition}'][i]['primary_contrast'] -
                result['cells'][f'{s}:frozen:{condition}'][i]['primary_contrast']
                for s in seeds for i in range(n))
            summary[f'moderation:{condition}'] = interval
    primary = summary.get('frozen:standard')
    if primary:
        effect = primary['primary_contrast']
        summary['prespecified_primary'] = {'partition': manifest['config']['partition'],
            'contrast': 'frozen standard wrong-majority selection: endpoint minus interior sham',
            'meaningful_threshold': .10, 'confirmatory': manifest['config']['partition'] == 'confirmatory',
            'meets_effect_and_interval_criteria': exact_estimate(effect, 'mean') >= Fraction(1, 10) and exact_estimate(effect, 'low') > 0,
            'secondary_inference': 'pointwise, exploratory; no inference identifying every shortcut'}
    return summary



def validate_connector_source(training, model, revision, seed):
    expected = {'model': model, 'model_revision': revision, 'objective': 'episode_remap',
                'task': 'shapes4', 'format': 'single', 'regions': 'connector',
                'components': 'both', 'rank': 8, 'steps': 150, 'dtype': 'bfloat16'}
    if any(training.get(key) != value for key, value in expected.items()) or seed not in training.get('seeds', []):
        raise ValueError('expected rank-8, 150-step episodic Shapes connector with matching seed/backbone')


def source_hashes():
    names = ('shortcut_controls', 'core', 'protocol', 'scoring', 'models',
             'frozen_features', 'option_order_eval', 'provenance', 'results', 'evaluation', 'paired_bootstrap')
    return {f'{name}.py': hashlib.sha256(Path(importlib.import_module(
        f'{__package__}.{name}').__file__).read_bytes()).hexdigest() for name in names}


def validate_result(result, manifest):
    seeds, n = manifest['config']['seeds'], manifest['config']['n']
    allowed = {f'{s}:{arm}:{c}' for s in seeds for arm in ('frozen', 'connector') for c in CONDITIONS}
    if not set(result['cells']).issubset(allowed) or not set(result['nearest']).issubset(manifest['episodes']):
        raise ValueError('unexpected saved result cells or features')
    for key, evidence in result['nearest'].items():
        validate_nearest(manifest['episodes'][key], evidence)
    for key, records in result['cells'].items():
        seed, _, condition = key.split(':')
        if len(records) > n:
            raise ValueError('too many saved records')
        for index, record in enumerate(records):
            episode = manifest['episodes'][f'{seed}:{index}']
            nearest = result['nearest'].get(episode['episode_id'])
            if nearest is None or record != score_outputs(episode, condition, record['raw_outputs'], nearest):
                raise ValueError('saved record does not reproduce from raw outputs and constraints')


def validate_payload(result, manifest, check_summary=True):
    """Independent replay audit; raw strings are evidence, predictions are derived.

    A partial checkpoint may have an empty summary. Any nonempty stored summary
    must reproduce exactly with the prespecified image bootstrap.
    """
    validate_result(result, manifest)
    expected = summarize(result, manifest)
    if check_summary and result.get('summary') and result['summary'] != expected:
        raise ValueError('stored summary does not reproduce from paired raw records')
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='Qwen/Qwen2-VL-2B-Instruct')
    parser.add_argument('--adapter', help='connector root containing seed_N saved adapter directories')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--partition', choices=PARTITIONS, required=True)
    parser.add_argument('--n', type=int)
    parser.add_argument('--seeds')
    parser.add_argument('--out')
    parser.add_argument('--device')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--preflight', action='store_true',
                        help='validate prepared data, saved checkpoints and resume config without models')
    parser.add_argument('--max-new-tokens', type=int, default=6)
    args = parser.parse_args()
    expected_n, expected_seeds = PARTITIONS[args.partition]
    n = args.n if args.n is not None else expected_n
    seeds = [int(x) for x in args.seeds.split(',')] if args.seeds else expected_seeds
    if n != expected_n or seeds != expected_seeds or args.max_new_tokens != 6:
        parser.error('frozen protocol requires development n16/seed0 or confirmatory n80/seeds0,1,2; six tokens')
    path = prepare_data(args.data_dir, args.partition, n, seeds)
    manifest, images = load_manifest(path)
    print(f'Prepared {args.partition}: {n} shared query images; {len(manifest["episodes"])} episodes', flush=True)
    if args.prepare_only:
        return
    if not args.out or not args.adapter:
        parser.error('evaluation requires --out and --adapter')
    if args.model != 'Qwen/Qwen2-VL-2B-Instruct':
        parser.error('the prespecified position study requires Qwen2-VL-2B-Instruct')
    revision = model_revision(args.model)
    if not revision:
        raise ValueError('pinned model revision required')
    checkpoints = checkpoint_manifest({'connector': Path(args.adapter)}, seeds)['connector']
    for seed, checkpoint in checkpoints.items():
        if checkpoint['base_model_name_or_path'] != args.model or checkpoint['training_source'] is None:
            raise ValueError('checkpoint base model or training provenance missing/mismatched')
        training = json.loads(Path(checkpoint['training_source']['path']).read_text())['run_config']
        validate_connector_source(training, args.model, revision, int(seed))
    config = {'schema_version': 1, 'experiment': NAMESPACE, 'model': args.model,
              'model_revision': revision, 'manifest_path': str(path.resolve()),
              'manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'partition': args.partition, 'n': n, 'seeds': seeds, 'k': 8,
              'arms': ['frozen', 'connector'], 'conditions': list(CONDITIONS), 'variants': list(VARIANTS),
              'dtype': 'bfloat16', 'requested_device': args.device, 'max_new_tokens': 6,
              'checkpoints': checkpoints, 'source_sha256': source_hashes(), 'software_versions': software_versions(),
              'primary': 'frozen standard endpoint minus sham wrong-majority probability',
              'meaningful_effect': .10, 'confirmation_rule': 'effect >= .10 and paired image CI95 lower > 0',
              'bootstrap_draws': 10000, 'bootstrap_seed': 90210,
              'nearest_selection': 'none; frozen features measured once; all role collisions retained'}
    result = load_result(args.out, config, {'nearest': {}, 'cells': {}, 'summary': {}, 'runtime': {}})
    validate_result(result, manifest)
    required = [f'{s}:{arm}:{c}' for s in seeds for arm in ('frozen', 'connector') for c in CONDITIONS]
    if args.preflight:
        print(json.dumps({'run_config': config, 'plan': {
            'paired_records': len(required) * n, 'generations': len(required) * n * 3,
            'unique_images': len(images), 'nearest_episodes': len(manifest['episodes']),
            'completed_paired_records': sum(len(rows) for rows in result['cells'].values()),
            'cells': required}}, indent=2), flush=True)
        return
    if all(len(result['cells'].get(key, [])) == n for key in required):
        result['summary'] = summarize(result, manifest)
        write_json_atomic(args.out, result)
        return
    def persist():
        write_json_atomic(args.out, result)
    def run_cells(model, processor, arm, selected_seeds):
        for seed in selected_seeds:
            for condition in CONDITIONS:
                key = f'{seed}:{arm}:{condition}'
                records = result['cells'].setdefault(key, [])
                for index in range(len(records), n):
                    episode = manifest['episodes'][f'{seed}:{index}']
                    records.append(evaluate_episode(model, processor, episode, images, condition,
                        result['nearest'][episode['episode_id']], args.max_new_tokens))
                    persist()
                print(f'Complete {key}: {len(records)} paired records', flush=True)
    need_base = len(result['nearest']) != len(manifest['episodes']) or any(
        len(result['cells'].get(f'{s}:frozen:{c}', [])) < n for s in seeds for c in CONDITIONS)
    if need_base:
        model, processor = load_model(args.model, dtype=torch.bfloat16, device=args.device)
        try:
            model.requires_grad_(False).eval()
            result['runtime']['frozen_device'] = str(model.device)
            embeddings = {}
            for key, episode in manifest['episodes'].items():
                if key in result['nearest']:
                    continue
                for identity in [episode['query_id'], *episode['support_ids']]:
                    if identity not in embeddings:
                        embeddings[identity] = image_embedding(model, processor, images[identity], 112)
                result['nearest'][key] = nearest_evidence(episode, embeddings[episode['query_id']],
                                                         [embeddings[x] for x in episode['support_ids']])
                persist()
            run_cells(model, processor, 'frozen', seeds)
        finally:
            del model, processor
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
    for seed in seeds:
        if all(len(result['cells'].get(f'{seed}:connector:{c}', [])) == n for c in CONDITIONS):
            continue
        base, processor = load_model(args.model, dtype=torch.bfloat16, device=args.device)
        model = None
        try:
            model = PeftModel.from_pretrained(base, checkpoints[str(seed)]['path']).eval()
            model.requires_grad_(False)
            result['runtime'][f'connector_{seed}_device'] = str(model.device)
            run_cells(model, processor, 'connector', [seed])
        finally:
            del model, base, processor
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
    validate_result(result, manifest)
    result['summary'] = summarize(result, manifest)
    persist()
    print('DONE', result['summary']['prespecified_primary'], flush=True)


if __name__ == '__main__':
    main()
