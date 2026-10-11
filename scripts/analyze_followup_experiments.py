#!/usr/bin/env python3
"""Audit and analyze the fixed October 10 studies without model inference."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
from fractions import Fraction
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from incontext_collapse.results import write_json_atomic


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def independent_prediction(raw, labels):
    require(isinstance(raw, str), 'missing raw answer')
    normalized = ' '.join(raw.casefold().strip().split())
    hits = [i for i, label in enumerate(labels)
            if re.search(r'(?<!\w)' + re.escape(label.casefold()) + r'(?!\w)', normalized)]
    return hits[0] if len(hits) == 1 else None


def seed_interval(values, confidence=.95):
    values = list(map(float, values))
    require(len(values) in (1, 3), 'this protocol uses one development or three confirmation seeds')
    mean = statistics.fmean(values)
    report = {'mean': mean, 'seed_values': values, 'n_seeds': len(values), 'confidence': confidence}
    if len(values) == 1:
        return {**report, 'low': None, 'high': None}
    # Exact inverse Student-t CDF for df=2: F(t)=1/2+t/(2*sqrt(2+t*t)).
    z = confidence
    critical = math.sqrt(2) * z / math.sqrt(1-z*z)
    half = critical * statistics.stdev(values) / math.sqrt(3)
    return {**report, 'low': mean-half, 'high': mean+half, 'df': 2, 'critical': critical}


def image_interval(matrix, confidence=.95, seed=20261010, draws=10000):
    matrix = np.asarray(matrix, dtype=float)
    require(matrix.ndim == 2 and matrix.shape[0] in (1, 3) and matrix.shape[1] > 0,
            'expected seed by image paired matrix')
    require(np.isfinite(matrix).all(), 'nonfinite estimates')
    values = matrix.mean(axis=0)
    rng = np.random.default_rng(seed)
    means = []
    for offset in range(0, draws, 500):
        count = min(500, draws-offset)
        means.extend(values[rng.integers(0, len(values), (count, len(values)))].mean(axis=1))
    tail = (1-confidence)/2
    return {'mean': float(values.mean()), 'low': float(np.quantile(means, tail)),
            'high': float(np.quantile(means, 1-tail)), 'confidence': confidence,
            'n_images': len(values), 'draws': draws, 'seed': seed,
            'seed_interval_95': seed_interval(matrix.mean(axis=1)),
            'scope': 'whole query images after averaging fixed support/checkpoint seeds'}


def check_source_hashes(recorded):
    for name, expected in recorded.items():
        require(Path(name).name == name, 'unexpected runtime source key')
        require(digest(ROOT/'incontext_collapse'/name) == expected, 'runtime source changed: '+name)


def check_checkpoint_bytes(checkpoints):
    for arm, by_seed in checkpoints.items():
        for seed, entry in by_seed.items():
            directory = Path(entry['path'])
            if not directory.is_absolute():
                directory = ROOT/directory
            for name, expected in entry['sha256'].items():
                require(Path(name).name == name, 'unsafe checkpoint basename')
                require(digest(directory/name) == expected, f'checkpoint changed: {arm}/{seed}/{name}')
            training = entry.get('training_source')
            require(training is not None, 'missing training provenance')
            source = Path(training['path'])
            if not source.is_absolute(): source = ROOT/source
            require(digest(source) == training['sha256'], 'source training result changed')


def audit_shortcut(payload):
    from incontext_collapse import shortcut_controls as runner
    config = payload['run_config']
    require(config['partition'] in ('development','confirmatory'), 'unknown shortcut partition')
    require(config['n'] == (16 if config['partition']=='development' else 80), 'unexpected shortcut n')
    require(config['seeds'] == ([0] if config['partition']=='development' else [0,1,2]), 'shortcut seeds')
    require(config['arms']==['frozen','connector'] and config['conditions']==['standard','remap'], 'shortcut arms/conditions')
    require(config['variants']==['base','endpoint','sham'] and config['k']==8, 'shortcut variants/K')
    check_source_hashes(config['source_sha256'])
    require(config['source_sha256']==runner.source_hashes(), 'incomplete shortcut source manifest')
    require(set(config['checkpoints'])==set(map(str,config['seeds'])), 'shortcut checkpoint seed grid')
    check_checkpoint_bytes({'connector': config['checkpoints']})
    require(digest(config['manifest_path']) == config['manifest_sha256'], 'shortcut manifest changed')
    manifest, _ = runner.load_manifest(config['manifest_path'])
    require(all(config[k] == manifest['config'][k] for k in ('partition','n','seeds')), 'shortcut config/frame mismatch')
    runner.validate_result(payload, manifest)
    expected = {f'{s}:{a}:{c}' for s in config['seeds'] for a in config['arms'] for c in config['conditions']}
    require(set(payload['cells']) == expected, 'incomplete shortcut cell grid')
    require(set(payload['nearest']) == set(manifest['episodes']), 'incomplete nearest-feature grid')
    contrast = {}; variants = {}
    for arm in config['arms']:
        for condition in config['conditions']:
            matrix=[]; all_metrics={v:[] for v in config['variants']}
            for seed in config['seeds']:
                rows=payload['cells'][f'{seed}:{arm}:{condition}']
                require(len(rows)==config['n'], 'incomplete shortcut records')
                values=[]
                for i,row in enumerate(rows):
                    episode=manifest['episodes'][f'{seed}:{i}']
                    require(row['query_id']==manifest['query_ids'][i], 'changed query pairing')
                    majority=episode['roles']['majority']
                    require(majority!=row['gold'], 'majority is not wrong')
                    require(Counter(episode['support_classes'])[majority]==5, 'incorrect majority frequency')
                    predictions={v:independent_prediction(row['raw_outputs'][v], row['label_space']) for v in config['variants']}
                    require(predictions==row['predictions'], 'shortcut independent parse mismatch')
                    value=int(predictions['endpoint']==majority)-int(predictions['sham']==majority)
                    require(value==row['primary_contrast'], 'wrong shortcut contrast')
                    values.append(value)
                    for variant,pred in predictions.items():
                        order=episode['variants'][variant]
                        metric={'accuracy':int(pred==row['gold']), 'invalid':int(pred is None),
                                'first_match':int(pred==episode['support_classes'][order[0]]),
                                'last_match':int(pred==episode['support_classes'][order[-1]]),
                                'majority_match':int(pred==majority),
                                'nearest_match':int(pred==row['nearest']['nearest_class'])}
                        require(metric==row['metrics'][variant], 'shortcut derived metrics differ')
                        all_metrics[variant].append(metric)
                matrix.append(values)
            label=f'{arm}:{condition}'
            contrast[label]=image_interval(matrix, seed=90210)
            variants[label]={v:{m:statistics.fmean(r[m] for r in rows) for m in rows[0]} for v,rows in all_metrics.items()}
    # The primary runner uses Python Random. Reproduce its declared bootstrap
    # separately so its exact decision is checked, not replaced by another RNG.
    matrix=[[r['primary_contrast'] for r in payload['cells'][f'{s}:frozen:standard']] for s in config['seeds']]
    # Independently retain integer seed sums through resampling and type-7 interpolation.
    values=[sum(column) for column in zip(*matrix)];rng=random.Random(90210)
    draws=sorted(sum(rng.choices(values,k=len(values))) for _ in range(10000))
    denominator=config['n']*len(config['seeds'])
    def exact_quantile(probability):
        position=probability*(len(draws)-1);lo=position.numerator//position.denominator
        return (Fraction(draws[lo])+(position-lo)*(draws[min(lo+1,len(draws)-1)]-draws[lo]))/denominator
    exact={'mean':Fraction(sum(values),denominator),'low':exact_quantile(Fraction(1,40)),
           'high':exact_quantile(Fraction(39,40))}
    primary={**{key:float(value) for key,value in exact.items()},
             'exact_rationals':{key:{'numerator':value.numerator,'denominator':value.denominator}
                                for key,value in exact.items()},
             'draws':10000,'seed':90210,'rng':'random.Random','n_images':config['n'],'threshold':.10}
    primary['criteria_met']=exact['mean']>=Fraction(1,10) and exact['low']>0
    recomputed=runner.summarize(payload,manifest)
    require(recomputed==payload['summary'], 'saved shortcut summary differs')
    require(primary['criteria_met']==recomputed['prespecified_primary']['meets_effect_and_interval_criteria'], 'primary decision differs')
    return {'primary':primary,'secondary_contrasts':contrast,'variant_metrics':variants,
            'nearest_four_distinct_fraction':statistics.fmean(v['four_distinct_roles'] for v in payload['nearest'].values()),
            'generation_records':len(expected)*config['n']*3,'unique_query_images':config['n'],
            'interpretation':'position intervention; frequency/nearest matches are competing predictions, not independently localized mechanisms'}


def audit_rule(payload):
    from incontext_collapse import rule_factorial as runner
    config=payload['run_config']; seeds=config['seeds']; ks=config['condition_ks']['explicit_natural']
    require(config['partition'] in ('development','confirmatory'), 'unknown factorial partition')
    require(config['arms']==list(runner.ARMS) and config['conditions']==list(runner.CONDITIONS)
            and config['rules']==list(runner.RULES), 'factorial factor membership differs')
    require(config['condition_ks']=={c:ks for c in runner.CONDITIONS}, 'factorial conditions have different K grids')
    expected_grid=(16,[0],[8]) if config['partition']=='development' else (80,[0,1,2],[8,16])
    require((config['n'],seeds,ks)==expected_grid,'factorial grid differs')
    check_source_hashes(config['source_sha256']);check_checkpoint_bytes(config['checkpoints'])
    require(config['source_sha256']==runner.source_hashes(), 'incomplete factorial source manifest')
    require(set(config['checkpoints'])==set(runner.ARMS[1:]) and
            all(set(v)==set(map(str,seeds)) for v in config['checkpoints'].values()), 'factorial checkpoint grid')
    require(digest(config['manifest'])==config['manifest_sha256'],'factorial manifest changed')
    require(digest(config['protocol_file'])==config['protocol_sha256'], 'factorial protocol changed')
    require(digest(config['exclusion_inventory'])==config['exclusion_inventory_sha256'], 'exclusion inventory changed')
    require('prior_data_root' in config and 'prior_excluded_directories' in config,
            'factorial relocated inventory root/policy must be explicit')
    inventory=runner.load_exclusion_inventory(config['exclusion_inventory'], prior_root=config['prior_data_root'])
    runner.validate_inventory_policy(inventory['config'], config['prior_data_root'], config['prior_excluded_directories'])
    manifest,_=runner.load_manifest(config['manifest'],inventory['excluded_identifiers'])
    require(config['n']==manifest['config']['n'] and config['partition']==manifest['config']['partition'], 'factorial config/frame mismatch')
    if config['development_manifest']:
        previous=config['development_manifest']
        require(digest(previous['path'])==previous['sha256'], 'factorial development frame changed')
        development,_=runner.load_manifest(previous['path'],inventory['excluded_identifiers'])
        require(not {s['image_id'] for s in development['scenes'].values()} &
                {s['image_id'] for s in manifest['scenes'].values()}, 'development/confirmation overlap')
    plan=runner.make_plan(seeds,ks)
    runner.validate_result(payload,manifest,plan)
    require(set(payload['cells'])=={runner.cell_key(x) for x in plan},'incomplete factorial grid')
    arrays={}; means={}
    for arm in config['arms']:
        for condition in config['conditions']:
            matrices=[]; invalid=0; total=0
            for seed in seeds:
                grid=[]
                for k in ks:
                    sides=[]
                    for rule in config['rules']:
                        rows=payload['cells'][runner.cell_key((seed,arm,condition,rule,k))]['records']
                        expected_query_ids=[manifest['scenes'][q['scene_id']]['image_id'] for q in manifest['queries']]
                        require([r['query_id'] for r in rows]==expected_query_ids,'factorial shared queries differ')
                        for row in rows:
                            pred=independent_prediction(row['raw_output'],row['label_space'])
                            require(pred==row['prediction'] and bool(row['correct'])==(pred==row['gold']),'independent factorial score differs')
                            invalid+=int(pred is None);total+=1
                            side=rule.removesuffix('_color')
                            require(row['gold']==row['query_colors'][side],'factorial semantic oracle differs')
                            require(row['demo_label_classes']==[c[side] for c in row['demo_colors']],'support semantic oracle differs')
                            require(row['query_id'] not in row['demo_source_ids'],'query-support leakage')
                            require(sorted(row['option_order'])==list(range(4)),'bad options')
                        frozen=payload['cells'][runner.cell_key((seed,'frozen',condition,rule,k))]['records']
                        for row,base in zip(rows,frozen):
                            require(row['prompt_sha256']==base['prompt_sha256'],'arms use different prompts')
                        sides.append([int(r['correct']) for r in rows])
                    grid.append(sides)
                matrices.append(grid)
            arr=np.asarray(matrices,dtype=float) # seed,K,rule,image
            arrays[arm,condition]=arr
            means[f'{arm}:{condition}']={'accuracy':image_interval(arr.mean(axis=(1,2))),
                'both_rules_correct':image_interval(arr.prod(axis=2).mean(axis=1)),
                'left_mean':float(arr[:,:,0,:].mean()),'right_mean':float(arr[:,:,1,:].mean()),
                'invalid_count':invalid,'record_count':total,'unparseable_rate':invalid/total}
    effects={}
    for arm in config['arms'][1:]:
        for condition in config['conditions']:
            delta=arrays[arm,condition]-arrays['frozen',condition]
            joint=arrays[arm,condition].prod(axis=2)-arrays['frozen',condition].prod(axis=2)
            accuracy=seed_interval(delta.mean(axis=(1,2,3)),.9875)
            joint_est=seed_interval(joint.mean(axis=(1,2)),.9875)
            accuracy['mean']=int(delta.sum())/delta.size
            joint_est['mean']=int(joint.sum())/joint.size
            sides=delta.mean(axis=(0,1,3)).tolist()
            effects[f'{arm}:{condition}']={'accuracy_effect_corrected_seed_interval':accuracy,
                'both_correct_effect_corrected_seed_interval':joint_est,
                'accuracy_effect_image_interval_pointwise':image_interval(delta.mean(axis=(1,2))),
                'rule_effects':dict(zip(config['rules'],sides)),
                'prespecified_secondary_transfer_supported':arm=='episode_connector' and len(seeds)==3
                 and accuracy['mean']>=.10 and accuracy['low']>0 and joint_est['low']>0 and min(sides)>0}
    interaction=((arrays['episode_connector','explicit_nonce']-arrays['frozen','explicit_nonce'])-
                 (arrays['episode_connector','implicit_nonce']-arrays['frozen','implicit_nonce'])).mean(axis=(1,2))
    return {'primary_interaction':{'seed_interval_95':seed_interval(interaction.mean(axis=1)),
                'image_interval_95':image_interval(interaction),
                'contrast':'(episode-frozen)_explicit_nonce - (episode-frozen)_implicit_nonce'},
            'condition_means':means,'condition_effects':effects,
            'generation_records':len(plan)*config['n'],'unique_query_images':config['n'],
            'interpretation':'factorial attribute selection, not open-ended VQA or general causal localization'}


def audit_vqa(payload, frame):
    from incontext_collapse import vqa_confirmatory as runner
    from incontext_collapse.vqa import vqa_norm
    config=payload['run_config'];partition=config['partition']
    require(partition in ('development','confirmatory'), 'unknown VQA partition')
    require(config['seeds']==[0,1,2] and config['model']==runner.MODEL, 'VQA model/seeds differ')
    require(config['arms']==(['frozen'] if partition=='development' else ['frozen',*runner.ADAPTER_ARMS]), 'VQA arm grid differs')
    require(config['policy']==runner.POLICY,'VQA policy differs')
    require(config['implementation_sha256']==runner.implementation_manifest(), 'incomplete VQA runtime manifest')
    require(config['model_revision']==runner.model_revision(runner.MODEL) and
            config['n']==runner.N_PARTITION[partition], 'VQA pinned model or sample size differs')
    require(set(config['checkpoints'])==set(runner.ADAPTER_ARMS) and
            all(set(v)=={'0','1','2'} for v in config['checkpoints'].values()), 'VQA checkpoint grid differs')
    check_source_hashes(config['implementation_sha256']);check_checkpoint_bytes(config['checkpoints'])
    require(digest(frame)==config['manifest_sha256'],'VQA frame changed')
    pool,partitions,_=runner.load_prepared(frame,ROOT/'data/revision_v25/vqa_image_frame/manifest.json')
    if partition=='confirmatory':
        development_path=Path(frame).parent.parent/'vqa_development.json'
        require(digest(development_path)==config.get('development_result_sha256'), 'VQA development gate artifact changed')
        development=json.loads(development_path.read_text())
        runner.assert_gate(development,config,pool,partitions['development'])
    queries=partitions[partition]
    require(len(queries)==(80 if partition=='development' else 400),'VQA partition size')
    runner.validate_records(payload,pool,queries,partition,config['arms'],require_complete=True)
    for row in payload['records'].values():
        refs=[vqa_norm(a) for a in row['answers']];pred=vqa_norm(row['prediction'])
        independent=(float(pred==refs[0]) if len(refs)==1 else statistics.fmean(
            min(sum(pred==value for j,value in enumerate(refs) if j!=i)/3,1) for i in range(len(refs))))
        require(abs(independent-row['score'])<1e-12,'independent VQA score differs')
        require(row['image_id'] not in row['demo_image_ids'],'VQA image leakage')
    ids=[runner.vqa_query_id(q[0],q[1]) for q in queries]
    analysis=runner.analyze_records(payload['records'],ids,arms=config['arms'])
    require(payload['analysis']==analysis,'stored VQA analysis differs')
    # Independent matrix assembly and bootstrap of the prespecified gate.
    matrix=[]
    for seed in config['seeds']:
        matrix.append([payload['records'][runner.record_key(seed,'frozen',0,q)]['score']-
                       payload['records'][runner.record_key(seed,'frozen',16,q)]['score'] for q in ids])
    drop=image_interval(matrix,seed=runner.BOOTSTRAP_SEED)
    numerator=np.rint(np.asarray(matrix)*10).astype(np.int64)
    require(np.allclose(numerator/10,np.asarray(matrix),rtol=0,atol=1e-12), 'VQA score does not lie on its exact tenth grid')
    drop['mean']=int(numerator.sum())/(10*numerator.size)
    image_numerators=numerator.sum(axis=0);rng=np.random.default_rng(runner.BOOTSTRAP_SEED);draws=[]
    for offset in range(0,10000,500):
        draws.extend(map(int,image_numerators[rng.integers(0,len(ids),(500,len(ids)))].sum(axis=1)))
    draws.sort();denominator=10*numerator.size
    def exact_quantile(probability):
        position=probability*(len(draws)-1);lo=position.numerator//position.denominator
        return (Fraction(draws[lo])+(position-lo)*(draws[min(lo+1,len(draws)-1)]-draws[lo]))/denominator
    low,high=exact_quantile(Fraction(1,40)),exact_quantile(Fraction(39,40))
    drop['low']=float(low);drop['high']=float(high)
    gate=Fraction(int(numerator.sum()),denominator)>=Fraction(1,10) and low>0
    require(gate==analysis['baseline_collapse_supported'],'independent VQA gate differs')
    return {'analysis':analysis,'independent_frozen_drop':drop,'gate_pass':gate,
            'generation_records':len(payload['records']),'unique_query_images':len(queries),
            'interpretation':'finite-frame VQA; stop after failed baseline screen; no repair claim without collapse'}


def audit(path,kind,frame=None):
    path=Path(path);payload=json.loads(path.read_text())
    if kind=='shortcut':result=audit_shortcut(payload)
    elif kind=='rule':result=audit_rule(payload)
    elif kind=='vqa':result=audit_vqa(payload,frame)
    else:raise ValueError('unknown study')
    return {'status':'complete','kind':kind,'artifact':str(path),'artifact_sha256':digest(path),
            'analyzer_sha256':digest(__file__),'partition':payload['run_config']['partition'],
            'config':payload['run_config'],'findings':result,'paper_modified':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifact');parser.add_argument('--kind',choices=['shortcut','rule','vqa'],required=True)
    parser.add_argument('--frame');parser.add_argument('--out',required=True)
    args=parser.parse_args()
    report=audit(args.artifact,args.kind,args.frame)
    write_json_atomic(args.out,report)
    print(json.dumps({'status':report['status'],'kind':args.kind,'records':report['findings']['generation_records'],'out':args.out}))


if __name__=='__main__':main()
