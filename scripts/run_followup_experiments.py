#!/usr/bin/env python3
"""Run three fixed checkpoint studies serially, with a prospective VQA gate."""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from incontext_collapse.results import write_json_atomic
OUT='data/followup_20261010'
PROTOCOL='docs/followup_experiments_protocol_2026_10_10.txt'
CONNECTOR='data/revision_v24/qwen2vl2b_shapes_connector_episode_remap150_n40_generation_v24_adapters/connector_both'
NATURAL='data/revision_v24_completion/qwen2vl2b_shapes_connector_natural150_n40_generation_v24_adapters/connector_both'
ALL='data/revision_v24_completion/qwen2vl2b_shapes_all_layer_episode_remap150_n40_generation_v24_adapters/all_both'
VQA='data/revision_v24_completion/qwen2vl2b_vqav2_connector_train150_n40_v24_adapters/connector_both'
MODEL='Qwen/Qwen2-VL-2B-Instruct'

def now():return datetime.now(timezone.utc).isoformat()

def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def jobs(root=ROOT):
    py=str(root/'.venv/bin/python')
    rule=['--data-dir',OUT+'/rule_images','--prior-exclude-dir',OUT,'--protocol-file',PROTOCOL,
          '--adapter','natural_connector='+NATURAL,'--adapter','episode_connector='+CONNECTOR,
          '--adapter','all_layer='+ALL]
    shortcut=['--data-dir',OUT+'/shortcut_images','--adapter',CONNECTOR]
    vqa=['--data-dir',OUT+'/vqa_images','--adapter','shapes_connector='+CONNECTOR,'--adapter','vqa_connector='+VQA]
    definitions=[
      ('rule_development','rule','rule_factorial',rule+['--partition','development','--n','16','--seeds','0','--ks','8'],512),
      ('shortcut_development','shortcut','shortcut_controls',shortcut+['--partition','development'],192),
      ('vqa_development','vqa','vqa_confirmatory',vqa+['--partition','development'],960),
      ('rule_confirmatory','rule','rule_factorial',rule+['--partition','confirmatory','--n','80','--seeds','0,1,2','--ks','8,16'],15360),
      ('shortcut_confirmatory','shortcut','shortcut_controls',shortcut+['--partition','confirmatory'],2880),
      ('vqa_confirmatory','vqa','vqa_confirmatory',vqa+['--partition','confirmatory','--development-result',OUT+'/vqa_development.json'],14400)]
    return [{'name':n,'kind':k,'output':f'{OUT}/{n}.json',
             'command':[py,'-m','incontext_collapse.'+m,*a,'--out',f'{OUT}/{n}.json'],
             'planned_generations':count,'conditional':n=='vqa_confirmatory'} for n,k,m,a,count in definitions]

def environment():
    return {**os.environ,'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
            'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1','MPLCONFIGDIR':'/private/tmp/incontext-followup-matplotlib'}

def run(command,root=ROOT):
    print(f'[{now()}] {shlex.join(command)}',flush=True)
    subprocess.run(command,cwd=root,env=environment(),check=True)

def source_manifest(root=ROOT):
    paths=list((root/'incontext_collapse').glob('*.py'))
    paths.extend(root/p for p in [PROTOCOL,'requirements.lock','scripts/run_followup_experiments.py',
        'scripts/analyze_followup_experiments.py','tests/test_followup_experiments.py',
        'tests/test_rule_factorial.py','tests/test_shortcut_controls.py','tests/test_vqa_confirmatory.py'])
    return {str(p.relative_to(root)):digest(p) for p in sorted(paths)}

def paper_manifest(root=ROOT):
    return {str(p.relative_to(root)):digest(p) for p in sorted((root/'paper').rglob('*'))
            if p.is_file() and '.git' not in p.parts}

def data_manifest(root=ROOT):
    paths=[]
    for subdir in ['rule_images','shortcut_images','vqa_images']:
        paths.extend(p for p in (root/OUT/subdir).rglob('*') if p.is_file())
    return {str(p.relative_to(root)):digest(p) for p in sorted(paths)}

def checkpoint_manifest(root=ROOT):
    from incontext_collapse.option_order_eval import checkpoint_manifest as describe
    return describe({n:root/p for n,p in [('episode_connector',CONNECTOR),('natural_connector',NATURAL),
                                        ('all_layer',ALL),('vqa_connector',VQA)]},[0,1,2])

def model_manifest():
    from scripts.run_revision_coverage_queue import cached_model_manifest
    cache=cached_model_manifest(MODEL);snapshot=Path(cache['snapshot'])
    cache['file_sha256']={p.name:digest(p) for p in sorted(snapshot.iterdir()) if p.is_file()}
    return cache

def hardware():
    import torch
    memory=subprocess.run(['sysctl','-n','hw.memsize'],capture_output=True,text=True,check=True).stdout.strip()
    return {'platform':platform.platform(),'machine':platform.machine(),'python':sys.version,
            'physical_memory_bytes':int(memory),'torch':torch.__version__,
            'mps_available':torch.backends.mps.is_available(),'command_line':sys.argv,'recorded_utc':now()}

def check_frozen(state,root=ROOT,check_data=True):
    if source_manifest(root)!=state['source_sha256']:raise RuntimeError('frozen experiment source changed')
    if paper_manifest(root)!=state['paper_sha256']:raise RuntimeError('paper changed during campaign')
    if check_data and data_manifest(root)!=state['data_sha256']:raise RuntimeError('prepared data changed')
    if checkpoint_manifest(root)!=state['checkpoints']:raise RuntimeError('saved checkpoints changed')

def audit_job(job,root=ROOT):
    from scripts.analyze_followup_experiments import audit
    frame=root/OUT/'vqa_images/manifest.json' if job['kind']=='vqa' else None
    report=audit(root/job['output'],job['kind'],frame)
    if report['findings']['generation_records']!=job['planned_generations']:
        raise RuntimeError('audited record count differs from fixed plan')
    path=root/OUT/'reports'/f"{job['name']}.json";write_json_atomic(path,report)
    return {'path':str(path.relative_to(root)),'sha256':digest(path),
            'artifact_sha256':digest(root/job['output']),'generation_records':report['findings']['generation_records']}

def verify_completed(job,root=ROOT):
    if (digest(root/job['output'])!=job['audit']['artifact_sha256'] or
            digest(root/job['audit']['path'])!=job['audit']['sha256']):
        raise RuntimeError('completed evidence or audit report changed')

def verify_stage_plan(state,plan,require_finished=False):
    if len(state['stages'])!=len(plan):raise RuntimeError('stage membership differs from fixed plan')
    for observed,expected in zip(state['stages'],plan):
        if any(observed.get(k)!=v for k,v in expected.items()):
            raise RuntimeError('stage command/configuration differs from fixed plan')
        status=observed.get('status')
        if status not in (None,'running','auditing','complete','stopped_by_prespecified_gate'):
            raise RuntimeError('unknown stage status')
        if status=='stopped_by_prespecified_gate' and not expected['conditional']:
            raise RuntimeError('unconditional study cannot be skipped by a gate')
        if require_finished and status not in ('complete','stopped_by_prespecified_gate'):
            raise RuntimeError('campaign marked complete with unfinished stages')

def checked_gate(state,root=ROOT):
    from scripts.analyze_followup_experiments import audit
    development=next(j for j in state['stages'] if j['name']=='vqa_development')
    if development.get('status')!='complete':raise RuntimeError('VQA gate requires audited complete development')
    verify_completed(development,root)
    fresh=audit(root/development['output'],'vqa',root/OUT/'vqa_images/manifest.json')
    if fresh['artifact_sha256']!=development['audit']['artifact_sha256']:
        raise RuntimeError('VQA raw artifact changed during gate audit')
    return fresh['findings']['gate_pass'],development['audit']['sha256']

def prepare(root=ROOT):
    for job in jobs(root):
        if job['kind']!='vqa':run([*job['command'],'--prepare-only'],root)
    from incontext_collapse.vqa_confirmatory import load_prepared
    load_prepared(root/OUT/'vqa_images/manifest.json',root/'data/revision_v25/vqa_image_frame/manifest.json')

def preflight(root=ROOT):
    for job in jobs(root):
        if not job['conditional']:run([*job['command'],'--preflight'],root)
    print(json.dumps({'preflight':'passed','unconditional_records':19904,'conditional_records':14400}),flush=True)

def execute(root=ROOT):
    from scripts.run_revision_coverage_queue import execution_lock
    control=root/OUT/'queue';control.mkdir(parents=True,exist_ok=True);path=control/'status.json'
    with execution_lock(control/'queue.lock'):
        old=json.loads(path.read_text()) if path.exists() else None;plan=jobs(root)
        if old:
            if old['fixed_plan']!=plan:raise RuntimeError('fixed plan changed; refusing resume')
            verify_stage_plan(old,plan,require_finished=old['status']=='complete')
            check_frozen(old,root);state=old
            for job in state['stages']:
                if job.get('status')=='complete':verify_completed(job,root)
                elif job.get('status')=='stopped_by_prespecified_gate':
                    gate_pass,gate_hash=checked_gate(state,root)
                    if gate_pass or gate_hash!=job['gate_report_sha256']:
                        raise RuntimeError('stored VQA stopping decision differs from verified raw evidence')
            if model_manifest()!=state['model_cache']:raise RuntimeError('backbone/processor cache changed before resume')
            if state['status']=='complete':
                print('Campaign already complete.',flush=True);return
            state['resumed_utc']=now()
        else:
            preflight(root)
            state={'schema_version':1,'started_utc':now(),'status':'preflight','fixed_plan':plan,
                   'stages':deepcopy(plan),'source_sha256':source_manifest(root),'paper_sha256':paper_manifest(root),
                   'data_sha256':data_manifest(root),'checkpoints':checkpoint_manifest(root),
                   'model_cache':model_manifest(),'hardware':hardware(),'publication_ready':False,'paper_modified':False}
            if not state['hardware']['mps_available']:raise RuntimeError('MPS unavailable; refusing implicit CPU campaign')
        state['pid']=os.getpid();write_json_atomic(path,state);caffeinate=None
        try:
            with execution_lock(root/'data/revision_v25/queue/run.lock',wait=True,
                                check=lambda:check_frozen(state,root,check_data=False)):
                check_frozen(state,root)
                caffeinate=subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                state['status']='running';write_json_atomic(path,state)
                for job in state['stages']:
                    if job.get('status') in ('complete','stopped_by_prespecified_gate'):continue
                    check_frozen(state,root)
                    if job['conditional']:
                        gate_pass,gate_hash=checked_gate(state,root)
                        if not gate_pass:
                            job.update(status='stopped_by_prespecified_gate',finished_utc=now(),
                                reason='Fresh frozen-model screen did not meet the prespecified collapse gate',
                                gate_report_sha256=gate_hash)
                            write_json_atomic(path,state);continue
                    state['current_stage']=job['name'];job.update(status='running',started_utc=now());write_json_atomic(path,state)
                    run(job['command'],root);check_frozen(state,root)
                    job['status']='auditing';write_json_atomic(path,state);job['audit']=audit_job(job,root)
                    job.update(status='complete',finished_utc=now());write_json_atomic(path,state)
                check_frozen(state,root)
                verify_stage_plan(state,plan,require_finished=True)
                for job in state['stages']:
                    if job['status']=='complete':verify_completed(job,root)
                    else:
                        gate_pass,gate_hash=checked_gate(state,root)
                        if gate_pass or gate_hash!=job['gate_report_sha256']:
                            raise RuntimeError('final stopping decision differs from verified evidence')
                if model_manifest()!=state['model_cache']:raise RuntimeError('backbone/processor cache changed')
                state.update(status='complete',finished_utc=now(),paper_modified=False);write_json_atomic(path,state)
                print(json.dumps({'status':'complete','stages':{j['name']:j['status'] for j in state['stages']}}),flush=True)
        except BaseException as error:
            state.update(status='failed',error=repr(error),failed_utc=now());write_json_atomic(path,state);raise
        finally:
            if caffeinate is not None:caffeinate.terminate();caffeinate.wait(timeout=10)

def main():
    parser=argparse.ArgumentParser(description=__doc__);actions=parser.add_mutually_exclusive_group(required=True)
    for name in ['plan','prepare','preflight','execute']:actions.add_argument('--'+name,action='store_true')
    args=parser.parse_args()
    if args.plan:print(json.dumps(jobs(),indent=2))
    elif args.prepare:prepare()
    elif args.preflight:preflight()
    else:execute()

if __name__ == "__main__":
    raise SystemExit("This historical queue is retired. Use scripts/reproduce.py plan --study all, then run --study ID --run-dir FRESH_PATH. Original source is retained in reproducibility/legacy/.")
