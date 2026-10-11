#!/usr/bin/env python3
"""Deterministic manuscript tables from the completed 2026-10-10 campaign.

Validates frozen queue/audit/raw/source hashes, independently reproduces the A
exact supplement and decision predicates, and exports small portable fragments.
No model calls or repeated full inference-artifact audits. Existing unequal files
are never overwritten; --check validates the generated package without writes.
"""
from __future__ import annotations
import argparse
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

C = 'data/followup_20261010'
PROTOCOL = 'docs/followup_experiments_protocol_2026_10_10.txt'
QUEUE_SHA = 'bac0d6357f06b28616068fa4be72253b7fe6b9b6d7eba11db45480025386d937'
REPORT_SHA = '8e600f7e75f74fcdf745771af188bceeee6a60dfaec5bed706b9f09dd650b66a'
GUARD_SHA = '8442772777a6556bd2cb9f2eddf3478f1243bb79902b3cc7fafcb03455895a64'
SUPPLEMENT_SHA = '24482a98083d532c1f062283598bf456768a4d4e0d1814bc7e1b47d338683dd8'
AMENDMENT_SHA = '235c1225415d9a23cb6c41625fa57d0df9801b64b7a658442c6e5264cf6d786b'
STAGES = (('rule_development','rule',16,[0],512), ('shortcut_development','shortcut',16,[0],192),
          ('vqa_development','vqa',80,[0,1,2],960), ('rule_confirmatory','rule',80,[0,1,2],15360),
          ('shortcut_confirmatory','shortcut',80,[0,1,2],2880), ('vqa_confirmatory','vqa',400,[0,1,2],14400))
ARMS = ('frozen','natural_connector','episode_connector','all_layer')
ARM_NAMES = {'frozen':'Frozen','natural_connector':'Natural connector','episode_connector':'Episodic connector',
             'connector':'Episodic connector','all_layer':'All-layer episodic',
             'shapes_connector':'Shapes connector','vqa_connector':'VQA connector'}
CONDS = ('explicit_natural','explicit_nonce','implicit_natural','implicit_nonce')
COND_NAMES = {key:key.replace('_',' / ').capitalize() for key in CONDS}
LABEL_NAMES = {'standard':'Natural','remap':'Nonce'}

def require(value, message):
    if not value: raise ValueError(message)

def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def encoded(value): return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()

def local_data_path(root, value):
    path=Path(value)
    require('..' not in path.parts, 'recorded path traversal')
    if path.is_absolute():
        starts=[i for i,part in enumerate(path.parts) if part=='data']
        require(len(starts)==1,'absolute evidence path lacks unique data subtree')
        path=Path(*path.parts[starts[0]:])
    require(path.parts[0]=='data','recorded evidence path is outside data')
    resolved=(root/path).resolve()
    require(resolved.is_relative_to(root/'data'),'evidence path escapes project data')
    return resolved

def number(value):
    require(type(value) in (int,float) and math.isfinite(value),'invalid numerical value')
    result=Decimal(str(value)).quantize(Decimal('.001'),rounding=ROUND_HALF_UP)
    return '0.000' if result==0 else format(result,'.3f')

def interval(value):
    return f"{number(value['mean'])} [{number(value['low'])}, {number(value['high'])}]"

def seed_interval(values, confidence=.95):
    require(len(values)==3,'three seeds required for t interval')
    mean=statistics.fmean(values);critical=math.sqrt(2)*confidence/math.sqrt(1-confidence**2)
    half=critical*statistics.stdev(values)/math.sqrt(3)
    return {'mean':mean,'low':mean-half,'high':mean+half,'confidence':confidence,'seed_values':list(values)}

def same_numbers(a,b):
    require(all(math.isclose(a[k],b[k],abs_tol=1e-12,rel_tol=0) for k in ('mean','low','high')),'interval differs from independently recomputed seed interval')

def b_criteria(row):
    a=row['accuracy_effect_corrected_seed_interval'];j=row['both_correct_effect_corrected_seed_interval']
    side=row['rule_effects']
    checks={'accuracy_mean_ge_0.10':a['mean']>=.10,'accuracy_lower_gt_0':a['low']>0,
            'joint_lower_gt_0':j['low']>0,'left_mean_gt_0':side['left_color']>0,'right_mean_gt_0':side['right_color']>0}
    return checks,all(checks.values())

def c_criteria(analysis):
    d=analysis['frozen_drop_D'];baseline={'D_mean_ge_0.10':d['mean']>=.10,'D_lower_gt_0':d['low']>0}
    result={'baseline':baseline,'baseline_met':all(baseline.values()),'adapters':{}}
    for arm,row in analysis['adapter_claims'].items():
        g,r,t,m=(row[k] for k in ('gain_G','collapse_reduction_R','zero_shot_retention','remaining_drop'))
        checks={'confirmed_D':result['baseline_met'],'G_mean_ge_0.05':g['mean']>=.05,'G_lower_gt_0':g['low']>0,
                'R_lower_gt_0':r['low']>0,'retention_lower_gt_minus0.03':t['low']>-.03}
        repair=all(checks.values());removed=repair and m['high']<.10
        result['adapters'][arm]={'components':checks,'repair_met':repair,'remaining_upper_lt_0.10':m['high']<.10,'removed_met':removed}
    return result

class Evidence:
    def __init__(self, root):
        self.root=Path(root).resolve();self.pins={};self.reports={};self.raw_a={}
        self.state=self.read(C+'/queue/status.json',QUEUE_SHA)
        require(self.state['status']=='complete' and self.state['paper_modified'] is False,'campaign incomplete')
        self.read_bytes(C+'/analysis/results_report.txt',REPORT_SHA)
        plan=self.state['fixed_plan'];stages=self.state['stages']
        require([x['name'] for x in stages]==[x[0] for x in STAGES]==[x['name'] for x in plan],'fixed stage grid changed')
        for path,digest in self.state['source_sha256'].items(): self.pin(path,digest)
        for stage,fixed,(name,kind,n,seeds,count) in zip(stages,plan,STAGES):
            require(all(stage[k]==v for k,v in fixed.items()),'stage differs from fixed plan')
            require(stage['status']=='complete' and stage['kind']==kind and stage['planned_generations']==count,'stage incomplete or wrong grid')
            require(stage['output']==f'{C}/{name}.json' and stage['conditional']==(name=='vqa_confirmatory'),'wrong fixed output/branch')
            audit=stage['audit'];require(audit['path']==f'{C}/reports/{name}.json' and audit['generation_records']==count,'wrong audit slot/count')
            self.pin(stage['output'],audit['artifact_sha256'])
            report=self.read(audit['path'],audit['sha256']);config=report['config'];f=report['findings']
            require(report['status']=='complete' and report['kind']==kind and report['partition']==name.split('_',1)[1],'invalid audit status/kind')
            require(config['n']==n and config['seeds']==seeds and config['partition']==report['partition'],'wrong audited sample/seed grid')
            require(f['generation_records']==count and f['unique_query_images']==n and report['paper_modified'] is False,'audited record counts differ')
            require(local_data_path(self.root,report['artifact'])==self.root/stage['output'],'audit refers to wrong artifact')
            require(report['artifact_sha256']==audit['artifact_sha256'] and report['analyzer_sha256']==self.state['source_sha256']['scripts/analyze_followup_experiments.py'],'audit provenance differs')
            self.reports[name]=report
            if kind=='shortcut': self.raw_a[name]=self.read(stage['output'],audit['artifact_sha256'])
        require(sum(r['findings']['generation_records'] for r in self.reports.values())==34304,'campaign record total differs')
        dev=self.reports['vqa_development'];confirm=self.reports['vqa_confirmatory']
        require(dev['findings']['gate_pass'] is True and dev['findings']['analysis']['baseline_collapse_supported'] is True,'completed branch lacks passed development gate')
        require(confirm['config']['development_result_sha256']==dev['artifact_sha256'],'C confirmation is not bound to development')
        self.guard=self.read(C+'/analysis/shortcut_primary_exact_guard.json',SUPPLEMENT_SHA)
        self.read(C+'/analysis/numerical_correction_v2/amendment.json',AMENDMENT_SHA)
        self.guard_source=C+'/analysis/tools_v2/exact_shortcut_bootstrap_guard.py';self.pin(self.guard_source,GUARD_SHA)
        require(self.guard['kind']=='exact_shortcut_primary_guard' and self.guard['guard_source_sha256']==GUARD_SHA and self.guard['amendment_sha256']==AMENDMENT_SHA,'wrong exact-primary supplement')
        spec=importlib.util.spec_from_file_location('reviewed_followup_exact_guard',self.root/self.guard_source)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        for name,payload in self.raw_a.items():
            config=payload['run_config'];manifest_path=local_data_path(self.root,config['manifest_path'])
            manifest=self.read(manifest_path.relative_to(self.root).as_posix(),config['manifest_sha256'])
            actual=module.corrected_primary(payload,manifest);saved=self.guard['stages'][name]
            require(all(saved[k]==v for k,v in actual.items()),'exact-primary supplement does not reproduce')
            require(saved['artifact_sha256']==self.reports[name]['artifact_sha256'] and saved['audit_report_sha256']==self.pins[f'{C}/reports/{name}.json'],'guard input provenance differs')
            require(saved['original_auditor_primary']==self.reports[name]['findings']['primary'],'guard auditor view differs')
            require(saved['auditor_criteria_match']==(actual['corrected']['criteria_met']==saved['original_auditor_primary']['criteria_met']),'guard comparison Boolean differs')
        self.decisions={}
        b=self.reports['rule_confirmatory']['findings'];primary=b['primary_interaction']['seed_interval_95']
        same_numbers(seed_interval(primary['seed_values']),primary)
        for name,row in b['condition_effects'].items():
            for metric in ('accuracy_effect_corrected_seed_interval','both_correct_effect_corrected_seed_interval'):
                v=row[metric];require(v['confidence']==.9875,'B corrected interval level differs');same_numbers(seed_interval(v['seed_values'],.9875),v)
            checks,met=b_criteria(row);require(met==row['prespecified_secondary_transfer_supported'],'B row decision differs')
            self.decisions['B:'+name]={'components':checks,'met':met,'scope':'prespecified episodic condition' if name.startswith('episode_connector:') else 'descriptive control; no substitute primary claim'}
        for partition in ('development','confirmatory'):
            f=self.reports['vqa_'+partition]['findings'];a=f['analysis'];decision=c_criteria(a)
            require(a['frozen_drop_D']['confidence']==.95,'D interval level differs')
            require(decision['baseline_met']==a['baseline_collapse_supported']==f['gate_pass'],'C baseline decision differs')
            for arm,row in a['adapter_claims'].items():
                require(all(row[k]['confidence']==.975 for k in ('gain_G','collapse_reduction_R','zero_shot_retention','remaining_drop')),'adapter interval level differs')
                d=decision['adapters'][arm];require(d['repair_met']==row['repair_supported'] and d['removed_met']==row['collapse_below_margin_supported'],'C conjunctive decision differs')
            self.decisions['C:'+partition]=decision

    def pin(self,rel,expected=None):
        path=(self.root/rel).resolve();require(path.is_relative_to(self.root),'input escapes project')
        actual=sha(path);require(expected is None or actual==expected,'input bytes changed: '+rel)
        self.pins[rel]=actual;return actual
    def read_bytes(self,rel,expected=None):
        data=(self.root/rel).read_bytes();digest=hashlib.sha256(data).hexdigest()
        require(expected is None or digest==expected,'input bytes changed: '+rel);self.pins[rel]=digest;return data
    def read(self,rel,expected=None):return json.loads(self.read_bytes(rel,expected))
    def recheck(self):
        for rel,digest in self.pins.items():require(sha(self.root/rel)==digest,'evidence changed during export: '+rel)

def build(e):
    files={};columns={};rows_count={}
    def add(name,headers,rows):
        require(all(len(row)==len(headers) for row in rows),'table column mismatch: '+name)
        # Journal-width presentation only: preserve all rounded numeric strings.
        def stack(*parts): return r'\shortstack[l]{'+r'\\'.join('{'+part+'}' for part in parts)+'}'
        def stack_interval(value):
            mean,bounds=value.split(' ',1)
            return stack(mean,bounds)
        if name in ('B_compact.tex','B_conditions.tex'):
            rows=[list(row) for row in rows]
            condition_col=0 if name=='B_compact.tex' else 1
            for row in rows:
                row[condition_col]=stack(*row[condition_col].split(' / '))
            if name=='B_compact.tex':
                headers=['Condition',stack(r'$\Delta$ accuracy',r'[98.75\% seed CI]'),
                         stack(r'$\Delta$ joint',r'[98.75\% seed CI]'),
                         r'$\Delta$ left',r'$\Delta$ right',stack('Transfer','criterion')]
                for row in rows:
                    for col in (1,2): row[col]=stack_interval(row[col])
            else:
                headers=['Arm','Condition',stack('Accuracy',r'[95\% image CI]'),
                         stack('Joint',r'[95\% image CI]'),'Left','Right','Invalid']
                for row in rows:
                    if ' ' in row[0]: row[0]=stack(*row[0].split(' ',1))
                    for col in (2,3): row[col]=stack_interval(row[col])
        elif name=='C_compact.tex':
            require([row[0] for row in rows]==['Shapes connector','VQA connector'],
                    'unexpected C adapter ordering')
            metrics=[r'$G$',r'$R$','Retention','Remaining drop','Repair criterion','Removal criterion']
            headers=['Quantity',stack('Shapes connector',r'[97.5\% image CI]'),
                     stack('VQA connector',r'[97.5\% image CI]')]
            rows=[[metric,*[stack_interval(row[i+1]) if i<4 else row[i+1] for row in rows]]
                  for i,metric in enumerate(metrics)]
        columns[name]=headers;rows_count[name]=len(rows)
        text='% Generated by export_followup_tables.py; tabular only; no manual numerical edits.\n'
        text+='\\begin{tabular}{'+'l'*len(headers)+'}\n\\toprule\n'
        text+=' & '.join(headers)+' \\\\\n\\midrule\n'
        text+='\n'.join(' & '.join(map(str,row))+' \\\\' for row in rows)
        text+='\n\\bottomrule\n\\end{tabular}\n';files[name]=text.encode()
    a=e.reports['shortcut_confirmatory']['findings'];b=e.reports['rule_confirmatory']['findings']
    ac=[];af=[];aseed=[]
    for arm in ('frozen','connector'):
        for label in ('standard','remap'):
            key=arm+':'+label;v=a['variant_metrics'][key]
            estimate=e.guard['stages']['shortcut_confirmatory']['corrected'] if key=='frozen:standard' else a['secondary_contrasts'][key]
            ac.append([ARM_NAMES[arm],LABEL_NAMES[label],interval(estimate),number(v['endpoint']['accuracy']-v['sham']['accuracy'])])
            for variant in ('base','endpoint','sham'):
                af.append([ARM_NAMES[arm],LABEL_NAMES[label],variant.capitalize(),*[number(v[variant][metric]) for metric in ('accuracy','invalid','first_match','last_match','majority_match','nearest_match')]])
            se=a['secondary_contrasts'][key]['seed_interval_95']
            aseed.append([ARM_NAMES[arm],LABEL_NAMES[label],interval(se),*[number(x) for x in se['seed_values']]])
    add('A_compact.tex',['Arm','Labels',r'$\Delta$ wrong majority [95\% image CI]',r'$\Delta$ accuracy'],ac)
    add('A_complete.tex',['Arm','Labels','Variant','Accuracy','Invalid','First','Last','Majority','Nearest'],af)
    add('A_seed_sensitivity.tex',['Arm','Labels',r'95\% seed interval','Seed 0','Seed 1','Seed 2'],aseed)
    moderation=[]
    for label in ('standard','remap'):
        raw=e.raw_a['shortcut_confirmatory']['summary']['moderation:'+label]
        estimate={'mean':raw['mean'],'low':raw['ci95_low'],'high':raw['ci95_high']}
        expected=a['secondary_contrasts']['connector:'+label]['mean']-a['secondary_contrasts']['frozen:'+label]['mean']
        require(math.isclose(raw['mean'],expected,abs_tol=1e-12),'A moderation mean differs')
        moderation.append([LABEL_NAMES[label],interval(estimate)])
    add('A_moderation.tex',['Labels',r'Connector--frozen position effect [95\% image CI]'],moderation)
    bc=[];bf=[];controls=[];criteria=[]
    for condition in CONDS:
        row=b['condition_effects']['episode_connector:'+condition];checks,met=b_criteria(row)
        bc.append([COND_NAMES[condition],interval(row['accuracy_effect_corrected_seed_interval']),interval(row['both_correct_effect_corrected_seed_interval']),number(row['rule_effects']['left_color']),number(row['rule_effects']['right_color']),'Met' if met else 'Not met'])
        criteria.append([COND_NAMES[condition],*['Yes' if yes else 'No' for yes in checks.values()],'Met' if met else 'Not met'])
    for arm in ARMS:
        for condition in CONDS:
            row=b['condition_means'][arm+':'+condition]
            bf.append([ARM_NAMES[arm],COND_NAMES[condition],interval(row['accuracy']),interval(row['both_rules_correct']),number(row['left_mean']),number(row['right_mean']),number(row['unparseable_rate'])])
    for arm in ('natural_connector','all_layer'):
        for condition in CONDS:
            row=b['condition_effects'][arm+':'+condition]
            acc=seed_interval(row['accuracy_effect_corrected_seed_interval']['seed_values'])
            joint=seed_interval(row['both_correct_effect_corrected_seed_interval']['seed_values'])
            controls.append([ARM_NAMES[arm],COND_NAMES[condition],interval(acc),interval(joint),number(row['rule_effects']['left_color']),number(row['rule_effects']['right_color'])])
    add('B_compact.tex',['Condition',r'$\Delta$ accuracy [98.75\% CI]',r'$\Delta$ joint [98.75\% CI]',r'$\Delta$ left',r'$\Delta$ right','Transfer criterion'],bc)
    add('B_conditions.tex',['Arm','Condition',r'Accuracy [95\% image CI]',r'Joint [95\% image CI]','Left','Right','Invalid'],bf)
    p=b['primary_interaction']['seed_interval_95']
    add('B_primary.tex',['Diagnostic',r'95\% seed interval','Seed 0','Seed 1','Seed 2'],[[r'$(E-F)_{\rm explicit,nonce}-(E-F)_{\rm implicit,nonce}$',interval(p),*[number(x) for x in p['seed_values']]]])
    add('B_complete.tex',['Condition',r'Mean $\geq.10$',r'Acc. lower $>0$',r'Joint lower $>0$',r'Left $>0$',r'Right $>0$','All criteria'],criteria)
    add('B_controls.tex',['Control arm','Condition',r'$\Delta$ accuracy [95\% seed CI]',r'$\Delta$ joint [95\% seed CI]',r'$\Delta$ left',r'$\Delta$ right'],controls)
    curves=[];cc=[];cd=[];gate_rows=[];cs=[]
    for partition in ('development','confirmatory'):
        f=e.reports['vqa_'+partition]['findings'];analysis=f['analysis'];dec=e.decisions['C:'+partition];part='Development' if partition=='development' else 'Confirmation'
        d=analysis['frozen_drop_D'];gate_rows.append([part,80 if partition=='development' else 400,interval(d),'Yes' if dec['baseline']['D_mean_ge_0.10'] else 'No','Yes' if dec['baseline']['D_lower_gt_0'] else 'No','Met' if dec['baseline_met'] else 'Not met'])
        cd.extend([[part,'Frozen',r'$D$',interval(d),r'Mean $\geq.10$','Yes' if dec['baseline']['D_mean_ge_0.10'] else 'No'],[part,'Frozen',r'$D$',interval(d),r'95\% lower $>0$','Yes' if dec['baseline']['D_lower_gt_0'] else 'No']])
        for arm in ('frozen','shapes_connector','vqa_connector'):
            if arm not in analysis['accuracies']:continue
            for k in ('0','4','8','16'):
                row=analysis['accuracies'][arm][k]
                curves.append([part,ARM_NAMES[arm],k,interval(row)])
                cs.append([part,ARM_NAMES[arm],k,*[number(x) for x in row['seed_means']]])
        for arm,row in analysis['adapter_claims'].items():
            decrow=dec['adapters'][arm]
            cc.append([ARM_NAMES[arm],*[interval(row[k]) for k in ('gain_G','collapse_reduction_R','zero_shot_retention','remaining_drop')],'Met' if decrow['repair_met'] else 'Not met','Met' if decrow['removed_met'] else 'Not met'])
            items=[('gain_G',r'$G$',r'Mean $\geq.05$','G_mean_ge_0.05'),('gain_G',r'$G$',r'Lower $>0$','G_lower_gt_0'),('collapse_reduction_R',r'$R$',r'Lower $>0$','R_lower_gt_0'),('zero_shot_retention',r'$A_0-F_0$',r'Lower $>-.03$','retention_lower_gt_minus0.03')]
            for field,symbol,test,key in items:cd.append([part,ARM_NAMES[arm],symbol,interval(row[field]),test,'Yes' if decrow['components'][key] else 'No'])
            cd.append([part,ARM_NAMES[arm],r'$A_0-A_{16}$',interval(row['remaining_drop']),r'Upper $<.10$','Yes' if decrow['remaining_upper_lt_0.10'] else 'No'])
    add('C_compact.tex',['Adapter',r'$G$ [97.5\% CI]',r'$R$ [97.5\% CI]',r'Retention [97.5\% CI]',r'Remaining drop [97.5\% CI]','Repair','Removed'],cc)
    add('C_gate.tex',['Partition','Images',r'$D$ [95\% image CI]',r'Mean $\geq.10$',r'Lower $>0$','Criterion'],gate_rows)
    add('C_curves.tex',['Partition','Arm',r'$K$',r'Soft accuracy [95\% image CI]'],curves)
    add('C_complete.tex',['Partition','Arm','Quantity','Mean [CI]','Component test','Pass'],cd)
    add('C_seed_values.tex',['Partition','Arm',r'$K$','Seed 0','Seed 1','Seed 2'],cs)
    # Development is transparent and kept separate from confirmation.
    ad=[]
    dev=e.reports['shortcut_development']['findings']
    for arm in ('frozen','connector'):
        for label in ('standard','remap'):
            key=arm+':'+label
            effect=e.guard['stages']['shortcut_development']['corrected'] if key=='frozen:standard' else dev['secondary_contrasts'][key]
            ad.append([ARM_NAMES[arm],LABEL_NAMES[label],interval(effect),*[number(dev['variant_metrics'][key][v]['accuracy']) for v in ('base','endpoint','sham')]])
    add('A_development.tex',['Arm','Labels',r'Wrong-majority effect [95\% image CI]','Base acc.','Endpoint acc.','Sham acc.'],ad)
    bd=[]
    for arm in ARMS:
        for condition in CONDS:
            row=e.reports['rule_development']['findings']['condition_means'][arm+':'+condition]
            bd.append([ARM_NAMES[arm],COND_NAMES[condition],number(row['accuracy']['mean']),number(row['both_rules_correct']['mean']),number(row['left_mean']),number(row['right_mean']),number(row['unparseable_rate'])])
    add('B_development.tex',['Arm','Condition','Accuracy','Joint','Left','Right','Invalid'],bd)
    numerical={'findings':{name:r['findings'] for name,r in e.reports.items()},'exact_A_primary':e.guard,'unrounded_criteria':e.decisions,
               'A_moderation_original_runner':{k:v for k,v in e.raw_a['shortcut_confirmatory']['summary'].items() if k.startswith('moderation:')},
               'display':{'units':'decimal proportion or absolute proportion difference','decimals':3,'rounding':'Decimal(str(value)), ROUND_HALF_UP','decisions_use_rounded_display':False}}
    files['numerical_results.json']=encoded(numerical)
    files['frozen_protocol.txt']=(e.root/PROTOCOL).read_bytes()
    files['numerical_correction.txt']=(
        'Prospective exact-arithmetic A supplement\n\n'
        'A synthetic zero-bound rounding issue was documented before A confirmation. The supplement uses the same primary estimand, samples, 10,000 Random(90210) resamples, type-7 quantiles and mean>=1/10 plus lower>0 criteria. Integer sums and exact rational interpolation change only numerical evaluation. Original inference sources, raw results and audits remain preserved.\n'
        'Both development and confirmation original/exact decisions agree in this campaign. Confirmation:13/40 [21/80,31/80], criterion met. The primary is an image-conditional position effect; it is not an accuracy-loss estimate or a unique copying mechanism.\n'
        'Immutable amendment SHA256:'+AMENDMENT_SHA+'\nExact guard source SHA256:'+GUARD_SHA+'\nSupplement SHA256:'+SUPPLEMENT_SHA+'\n').encode()
    readme='''Follow-up manuscript tables — frozen 2026-10-10 campaign

These tabular-only LaTeX fragments require booktabs; the manuscript owns table
floats, captions, labels and blue revision formatting. Input with \\input{...}
inside the intended blue scope. Scale/wrap long tables in the manuscript.

Every displayed number is a decimal proportion (or absolute difference), not a
percentage. Values use three decimals and half-up rounding. All decision flags
are calculated and checked at full precision before formatting. Exact A
fractions and every unrounded audited estimate are in numerical_results.json.
"Not met" means the full declared positive criterion was not met; it is not
proof of no effect, equivalence or complete failure on every component.

A: A_compact/A_complete contain all four confirmation arm/label cells and all
three variants. Only frozen/natural wrong-majority endpoint-minus-sham is the
formal primary. The other effects, accuracy changes, moderation and seed
sensitivities are secondary/pointwise. The exact supplement supplies the primary
interval; original independent-auditor NumPy intervals supply secondary rows.
The two moderation rows reproduce the original runner's saved Python-Random
intervals and are secondary. First/last/majority/nearest are prediction matches,
not identified circuits. Accuracy change is not the wrong-majority effect.

B: B_primary reports the adapter-by-instruction diagnostic within nonce labels,
not the complete instruction-by-vocabulary interaction. B_compact/B_complete
show the four prespecified episodic condition criteria using98.75% df=2 seed-t
intervals; gain>=.10, both accuracy/joint lower bounds>0 and both side effects>0
are conjunctive. B_conditions includes all16 arm/condition means:95% whole-image
intervals conditional on fixed seeds, not the fitting-seed claim intervals.
Joint success is both selectors correct for a query/seed/K. Invalid outputs stay
in all denominators. Means average both selectors and K8/16. B_controls uses
pointwise95% seed intervals for natural/all-layer comparisons, solely descriptive;
these arms cannot replace a failed primary. All-layer has a larger budget.

C: C_gate separates development80 images from confirmation400 images. C_curves
contains all16 arm/K rows at95% image intervals. Adapter G/R/retention/remaining
intervals are97.5%, for two adapter claims with conjunctive requirements.
C_complete shows individual components; passing a remaining-drop component
alone does not establish removal. Repair requires confirmed D>=.10 and lower>0,
G>=.05 and lower>0, R lower>0 and retention lower>-.03. Removal additionally
requires remaining-drop upper<.10. Confirmation D=0.093166... misses the .10 mean
margin despite a positive lower bound; neither adapter meets the full repair
criterion. The support pool has18 images and the fresh image frame is finite.

Development A/B files are kept separate; their one seed does not support a
fitting-seed interval. A/B use80 shared confirmatory images; C uses400. Repeated
answers and support/checkpoint seeds are not extra independent images. No new
training, model inference, raw-output selection or inferential test is introduced.

Provenance records exact queue, six audit/raw, protocol, source, report and guard
hashes and every output hash. This exporter checks the already completed audits
and raw identities and reproduces exact A arithmetic and decision logic; it does
not rerun the full inference artifact auditor, model/checkpoint verification or
model inference. Those were performed before the frozen report was issued.
No raw images, checkpoints or large datasets are included in this small package.

Reproduce on Mac (from project root; requires locally retained data):
python scripts/export_followup_tables.py --project-root . --output-dir paper/followup_results --check
For a new output directory, omit --check. Unequal existing files are rejected.
'''
    readme+='\nFragments and columns:\n'+'\n'.join(name+': '+' | '.join(headers) for name,headers in columns.items())+'\n'
    files['README.txt']=readme.encode()
    provenance={'schema_version':1,'campaign':'followup_20261010','exporter_source_sha256':sha(Path(__file__)),
                'inputs_sha256':e.pins,'tables':{name:{'columns':columns[name],'rows':rows_count[name]} for name in columns},
                'output_sha256':{name:hashlib.sha256(data).hexdigest() for name,data in files.items()},
                'decisions_before_display_rounding':True,'exact_A_supplement_recomputed':True,
                'full_inference_audits_repeated':False,'model_inference':False,'generation_records':34304,
                'portable_paths':True,'frozen_sources_raw_audits_modified':False}
    files['provenance.json']=encoded(provenance)
    return files

def write(files,directory,check=False):
    directory=Path(directory)
    if check:
        require(directory.is_dir(),'output directory missing')
        for name,data in files.items():require((directory/name).is_file() and (directory/name).read_bytes()==data,'export differs: '+name)
        return
    # Preflight every existing target before creating any file. Existing equal
    # content is safe; unequal evidence/table bytes require a fresh directory.
    for name,data in files.items():
        p=directory/name;require(not p.exists() or p.read_bytes()==data,'preserve existing output; choose a new directory: '+name)
    directory.mkdir(parents=True,exist_ok=True)
    created=[]
    try:
        for name,data in files.items():
            p=directory/name
            if p.exists():continue
            with p.open('xb') as f:f.write(data)
            created.append(p)
    except BaseException:
        for p in created:p.unlink()
        raise

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--check',action='store_true');a=p.parse_args()
    e=Evidence(a.project_root);files=build(e);e.recheck();write(files,a.output_dir,a.check)
    print(json.dumps({'status':'checked' if a.check else 'exported','outputs':len(files),'tables':sum(x.endswith('.tex') for x in files),'records':34304,'output_dir':str(a.output_dir),'model_inference':False}))
if __name__=='__main__':main()
