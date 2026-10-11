"""Independent numerical-boundary and raw-evidence regressions; no model calls."""
from copy import deepcopy
from fractions import Fraction
import math
import random
import statistics
import json
import shutil
import tempfile
from types import SimpleNamespace
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from incontext_collapse.paired_bootstrap import integer_image_interval, exact_estimate, type7_integer_quantile
from incontext_collapse.shortcut_controls import summarize, validate_connector_source
from incontext_collapse.vqa import adapt, validate_evaluation_request, vqa_score, agg_kcurve
from incontext_collapse.vqa_repair import evaluate_cell, validate_evidence_cell
from incontext_collapse.vqa_train import train_connector
from incontext_collapse.vqa_diverse import paired_image_interval
from incontext_collapse import rule_factorial as rf
from scripts.audit_revision_artifacts import audit_cell, normalize_vqa_answer, official_vqa_score
from scripts.summarize_cama import summarize as summarize_cama
from tests.test_cama_summary import artifact as cama_artifact


class ScientificAuditTests(unittest.TestCase):
    def test_A_real_zero_boundary_fixture_uses_the_same_random_indices(self):
        # Archived synthetic edge: 80 images × three integer contrasts.
        columns = [3]*29 + [-1]*48 + [-2]*3
        values = [x/3 for x in columns]
        rng = random.Random(90210)
        old = sorted(statistics.fmean(rng.choices(values, k=80)) for _ in range(10000))
        self.assertGreater(float(np.quantile(old,.025)),0)
        result = integer_image_interval(columns,3)
        self.assertEqual(exact_estimate(result,'mean'),Fraction(11,80))
        self.assertEqual(exact_estimate(result,'low'),0)
        self.assertFalse(exact_estimate(result,'mean')>=Fraction(1,10) and exact_estimate(result,'low')>0)
        matrix=[[1]*29+[-1]*51,[1]*29+[0]*48+[-1]*3,[1]*29+[0]*51]
        manifest={'config':{'n':80,'seeds':[0,1,2],'partition':'confirmatory'},'query_ids':list(map(str,range(80)))}
        cells={}
        for seed in range(3):
            for arm in ('frozen','connector'):
                for condition in ('standard','remap'):
                    cells[f'{seed}:{arm}:{condition}']=[{'query_id':str(i),'primary_contrast':value,
                        'metrics':{v:{'accuracy':0} for v in ('base','endpoint','sham')},
                        'nearest':{'collides_with':[]}} for i,value in enumerate(matrix[seed])]
        actual=summarize({'cells':cells},manifest)
        self.assertFalse(actual['prespecified_primary']['meets_effect_and_interval_criteria'])
        self.assertEqual(actual['frozen:standard']['primary_contrast']['ci95_low'],0)

    def test_exact_type7_does_not_promote_a_zero_quantile(self):
        # A valid integer-sum quantile boundary; the old floating tail drifts upward.
        ordered=[-79]*125+[1]*9875
        self.assertEqual(type7_integer_quantile(ordered,Fraction(1,80),12000),0)
        self.assertGreater(np.quantile(np.array(ordered)/12000,(1-.975)/2),0)
        self.assertEqual(type7_integer_quantile([-80,0],Fraction(77,80),100),Fraction(-3,100))

    def test_numpy_stream_and_exact_interpolation_match_independent_oracle(self):
        values=[-2,0,1,4,7]
        result=integer_image_interval(values,30,confidence=.975,draws=101,seed=17,generator='numpy')
        rng=np.random.default_rng(17)
        totals=sorted(sum(values[int(i)] for i in rng.integers(0,5,5)) for _ in range(101))
        position=Fraction(1,80)*100;left=position.numerator//position.denominator
        expected=(Fraction(totals[left])+(position-left)*(totals[left+1]-totals[left]))/150
        self.assertEqual(exact_estimate(result,'low'),expected)
        self.assertEqual(exact_estimate(result,'mean'),Fraction(sum(values),150))
        for args in (([],1),([1.0],1),([True],1),([1],0)):
            with self.assertRaises(ValueError):integer_image_interval(*args)
        with self.assertRaises(ValueError):integer_image_interval([1],1,confidence=1)
        with self.assertRaises(ValueError):integer_image_interval([2**63],1,generator='numpy')

    def test_independent_vqa_aliases_receive_leave_one_out_credit(self):
        for source,target in [('hed',"he'd"),('couldve',"could've"),('whyll',"why'll"),
                              ('somebodyll',"somebody'll"),("youd've","you'd've")]:
            self.assertEqual(normalize_vqa_answer(source),target)
            answers=[target]*3+['unmatched']*7
            self.assertAlmostEqual(official_vqa_score(source,answers),.9)
            self.assertAlmostEqual(vqa_score(source,answers),.9)

    def test_generation_score_and_shortcut_flags_must_follow_raw_answer(self):
        row={'query_index':0,'query_id':'q','gold':0,'prediction':0,'correct':True,
             'label_space':['amber','birch'],'raw_output':'birch','generation_prediction':1,
             'parse_status':'exact','unparseable':False,'demo_label_classes':[1,1,0],
             'copies_first':True,'copies_last':False,'copies_any_demo':True,'copies_majority':True}
        errors=[];audit_cell('cell',{'score_mode':'generation','records':[row],'summary':{}},errors,[])
        self.assertTrue(any('scored prediction' in x for x in errors))
        self.assertTrue(any('copies_majority' in x for x in errors))

    def test_cama_rejects_outcomes_forged_independently_of_raw_generation(self):
        good=cama_artifact();self.assertEqual(summarize_cama(good)['comparisons'][0]['wins'],1)
        altered=deepcopy(good);altered['cells'][0]['records'][0]['correct']=True
        with self.assertRaisesRegex(ValueError,'raw generation'):summarize_cama(altered)
        altered=deepcopy(good);altered['cells'][0]['records'][0]['shuffle_map']={'0':1,'1':0}
        with self.assertRaisesRegex(ValueError,'shuffle_map'):summarize_cama(altered)
        altered=deepcopy(good);altered['run_config']['seeds']=[0,0]
        with self.assertRaisesRegex(ValueError,'duplicate'):summarize_cama(altered)

    def test_connector_contract_rejects_all_layer_checkpoint_mislabeled_connector(self):
        source={'model':'model','model_revision':'revision','objective':'episode_remap',
                'task':'shapes4','format':'single','regions':'connector','components':'both',
                'rank':8,'steps':150,'dtype':'bfloat16','seeds':[0,1,2]}
        validate_connector_source(source,'model','revision',1)
        for field,value in [('regions','all'),('rank',16),('steps',250),('seeds',[0])]:
            changed={**source,field:value}
            with self.assertRaises(ValueError):validate_connector_source(changed,'model','revision',1)

    def test_legacy_vqa_does_not_silently_reduce_requested_samples_or_shots(self):
        for ks,n,seeds in [([3],1,[0]),([1],3,[0]),([1,1],1,[0]),([0],1,[0,0])]:
            with self.assertRaises(ValueError):validate_evaluation_request([1,2],[3,4],ks,n,seeds)
        with self.assertRaises(ValueError):train_connector(None,None,[1,2],['target'],0,0,8,1,112)
        image=Image.new('RGB',(2,2),'red')
        self.assertEqual(adapt('gqa',{'image':image,'question':'q','answers':'red'})[2],['red'])
        self.assertEqual(adapt('gqa',{'image':image,'question':'q','answers':{'answer':['red']}})[2],['red'])
        self.assertEqual(adapt('gqa',{'image':image,'question':'q','multiple_choice_answer':0})[2],['0'])
        with patch('incontext_collapse.vqa.eval_kcurve',return_value={0:.5}):
            self.assertIsNone(agg_kcurve(None,None,[1],[2],[0],1,[0])['ci']['0'])

    def test_resumed_vqa_records_are_rescored_and_pairing_replayed(self):
        pool=[(Image.new('RGB',(2,2),'red'),'support',['yes']*10)]
        test=[(Image.new('RGB',(2,2),'blue'),'query',['yes']*10)]
        with patch('incontext_collapse.vqa_repair._generate',return_value='yes'):
            good=evaluate_cell(None,None,pool,test,1,1,112,0)
        validate_evidence_cell(good,pool,test,1,1,0)
        for field,value in [('score',0),('demo_ids',['other']),('question','changed')]:
            altered=deepcopy(good);altered['records'][0][field]=value
            with self.assertRaises(ValueError):validate_evidence_cell(altered,pool,test,1,1,0)
        duplicate=deepcopy(good);duplicate['records'].append(deepcopy(duplicate['records'][0]))
        with self.assertRaisesRegex(ValueError,'duplicate'):paired_image_interval([duplicate],[duplicate],draws=10)

    def test_rule_inventory_relocates_explicitly_without_reading_old_root(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(rf,'_source_shapes_hashes',return_value=[]):
            top=Path(tmp);old=top/'old'/'data';new=top/'mirror'/'data';old.mkdir(parents=True)
            (old/'cached.json').write_text(json.dumps({'image_id':'a'*64}))
            (old/'followup_20261010'/'rule_images').mkdir(parents=True)
            inventory=top/'inventory.json'
            rf.build_exclusion_inventory(inventory,old,[old/'followup_20261010',old/'followup_20261010'/'rule_images'])
            original=inventory.read_bytes();shutil.copytree(old,new);shutil.rmtree(old.parent)
            loaded=rf.load_exclusion_inventory(inventory,prior_root=new)
            self.assertEqual(loaded['excluded_identifiers'],['a'*64])
            self.assertEqual(rf.build_exclusion_inventory(inventory,new,[new/'followup_20261010',top/'new_rule_images']),inventory.resolve())
            self.assertEqual(inventory.read_bytes(),original)
            with self.assertRaisesRegex(ValueError,'configuration differs'):
                rf.build_exclusion_inventory(inventory,new,[])
            (new/'cached.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'checksum'):
                rf.load_exclusion_inventory(inventory,prior_root=new)

    def test_rule_inventory_rejects_relocated_missing_escape_and_duplicate_files(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(rf,'_source_shapes_hashes',return_value=[]):
            top=Path(tmp);old=top/'old';new=top/'new';old.mkdir();new.mkdir()
            (old/'cached.json').write_text('{}');inventory=top/'inventory.json'
            rf.build_exclusion_inventory(inventory,old)
            original=json.loads(inventory.read_text())
            with self.assertRaises(FileNotFoundError):rf.load_exclusion_inventory(inventory,prior_root=new)
            (new/'cached.json').symlink_to(old/'cached.json')
            with self.assertRaisesRegex(ValueError,'escaping'):rf.load_exclusion_inventory(inventory,prior_root=new)
            (new/'cached.json').unlink();shutil.copyfile(old/'cached.json',new/'cached.json')
            for value in (str(top/'outside.json'),str(old/'..'/'outside.json')):
                changed=deepcopy(original);changed['files'][0]['path']=value
                inventory.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError,'escapes|unsafe'):rf.load_exclusion_inventory(inventory,prior_root=new)
            changed=deepcopy(original);changed['files']*=2;inventory.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'duplicate'):rf.load_exclusion_inventory(inventory,prior_root=new)

    def test_all_direct_generators_override_inherited_beams_and_decode_only_continuation(self):
        from incontext_collapse.vqa import _generate
        from incontext_collapse.shortcut_controls import generate
        from incontext_collapse.vqa_confirmatory import generate_with_metadata
        class Inputs(dict):
            def to(self,device):return self
        calls=[];decoded=[]
        def fake_generate(**kwargs):
            calls.append(kwargs)
            return torch.tensor([[10,11,12,7,2]])
        def decode(tokens,**kwargs):
            decoded.append(tokens.tolist());return ['yes']
        model=SimpleNamespace(device='cpu',generate=fake_generate,
            generation_config=SimpleNamespace(num_beams=5,num_return_sequences=3,eos_token_id=2))
        processor=SimpleNamespace(apply_chat_template=lambda *a,**kw:Inputs(input_ids=torch.tensor([[10,11,12]])),batch_decode=decode)
        self.assertEqual(_generate(model,processor,[]),'yes')
        self.assertEqual(generate(model,processor,[]),'yes')
        prediction,metadata=generate_with_metadata(model,processor,[])
        self.assertEqual(prediction,'yes');self.assertTrue(metadata['ended_with_eos'])
        self.assertEqual(decoded,[[[7,2]]]*3)
        self.assertEqual([c['max_new_tokens'] for c in calls],[10,6,10])
        for call in calls:
            self.assertFalse(call['do_sample']);self.assertEqual(call['num_beams'],1)
            self.assertEqual(call['num_return_sequences'],1)
        self.assertTrue(calls[1]['use_cache'])
        self.assertNotIn('use_cache',calls[0]);self.assertNotIn('use_cache',calls[2])


if __name__=='__main__':unittest.main()
