import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
from copy import deepcopy
from scripts.analyze_followup_experiments import independent_prediction,seed_interval,image_interval
from scripts import run_followup_experiments as queue

class StatisticalTests(unittest.TestCase):
    def test_ambiguous_echo_does_not_receive_first_option_credit(self):
        self.assertIsNone(independent_prediction('circle, square, triangle, cross',['circle','square','triangle','cross']))
        self.assertIsNone(independent_prediction('squares',['square','circle']))
        self.assertEqual(independent_prediction('Answer: square.',['square','circle']),0)

    def test_exact_three_seed_t_and_correction(self):
        ordinary=seed_interval([0,.1,.2]);corrected=seed_interval([0,.1,.2],.9875)
        self.assertAlmostEqual(ordinary['critical'],4.3026527297,places=8)
        self.assertLess(corrected['low'],ordinary['low'])
        self.assertIsNone(seed_interval([.1])['low'])
        with self.assertRaises(ValueError):seed_interval([0,.1])

    def test_repeating_support_seeds_does_not_increase_image_sample_size(self):
        one=image_interval([[0,1,0,1]],draws=1000);repeated=image_interval([[0,1,0,1]]*3,draws=1000)
        self.assertEqual(one['n_images'],4)
        self.assertEqual((one['mean'],one['low'],one['high']),(repeated['mean'],repeated['low'],repeated['high']))

    def test_fixed_budget_and_gate(self):
        jobs=queue.jobs()
        self.assertEqual(sum(j['planned_generations'] for j in jobs if not j['conditional']),19904)
        self.assertEqual([j['name'] for j in jobs if j['conditional']],['vqa_confirmatory'])
        self.assertEqual(sum(j['planned_generations'] for j in jobs),34304)
        self.assertTrue(all(j['output'].startswith('data/followup_20261010/') for j in jobs))
        self.assertIn('--development-result',jobs[-1]['command'])

    def test_paper_change_stops_continuation(self):
        state={'source_sha256':{},'paper_sha256':{'paper/Main.tex':'old'},'data_sha256':{},'checkpoints':{}}
        with patch.object(queue,'source_manifest',return_value={}),patch.object(queue,'paper_manifest',return_value={'paper/Main.tex':'new'}):
            with self.assertRaisesRegex(RuntimeError,'paper changed'):queue.check_frozen(state)

    def test_changed_audit_report_is_rejected_before_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'raw.json').write_text('{}');(root/'audit.json').write_text('{"gate_pass":true}')
            job={'output':'raw.json','audit':{'path':'audit.json','artifact_sha256':queue.digest(root/'raw.json'),
                 'sha256':queue.digest(root/'audit.json')}}
            queue.verify_completed(job,root)
            (root/'audit.json').write_text('{"gate_pass":false}')
            with self.assertRaisesRegex(RuntimeError,'audit report changed'):queue.verify_completed(job,root)

    def test_gate_is_recomputed_from_complete_raw_evidence(self):
        job={'name':'vqa_development','status':'complete','output':'raw.json',
             'audit':{'artifact_sha256':'rawhash','sha256':'reporthash'}}
        with patch.object(queue,'verify_completed'),patch('scripts.analyze_followup_experiments.audit',
            return_value={'artifact_sha256':'rawhash','findings':{'gate_pass':True}}) as recompute:
            self.assertEqual(queue.checked_gate({'stages':[job]},Path('/tmp')),(True,'reporthash'))
            self.assertEqual(recompute.call_count,1)

    def test_resume_rejects_changed_stage_command_and_false_completion(self):
        plan=queue.jobs();state={'stages':deepcopy(plan)}
        queue.verify_stage_plan(state,plan)
        with self.assertRaisesRegex(RuntimeError,'unfinished'):queue.verify_stage_plan(state,plan,True)
        state['stages'][0]['command'].append('--unexpected')
        with self.assertRaisesRegex(RuntimeError,'fixed plan'):queue.verify_stage_plan(state,plan)

if __name__=='__main__':unittest.main()
