"""Canonical dispatcher regression tests; no models, datasets, or external writes."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('reproduce_dispatch_test',
    Path(__file__).resolve().parents[1] / 'scripts/reproduce.py')
reproduce = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reproduce)


class DispatchTests(unittest.TestCase):
    def test_extra_families_check_grid_then_independent_raw_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'raw.json').write_text('{}')
            for family in ('paired','panel','adapter_eval'):
                study={'id':family,'audit_family':family,'kind':'evaluation','output':'raw.json',
                       'arguments':['--cells-dir','cells']}
                with patch.object(reproduce,'run_command') as command:
                    reproduce.audit_result(root,study,root)
                calls=[call.args[0] for call in command.call_args_list]
                self.assertEqual(calls[0][2:4],['incontext_collapse.queue_check',family.replace('_','-')])
                self.assertEqual(calls[1][1],'scripts/audit_revision_artifacts.py')
                self.assertEqual(calls[1][2],'cells' if family=='panel' else 'raw.json')

    def test_retraining_only_dependency_runs_before_confirmation(self):
        studies = [
            {'id':'train','kind':'training','dependencies':[]},
            {'id':'earlier80','kind':'evaluation','dependencies':['train']},
            {'id':'confirmation','kind':'evaluation','dependencies':['train'],
             'retrain_dependencies':['earlier80']}]
        manifest={'studies':studies}
        self.assertEqual([s['id'] for s in reproduce.select_studies(manifest,'confirmation')],['confirmation'])
        self.assertEqual([s['id'] for s in reproduce.select_studies(manifest,'confirmation',True)],
                         ['train','earlier80','confirmation'])

    def test_rule_prior_mirror_preserves_relative_files_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'source';workspace=Path(directory)/'new'
            (root/'data/followup_20261010/rule_images').mkdir(parents=True)
            original=root/'data/training.json';original.write_text('old training')
            inventory={'config':{'prior_root':'/original/project/data'},'files':[
                {'path':'/original/project/data/training.json','sha256':reproduce.digest(original)}]}
            (root/'data/followup_20261010/rule_images/prior_image_inventory.json').write_text(json.dumps(inventory))
            reproduce.preserve_rule_prior(root,workspace)
            mirror=workspace/'reproduction_prior/data/training.json'
            self.assertEqual(mirror.read_bytes(),original.read_bytes())
            original.write_text('new training')
            self.assertEqual(mirror.read_text(),'old training')

    def test_gate_checks_bytes_and_requires_identical_fresh_audit(self):
        for passed in (True,False):
            with tempfile.TemporaryDirectory() as directory:
                workspace=Path(directory);logs=workspace/'logs';logs.mkdir()
                study={'id':'followup.vqa_development','output':'dev.json'}
                raw=workspace/'dev.json';raw.write_text('{}')
                report=logs/(study['id']+'.audit.json')
                report.write_text(json.dumps({'status':'complete','artifact_sha256':reproduce.digest(raw),
                                              'findings':{'gate_pass':passed}}))
                completed={'status':'complete','output_sha256':reproduce.digest(raw),
                           'audit_report_sha256':reproduce.digest(report)}
                def reaudit(workspace,study,target):
                    (target/report.name).write_bytes(report.read_bytes())
                    return reproduce.digest(raw)
                with patch.object(reproduce,'audit_result',side_effect=reaudit):
                    gate=reproduce.checked_vqa_gate(workspace,study,completed,logs)
                self.assertEqual(gate['gate_pass'],passed)
                self.assertEqual(gate['audit_sha256'],gate['reaudit_sha256'])
                raw.write_text('{"tampered":true}')
                with self.assertRaisesRegex(ValueError,'changed'):
                    reproduce.checked_vqa_gate(workspace,study,completed,logs)

    def test_device_override_only_for_supported_cli(self):
        study = {'id': 'cama', 'module': 'incontext_collapse.cama',
                 'arguments': ['--device', 'mps', '--out', 'new.json']}
        self.assertNotIn('--device', reproduce.commands_for([study])[0])
        self.assertIn('cuda', reproduce.commands_for([study], 'cuda')[0])
        unsupported = {**study, 'module': 'incontext_collapse.circuit', 'arguments': []}
        with self.assertRaisesRegex(ValueError, 'no device override'):
            reproduce.commands_for([unsupported], 'cpu')
        self.assertEqual(reproduce.commands_for([unsupported])[0][-1], 'incontext_collapse.circuit')

    def test_workspace_cannot_recurse_into_source_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, 'outside the source'):
                reproduce.prepare_workspace(root, {}, root / 'reproducibility/new/workspace')
            self.assertFalse((root / 'reproducibility').exists())

    def test_revision_evaluation_uses_complete_grid_reducer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'raw.json').write_text('{}')
            study = {'id': 'revision.example', 'output': 'raw.json',
                     'kind': 'evaluation', 'audit_family': 'revision'}
            with patch.object(reproduce, 'run_command') as command:
                reproduce.audit_result(root, study, root)
            self.assertEqual(command.call_args.args[0][1], 'scripts/summarize_revision_followup.py')

    def test_followup_requires_planned_records_and_exact_raw_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); raw = root / 'raw.json'; raw.write_text('{}')
            study = {'id': 'followup.shortcut_confirmatory', 'output': 'raw.json',
                     'kind': 'evaluation', 'audit_family': 'followup', 'planned_generations': 12}
            report = root / (study['id'] + '.audit.json')
            good = {'status': 'complete', 'artifact_sha256': reproduce.digest(raw),
                    'findings': {'generation_records': 12}}
            with patch.object(reproduce, 'run_command'):
                for value in [good, {**good, 'status': 'partial'},
                              {**good, 'artifact_sha256': 'changed'},
                              {**good, 'findings': {'generation_records': 11}}]:
                    report.write_text(json.dumps(value))
                    if value == good:
                        self.assertEqual(reproduce.audit_result(root, study, root), reproduce.digest(raw))
                    else:
                        with self.assertRaisesRegex(ValueError, 'planned generation count'):
                            reproduce.audit_result(root, study, root)

    def test_training_seed_completeness_precedes_generic_audit(self):
        study = {'module': 'incontext_collapse.dara', 'arguments': ['--seeds', '0,1']}
        good = {'run_config': {'seeds': [0, 1], 'ks': [4], 'n': 2},
                'seeds_done': [0, 1], 'training_runs': [{'torch_seed': 0}, {'torch_seed': 1}]}
        reproduce.validate_training_complete(good, study)
        with self.assertRaisesRegex(ValueError, 'incomplete or duplicate'):
            reproduce.validate_training_complete({**good, 'seeds_done': [0]}, study)
        with self.assertRaisesRegex(ValueError, 'incomplete or duplicate'):
            reproduce.validate_training_complete({**good, 'training_runs': [{'torch_seed': 0}]}, study)

    def test_vqa_training_requires_both_arms_complete(self):
        study = {'module': 'incontext_collapse.vqa_train', 'arguments': ['--seeds', '0']}
        cell = {'4': {'records': [{}, {}]}}
        good = {'run_config': {'seeds': [0], 'ks': [4], 'n': 2}, 'seeds_done': [0],
                'training_runs': {'0': {'torch_seed': 0}},
                'evidence': {'0': {'baseline': cell, 'repaired': cell}}}
        reproduce.validate_training_complete(good, study)
        with self.assertRaisesRegex(ValueError, 'evaluation arms'):
            reproduce.validate_training_complete({**good, 'evidence': {'0': {'baseline': cell}}}, study)


if __name__ == '__main__':
    unittest.main()
