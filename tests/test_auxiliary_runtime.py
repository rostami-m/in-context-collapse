"""CPU/mocked regression tests for future-run integrity; no model/API calls."""
import contextlib
import hashlib
import io
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from incontext_collapse import panel, vaccine, continual, dara, cama, figures, frontier, inject, models
from incontext_collapse.core import TASKS, build_messages
from incontext_collapse.evaluation import evaluate_classification_cell
from incontext_collapse.protocol import ProtocolConfig
from runner import storage


class PanelIntegrityTests(unittest.TestCase):
    def test_legacy_loader_name_preserves_exact_model_or_raises(self):
        identity='meta-llama/Llama-3.2-11B-Vision-Instruct'
        with patch.object(models,'load_model',return_value=('model','processor')) as load:
            self.assertEqual(models.load_model_with_fallback(identity,'cpu'),('model','processor',identity))
            load.assert_called_once_with(identity,device='cpu')
        with patch.object(models,'load_model',side_effect=RuntimeError('gated')) as load:
            with self.assertRaisesRegex(RuntimeError,'gated'): models.load_model_with_fallback(identity)
            load.assert_called_once_with(identity,device=None)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        names, pool, test = TASKS['shapes4'](4, 4)
        self.config = {'n_test': 4, 'score_mode': 'generation', 'protocol': ProtocolConfig().metadata()}
        self.cell = evaluate_classification_cell(None, None, names, pool, test, 'standard', 0, 4, 112, 0,
                    build_messages, lambda *_: 'circle', protocol=ProtocolConfig(), score_mode='generation')
        self.cell.update(model='org/model', task='shapes4', cond='standard', K=0, seed=0)
        self.plan = [('org/model', 'shapes4', 'standard', 0, 0)]
        self.filename = lambda *identity: 'cell.json'

    def write(self):
        (self.path/'cell.json').write_text(json.dumps(self.cell))

    def check(self, complete=True):
        return panel.load_planned_cells(self.path,self.plan,self.filename,self.config,require_complete=complete)

    def test_only_complete_matching_cells_aggregate(self):
        self.assertEqual(self.check(False), [])
        with self.assertRaisesRegex(ValueError,'missing'): self.check()
        self.write(); self.assertEqual(len(self.check()),1)
        (self.path/'foreign.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'extra'): self.check()

    def test_wrong_identity_protocol_count_and_summary_rejected(self):
        for mutate in [lambda: self.cell.update(model='other/model'),
                       lambda: self.cell.update(k=4),
                       lambda: self.cell.update(score_mode='both'),
                       lambda: self.cell['records'].pop(),
                       lambda: self.cell['summary'].update(accuracy=.999)]:
            original=json.loads(json.dumps(self.cell));mutate();self.write()
            with self.assertRaises(ValueError): self.check()
            self.cell=original

    def test_exact_loader_failure_never_substitutes_or_writes_summary(self):
        output=self.path/'result.json'; cells=self.path/'cells'
        argv=['panel','--models','org/model','--tasks','shapes4','--conditions','standard',
              '--k-grid','0','--seeds','1','--n-test','4','--out',str(output),'--cells-dir',str(cells)]
        with patch.object(sys,'argv',argv),patch.object(panel,'pick_device',return_value='cpu'),\
             patch.object(panel,'load_model',side_effect=RuntimeError('unavailable')) as loader,\
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'unavailable'): panel.main()
        loader.assert_called_once_with('org/model',device='cpu');self.assertFalse(output.exists())


class SnapshotAndSupervisionTests(unittest.TestCase):
    def test_fresh_dara_cli_evaluates_final_and_roundtrips_selected(self):
        attention=SimpleNamespace(dara_factors=torch.nn.Parameter(torch.ones(2,8)))
        training={'torch_seed':0,'trainable_parameters':16,'factors':[[1.1]*8]*2,
                  'final_factors':[[1.2]*8]*2,'best_step':1}
        result={'seeds_done':[],'seed_curves':[],'training_runs':[],'evidence':[]}
        evaluated=[]
        def evaluate(*args,**kwargs):
            evaluated.append(float(attention.dara_factors[0,0].detach()))
            return {4:.5},{4:{'records':[],'summary':{}}}
        argv=['dara','--model','tiny','--ks','4','--seeds','0','--out','/tmp/mocked_only.json']
        with patch.object(sys,'argv',argv),patch.object(dara,'model_revision',return_value='pinned'),\
             patch.object(dara,'load_result',return_value=result),patch.object(dara,'write_json_atomic'),\
             patch.object(dara.os,'makedirs'),patch.object(dara,'load_model',return_value=(Mock(),Mock())),\
             patch.object(dara,'install_dara',return_value=(attention,Mock())),\
             patch.dict(dara.TASKS,{'shapes4':lambda *_:(['a','b'],{},[])}),\
             patch.object(dara,'eval_kcurve',side_effect=evaluate),\
             patch.object(dara,'train_dara',return_value=training),contextlib.redirect_stdout(io.StringIO()):
            dara.main()
        self.assertEqual(evaluated[:2],[1.,1.])
        self.assertTrue(all(abs(value-1.2)<1e-6 for value in evaluated[2:]))
        source={'run_config':{'method':'DARA compatibility port','implementation_version':3,
                 'model':'tiny','model_revision':'pinned','task':'shapes4','protocol':{}},
                'training_runs':result['training_runs']}
        with patch.object(dara.Path,'read_bytes',return_value=json.dumps(source).encode()),\
             patch.object(dara,'model_revision',return_value='pinned'):
            _,selected,_=dara.load_factor_source('mocked.json',model_id='tiny',task='shapes4',seeds=[0],snapshot='selected')
            _,final,_=dara.load_factor_source('mocked.json',model_id='tiny',task='shapes4',seeds=[0],snapshot='final')
        self.assertAlmostEqual(selected[0]['factors'][0][0],1.1)
        self.assertAlmostEqual(final[0]['factors'][0][0],1.2)
        self.assertAlmostEqual(final[0]['source_selected_factors'][0][0],1.1)

    def test_fresh_dara_respects_final_and_selected_without_losing_either(self):
        training={'factors':[[1.1,1.2]],'final_factors':[[2.1,2.2]],'best_step':1}
        original=json.loads(json.dumps(training))
        for choice,expected in [('final',training['final_factors']),('selected',training['factors'])]:
            attention=SimpleNamespace(dara_factors=torch.nn.Parameter(torch.ones(1,2)))
            result=dara.select_training_snapshot(attention,training,choice)
            self.assertTrue(torch.allclose(attention.dara_factors,torch.tensor(expected)))
            self.assertEqual(result['evaluation_snapshot'],choice)
            self.assertEqual(result['source_selected_factors'],original['factors'])
        self.assertEqual(training,original)
        with self.assertRaises(ValueError): dara.select_training_snapshot(attention,training,'best')

    def test_cama_rejects_nonfinite_weights_and_epsilon(self):
        for name in ['query_weight','epsilon']:
            for value in [float('nan'),float('inf')]:
                with self.assertRaises(ValueError): cama.CamaConfig(**{name:value})

    def test_cl_requires_distinct_single_token_letters(self):
        tokenizer=Mock();proc=SimpleNamespace(tokenizer=tokenizer)
        tokenizer.encode.side_effect=[[1],[2],[3],[4]]
        self.assertEqual(continual.label_token_ids(proc),dict(zip('ABCD',[1,2,3,4])))
        for values in [[[1,2],[2],[3],[4]],[[1],[1],[3],[4]],[[],[2],[3],[4]]]:
            tokenizer.encode.side_effect=values
            with self.assertRaises(ValueError): continual.label_token_ids(proc)

    def test_cl_rejects_supervising_prompt_or_eos_instead_of_answer(self):
        class Inputs(dict):
            def to(self,*_): return self
        model=Mock(device='cpu');model.parameters.return_value=[torch.nn.Parameter(torch.ones(1))]
        proc=Mock();proc.return_value=Inputs(input_ids=torch.tensor([[9,8,7]]))
        proc.apply_chat_template.return_value='prompt'
        with self.assertRaisesRegex(ValueError,'final single token'):
            continual.train_task(model,proc,{0:[object()]},1,{'A':1},1)
        model.assert_not_called()


class RetiredLaunchAndStorageTests(unittest.TestCase):
    def test_storage_check_preserves_existing_probe_and_removes_own(self):
        with tempfile.TemporaryDirectory() as directory:
            original=Path(directory)/'.storage_check'; original.write_text('preserve')
            with contextlib.redirect_stdout(io.StringIO()): self.assertEqual(storage.check(directory),0)
            self.assertEqual(original.read_text(),'preserve')
            self.assertEqual([p.name for p in Path(directory).iterdir()],['.storage_check'])

    def test_frontier_and_standalone_inject_stop_without_loads(self):
        for module in [frontier,inject]:
            with patch.object(sys,'argv',['old','--model','arbitrary']),contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught: module.main()
                self.assertEqual(caught.exception.code,2)

    def test_retired_shell_entries_stop_before_tools(self):
        root=Path(__file__).resolve().parents[1]
        paths=[root/'runner/launch.sh',root/'runner/slurm_array.sbatch',*sorted((root/'scripts').glob('*.sh'))]
        for path in paths:
            with self.subTest(path=path.name):
                result=subprocess.run(['/bin/bash',str(path)],text=True,capture_output=True,timeout=3)
                self.assertEqual(result.returncode,2);self.assertIn('retired',result.stderr)


class FigureOutputTests(unittest.TestCase):
    def test_wrapper_preserves_dataset_cache_when_moving_plot_caches(self):
        path=Path(__file__).resolve().parents[1]/'scripts/make_figures.py'
        spec=importlib.util.spec_from_file_location('figure_wrapper_test',path)
        wrapper=importlib.util.module_from_spec(spec);spec.loader.exec_module(wrapper)
        for original,expected in [({'XDG_CACHE_HOME':'/original/cache'},'/original/cache/huggingface'),
                                  ({'HF_HOME':'/explicit/hf'},'/explicit/hf')]:
            with patch.dict(os.environ,original,clear=True),patch.object(sys,'argv',['wrapper']),\
                 patch.object(wrapper.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as run:
                self.assertEqual(wrapper.main(),0)
                env=run.call_args.kwargs['env']
                self.assertEqual(env['HF_HOME'],expected)
                self.assertNotEqual(env['XDG_CACHE_HOME'],original.get('XDG_CACHE_HOME'))

    def test_saving_requires_explicit_output_and_never_overwrites(self):
        with patch.object(figures,'OUT',None), self.assertRaisesRegex(RuntimeError,'explicit'):
            figures.save_figure(Mock(),'figure.pdf')
        with tempfile.TemporaryDirectory() as directory,patch.object(figures,'OUT',Path(directory)):
            target=Path(directory)/'figure.pdf';target.write_bytes(b'preserve')
            fig=Mock()
            with self.assertRaises(FileExistsError): figures.save_figure(fig,'figure.pdf')
            fig.savefig.assert_not_called();self.assertEqual(target.read_bytes(),b'preserve')

    def test_no_implicit_paper_sync(self):
        with self.assertRaisesRegex(RuntimeError,'retired'): figures.sync_paper_figures()

    def test_existing_or_protected_output_rejected_before_render(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(figures,'fig_concept') as render,\
             contextlib.redirect_stderr(io.StringIO()):
            for output in [Path(directory),figures.ROOT/'paper/new-output']:
                with self.assertRaises(SystemExit): figures.main(['--output-dir',str(output)])
            render.assert_not_called()

    def test_mock_render_manifest_labels_archival_copy_and_no_paper_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'figures').mkdir();(root/'data/panel_cells').mkdir(parents=True)
            (root/'data/figdata').mkdir();(root/'figures/fig3_frontier_cliff.pdf').write_bytes(b'historical')
            (root/'paper').mkdir()
            manuscript=root/'paper/Main.tex'
            manuscript.write_text(r'\includegraphics{figures/fig3_frontier_cliff.pdf}')
            original_manuscript=manuscript.read_bytes()
            output=root/'new-output'
            functions=[n for n in vars(figures) if n.startswith('fig_') and callable(getattr(figures,n))]
            with contextlib.ExitStack() as stack:
                for name in functions: stack.enter_context(patch.object(figures,name))
                for name,value in [('ROOT',root),('FD',root/'data/figdata'),('V2',root/'data/panel_cells')]:
                    stack.enter_context(patch.object(figures,name,value))
                stack.enter_context(patch.dict('os.environ',{},clear=False))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                figures.main(['--output-dir',str(output),'--skip-qualitative'])
            manifest=json.loads((output/'figures_manifest.json').read_text())
            self.assertEqual(manifest['outputs']['fig3_frontier_cliff.pdf']['status'],'copied_historical_asset')
            self.assertFalse(manifest['paper_sync']);self.assertFalse((root/'paper/figures').exists())
            self.assertEqual(manuscript.read_bytes(),original_manuscript)
            self.assertEqual((output/'fig3_frontier_cliff.pdf').read_bytes(),b'historical')
            self.assertIsNone(figures.OUT)

    def test_missing_historical_curve_is_not_an_empty_successful_plot(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(figures,'V2',Path(directory)):
            with self.assertRaisesRegex(ValueError,'missing historical'):
                figures.vcurve('absent-model','cifar4','standard')


if __name__=='__main__': unittest.main()
