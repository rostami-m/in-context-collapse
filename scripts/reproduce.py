#!/usr/bin/env python3
"""Reproduce the acd010a paper without writing into its retained evidence.

Saved numerical analyses run from a hash-verified source snapshot. New inference
and fitting use maintained code in a fresh, isolated workspace. Neither path
overwrites the paper, its recorded results, or a previous run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'reproducibility/paper_manifest.json'
DEVICE_ARGUMENT_MODULES = {'incontext_collapse.cama', 'incontext_collapse.dara',
                           'incontext_collapse.shortcut_controls'}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def contained(root, relative):
    """Reject absolute paths, symlink escapes, and parent traversal in catalogs."""
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'unsafe relative path: {relative}')
    target = (root / path).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f'path escapes root: {relative}')
    return target


def fresh(path):
    path = Path(path).expanduser().resolve()
    if path.is_relative_to(ROOT.resolve()):
        raise ValueError('choose an output directory outside the source checkout')
    if path.exists():
        raise ValueError(f'output already exists; choose a new path: {path}')
    path.mkdir(parents=True)
    return path


def check_hashes(root, entries):
    errors = []
    for name, entry in entries.items():
        expected = entry['sha256'] if isinstance(entry, dict) else entry
        path = contained(root, name)
        if not path.is_file():
            errors.append(f'missing: {name}')
        elif digest(path) != expected:
            errors.append(f'changed: {name}')
    if errors:
        raise ValueError('frozen evidence check failed:\n' + '\n'.join(errors[:40]))
    return len(entries)


def verify_files(root, manifest):
    counts = {key: check_hashes(root, manifest[key]) for key in
              ('paper_files', 'evidence_files', 'figure_files')}
    snapshot = contained(root, manifest['snapshot'])
    if digest(snapshot / 'manifest.json') != manifest['snapshot_manifest_sha256']:
        raise ValueError('frozen source manifest changed')
    counts['snapshot_files'] = check_hashes(snapshot / 'root', read_json(snapshot / 'manifest.json')['files'])
    if 'maintenance_manifest_sha256' in manifest:
        maintenance = root / 'reproducibility/maintenance_20261011.json'
        if digest(maintenance) != manifest['maintenance_manifest_sha256']:
            raise ValueError('maintained source manifest changed')
        counts['maintained_files'] = check_hashes(root, read_json(maintenance)['files'])
    return counts


def copy_entries(source, target, entries):
    for name in entries:
        src, dst = contained(source, name), contained(target, name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def runtime_files(root):
    files = []
    for name in ('incontext_collapse', 'scripts', 'tests', 'runner', 'docs', 'reproducibility'):
        base = root / name
        if base.exists():
            files.extend(p.relative_to(root).as_posix() for p in base.rglob('*')
                         if p.is_file() and '__pycache__' not in p.parts
                         and not p.name.endswith(('.pyc', '.log')))
    files.extend(p.name for p in root.iterdir() if p.is_file() and
                 p.name in ('README.md', 'Makefile', 'requirements.txt', 'requirements.lock',
                            'CITATION.cff', 'LICENSE', '.gitignore'))
    return sorted(set(files))


def environment():
    # CPU analysis is offline. Users prepare the pinned model/data cache separately.
    return {**os.environ, 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
            'TRANSFORMERS_OFFLINE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
            'PYTHONUNBUFFERED': '1'}


def run_command(command, cwd, log):
    with Path(log).open('w') as stream:
        result = subprocess.run(command, cwd=cwd, env=environment(), stdout=stream,
                                stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f'command failed ({result.returncode}); see {log}')


def prepare_workspace(root, manifest, target, frozen=False):
    if target.resolve().is_relative_to(root.resolve()):
        raise ValueError('isolated run/bundle workspaces must lie outside the source checkout')
    target.mkdir(parents=True, exist_ok=False)
    if frozen:
        snap = contained(root, manifest['snapshot'])
        copy_entries(snap / 'root', target, read_json(snap / 'manifest.json')['files'])
    else:
        copy_entries(root, target, runtime_files(root))
    for key in ('evidence_files', 'paper_files', 'figure_files'):
        copy_entries(root, target, manifest[key])
    return target


def export_tables(root, manifest, output):
    """Use the actual paper-time analysis, including its recorded amendments."""
    with tempfile.TemporaryDirectory(prefix='incontext-paper-analysis-') as temp:
        workspace = prepare_workspace(root, manifest, Path(temp) / 'workspace', frozen=True)
        commands = [
            ['scripts/export_revision_tables.py', '--out-dir', str(output / 'revision_results')],
            ['scripts/export_strengthening_tables.py', '--out-dir', str(output / 'revision_results')],
            ['scripts/export_region_aggregate_tables.py', '--out-dir', str(output / 'revision_results')],
            ['scripts/export_followup_tables.py', '--project-root', str(workspace),
             '--output-dir', str(output / 'followup_results')],
        ]
        for index, args in enumerate(commands):
            run_command([sys.executable, *args], workspace, output / f'export_{index + 1}.log')
    compared, errors = [], []
    for group in ('revision_results', 'followup_results'):
        for path in sorted((root / 'paper' / group).glob('*')):
            if not path.is_file():
                continue
            generated = output / group / path.name
            if not generated.is_file() or generated.read_bytes() != path.read_bytes():
                errors.append(f'{group}/{path.name}')
            compared.append(f'{group}/{path.name}')
    if errors:
        raise ValueError('regenerated files differ from paper: ' + ', '.join(errors))
    if not compared:
        raise ValueError('paper contains no generated numerical tables')
    receipt = {'paper_commit': manifest['paper_commit'], 'matched_files': compared,
               'snapshot': manifest['snapshot'], 'model_inference': False}
    write_json(output / 'table_verification.json', receipt)
    return receipt


def select_studies(manifest, requested, from_scratch=False):
    studies = {study['id']: study for study in manifest['studies']}
    requested_ids = ([key for key, study in studies.items() if study['kind'] != 'training'
                      and key != 'smoke'] if requested == 'all' else requested.split(','))
    selected, visiting = [], set()

    def visit(name):
        if name not in studies:
            raise ValueError(f'unknown study {name}; use plan --study all')
        if name in visiting:
            raise ValueError(f'cyclic study dependency: {name}')
        if name in selected:
            return
        visiting.add(name)
        study = studies[name]
        dependencies = list(study.get('dependencies', []))
        if from_scratch:
            dependencies.extend(study.get('retrain_dependencies', []))
        for dep in dependencies:
            if studies[dep]['kind'] != 'training' or from_scratch:
                visit(dep)
        visiting.remove(name)
        selected.append(name)

    for name in requested_ids:
        visit(name)
    return [studies[name] for name in selected]


def commands_for(studies, device=None):
    commands = []
    for study in studies:
        args = list(study['arguments'])
        if device and study['module'] not in DEVICE_ARGUMENT_MODULES:
            raise ValueError(f"{study['id']}: this module selects its device automatically; "
                             'omit --device (the module has no device override argument)')
        if '--device' in args:
            index = args.index('--device')
            if device:
                args[index + 1] = device
            else:
                del args[index:index + 2]  # Recorded MPS preference is not a portable default.
        elif device:
            args.extend(['--device', device])
        if study['module'] == 'incontext_collapse.rule_factorial':
            # The original exclusion evidence is mirrored before any new fitting.
            if '--prior-data-root' in args:
                args[args.index('--prior-data-root') + 1] = 'reproduction_prior/data'
            else:
                args.extend(['--prior-data-root', 'reproduction_prior/data'])
            if '--prior-exclude-dir' in args:
                args[args.index('--prior-exclude-dir') + 1] = 'reproduction_prior/data/followup_20261010'
        commands.append([sys.executable, '-m', study['module'], *args])
    return commands


def audit_result(workspace, study, logs):
    result = study['output']
    family = study['audit_family']
    if study.get('kind') == 'training':
        validate_training_complete(read_json(workspace / result), study)
    if family == 'followup':
        kind = next(kind for kind in ('rule', 'shortcut', 'vqa') if kind in study['id'])
        report = logs / (study['id'] + '.audit.json')
        command = [sys.executable, 'scripts/analyze_followup_experiments.py', result,
                   '--kind', kind, '--out', str(report)]
        if kind == 'vqa':
            command.extend(['--frame', 'data/followup_20261010/vqa_images/manifest.json'])
    elif family == 'strengthening':
        command = [sys.executable, 'scripts/audit_strengthening_artifacts.py', result, '--json']
    elif family in ('paired', 'panel', 'adapter_eval'):
        check = [sys.executable, '-m', 'incontext_collapse.queue_check',
                 family.replace('_', '-'), result]
        audit_path = result
        if family == 'panel':
            arguments = study['arguments']
            audit_path = arguments[arguments.index('--cells-dir') + 1]
            check.extend(['--cells-dir', audit_path])
        run_command(check, workspace, logs / (study['id'] + '.grid_check.log'))
        command = [sys.executable, 'scripts/audit_revision_artifacts.py', audit_path, '--summary']
    elif study.get('kind') == 'training':
        command = [sys.executable, 'scripts/audit_revision_artifacts.py', result, '--summary']
    elif family in ('revision', 'coverage'):
        # This reducer combines independent rescoring with exact declared-grid checks.
        command = [sys.executable, 'scripts/summarize_revision_followup.py', result,
                   '--out', str(logs / (study['id'] + '.audit.json'))]
    else:
        raise ValueError(f'unsupported audit family: {family}')
    run_command(command, workspace, logs / (study['id'] + '.audit.log'))
    if family == 'followup':
        audited = read_json(report)
        if (audited.get('status') != 'complete'
                or audited.get('artifact_sha256') != digest(workspace / result)
                or audited.get('findings', {}).get('generation_records') != study['planned_generations']):
            raise ValueError(f"{study['id']}: audited artifact is incomplete or differs from the planned generation count")
    return digest(workspace / result)


def validate_training_complete(payload, study):
    """A successful fitting subprocess must include every requested seed and cell."""
    config = payload['run_config']
    args = study['arguments']
    seeds = [int(value) for value in args[args.index('--seeds') + 1].split(',')]
    if config.get('seeds') != seeds:
        raise ValueError('training seeds differ from registered command')
    ks, n = set(map(str, config['ks'])), config['n']

    def seed_list(values, key=None):
        found = [value[key] if key else value for value in values]
        if sorted(found) != sorted(seeds):
            raise ValueError('incomplete or duplicate training seeds')

    def shots(values):
        if set(values) != ks or any(len(cell['records']) != n for cell in values.values()):
            raise ValueError('incomplete training evaluation grid')

    module = study['module']
    if module == 'incontext_collapse.circuit':
        for region in config['regions'].split(','):
            for component in config['components'].split(','):
                arm = payload[f'{region}:{component}']
                if arm.get('error') or arm.get('seed_errors'):
                    raise ValueError('training arm contains an error')
                seed_list(arm['seeds_done'])
                seed_list(arm['training_runs'], 'torch_seed')
                seed_list(arm['evidence'], 'seed')
                for block in arm['evidence']:
                    for condition in ('standard', 'remap'):
                        shots(block[condition])
        seed_list(payload['baseline_evidence'], 'seed')
        for block in payload['baseline_evidence']:
            for condition in ('standard', 'remap'):
                shots(block[condition])
    elif module == 'incontext_collapse.vqa_train':
        seed_list(payload['seeds_done'])
        if set(payload['training_runs']) != set(map(str, seeds)) or set(payload['evidence']) != set(map(str, seeds)):
            raise ValueError('incomplete VQA training seeds')
        seed_list(payload['training_runs'].values(), 'torch_seed')
        for block in payload['evidence'].values():
            if set(block) != {'baseline', 'repaired'}:
                raise ValueError('incomplete VQA training evaluation arms')
            for values in block.values():
                shots(values)
    elif module == 'incontext_collapse.dara':
        seed_list(payload['seeds_done'])
        seed_list(payload['training_runs'], 'torch_seed')
        # The DARA-specific independent auditor checks both attention grids.
    else:
        raise ValueError(f'unsupported training audit module: {module}')


def execute_studies(root, manifest, studies, output, device=None):
    commands = commands_for(studies, device)
    workspace = prepare_workspace(root, manifest, output / 'workspace')
    logs = output / 'logs'
    logs.mkdir()
    if any(study['module'] == 'incontext_collapse.rule_factorial' for study in studies):
        preserve_rule_prior(root, workspace)
    # Remove only copied outputs of the selected jobs; retained source is untouched.
    for study in studies:
        for name in study.get('produces', [study['output']]):
            path = contained(workspace, name)
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
    receipt = {'created_utc': datetime.now(timezone.utc).isoformat(),
               'paper_commit': manifest['paper_commit'], 'python': sys.version,
               'platform': platform.platform(), 'status': 'running',
               'source_sha256': {name: digest(workspace / name) for name in runtime_files(workspace)},
               'input_manifest_sha256': digest(root / MANIFEST),
               'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
               'stages': []}
    receipt_path = output / 'run.json'
    write_json(receipt_path, receipt)
    try:
        for study, command in zip(studies, commands):
            check_hashes(workspace, receipt['source_sha256'])
            stage = {'id': study['id'], 'command': command, 'status': 'running'}
            receipt['stages'].append(stage)
            write_json(receipt_path, receipt)
            if study['id'] == 'followup.vqa_confirmatory':
                development = next(item for item in studies if item['id'] == 'followup.vqa_development')
                completed = next(item for item in receipt['stages'] if item['id'] == development['id'])
                gate = checked_vqa_gate(workspace, development, completed, logs)
                stage['development_gate'] = gate
                if not gate['gate_pass']:
                    stage['status'] = 'stopped_by_prespecified_gate'
                    write_json(receipt_path, receipt)
                    continue
            print('Running ' + study['id'], flush=True)
            run_command(command, workspace, logs / (study['id'] + '.log'))
            stage['output_sha256'] = audit_result(workspace, study, logs)
            report_path = logs / (study['id'] + '.audit.json')
            if report_path.exists():
                stage['audit_report_sha256'] = digest(report_path)
            stage['status'] = 'complete'
            write_json(receipt_path, receipt)
        receipt['status'] = 'complete'
        check_hashes(workspace, receipt['source_sha256'])
    except Exception as error:
        receipt['status'] = 'failed'
        receipt['error'] = str(error)
        raise
    finally:
        receipt['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(receipt_path, receipt)
    return receipt


def checked_vqa_gate(workspace, development, completed, logs):
    """Bind the locked screen to saved raw/audit bytes and a fresh independent audit."""
    raw = workspace / development['output']
    report = logs / (development['id'] + '.audit.json')
    raw_sha, audit_sha = digest(raw), digest(report)
    if (completed.get('status') != 'complete' or completed.get('output_sha256') != raw_sha
            or completed.get('audit_report_sha256') != audit_sha):
        raise ValueError('development gate artifact or audited receipt changed')
    audit = read_json(report)
    if audit.get('artifact_sha256') != raw_sha or audit.get('status') != 'complete':
        raise ValueError('development gate audit does not bind complete raw evidence')
    recheck_logs = logs / 'vqa_development_gate_recheck'
    recheck_logs.mkdir(exist_ok=False)
    if audit_result(workspace, development, recheck_logs) != raw_sha:
        raise ValueError('development raw evidence changed during gate recheck')
    rechecked = recheck_logs / report.name
    if digest(rechecked) != audit_sha or digest(report) != audit_sha or digest(raw) != raw_sha:
        raise ValueError('development independent gate recheck differs')
    passed = audit.get('findings', {}).get('gate_pass')
    if type(passed) is not bool:
        raise ValueError('development gate decision is missing or invalid')
    return {'gate_pass': passed, 'raw_sha256': raw_sha, 'audit_sha256': audit_sha,
            'reaudit_sha256': digest(rechecked)}


def preserve_rule_prior(root, workspace):
    """Copy exactly the frozen B exclusion files, retaining their original bytes."""
    inventory = read_json(root / 'data/followup_20261010/rule_images/prior_image_inventory.json')
    original_root = Path(inventory['config']['prior_root'])
    target = workspace / 'reproduction_prior/data'
    entries = {}
    for record in inventory['files']:
        relative = Path(record['path']).relative_to(original_root).as_posix()
        if relative in entries:
            raise ValueError('duplicate prior evidence path')
        entries[relative] = record['sha256']
    check_hashes(root / 'data', entries)
    target.mkdir(parents=True, exist_ok=False)
    copy_entries(root / 'data', target, entries)
    check_hashes(target, entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    verify = sub.add_parser('verify', help='Hash checks and exact saved-evidence numerical regeneration; no inference')
    verify.add_argument('--output-dir', type=Path)
    for name in ('tables', 'figures', 'bundle'):
        item = sub.add_parser(name)
        item.add_argument('--output-dir', type=Path, required=True)
    for name in ('plan', 'run'):
        item = sub.add_parser(name)
        item.add_argument('--study', required=True, help='ID, comma-separated IDs, or all')
        item.add_argument('--from-scratch', action='store_true', help='Include the recorded fitting dependencies')
        item.add_argument('--device', choices=['cpu', 'mps', 'cuda'])
        if name == 'run':
            item.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(ROOT / MANIFEST)
    if args.action in ('plan', 'run'):
        studies = select_studies(manifest, args.study, args.from_scratch)
        if args.action == 'plan':
            print(json.dumps({'studies': studies, 'commands': commands_for(studies, args.device),
                              'limits': manifest['limits']}, indent=2))
            return 0
    counts = verify_files(ROOT, manifest)
    if args.action == 'verify':
        if args.output_dir:
            output = fresh(args.output_dir)
            tables = export_tables(ROOT, manifest, output)
        else:
            with tempfile.TemporaryDirectory(prefix='incontext-verify-') as temp:
                tables = export_tables(ROOT, manifest, Path(temp))
        print(json.dumps({'status': 'passed', 'files': counts,
                          'matched_numerical_files': len(tables['matched_files']),
                          'limits': manifest['limits']}, indent=2))
    elif args.action == 'tables':
        receipt = export_tables(ROOT, manifest, fresh(args.output_dir))
        print(f"Matched {len(receipt['matched_files'])} paper numerical files")
    elif args.action == 'figures':
        if args.output_dir.exists():
            raise ValueError('figure output must be a fresh path')
        subprocess.run([sys.executable, 'scripts/make_figures.py', '--output-dir',
                        str(args.output_dir.resolve())], cwd=ROOT, env=environment(), check=True)
    elif args.action == 'bundle':
        output = fresh(args.output_dir)
        workspace = prepare_workspace(ROOT, manifest, output / 'incontext_vlm')
        write_json(output / 'bundle.json', {'paper_commit': manifest['paper_commit'],
                   'manifest_sha256': digest(ROOT / MANIFEST), 'files': counts,
                   'limits': manifest['limits'], 'hub_model_weights_included': False})
        print(workspace)
    elif args.action == 'run':
        execute_studies(ROOT, manifest, studies, fresh(args.run_dir), args.device)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        raise SystemExit(1)
