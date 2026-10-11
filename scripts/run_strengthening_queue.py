#!/usr/bin/env python3
"""Run the fixed saved-checkpoint strengthening studies serially and offline."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from incontext_collapse.results import write_json_atomic
from incontext_collapse.provenance import model_revision
from scripts import run_revision_followup_queue as previous_queue
from scripts import run_revision_coverage_queue as coverage_queue

OUT = "data/strengthening_v1"
CONTROL = OUT + "/queue"
FRAME = OUT + "/rule_images"
MODEL = "Qwen/Qwen2-VL-2B-Instruct"
ADAPTERS = {
    "natural_connector": "data/revision_v24_completion/qwen2vl2b_shapes_connector_natural150_n40_generation_v24_adapters/connector_both",
    "episode_connector": previous_queue.CONNECTOR + "/connector_both",
    "all_layer": previous_queue.ALL_LAYER + "/all_both",
}
PROTOCOL = "docs/strengthening_protocol_2026_10_05.md"


def stages(root=ROOT):
    python = str(root / ".venv/bin/python")
    adapters = [part for name, path in ADAPTERS.items() for part in ("--adapter", f"{name}={path}")]
    common = ["--model", MODEL, *adapters, "--img-size", "112", "--max-new-tokens", "6"]
    arms = ["frozen", *ADAPTERS]
    jobs = []

    def target(name, tasks, seeds, n, ks, purpose):
        output = f"{OUT}/{name}.json"
        conditions = ["standard", "shuffled", "remap", "remap_shuffled"]
        command = [python, "-m", "incontext_collapse.option_order_eval", *common,
                   "--tasks", ",".join(tasks), "--conditions", ",".join(conditions),
                   "--seeds", ",".join(map(str, seeds)), "--n", str(n),
                   "--ks", ",".join(map(str, ks)), "--option-orders", "random", "--out", output]
        jobs.append({"name": name, "kind": "target_transfer", "purpose": purpose,
                     "command": command, "output": output,
                     "planned_generations": len(tasks) * len(seeds) * len(arms) * len(conditions) * len(ks) * n,
                     "expected_config": {"experiment": "paired_option_order_saved_adapters",
                         "model": MODEL, "model_revision": model_revision(MODEL), "tasks": tasks,
                         "arms": arms, "conditions": conditions, "ks": ks, "seeds": seeds,
                         "n": n, "option_orders": ["random"], "img_size": 112,
                         "max_new_tokens": 6, "score_mode": "generation"}})

    def rules(name, partition, seeds, n, ks, purpose):
        output = f"{OUT}/{name}.json"
        command = [python, "-m", "incontext_collapse.rule_transfer", *common,
                   "--data-dir", FRAME, "--partition", partition, "--n", str(n),
                   "--seeds", ",".join(map(str, seeds)), "--ks", ",".join(map(str, ks)),
                   "--out", output]
        jobs.append({"name": name, "kind": "rule_transfer", "purpose": purpose,
                     "command": command, "output": output,
                     "planned_generations": len(seeds) * len(arms) * 2 * (len(ks) + 1) * n,
                     "expected_config": {"experiment": "two_object_rule_transfer",
                         "family": "two_object_color_selection", "model": MODEL,
                         "model_revision": model_revision(MODEL), "partition": partition,
                         "arms": arms, "condition_ks": {"explicit_natural": [0], "implicit_remap": ks},
                         "conditions": ["explicit_natural", "implicit_remap"],
                         "rules": ["left_color", "right_color"], "seeds": seeds,
                         "n": n, "img_size": 112, "max_new_tokens": 6,
                         "option_order": "random", "score_mode": "generation", "dtype": "bfloat16"}})

    target("target_smoke", ["cifar4", "fashion4"], [0], 4, [0, 4], "technical smoke; not confirmatory evidence")
    rules("rule_development", "development", [0], 8, [0, 8], "technical smoke on separate development images")
    target("natural_transfer_cifar", ["cifar4"], [0, 1, 2], 40, [0, 4, 8, 16], "fixed comparative target evaluation")
    target("natural_transfer_fashion", ["fashion4"], [0, 1, 2], 40, [0, 4, 8, 16], "fixed comparative target evaluation")
    rules("rule_confirmatory", "confirmatory", [0, 1, 2], 40, [0, 8, 16], "held-out rule-transfer evaluation")
    return jobs


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_fingerprint(root=ROOT):
    paths = ["scripts/run_strengthening_queue.py", "scripts/audit_strengthening_artifacts.py",
             "scripts/summarize_revision_followup.py", "scripts/run_revision_coverage_queue.py",
             "incontext_collapse/rule_transfer.py", PROTOCOL,
             "tests/test_strengthening_queue.py", "tests/test_strengthening_audit.py",
             "tests/test_rule_transfer.py"]
    return {"previous": previous_queue.source_fingerprint(),
            "new": {path: digest(root / path) for path in paths}}


def checkpoint_identity(checkpoints):
    """Compare checkpoint contents and provenance independently of path spelling."""
    return {arm: {seed: {
        "sha256": entry["sha256"], "base_model_name_or_path": entry["base_model_name_or_path"],
        "training_source": {key: value for key, value in (entry.get("training_source") or {}).items()
                            if key != "path"},
    } for seed, entry in by_seed.items()} for arm, by_seed in checkpoints.items()}


def check_previous_completion(root=ROOT):
    previous = json.loads((root / "data/revision_v25/queue/status.json").read_text())
    coverage = json.loads((root / "data/revision_v25/coverage_queue/status.json").read_text())
    for state, jobs, fingerprint in (
        (previous, previous_queue.stages(), previous_queue.source_fingerprint()),
        (coverage, coverage_queue.stages(), coverage_queue.source_fingerprint()),
    ):
        if state.get("status") != "complete" or state.get("source_sha256") != fingerprint:
            raise RuntimeError("previous completed evidence or its frozen sources have changed")
        observed = state.get("stages", [])
        if (len(observed) != len(jobs) or {item["name"] for item in observed} != {item["name"] for item in jobs}
                or any(item.get("status") != "complete" for item in observed)):
            raise RuntimeError("previous experiment stage grid is incomplete")


def prepare_frames(root=ROOT):
    from incontext_collapse.rule_transfer import build_manifest
    return [build_manifest(root / FRAME, partition=partition, n=n, size=112)
            for partition, n in (("development", 8), ("confirmatory", 40))]


def preflight(root=ROOT):
    from incontext_collapse.option_order_eval import checkpoint_manifest
    from incontext_collapse.rule_transfer import load_manifest, validate_disjoint
    check_previous_completion(root)
    checkpoints = checkpoint_manifest({name: root / path for name, path in ADAPTERS.items()}, [0, 1, 2])
    for name, by_seed in checkpoints.items():
        objective = "natural" if name == "natural_connector" else "episode_remap"
        for checkpoint in by_seed.values():
            training = checkpoint.get("training_source") or {}
            if (checkpoint["base_model_name_or_path"] != MODEL or training.get("objective") != objective
                    or training.get("rank") != 8 or training.get("steps") != 150):
                raise RuntimeError("checkpoint model, objective, rank, or steps differ from the fixed study")
    manifests, frames = {}, []
    for partition, n in (("development", 8), ("confirmatory", 40)):
        path = root / FRAME / f"manifest_{partition}.json"
        manifest, _ = load_manifest(path)
        if manifest["config"]["n"] != n or manifest["config"]["img_size"] != 112:
            raise RuntimeError("prepared rule frame differs from the fixed study")
        frames.append(manifest)
        manifests[partition] = {"path": str(path.relative_to(root)), "sha256": digest(path)}
    validate_disjoint(*frames)
    cache = coverage_queue.cached_model_manifest(MODEL)
    snapshot = Path(cache["snapshot"])
    cached_files = {snapshot / name for name in cache["shard_bytes"]}
    cached_files.update(path for path in snapshot.iterdir()
                        if path.is_file() and path.suffix in {".json", ".jinja", ".txt"})
    cache["sha256"] = {path.name: digest(path) for path in sorted(cached_files)}
    return {"checkpoints": checkpoint_identity(checkpoints), "manifests": manifests,
            "model_cache": cache}


def validate_stage_config(job, payload, inputs):
    config = payload.get("run_config", {})
    for key, expected in job["expected_config"].items():
        if config.get(key) != expected:
            raise ValueError(f"{job['name']}: output configuration mismatch: {key}")
    expected_checkpoints = {arm: {str(seed): by_seed[str(seed)] for seed in config["seeds"]}
                            for arm, by_seed in inputs["checkpoints"].items()}
    if checkpoint_identity(config.get("checkpoints", {})) != expected_checkpoints:
        raise ValueError(f"{job['name']}: checkpoint provenance differs from preflight")
    if job["kind"] == "rule_transfer":
        if config.get("manifest_sha256") != inputs["manifests"][config["partition"]]["sha256"]:
            raise ValueError(f"{job['name']}: rule image manifest differs from preflight")
        if config["partition"] == "confirmatory" and (
                (config.get("development_manifest") or {}).get("sha256")
                != inputs["manifests"]["development"]["sha256"]):
            raise ValueError(f"{job['name']}: development image manifest differs from preflight")


def audit_stage(job, inputs, root=ROOT):
    path = root / job["output"]
    payload = json.loads(path.read_text())
    validate_stage_config(job, payload, inputs)
    from scripts.audit_strengthening_artifacts import audit_file
    report = audit_file(path, expected_config=job["expected_config"], root=root)
    if (report["status"] != "complete" or report.get("errors") or report.get("warnings")
            or report.get("audit", {}).get("warnings")):
        raise ValueError(f"{job['name']}: strengthening audit failed: {report.get('errors')}")
    report["queue_purpose"] = job["purpose"]
    report["publication_ready"] = False
    destination = root / OUT / "reports" / f"{job['name']}.json"
    write_json_atomic(destination, report)
    return {"path": str(destination.relative_to(root)), "sha256": digest(destination),
            "artifact_sha256": digest(path)}


def execute(jobs, root=ROOT, *, stop_after=None):
    control = root / CONTROL
    control.mkdir(parents=True, exist_ok=True)
    with coverage_queue.execution_lock(control / "queue.lock"):
        path = control / "status.json"
        fingerprint = source_fingerprint(root)
        previous = json.loads(path.read_text()) if path.exists() else {}
        if previous and previous.get("source_sha256") != fingerprint:
            raise RuntimeError("strengthening source changed; refusing to mix resumed evidence")
        if previous and previous.get("fixed_plan") != jobs:
            raise RuntimeError("strengthening plan changed; refusing resume")
        inputs = preflight(root)
        if previous and previous.get("inputs") != inputs:
            raise RuntimeError("strengthening checkpoint, image, or cache inputs changed")
        completed = {job["name"] for job in previous.get("stages", []) if job.get("status") == "complete"}
        for job in previous.get("stages", []):
            if job["name"] in completed and (
                    not (root / job["output"]).is_file()
                    or digest(root / job["output"]) != job.get("report", {}).get("artifact_sha256")):
                raise RuntimeError("completed strengthening artifact changed; refusing resume")
        state = {"started": previous.get("started", previous_queue.now()),
                 "resumed": previous_queue.now() if previous else None, "pid": os.getpid(),
                 "status": "waiting", "source_sha256": fingerprint, "inputs": inputs,
                 "fixed_plan": jobs, "stages": deepcopy(jobs), "publication_ready": False}
        write_json_atomic(path, state)
        environment = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                       "TRANSFORMERS_OFFLINE": "1", "PYTHONUNBUFFERED": "1",
                       "MPLCONFIGDIR": "/private/tmp/incontext-vlm-matplotlib"}

        def check_sources():
            if source_fingerprint(root) != fingerprint:
                raise RuntimeError("frozen strengthening sources changed during the queue")

        def run(command):
            print(f"[{previous_queue.now()}] {shlex.join(command)}", flush=True)
            subprocess.run(command, cwd=root, env=environment, check=True)

        try:
            with coverage_queue.execution_lock(root / "data/revision_v25/queue/run.lock",
                                               wait=True, check=check_sources):
                check_sources()
                if preflight(root) != inputs:
                    raise RuntimeError("inputs changed while waiting for the shared model lock")
                state["status"] = "running"
                python = str(root / ".venv/bin/python")
                run([python, "-c", "import torch; assert torch.backends.mps.is_available(), 'MPS unavailable'"])
                for job in state["stages"]:
                    check_sources()
                    if preflight(root) != inputs:
                        raise RuntimeError("checkpoint, dataset, or model-cache inputs changed between stages")
                    state["current_stage"] = job["name"]
                    job.update(status="testing", started=previous_queue.now())
                    write_json_atomic(path, state)
                    run([python, "-m", "unittest", "discover", "-s", "tests", "-q"])
                    check_sources()
                    if job["name"] not in completed:
                        job["status"] = "evaluating"
                        write_json_atomic(path, state)
                        run(job["command"])
                    else:
                        job["resumed_complete"] = True
                    check_sources()
                    job["status"] = "auditing"
                    write_json_atomic(path, state)
                    job["report"] = audit_stage(job, inputs, root)
                    check_sources()
                    if preflight(root) != inputs:
                        raise RuntimeError("checkpoint or image inputs changed during a stage")
                    job.update(status="complete", finished=previous_queue.now())
                    write_json_atomic(path, state)
                    if job["name"] == stop_after:
                        state.update(status="pilot_complete", stopped=previous_queue.now())
                        return
                state.update(status="complete", finished=previous_queue.now())
                write_json_atomic(root / OUT / "completed_summary.json", {
                    "status": "complete", "publication_ready": False,
                    "source_sha256": fingerprint, "inputs": inputs,
                    "reports": [job["report"] for job in state["stages"]],
                    "scope": "Technical completion only. Retain all outcomes; scientific inclusion requires interpretation.",
                })
        except BaseException as error:
            state.update(status="failed", stopped=previous_queue.now(), error=str(error))
            raise
        finally:
            write_json_atomic(path, state)


def status(root=ROOT):
    path = root / CONTROL / "status.json"
    if not path.exists():
        return {"status": "not_started"}
    state = json.loads(path.read_text())
    jobs = []
    for job in state["stages"]:
        output = root / job["output"]
        payload = json.loads(output.read_text()) if output.exists() else {}
        cells = payload.get("cells", {})
        records = sum(len(cell.get("records", [])) for cell in cells.values())
        jobs.append({"name": job["name"], "status": job.get("status", "queued"),
                     "saved_records": records, "planned_generations": job["planned_generations"]})
    return {"status": state["status"], "pid": state["pid"], "current_stage": state.get("current_stage"),
            "error": state.get("error"), "stages": jobs,
            "publication_ready": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--prepare", action="store_true", help="prepare and verify fixed images/checkpoints; no model load")
    mode.add_argument("--preflight", action="store_true", help="read-only fixed-input verification")
    mode.add_argument("--detach", action="store_true")
    mode.add_argument("--status", action="store_true")
    parser.add_argument("--stop-after", choices=[job["name"] for job in stages()],
                        help="technical pilot boundary; resume without this option to run the same full plan")
    args = parser.parse_args()
    jobs = stages()
    if args.dry_run:
        print(json.dumps({"stages": jobs, "planned_generations": sum(job["planned_generations"] for job in jobs),
                          "training_steps": 0, "paid_api_calls": 0}, indent=2))
    elif args.status:
        print(json.dumps(status(), indent=2))
    elif args.prepare or args.preflight:
        if args.prepare:
            prepare_frames()
        print(json.dumps(preflight(), indent=2))
    elif args.detach:
        preflight()
        control = ROOT / CONTROL
        control.mkdir(parents=True, exist_ok=True)
        command = [str(ROOT / ".venv/bin/python"), str(Path(__file__).resolve())]
        if args.stop_after:
            command += ["--stop-after", args.stop_after]
        if sys.platform == "darwin":
            command = ["/usr/bin/caffeinate", "-i", *command]
        with (control / "run.log").open("a") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"launcher_pid": process.pid, "log": str(control / "run.log"),
                          "status": str(control / "status.json")}), flush=True)
    else:
        execute(jobs, stop_after=args.stop_after)


if __name__ == "__main__":
    raise SystemExit("This historical queue is retired. Use scripts/reproduce.py plan --study all, then run --study ID --run-dir FRESH_PATH. Original source is retained in reproducibility/legacy/.")
