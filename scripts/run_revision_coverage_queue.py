#!/usr/bin/env python3
"""Run second-wave coverage only after the full follow-up queue succeeds.

The two queues share the first queue's execution lock. Preflight only reads
checkpoint metadata and the pinned local Hugging Face cache; it loads no model.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from incontext_collapse.provenance import model_revision
from incontext_collapse.protocol import ProtocolConfig
from incontext_collapse.results import write_json_atomic
from scripts import run_revision_followup_queue as first_queue

OUT = "data/revision_v25/coverage"
CONTROL = "data/revision_v25/coverage_queue"
FIRST_CONTROL = "data/revision_v25/queue"
QWEN = "Qwen/Qwen2-VL-2B-Instruct"
QWEN7 = "Qwen/Qwen2-VL-7B-Instruct"


def stages(root=ROOT):
    python = str(root / ".venv/bin/python")
    depth = "data/revision_v24/qwen2vl2b_shapes_depth_q_up_v_episode_remap150_n40_generation_v24_adapters"
    natural = "data/revision_v24_completion/qwen2vl2b_shapes_connector_natural150_n40_generation_v24_adapters"
    adapters = {region: f"{depth}/{region}_both" for region in ("early", "mid", "late")}
    adapters["natural_connector"] = f"{natural}/connector_both"
    jobs = []

    def add(name, module, arguments, expected, *, generations=0, scoring=0, records=0,
            checkpoints=None):
        output = f"{OUT}/{name}_n40.json"
        jobs.append({
            "name": name, "command": [python, "-m", f"incontext_collapse.{module}",
                                       *arguments, "--out", output],
            "output": output, "expected_config": expected,
            "planned_generations": generations, "planned_scoring_queries": scoring,
            "planned_feature_records": records, "adapters": checkpoints or {},
        })

    common = ["--seeds", "0,1,2", "--n", "40", "--img-size", "112"]
    for name, model, arm_paths, conditions, ks in (
        ("options_depth_natural_shapes", QWEN, adapters,
         ["standard", "remap", "remap_shuffled"], [0, 4, 8, 16]),
        ("options_qwen7_shapes", QWEN7, {},
         ["standard", "shuffled", "remap", "remap_shuffled"], [0, 1, 2, 4, 8, 16]),
    ):
        arguments = ["--model", model, "--tasks", "shapes4", "--conditions", ",".join(conditions),
                     "--ks", ",".join(map(str, ks)), "--option-orders", "random",
                     "--max-new-tokens", "6", *common]
        for arm, path in arm_paths.items():
            arguments += ["--adapter", f"{arm}={path}"]
        expected = {"experiment": "paired_option_order_saved_adapters", "model": model,
                    "model_revision": model_revision(model), "tasks": ["shapes4"],
                    "arms": ["frozen", *arm_paths], "conditions": conditions, "ks": ks,
                    "seeds": [0, 1, 2], "n": 40, "img_size": 112,
                    "option_orders": ["random"], "max_new_tokens": 6,
                    "score_mode": "generation",
                    "protocols": {"random": ProtocolConfig(option_order="random").metadata()}}
        add(name, "option_order_eval", arguments, expected,
            generations=3 * len(expected["arms"]) * len(conditions) * len(ks) * 40,
            checkpoints=arm_paths)

    for module, baseline, conditions, ks in (
        ("calibration", "contextual_calibration_neutral_query", ["standard", "remap"], [0, 4, 8, 16]),
        ("frozen_features", "frozen_connector_feature_1nn", ["remap", "remap_shuffled"], [4, 8, 16]),
    ):
        for task in ("cifar4", "fashion4"):
            arguments = ["--model", QWEN, "--task", task, "--conditions", ",".join(conditions),
                         "--ks", ",".join(map(str, ks)), *common]
            if module == "calibration":
                arguments += ["--option-order", "random"]
            expected = {"baseline": baseline, "model": QWEN,
                        "model_revision": model_revision(QWEN), "task": task,
                        "conditions": conditions, "ks": ks, "seeds": [0, 1, 2],
                        "n": 40, "img_size": 112,
                        "protocol": ProtocolConfig(option_order=(
                            "random" if module == "calibration" else "class")).metadata()}
            count = 3 * len(conditions) * len(ks) * 40
            add(f"{module}_{task}", module, arguments, expected,
                scoring=count if module == "calibration" else 0,
                records=count if module == "frozen_features" else 0)
    return jobs


def source_fingerprint():
    paths = [Path(__file__).resolve(), ROOT / "tests/test_coverage_queue.py",
             ROOT / "incontext_collapse/frozen_features.py"]
    return {
        "first_queue": first_queue.source_fingerprint(),
        "coverage": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in paths},
    }


def cached_model_manifest(model, cache_dir=None):
    from huggingface_hub.constants import HF_HUB_CACHE

    revision = model_revision(model)
    if revision is None:
        raise ValueError(f"model has no pinned revision: {model}")
    snapshot = Path(cache_dir or HF_HUB_CACHE) / ("models--" + model.replace("/", "--")) / "snapshots" / revision
    required = ["config.json", "preprocessor_config.json", "tokenizer_config.json",
                "tokenizer.json", "vocab.json", "merges.txt", "generation_config.json",
                "model.safetensors.index.json"]
    for name in required:
        path = snapshot / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"pinned cached model file missing: {path}")
    tokenizer = json.loads((snapshot / "tokenizer_config.json").read_text())
    if not tokenizer.get("chat_template") and not any(
            (snapshot / name).is_file() for name in ("chat_template.json", "chat_template.jinja")):
        raise FileNotFoundError(f"pinned cached chat template missing: {snapshot}")
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    shards = sorted(set(index.get("weight_map", {}).values()))
    if not shards:
        raise ValueError("cached weight index has no shards")
    sizes = {}
    for name in shards:
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("unexpected cached shard path")
        path = snapshot / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"pinned cached weight shard missing: {path}")
        sizes[name] = path.stat().st_size
    return {"model": model, "revision": revision, "snapshot": str(snapshot),
            "shard_bytes": sizes}


def preflight(jobs, root=ROOT, cache_dir=None):
    from incontext_collapse.option_order_eval import checkpoint_manifest

    checked = {}
    for job in jobs:
        if not job["adapters"]:
            continue
        manifest = checkpoint_manifest(
            {name: root / path for name, path in job["adapters"].items()}, [0, 1, 2])
        for arm in manifest.values():
            for checkpoint in arm.values():
                if checkpoint["base_model_name_or_path"] != QWEN:
                    raise ValueError("coverage checkpoint has the wrong base model")
                if checkpoint["training_source"] is None:
                    raise ValueError("coverage checkpoint lacks source training provenance")
        checked[job["name"]] = manifest
    return {"checkpoints": checked, "cached_model": cached_model_manifest(QWEN7, cache_dir)}


def first_queue_complete(path, expected_sources, expected_stages):
    if not path.exists():
        return False
    manifest = json.loads(path.read_text())
    if manifest.get("source_sha256") != expected_sources:
        raise RuntimeError("first queue source fingerprint differs from frozen coverage inputs")
    if manifest.get("status") == "failed":
        raise RuntimeError(f"first queue failed: {manifest.get('error', 'unknown failure')}")
    if manifest.get("status") != "complete":
        if manifest.get("status") != "running":
            raise RuntimeError("first queue has an unexpected status")
        return False
    observed = manifest.get("stages", [])
    names = [job.get("name") for job in observed]
    if (len(names) != len(set(names)) or set(names) != set(expected_stages)
            or any(job.get("status") != "complete" for job in observed)):
        raise RuntimeError("first queue did not complete its entire required stage grid")
    return True


def wait_for_first_success(path, expected_sources, expected_stages, check_sources,
                           *, sleep=time.sleep, poll_seconds=60):
    while True:
        check_sources()
        if first_queue_complete(path, expected_sources, expected_stages):
            return
        print(f"[{first_queue.now()}] waiting for full first-queue success", flush=True)
        sleep(poll_seconds)


@contextmanager
def execution_lock(path, *, wait=False, check=None, sleep=time.sleep, poll_seconds=10):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        while True:
            if check is not None:
                check()
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not wait:
                    raise RuntimeError(f"another process owns queue lock: {path}")
                sleep(poll_seconds)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def validate_output(job, root=ROOT):
    payload = json.loads((root / job["output"]).read_text())
    config = payload.get("run_config", {})
    for name, expected in job["expected_config"].items():
        if config.get(name) != expected:
            raise ValueError(f"coverage output configuration mismatch: {name}")
    if config.get("baseline") != "frozen_connector_feature_1nn":
        return
    seeds, conditions, ks, n = config["seeds"], config["conditions"], config["ks"], config["n"]
    if sorted(payload.get("seeds_done", [])) != sorted(seeds):
        raise ValueError("frozen-feature seed grid incomplete or duplicated")
    evidence = payload.get("evidence", {})
    if set(evidence) != set(map(str, seeds)):
        raise ValueError("frozen-feature evidence seed grid mismatch")
    for seed in seeds:
        if set(evidence[str(seed)]) != set(conditions):
            raise ValueError("frozen-feature condition grid mismatch")
        for condition in conditions:
            shots = evidence[str(seed)][condition]
            if set(shots) != set(map(str, ks)):
                raise ValueError("frozen-feature K grid mismatch")
            for k in ks:
                cell = shots[str(k)]
                if (cell.get("seed"), cell.get("condition"), cell.get("k")) != (seed, condition, k):
                    raise ValueError("frozen-feature cell identity mismatch")
                records = cell.get("records", [])
                if (len(records) != n or cell.get("summary", {}).get("n") != n
                        or [record.get("query_index") for record in records] != list(range(n))):
                    raise ValueError("frozen-feature query grid mismatch")
    if set(payload.get("summary", {})) != set(conditions) or any(
            set(payload["summary"][condition]) != set(map(str, ks)) for condition in conditions):
        raise ValueError("frozen-feature aggregate grid incomplete")


def execute(jobs, root=ROOT):
    control = root / CONTROL
    control.mkdir(parents=True, exist_ok=True)
    with execution_lock(control / "queue.lock"):
        path = control / "status.json"
        fingerprint = source_fingerprint()
        previous = json.loads(path.read_text()) if path.exists() else {}
        if previous and previous.get("source_sha256") != fingerprint:
            raise RuntimeError("coverage source changed since previous run; refusing resume")
        inputs = preflight(jobs, root)
        if previous and previous.get("preflight") != inputs:
            raise RuntimeError("coverage checkpoint or cached-model inputs changed since previous run")
        completed = {job["name"] for job in previous.get("stages", []) if job.get("status") == "complete"}
        manifest = {"started": first_queue.now(), "pid": os.getpid(), "status": "waiting",
                    "source_sha256": fingerprint, "preflight": inputs, "stages": deepcopy(jobs)}
        write_json_atomic(path, manifest)
        first_path = root / FIRST_CONTROL / "status.json"
        required_stages = [job["name"] for job in first_queue.stages()]
        environment = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                       "TRANSFORMERS_OFFLINE": "1", "PYTHONUNBUFFERED": "1",
                       "MPLCONFIGDIR": "/private/tmp/incontext-vlm-matplotlib"}

        def check_sources():
            if source_fingerprint() != fingerprint:
                raise RuntimeError("frozen experiment or coverage sources changed")

        def check_ready():
            check_sources()
            if not first_queue_complete(first_path, fingerprint["first_queue"], required_stages):
                raise RuntimeError("first queue is no longer complete")

        def run(command):
            print(f"[{first_queue.now()}] {shlex.join(command)}", flush=True)
            subprocess.run(command, cwd=root, env=environment, check=True)

        try:
            wait_for_first_success(first_path, fingerprint["first_queue"], required_stages, check_sources)
            with execution_lock(root / FIRST_CONTROL / "run.lock", wait=True, check=check_ready):
                check_ready()
                if preflight(jobs, root) != inputs:
                    raise RuntimeError("coverage inputs changed while waiting")
                manifest["status"] = "running"
                python = str(root / ".venv/bin/python")
                run([python, "-c", "import torch; assert torch.backends.mps.is_available(), 'MPS unavailable'"])
                for job in manifest["stages"]:
                    check_ready()
                    manifest["current_stage"] = job["name"]
                    job.update(status="testing", started=first_queue.now())
                    write_json_atomic(path, manifest)
                    run([python, "-m", "unittest", "discover", "-s", "tests", "-q"])
                    check_ready()
                    if job["name"] not in completed:
                        job["status"] = "evaluating"
                        write_json_atomic(path, manifest)
                        run(job["command"])
                    else:
                        job["resumed_complete"] = True
                    job["status"] = "auditing"
                    write_json_atomic(path, manifest)
                    run([python, "scripts/audit_revision_artifacts.py", job["output"], "--summary"])
                    validate_output(job, root)
                    check_ready()
                    job.update(status="complete", finished=first_queue.now())
                    write_json_atomic(path, manifest)
                manifest.update(status="complete", finished=first_queue.now())
        except BaseException as error:
            manifest.update(status="failed", stopped=first_queue.now(), error=str(error))
            raise
        finally:
            write_json_atomic(path, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--preflight", action="store_true", help="read-only cached-input checks; no model load")
    mode.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    jobs = stages()
    if args.dry_run:
        totals = {key: sum(job[key] for job in jobs) for key in
                  ("planned_generations", "planned_scoring_queries", "planned_feature_records")}
        print(json.dumps({"stages": jobs, **totals}, indent=2))
        return
    if args.preflight:
        print(json.dumps(preflight(jobs), indent=2))
        return
    if args.detach:
        control = ROOT / CONTROL
        control.mkdir(parents=True, exist_ok=True)
        command = [str(ROOT / ".venv/bin/python"), str(Path(__file__).resolve())]
        if sys.platform == "darwin":
            command = ["/usr/bin/caffeinate", "-i", *command]
        with (control / "run.log").open("a") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"launcher_pid": process.pid, "log": str(control / "run.log"),
                          "status": str(control / "status.json")}), flush=True)
        return
    execute(jobs)


if __name__ == "__main__":
    raise SystemExit("This historical queue is retired. Use scripts/reproduce.py plan --study all, then run --study ID --run-dir FRESH_PATH. Original source is retained in reproducibility/legacy/.")
