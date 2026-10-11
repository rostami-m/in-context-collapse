#!/usr/bin/env python3
"""Serialize the audit follow-up experiments, with tests and evidence gates."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
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

PY = str(ROOT / ".venv/bin/python")
OUT = "data/revision_v25"
CONTROL = ROOT / OUT / "queue"
QWEN = "Qwen/Qwen2-VL-2B-Instruct"
CONNECTOR = "data/revision_v24/qwen2vl2b_shapes_connector_episode_remap150_n40_generation_v24_adapters"
ALL_LAYER = "data/revision_v24_completion/qwen2vl2b_shapes_all_layer_episode_remap150_n40_generation_v24_adapters"
QWEN25 = "data/revision_v24_completion/qwen25vl3b_shapes_connector_episode_remap150_n40_generation_v24_adapters"
VQA = "data/revision_v24_completion/qwen2vl2b_vqav2_connector_train150_n40_v24_adapters"


def stages():
    options = [PY, "-m", "incontext_collapse.option_order_eval", "--model", QWEN,
               "--adapter", f"connector={CONNECTOR}/connector_both",
               "--adapter", f"all_layer={ALL_LAYER}/all_both",
               "--conditions", "standard,shuffled,remap,remap_shuffled",
               "--seeds", "0,1,2", "--n", "40", "--max-new-tokens", "6"]
    jobs = []

    def add(name, command, output, generations):
        jobs.append({"name": name, "command": [*command, "--out", output],
                     "output": output, "planned_generations": generations})

    add("option_order_shapes", [*options, "--tasks", "shapes4", "--ks", "0,1,4,8,16"],
        f"{OUT}/options_shapes_n40.json", 14400)
    add("matched_transfer", [*options, "--tasks", "cifar4,fashion4", "--ks", "0,4,8,16",
                             "--option-orders", "random"],
        f"{OUT}/options_transfer_n40.json", 11520)
    add("standard_calibration", [PY, "-m", "incontext_collapse.calibration", "--model", QWEN,
                                  "--task", "shapes4", "--conditions", "standard",
                                  "--ks", "0,4,8,16", "--seeds", "0,1,2", "--n", "40",
                                  "--option-order", "random"],
        f"{OUT}/standard_calibration_shapes_n40.json", 0)
    jobs[-1]["planned_scoring_queries"] = 480
    add("second_model", [PY, "-m", "incontext_collapse.option_order_eval", "--model",
                         "Qwen/Qwen2.5-VL-3B-Instruct", "--adapter",
                         f"connector={QWEN25}/connector_both", "--tasks", "shapes4",
                         "--conditions", "standard,shuffled,remap,remap_shuffled",
                         "--ks", "0,4,8,16", "--seeds", "0,1,2", "--n", "40",
                         "--option-orders", "random", "--max-new-tokens", "6"],
        f"{OUT}/options_qwen25_shapes_n40.json", 3840)
    add("image_disjoint_vqa", [PY, "-m", "incontext_collapse.vqa_diverse", "--model", QWEN,
                               "--data-dir", f"{OUT}/vqa_image_frame", "--n", "80",
                               "--scan", "4096", "--adapters-dir", VQA,
                               "--shapes-adapters-dir", CONNECTOR, "--ks", "0,4,8",
                               "--seeds", "0,1,2"],
        f"{OUT}/vqav2_distinct80.json", 2160)
    for task, ks, generations in (("shapes4", "4,8,16", 1440),
                                  ("cifar4", "16", 480), ("fashion4", "16", 480)):
        add(f"cama_classification_{task}", [PY, "-m", "incontext_collapse.cama",
            "--device", "mps", "--task", task, "--prompt", "classification",
            "--option-order", "random", "--ks", ks, "--seeds", "0,1,2", "--n", "40",
            "--max-new-tokens", "6"],
            f"{OUT}/cama_classification_{task}_n40.json", generations)
    for objective in ("natural", "episode_remap"):
        source = ("data/revision_v24_completion/dara_qwen2vl2b_"
                  f"{objective}150_n40_generation_causal_v24.json")
        for snapshot in ("selected", "final"):
            add(f"dara_{objective}_{snapshot}", [PY, "-m", "incontext_collapse.dara",
                "--device", "mps", "--task", "shapes4", "--ks", "4,8,16", "--seeds",
                "0,1,2", "--n", "40", "--eval-factors-from", source,
                "--factor-snapshot", snapshot, "--option-order", "random"],
                f"{OUT}/dara_{objective}_{snapshot}_n40.json", 1440)
    return jobs


def source_fingerprint():
    modules = ("core", "protocol", "evaluation", "scoring", "models", "adapters",
               "circuit", "results", "provenance", "option_order_eval", "vqa",
               "vqa_repair", "vqa_diverse", "cama", "dara", "calibration")
    paths = [ROOT / "incontext_collapse" / f"{name}.py" for name in modules]
    paths += [ROOT / "requirements.lock", Path(__file__).resolve(),
              ROOT / "scripts/audit_revision_artifacts.py"]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def now():
    return datetime.now(timezone.utc).isoformat()


def execute(jobs):
    CONTROL.mkdir(parents=True, exist_ok=True)
    with (CONTROL / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("another follow-up queue owns the model-run lock")
        state_path = CONTROL / "status.json"
        fingerprint = source_fingerprint()
        if state_path.exists():
            previous = json.loads(state_path.read_text())
            if previous.get("source_sha256") != fingerprint and any(
                    (ROOT / job["output"]).exists() for job in jobs):
                raise RuntimeError("source changed since previous queue; do not mix resumed evidence")
        manifest = {"started": now(), "pid": os.getpid(), "status": "running",
                    "source_sha256": fingerprint, "stages": jobs}
        write_json_atomic(state_path, manifest)
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "PYTHONUNBUFFERED": "1", "MPLCONFIGDIR": "/private/tmp/incontext-vlm-matplotlib"}

        def run(command):
            print(f"[{now()}] {shlex.join(command)}", flush=True)
            subprocess.run(command, cwd=ROOT, env=env, check=True)

        try:
            run([PY, "-c", "import torch; assert torch.backends.mps.is_available(), 'MPS unavailable'"])
            for job in jobs:
                if source_fingerprint() != manifest["source_sha256"]:
                    raise RuntimeError("experiment code changed during queue; review before resuming")
                manifest["current_stage"] = job["name"]
                job.update(status="testing", started=now())
                write_json_atomic(state_path, manifest)
                run([PY, "-m", "unittest", "discover", "-s", "tests", "-q"])
                job["status"] = "evaluating"
                write_json_atomic(state_path, manifest)
                run(job["command"])
                job["status"] = "auditing"
                write_json_atomic(state_path, manifest)
                run([PY, "scripts/audit_revision_artifacts.py", job["output"], "--summary"])
                if source_fingerprint() != manifest["source_sha256"]:
                    raise RuntimeError("experiment code changed during active stage")
                job.update(status="complete", finished=now())
                write_json_atomic(state_path, manifest)
            manifest.update(status="complete", finished=now())
        except BaseException as error:
            manifest.update(status="failed", stopped=now(), error=str(error))
            raise
        finally:
            write_json_atomic(state_path, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--stages", help="comma-separated stage names; default all")
    args = parser.parse_args()
    jobs = stages()
    if args.stages:
        requested = args.stages.split(",")
        if len(set(requested)) != len(requested) or set(requested) - {job["name"] for job in jobs}:
            parser.error("stage names must be known and unique")
        jobs = [job for job in jobs if job["name"] in requested]
    if args.dry_run:
        print(json.dumps({"stages": jobs,
                          "planned_generations": sum(job["planned_generations"] for job in jobs)}, indent=2))
        return
    if args.detach:
        CONTROL.mkdir(parents=True, exist_ok=True)
        command = [PY, str(Path(__file__).resolve())]
        if args.stages:
            command += ["--stages", args.stages]
        if sys.platform == "darwin":
            command = ["/usr/bin/caffeinate", "-i", *command]
        with (CONTROL / "run.log").open("a") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"launcher_pid": process.pid, "log": str(CONTROL / "run.log"),
                          "status": str(CONTROL / "status.json")}), flush=True)
        return
    execute(jobs)


if __name__ == "__main__":
    raise SystemExit("This historical queue is retired. Use scripts/reproduce.py plan --study all, then run --study ID --run-dir FRESH_PATH. Original source is retained in reproducibility/legacy/.")
