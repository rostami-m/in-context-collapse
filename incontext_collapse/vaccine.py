"""Exploratory natural-label composite-adapter transfer runner.

This runner trains with objective="natural", then evaluates Shapes4, CIFAR4,
and Fashion4 under remapped labels. These are domain transfers
within the same four-class image-to-label format, not demonstrated transfer to
general task families. Defaults are connector + early + middle regions, rank 16,
and 250 training steps; the selected regions may be changed by CLI arguments.

The final controlled paper recipe instead uses circuit.py with an episodic-remap
objective, a rank-8 connector, and 150 steps, followed by saved-checkpoint
evaluation in option_order_eval.py. See README.md and
docs/CURRENT_EVIDENCE_GUIDE_2026_10_09.txt for exact artifact/recipe mappings.

Current evaluation options default to randomized order. Historical composite
results retain their recorded objective, mapping, options, and heterogeneous
capacities; current defaults do not retroactively reproduce those protocols.
Use a new output identity for any future experiment and preserve historical
result files. No positive transfer or robustness outcome is assumed by this CLI.
"""
from __future__ import annotations

import argparse
import gc
import json
import os

import torch

from .core import TASKS, region_targets
from .circuit import eval_kcurve, train_region
from .models import load_model
from .protocol import ProtocolConfig
from .results import load_result, write_json_atomic
from .provenance import model_revision, software_versions


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--train-task", default="shapes4")
    ap.add_argument("--eval-tasks", default="shapes4,cifar4,fashion4")
    ap.add_argument("--regions", default="connector,early,mid", help="vaccine locus; use 'late' for the readout control")
    ap.add_argument("--ks", default="0,1,2,4,8,16")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--score-mode", choices=["generation", "likelihood", "both"],
                    default="both")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--option-order", choices=["random", "class"], default="random")
    ap.add_argument("--out", default="data/figdata/vaccine_v2.json")
    a = ap.parse_args()
    ks = [int(x) for x in a.ks.split(",")]
    seeds = [int(s) for s in a.seeds.split(",")]
    evs = a.eval_tasks.split(",")
    for values, label in ((ks, "K grid"), (seeds, "seeds"), (evs, "evaluation tasks")):
        if not values or len(set(values)) != len(values):
            ap.error(f"{label} must be nonempty and unique")
    if min(ks) < 0 or min(a.n, a.steps, a.rank) < 1 or a.n > 120:
        ap.error("invalid K, evaluation count, steps or rank")
    if a.train_task not in TASKS or any(t not in TASKS for t in evs):
        ap.error("unknown task")
    regions = a.regions.split(",")
    if len(set(regions)) != len(regions) or any(r not in {"connector", "early", "mid", "late", "all"} for r in regions):
        ap.error("regions must be unique valid adaptation regions")
    protocol = ProtocolConfig(option_order=a.option_order)
    run_config = {
        "schema_version": 3, "model": a.model, "model_revision": model_revision(a.model),
        "software_versions": software_versions(), "dtype": "bfloat16",
        "img_size": 112, "train_k": 8, "learning_rate": 1e-4, "train_task": a.train_task,
        "eval_tasks": evs, "regions": a.regions, "ks": ks, "n": a.n,
        "steps": a.steps, "rank": a.rank, "score_mode": a.score_mode,
        "seeds": seeds, "objective": "natural", "protocol": protocol.metadata(),
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    R = load_result(a.out, run_config, {
        "model": a.model, "regions": a.regions, "ks": ks, "n": a.n, "seeds_done": [],
        "objective": "natural", "protocol": protocol.metadata(),
        "score_mode": a.score_mode,
        "baseline": {t: {str(k): [] for k in ks} for t in evs},
        "vaccine": {t: {str(k): [] for k in ks} for t in evs}})

    for s in seeds:
        if s in R["seeds_done"]:
            continue
        model, proc = load_model(a.model, dtype=torch.bfloat16)
        # Pre-vaccine (baseline) transfer curves.
        for t in evs:
            names, pool, test = TASKS[t](96, 120)
            cur, details = eval_kcurve(
                model, proc, names, pool, test, ks, True, a.n, seed=s,
                protocol=protocol, score_mode=a.score_mode, return_details=True)
            R.setdefault("evidence", {}).setdefault("baseline", {}).setdefault(t, {})[str(s)] = details
            for k in ks:
                R["baseline"][t][str(k)].append(cur[k])
        # Train the vaccine on the held-out train task at the chosen locus.
        tn, tp, tt = TASKS[a.train_task](96, 120)
        targets = sorted({target for r in regions for target in region_targets(model, r)})
        pm = train_region(model, proc, tn, tp, tt, targets, "remove",
                          steps=a.steps, rank=a.rank, seed=s, objective="natural",
                          option_order=protocol.option_order)
        R.setdefault("training_runs", []).append(pm.revision_training_metadata)
        # Post-vaccine (transfer) curves.
        for t in evs:
            names, pool, test = TASKS[t](96, 120)
            cur, details = eval_kcurve(
                pm, proc, names, pool, test, ks, True, a.n, seed=s,
                protocol=protocol, score_mode=a.score_mode, return_details=True)
            R.setdefault("evidence", {}).setdefault("vaccine", {}).setdefault(t, {})[str(s)] = details
            for k in ks:
                R["vaccine"][t][str(k)].append(cur[k])
        R["seeds_done"].append(s)
        write_json_atomic(a.out, R)
        print("seed", s, "done", flush=True)
        del pm, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
