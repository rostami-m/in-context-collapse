"""Classification behavior along robustness and mapping-learning axes.

Sweeps a panel of VLMs over tasks, conditions, shot counts ``K``, and seeds,
writing one JSON per (model, task, condition, K, seed) cell so the run is fully
resumable and safe for a long local job. The per-cell files in ``--cells-dir``
can be summarized without rerunning the models. Historical manuscript figures
retain their original artifacts and are not reconstructed by this default grid.

For each cell we record balanced accuracy, the copy rate (fraction of queries
answered with the most-recent demonstration's label; a descriptive agreement
rate, not by itself evidence of a copying mechanism), and the count of unparseable outputs.

Conditions:
  standard  natural labels (accuracy may come from prior or demonstrations)
  shuffled  demo labels permuted within the prompt (content-vs-format control)
  remap     arbitrary labels (mapping sensitivity requires matched controls)
  remap_shuffled arbitrary labels with a deliberately wrong demonstration mapping

Run from scratch, e.g.::

    python -m incontext_collapse.panel \\
        --models Qwen/Qwen2-VL-2B-Instruct,Qwen/Qwen2-VL-7B-Instruct \\
        --tasks cifar4,shapes4,fashion4 --k-grid 0,1,2,4,8,16 --seeds 3

Or a fast end-to-end smoke test on one small model::

    python -m incontext_collapse.panel --quick
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from .core import TASKS, build_messages, predict, pick_device
from .evaluation import evaluate_classification_cell, mean_t_ci, summarize_records
from .models import load_model
from .protocol import ProtocolConfig
from .provenance import model_revision, software_versions
from .results import ensure_run_config, write_json_atomic

# Exploratory default panel (0.5B-11B, several connectors); not a paper replay.
# MLP-projector (Qwen2-VL / Qwen2.5-VL / LLaVA families), pixel-shuffle (SmolVLM),
# and cross-attention (Llama-3.2-Vision) connectors are all represented.
DEFAULT_MODELS = [
    "Qwen/Qwen2-VL-2B-Instruct", "Qwen/Qwen2-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-3B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct",
    "llava-hf/llava-1.5-7b-hf", "llava-hf/llava-v1.6-mistral-7b-hf",
    "llava-hf/llava-onevision-qwen2-0.5b-ov-hf", "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    "HuggingFaceTB/SmolVLM-Instruct", "HuggingFaceTB/SmolVLM-500M-Instruct",
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
]
# Slow image-tiling / large models: cap K at 8 to keep runtime bounded.
_HEAVY = ["7b-ov", "7B-Instruct", "mistral-7b", "1.5-7b", "11B-Vision"]


def run_cell(model, processor, cond, k, seed, names, pool, test, n_test, size,
             protocol=None, score_mode="both"):
    """Evaluate one cell while retaining auditable per-query predictions."""
    return evaluate_classification_cell(
        model, processor, names, pool, test, cond, k, n_test, size, seed,
        build_messages, predict, protocol=protocol, score_mode=score_mode)


def load_planned_cells(directory, plan, filename, config, *, require_complete):
    """Validate exact cell identities and recomputed summaries before resume/aggregation."""
    expected = {filename(*identity): identity for identity in plan}
    found = {p.name for p in Path(directory).glob("*.json") if not p.name.startswith("_")}
    extra = found - expected.keys()
    missing = expected.keys() - found
    if extra or (require_complete and missing):
        raise ValueError(f"panel cell inventory mismatch: extra={sorted(extra)}, missing={sorted(missing)}")
    cells = []
    for name in sorted(found):
        cell = json.loads((Path(directory) / name).read_text())
        model, task, condition, k, seed = expected[name]
        actual = tuple(cell.get(key) for key in ("model", "task", "cond", "K", "seed"))
        if actual != (model, task, condition, k, seed):
            raise ValueError(f"panel cell identity mismatch: {name}")
        if cell.get("condition") != condition or cell.get("k") != k:
            raise ValueError(f"panel cell condition/K mismatch: {name}")
        if cell.get("protocol") != config["protocol"] or cell.get("score_mode") != config["score_mode"]:
            raise ValueError(f"panel cell protocol mismatch: {name}")
        records = cell.get("records", [])
        if len(records) != config["n_test"] or cell.get("summary") != summarize_records(records):
            raise ValueError(f"panel cell count/summary mismatch: {name}")
        cells.append(cell)
    return cells


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--tasks", default="cifar4,shapes4,fashion4")
    ap.add_argument("--conditions", default="standard,shuffled,remap")
    ap.add_argument("--k-grid", default="0,1,2,4,8,16")
    ap.add_argument("--n-train", type=int, default=64)
    ap.add_argument("--n-test", type=int, default=25)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--img-size", type=int, default=112)
    ap.add_argument("--demo-class-policy", choices=["random_balanced", "legacy_cycle"],
                    default="random_balanced")
    ap.add_argument("--mapping-mode", choices=["episode", "fixed"], default="episode")
    ap.add_argument("--remap-token-set", choices=["eval", "train", "legacy"], default="eval")
    ap.add_argument("--option-order", choices=["random", "class"], default="random")
    ap.add_argument("--score-mode", choices=["generation", "likelihood", "both"], default="both")
    ap.add_argument(
        "--disable-heavy-k-cap", action="store_true",
        help="Run the full requested K grid for every model, including slower backbones")
    ap.add_argument("--cells-dir", default="data/panel_cells_v2")
    ap.add_argument("--out", default="data/figdata/panel_summary_v2.json")
    ap.add_argument("--quick", action="store_true", help="one small model, tiny grid: end-to-end smoke test")
    a = ap.parse_args()
    if a.quick:
        a.models = a.models.split(",")[0]
        a.tasks, a.k_grid, a.n_test, a.seeds = "shapes4", "0,2,8", 8, 1

    models = a.models.split(",")
    tasks = a.tasks.split(",")
    conds = a.conditions.split(",")
    ks = [int(x) for x in a.k_grid.split(",")]
    kg = lambda m: ks if a.disable_heavy_k_cap else (
        [k for k in ks if k <= 8] if any(h in m for h in _HEAVY) else ks)
    for values, label in ((models, "models"), (tasks, "tasks"), (conds, "conditions"), (ks, "K grid")):
        if not values or len(set(values)) != len(values):
            ap.error(f"{label} must be nonempty and unique")
    if len({m.split('/')[-1] for m in models}) != len(models):
        ap.error("model basenames must be unique for cell filenames")
    if any(t not in TASKS for t in tasks) or any(c not in {"standard", "shuffled", "remap", "remap_shuffled"} for c in conds):
        ap.error("unknown task or condition")
    if min(ks) < 0 or min(a.n_train, a.n_test, a.seeds, a.img_size) < 1:
        ap.error("K must be nonnegative and sample counts, seeds and image size positive")
    if any(not kg(model) for model in models):
        ap.error("heavy-model K cap would leave an empty grid; request K<=8 or disable the cap")
    cdir = Path(a.cells_dir)
    if Path(a.out).resolve().is_relative_to(cdir.resolve()):
        ap.error("summary output must lie outside the cell directory")
    cdir.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    protocol = ProtocolConfig(a.demo_class_policy, a.mapping_mode, a.remap_token_set,
                              a.option_order)
    run_config = {
        "schema_version": 3, "model_loading": "exact_requested_model_no_fallback",
        "models": models, "tasks": tasks, "conditions": conds, "k_grid": ks,
        "n_train": a.n_train, "n_test": a.n_test, "seeds": a.seeds,
        "img_size": a.img_size, "protocol": protocol.metadata(),
        "score_mode": a.score_mode,
        "model_revisions": {model: model_revision(model) for model in models},
        "dtype": str(torch.float32 if device == "cpu" else torch.float16),
        "software_versions": software_versions(),
        "disable_heavy_k_cap": a.disable_heavy_k_cap,
        "cells_dir": str(Path(a.cells_dir)),
    }
    ensure_run_config(cdir / "_run_config.json", run_config)

    cid = lambda m, t, c, k, s: f"{m.split('/')[-1]}__{t}__{c}__K{k}__s{s}.json"
    plan = [(m, t, c, k, s) for m in models for t in tasks for c in conds for k in kg(m) for s in range(a.seeds)]
    load_planned_cells(cdir, plan, cid, run_config, require_complete=False)
    todo_models = [m for m in models if any(not (cdir / cid(m, t, c, k, s)).exists()
                                            for t in tasks for c in conds for k in kg(m) for s in range(a.seeds))]
    print(f"device={device} | {len(plan)} cells | {sum((cdir / cid(*p)).exists() for p in plan)} already done")

    for m0 in todo_models:
        # A missing/gated model is an explicit failure, never another panel member.
        model, proc = load_model(m0, device=device)
        m = m0
        print(f"\n=== {m} ===", flush=True)
        for t in tasks:
            names, pool, test = TASKS[t](a.n_train, a.n_test)
            for c in conds:
                for k in kg(m):
                    for s in range(a.seeds):
                        f = cdir / cid(m, t, c, k, s)
                        if f.exists():
                            continue
                        r = run_cell(model, proc, c, k, s, names, pool, test, a.n_test,
                                     a.img_size, protocol, a.score_mode)
                        r.update(model=m, task=t, cond=c, K=k, seed=s)
                        write_json_atomic(f, r)
                        summary = r["summary"]
                        print(f"{m.split('/')[-1]:22s} {t:8s} {c:8s} K={k:<3d} s={s} "
                              f"gen={summary['generation_accuracy']} "
                              f"ll={summary['likelihood_accuracy']} "
                              f"ll_sum={summary['likelihood_sum_accuracy']} "
                              f"copy={summary['copy_last_rate']:.3f}", flush=True)
        del model, proc
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

    # Aggregate cells -> per-condition mean / sd over seeds.
    cells = load_planned_cells(cdir, plan, cid, run_config, require_complete=True)
    agg: dict[str, list] = {}
    for cell in cells:
        key = f"{cell['model'].split('/')[-1]}|{cell['task']}|{cell['cond']}|K{cell['K']}"
        agg.setdefault(key, []).append(cell)
    summary = {}
    for key, v in sorted(agg.items()):
        accuracies = [x["summary"]["accuracy"] for x in v]
        generation_accuracies = [x["summary"]["generation_accuracy"] for x in v
                                 if x["summary"]["generation_accuracy"] is not None]
        likelihood_accuracies = [x["summary"]["likelihood_accuracy"] for x in v
                                 if x["summary"]["likelihood_accuracy"] is not None]
        likelihood_sum_accuracies = [x["summary"]["likelihood_sum_accuracy"] for x in v
                                     if x["summary"]["likelihood_sum_accuracy"] is not None]
        copy_last = [x["summary"]["copy_last_rate"] for x in v]
        unparseable = [x["summary"]["unparseable_rate"] for x in v]
        summary[key] = {
            "accuracy": mean_t_ci(accuracies),
            "generation_accuracy": mean_t_ci(generation_accuracies),
            "likelihood_accuracy": mean_t_ci(likelihood_accuracies),
            "likelihood_sum_accuracy": mean_t_ci(likelihood_sum_accuracies),
            "copy_last": mean_t_ci(copy_last),
            "unparseable": mean_t_ci(unparseable),
            "seed_values": {"accuracy": accuracies,
                            "generation_accuracy": generation_accuracies,
                            "likelihood_accuracy": likelihood_accuracies,
                            "likelihood_sum_accuracy": likelihood_sum_accuracies,
                            "copy_last": copy_last,
                            "unparseable": unparseable},
        }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(a.out, {"run_config": run_config, "cells": summary})
    print(f"\nsaved {a.out} ({len(summary)} aggregated cells)")


if __name__ == "__main__":
    main()
