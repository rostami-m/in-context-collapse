"""Evaluate saved revision adapters without retraining them.

This driver is deliberately separate from ``circuit`` so additional controls and
scorers can be applied to the exact same checkpoint after training.
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch
from peft import PeftModel

from .core import TASKS, build_messages, pick_device, predict
from .evaluation import evaluate_classification_cell, mean_t_ci
from .models import load_model
from .option_order_eval import checkpoint_manifest, validate_checkpoint_models
from .protocol import validate_evaluation_grid, validate_named_axis
from .protocol import ProtocolConfig
from .provenance import model_revision, software_versions
from .results import load_result, write_json_atomic


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapters-dir")
    ap.add_argument("--adapter-key", default="connector_both")
    ap.add_argument("--frozen", action="store_true",
                    help="evaluate the frozen base model instead of a saved adapter")
    ap.add_argument("--task", default="shapes4")
    ap.add_argument("--conditions", default="standard,shuffled,remap")
    ap.add_argument("--ks", default="0,1,4,8,16")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--img-size", type=int, default=112)
    ap.add_argument("--score-mode", choices=["generation", "likelihood", "both"],
                    default="both")
    ap.add_argument("--option-order", choices=["random", "class"], default="random",
                    help="independent displayed-option permutation; class reproduces Protocol 2.4")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.frozen and a.adapters_dir:
        ap.error("--frozen and --adapters-dir are mutually exclusive")
    if not a.frozen and not a.adapters_dir:
        ap.error("--adapters-dir is required unless --frozen is set")

    conditions = a.conditions.split(",")
    validate_named_axis(conditions, "conditions", {"standard", "shuffled", "remap", "remap_shuffled"})
    ks = [int(value) for value in a.ks.split(",")]
    seeds = [int(value) for value in a.seeds.split(",")]
    validate_evaluation_grid(seeds, ks, a.n, min_k=0)
    protocol = ProtocolConfig(option_order=a.option_order)
    checkpoints = {} if a.frozen else checkpoint_manifest(
        {a.adapter_key: Path(a.adapters_dir) / a.adapter_key}, seeds)
    validate_checkpoint_models(checkpoints, a.model)
    run_config = {
        "schema_version": 3,
        "checkpoints": checkpoints,
        "generation": {"max_new_tokens": 8, "do_sample": False,
                       "num_beams": 1, "num_return_sequences": 1},
        "model": a.model, "adapters_dir": a.adapters_dir,
        "adapter_key": None if a.frozen else a.adapter_key,
        "evaluation_arm": "frozen" if a.frozen else "repaired",
        "task": a.task, "conditions": conditions, "ks": ks, "n": a.n,
        "seeds": seeds, "img_size": a.img_size, "score_mode": a.score_mode,
        "protocol": protocol.metadata(),
        "model_revision": model_revision(a.model),
        "dtype": "bfloat16",
        "software_versions": software_versions(),
    }
    result = load_result(a.out, run_config, {
        "seeds_done": [], "per_seed": {condition: {str(k): [] for k in ks}
                                        for condition in conditions},
        "evidence": {},
    })
    names, pool, test = TASKS[a.task](96, 120)
    device = pick_device()

    for seed in seeds:
        if seed in result["seeds_done"]:
            continue
        base, processor = load_model(a.model, device=device, dtype=torch.bfloat16)
        if a.frozen:
            model = base
        else:
            adapter_path = Path(a.adapters_dir) / a.adapter_key / f"seed_{seed}"
            if not adapter_path.exists():
                raise FileNotFoundError(f"missing adapter checkpoint: {adapter_path}")
            model = PeftModel.from_pretrained(base, adapter_path).eval()
        seed_evidence = {}
        for condition in conditions:
            condition_evidence = {}
            for k in ks:
                cell = evaluate_classification_cell(
                    model, processor, names, pool, test, condition, k, a.n,
                    a.img_size, seed, build_messages, predict, protocol=protocol,
                    score_mode=a.score_mode)
                condition_evidence[str(k)] = cell
                result["per_seed"][condition][str(k)].append(
                    cell["summary"]["accuracy"])
            seed_evidence[condition] = condition_evidence
        result["evidence"][str(seed)] = seed_evidence
        result["seeds_done"].append(seed)
        write_json_atomic(a.out, result)
        print(f"seed {seed} done", flush=True)
        del model, base, processor
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

    result["summary"] = {
        condition: {str(k): mean_t_ci(result["per_seed"][condition][str(k)])
                    for k in ks}
        for condition in conditions
    }
    write_json_atomic(a.out, result)
    print("DONE", result["summary"], flush=True)


if __name__ == "__main__":
    main()
