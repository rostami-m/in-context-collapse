"""Paired prompt interventions for testing final-demonstration recency effects.

Each intervention keeps the query, demonstration images, and displayed-label
multiset fixed. ``order_swap`` moves a complete image/label pair into the final
position; ``label_swap`` leaves image order fixed and exchanges the displayed
labels of two demonstrations. Comparing predictions query by query is stronger
evidence about positional shortcuts than an unpaired copy-last correlation.
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import random

import torch
from peft import PeftModel

from .core import TASKS, build_messages, pick_device, predict
from .evaluation import mean_t_ci
from .models import load_model
from .option_order_eval import checkpoint_manifest, validate_checkpoint_models
from .protocol import validate_evaluation_grid, validate_named_axis
from .protocol import (ProtocolConfig, image_id, sample_balanced_demos,
                       sample_balanced_queries, sample_label_space, sample_option_order)
from .provenance import model_revision, software_versions
from .results import load_result, write_json_atomic
from .scoring import parse_answer


VARIANTS = ("original", "order_swap", "label_swap")
INTERVENTIONS = VARIANTS[1:]


def make_paired_interventions(demos: list[tuple], rng: random.Random) -> tuple[dict, dict]:
    """Construct invariant-preserving variants with a different final label."""
    if len(demos) < 2:
        raise ValueError("paired intervention requires at least two demonstrations")
    last_class = demos[-1][1]
    candidates = [i for i, (_, c) in enumerate(demos[:-1]) if c != last_class]
    if not candidates:
        raise ValueError("paired intervention requires two displayed classes")
    swap_index = rng.choice(candidates)

    order_swap = list(demos)
    order_swap[swap_index], order_swap[-1] = order_swap[-1], order_swap[swap_index]

    label_swap = list(demos)
    earlier_image, earlier_class = label_swap[swap_index]
    last_image, final_class = label_swap[-1]
    label_swap[swap_index] = (earlier_image, final_class)
    label_swap[-1] = (last_image, earlier_class)

    variants = {
        "original": list(demos),
        "order_swap": order_swap,
        "label_swap": label_swap,
    }
    metadata = {
        "swap_index": swap_index,
        "old_last_class": last_class,
        "new_last_class": earlier_class,
    }
    return variants, metadata


def _rate(records: list[dict], predicate) -> float:
    return sum(bool(predicate(record)) for record in records) / len(records) if records else 0.0


def summarize_paired_records(records: list[dict]) -> dict:
    """Summarize variant performance and paired effects with clear denominators."""
    variant_summary = {}
    for variant in VARIANTS:
        variant_summary[variant] = {
            "n": len(records),
            "accuracy": _rate(records, lambda r, v=variant: r["predictions"][v] == r["gold"]),
            "copy_last_rate": _rate(
                records, lambda r, v=variant: r["predictions"][v] == r["last_classes"][v]),
            "unparseable_rate": _rate(
                records, lambda r, v=variant: r["predictions"][v] is None),
        }

    effects = {}
    for intervention in INTERVENTIONS:
        comparable = [
            record for record in records
            if record["predictions"]["original"] is not None
            and record["predictions"][intervention] is not None
        ]
        changed = [
            record for record in comparable
            if record["predictions"]["original"] != record["predictions"][intervention]
        ]
        moved = [
            record for record in comparable
            if record["predictions"][intervention] == record["new_last_class"]
            and record["predictions"]["original"] != record["new_last_class"]
        ]
        gold_flips = [
            record for record in comparable
            if record["predictions"]["original"] == record["gold"]
            and record["predictions"][intervention] == record["new_last_class"]
            and record["new_last_class"] != record["gold"]
        ]
        effects[intervention] = {
            "n_total": len(records),
            "n_comparable": len(comparable),
            "n_changed": len(changed),
            "prediction_change_rate": len(changed) / len(comparable) if comparable else None,
            "moves_to_new_last_rate": len(moved) / len(comparable) if comparable else None,
            "moves_to_new_last_given_change": len(moved) / len(changed) if changed else None,
            "gold_to_new_last_flip_rate": len(gold_flips) / len(comparable) if comparable else None,
        }
    return {"variants": variant_summary, "effects": effects}


def evaluate_paired_cell(model, processor, names, pool, test, condition: str, k: int,
                         n: int, size: int, seed: int,
                         protocol: ProtocolConfig | None = None) -> dict:
    """Run all three prompt variants on the same sampled episodes."""
    if condition not in {"standard", "remap"}:
        raise ValueError("paired interventions support standard and remap conditions")
    protocol = protocol or ProtocolConfig()
    rng = random.Random(1000 * seed + k)
    items = sample_balanced_queries(test, n, rng)
    records = []
    for query_index, (query_image, gold) in enumerate(items):
        demos = sample_balanced_demos(
            pool, k, rng, class_policy=protocol.demo_class_policy)
        label_space = sample_label_space(
            names, condition, rng, mapping_mode=protocol.mapping_mode,
            token_set=protocol.remap_token_set)
        option_order = sample_option_order(len(names), seed, k, query_index,
                                           protocol.option_order)
        variants, intervention = make_paired_interventions(demos, rng)
        predictions = {}
        raw_outputs = {}
        parse_statuses = {}
        for variant, shown_demos in variants.items():
            raw = predict(model, processor, build_messages(
                shown_demos, query_image, label_space, size, option_order=option_order))
            parsed = parse_answer(raw, label_space)
            predictions[variant] = parsed.prediction
            raw_outputs[variant] = raw
            parse_statuses[variant] = parsed.status
        records.append({
            "query_index": query_index,
            "query_id": image_id(query_image),
            "gold": gold,
            "condition": condition,
            "k": k,
            "label_space": label_space,
            "option_order": option_order,
            "displayed_options": [label_space[index] for index in option_order],
            "demo_classes": [c for _, c in demos],
            "demo_ids": [image_id(image) for image, _ in demos],
            "predictions": predictions,
            "raw_outputs": raw_outputs,
            "parse_statuses": parse_statuses,
            "last_classes": {
                variant: shown_demos[-1][1] for variant, shown_demos in variants.items()
            },
            **intervention,
        })
    return {
        "protocol": protocol.metadata(),
        "condition": condition,
        "k": k,
        "seed": seed,
        "summary": summarize_paired_records(records),
        "records": records,
    }


def _metric_summary(per_seed: dict, conditions: list[str], ks: list[int]) -> dict:
    summary = {}
    for state, state_results in per_seed.items():
        summary[state] = {}
        for condition in conditions:
            summary[state][condition] = {}
            for k in ks:
                cells = state_results[condition][str(k)]
                metric_values = {}
                for cell in cells:
                    for variant, values in cell["variants"].items():
                        for metric in ("accuracy", "copy_last_rate", "unparseable_rate"):
                            metric_values.setdefault(f"{variant}.{metric}", []).append(values[metric])
                    for intervention, values in cell["effects"].items():
                        for metric in ("prediction_change_rate", "moves_to_new_last_rate",
                                       "moves_to_new_last_given_change",
                                       "gold_to_new_last_flip_rate"):
                            value = values[metric]
                            if value is not None:
                                metric_values.setdefault(f"{intervention}.{metric}", []).append(value)
                summary[state][condition][str(k)] = {
                    metric: mean_t_ci(values) for metric, values in metric_values.items()
                }
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapters-dir", required=True)
    ap.add_argument("--adapter-key", default="connector_both")
    ap.add_argument("--task", default="shapes4")
    ap.add_argument("--conditions", default="standard,remap")
    ap.add_argument("--ks", default="4,8,16")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--img-size", type=int, default=112)
    ap.add_argument("--option-order", choices=["random", "class"], default="random")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    conditions = args.conditions.split(",")
    validate_named_axis(conditions, "conditions", {"standard", "remap"})
    ks = [int(value) for value in args.ks.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]
    validate_evaluation_grid(seeds, ks, args.n, min_k=2)
    if any(k < 2 for k in ks):
        ap.error("all K values must be at least 2")
    protocol = ProtocolConfig(option_order=args.option_order)
    checkpoints = checkpoint_manifest(
        {args.adapter_key: Path(args.adapters_dir) / args.adapter_key}, seeds)
    validate_checkpoint_models(checkpoints, args.model)
    run_config = {
        "schema_version": 3,
        "checkpoints": checkpoints,
        "generation": {"max_new_tokens": 8, "do_sample": False,
                       "num_beams": 1, "num_return_sequences": 1},
        "model": args.model,
        "adapters_dir": args.adapters_dir,
        "adapter_key": args.adapter_key,
        "task": args.task,
        "conditions": conditions,
        "ks": ks,
        "n": args.n,
        "seeds": seeds,
        "img_size": args.img_size,
        "protocol": protocol.metadata(),
        "model_revision": model_revision(args.model),
        "dtype": "bfloat16",
        "software_versions": software_versions(),
        "interventions": {
            "order_swap": "swap complete final and earlier different-class pairs",
            "label_swap": "swap displayed labels only at the same two positions",
        },
    }
    initial_per_seed = {
        state: {condition: {str(k): [] for k in ks} for condition in conditions}
        for state in ("frozen", "repaired")
    }
    result = load_result(args.out, run_config, {
        "seeds_done": [], "cells_done": [], "per_seed": initial_per_seed,
        "evidence": {},
    })
    names, pool, test = TASKS[args.task](96, 120)
    device = pick_device()

    for seed in seeds:
        if seed in result["seeds_done"]:
            continue
        adapter_path = Path(args.adapters_dir) / args.adapter_key / f"seed_{seed}"
        if not adapter_path.exists():
            raise FileNotFoundError(f"missing adapter checkpoint: {adapter_path}")
        base, processor = load_model(args.model, device=device, dtype=torch.bfloat16)
        seed_evidence = result["evidence"].setdefault(
            str(seed), {"frozen": {}, "repaired": {}})
        for condition in conditions:
            seed_evidence["frozen"].setdefault(condition, {})
            for k in ks:
                cell_key = f"{seed}:frozen:{condition}:{k}"
                if cell_key in result["cells_done"]:
                    continue
                cell = evaluate_paired_cell(
                    base, processor, names, pool, test, condition, k, args.n,
                    args.img_size, seed, protocol)
                seed_evidence["frozen"][condition][str(k)] = cell
                result["per_seed"]["frozen"][condition][str(k)].append(cell["summary"])
                result["cells_done"].append(cell_key)
                result["summary"] = _metric_summary(result["per_seed"], conditions, ks)
                write_json_atomic(args.out, result)
                print(f"cell {cell_key} done", flush=True)

        model = PeftModel.from_pretrained(base, adapter_path).eval()
        for condition in conditions:
            seed_evidence["repaired"].setdefault(condition, {})
            for k in ks:
                cell_key = f"{seed}:repaired:{condition}:{k}"
                if cell_key in result["cells_done"]:
                    continue
                cell = evaluate_paired_cell(
                    model, processor, names, pool, test, condition, k, args.n,
                    args.img_size, seed, protocol)
                seed_evidence["repaired"][condition][str(k)] = cell
                result["per_seed"]["repaired"][condition][str(k)].append(cell["summary"])
                result["cells_done"].append(cell_key)
                result["summary"] = _metric_summary(result["per_seed"], conditions, ks)
                write_json_atomic(args.out, result)
                print(f"cell {cell_key} done", flush=True)
        result["seeds_done"].append(seed)
        result["summary"] = _metric_summary(result["per_seed"], conditions, ks)
        write_json_atomic(args.out, result)
        print(f"seed {seed} done", flush=True)
        del model, base, processor
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

    result["summary"] = _metric_summary(result["per_seed"], conditions, ks)
    write_json_atomic(args.out, result)
    print("DONE", result["summary"], flush=True)


if __name__ == "__main__":
    main()
