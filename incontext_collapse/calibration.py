"""Training-free contextual-calibration baseline for multimodal ICL.

For each query, this evaluator computes token-normalized restricted-label log
scores twice: once with the real query image and once with a neutral gray query
image while preserving the demonstrations, label map, and prompt wording. The
content-free scores are subtracted in log space before choosing a label.
"""
from __future__ import annotations

import argparse
import random

from PIL import Image
import torch

from .core import TASKS, build_messages
from .evaluation import mean_t_ci
from .models import load_model
from .protocol import validate_evaluation_grid, validate_named_axis
from .protocol import (ProtocolConfig, corrupt_demo_labels, sample_balanced_demos,
                       image_id, sample_balanced_queries, sample_label_space, sample_option_order)
from .provenance import model_revision, software_versions
from .results import load_result, write_json_atomic
from .scoring import restricted_label_scores, subtract_contextual_bias


def evaluate_cell(model, processor, names, pool, test, condition, k, n, size, seed,
                  protocol):
    if condition not in {"standard", "shuffled", "remap", "remap_shuffled"}:
        raise ValueError(f"unknown condition: {condition}")
    rng = random.Random(1000 * seed + k)
    items = sample_balanced_queries(test, n, rng)
    neutral = Image.new("RGB", (size, size), (127, 127, 127))
    records = []
    for query_index, (query_image, gold) in enumerate(items):
        source_demos = sample_balanced_demos(
            pool, k, rng, class_policy=protocol.demo_class_policy)
        shown_demos = list(source_demos)
        shuffle_map = None
        if condition in {"shuffled", "remap_shuffled"} and shown_demos:
            shown_demos, shuffle_map = corrupt_demo_labels(
                shown_demos, range(len(names)), rng)
        remapped = condition in {"remap", "remap_shuffled"}
        label_space = sample_label_space(
            names, "remap" if remapped else "standard", rng,
            mapping_mode=protocol.mapping_mode, token_set=protocol.remap_token_set)
        option_order = sample_option_order(len(names), seed, k, query_index,
                                           protocol.option_order)
        query_messages = build_messages(shown_demos, query_image, label_space, size,
                                        option_order=option_order)
        neutral_messages = build_messages(shown_demos, neutral, label_space, size,
                                          option_order=option_order)
        query = restricted_label_scores(model, processor, query_messages, label_space)
        content_free = restricted_label_scores(
            model, processor, neutral_messages, label_space)
        calibrated = subtract_contextual_bias(query["mean"], content_free["mean"])
        raw_prediction = max(range(len(label_space)), key=query["mean"].__getitem__)
        prediction = max(range(len(label_space)), key=calibrated.__getitem__)
        records.append({
            "query_index": query_index,
            "query_id": image_id(query_image),
            "gold": gold,
            "condition": condition,
            "k": k,
            "label_space": label_space,
            "option_order": option_order,
            "displayed_options": [label_space[index] for index in option_order],
            "demo_source_classes": [c for _, c in source_demos],
            "demo_source_ids": [image_id(image) for image, _ in source_demos],
            "demo_label_classes": [c for _, c in shown_demos],
            "shuffle_map": shuffle_map,
            "raw_prediction": raw_prediction,
            "prediction": prediction,
            "raw_correct": raw_prediction == gold,
            "correct": prediction == gold,
            "query_mean_scores": query["mean"],
            "content_free_mean_scores": content_free["mean"],
            "calibrated_scores": calibrated,
            "token_counts": query["token_counts"],
            "follows_corrupted_mapping": (
                shuffle_map is not None and prediction == shuffle_map[gold]),
        })
    count = len(records)
    return {
        "condition": condition,
        "k": k,
        "seed": seed,
        "protocol": protocol.metadata(),
        "summary": {
            "n": count,
            "raw_accuracy": sum(r["raw_correct"] for r in records) / count,
            "calibrated_accuracy": sum(r["correct"] for r in records) / count,
            "follows_corrupted_mapping_rate": sum(
                r["follows_corrupted_mapping"] for r in records) / count,
        },
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", default="shapes4")
    parser.add_argument("--conditions", default="remap,remap_shuffled")
    parser.add_argument("--ks", default="4,8,16")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--img-size", type=int, default=112)
    parser.add_argument("--option-order", choices=["random", "class"], default="random")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    conditions = args.conditions.split(",")
    validate_named_axis(conditions, "conditions", {"standard", "shuffled", "remap", "remap_shuffled"})
    ks = [int(value) for value in args.ks.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]
    validate_evaluation_grid(seeds, ks, args.n, min_k=0)
    protocol = ProtocolConfig(option_order=args.option_order)
    run_config = {
        "schema_version": 2,
        "baseline": "contextual_calibration_neutral_query",
        "model": args.model,
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
    }
    result = load_result(args.out, run_config, {
        "seeds_done": [], "cells_done": [], "evidence": {}})
    names, pool, test = TASKS[args.task](96, 120)
    model, processor = load_model(args.model, dtype=torch.bfloat16)
    for seed in seeds:
        if seed in result["seeds_done"]:
            continue
        seed_evidence = result["evidence"].setdefault(str(seed), {})
        for condition in conditions:
            condition_evidence = seed_evidence.setdefault(condition, {})
            for k in ks:
                cell_key = f"{seed}:{condition}:{k}"
                if cell_key in result["cells_done"]:
                    continue
                condition_evidence[str(k)] = evaluate_cell(
                    model, processor, names, pool, test, condition, k, args.n,
                    args.img_size, seed, protocol)
                result["cells_done"].append(cell_key)
                write_json_atomic(args.out, result)
                print(f"cell {cell_key} done", flush=True)
        result["seeds_done"].append(seed)
        write_json_atomic(args.out, result)
        print(f"seed {seed} done", flush=True)

    result["summary"] = {}
    for condition in conditions:
        result["summary"][condition] = {}
        for k in ks:
            cells = [result["evidence"][str(seed)][condition][str(k)]["summary"]
                     for seed in seeds]
            result["summary"][condition][str(k)] = {
                metric: mean_t_ci([cell[metric] for cell in cells])
                for metric in ("raw_accuracy", "calibrated_accuracy",
                               "follows_corrupted_mapping_rate")
            }
    write_json_atomic(args.out, result)
    print("DONE", result["summary"], flush=True)


if __name__ == "__main__":
    main()
