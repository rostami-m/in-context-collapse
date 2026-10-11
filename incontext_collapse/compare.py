"""Matched exploratory fixed-map comparison of Vaccine, Inject and a shot cap.

All arms share balanced held-out queries, a fixed Greek class-to-label map,
displayed options, prompt template, image size, parser and decoding budget.
Many-shot and Vaccine receive identical full demonstration blocks. Zero-shot
and Inject receive no raw demonstrations; the fixed-shot cap receives their
exact prefix plus the vector. This is NOT the proposed copy-rate Gate policy.

Inject averages raw last-prefill states from the first up-to-eight evaluation
queries with full support blocks, without using query labels. Its fixed-map,
transductive vector includes the full support, not just the omitted prefix.
This diagnostic is not the primary episode-randomized REMAP evaluation, and
does not by itself establish repair or amortization. Vaccine can be freshly
trained on a different task or loaded from byte-hashed saved checkpoints.
Schema-v3 outputs retain per-query evidence and cannot resume older results.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import random

import torch
from peft import PeftModel

from .core import TASKS, build_messages, region_targets
from .circuit import train_region
from .evaluation import mean_t_ci, summarize_records
from .inject import _generate, get_layers, locus_idx
from .models import load_model
from .option_order_eval import checkpoint_manifest
from .protocol import (ProtocolConfig, image_id, sample_balanced_demos,
                       sample_balanced_queries, sample_label_space,
                       sample_option_order)
from .provenance import model_revision, software_versions
from .results import load_result, write_json_atomic
from .scoring import parse_answer, shortcut_diagnostics

METHODS = ("zeroshot", "manyshot", "inject", "gate", "vaccine")


def comparison_episodes(names, pool, test, *, k, n, seed, protocol):
    """Materialize one common query/support plan before evaluating any arm."""
    if protocol.mapping_mode != "fixed":
        raise ValueError("a shared task vector requires this diagnostic's fixed mapping")
    rng = random.Random(1000 * seed + k)
    items = sample_balanced_queries(test, n, rng)
    pool_ids = {image_id(image) for images in pool.values() for image in images}
    episodes = []
    for index, (query, gold) in enumerate(items):
        query_id = image_id(query)
        if query_id in pool_ids:
            raise ValueError("held-out query occurs in the demonstration pool")
        demos = sample_balanced_demos(pool, k, rng, protocol.demo_class_policy)
        labels = sample_label_space(names, "remap", rng,
                                    protocol.mapping_mode, protocol.remap_token_set)
        order = sample_option_order(len(names), seed, k, index, protocol.option_order)
        episodes.append({
            "query_image": query, "demos": demos,
            "evidence": {
                "query_index": index, "query_id": query_id, "gold": gold,
                "label_space": labels, "option_order": order,
                "displayed_options": [labels[c] for c in order],
                "support_ids": [image_id(image) for image, _ in demos],
                "support_classes": [c for _, c in demos],
            },
        })
    return episodes


def episode_messages(episode, demos, size):
    info = episode["evidence"]
    return build_messages(demos, episode["query_image"], info["label_space"], size,
                          option_order=info["option_order"])


def extract_matched_vector(model, proc, episodes, layer, *, size, max_new_tokens, n=8):
    """Average full-support states; missing extraction is an error, never a no-op."""
    vectors, sources = [], []
    for episode in episodes[:n]:
        messages = episode_messages(episode, episode["demos"], size)
        raw, vector = _generate(model, proc, messages, hook_layer=layer, grab=True,
                                max_new_tokens=max_new_tokens)
        if vector is None or not torch.isfinite(vector).all():
            raise RuntimeError("task-vector extraction produced no finite prefill state")
        vectors.append(vector)
        sources.append({**episode["evidence"], "raw_output": raw})
    if not vectors:
        raise ValueError("task-vector extraction requires at least one query")
    vector = torch.stack(vectors).mean(0)[0]
    return vector, {
        "estimator": "mean_raw_last_prefill_state", "transductive": True,
        "query_labels_used": False, "support": "full_matched_demo_block",
        "empty_prompt_subtraction": False, "gain": 1.0, "sources": sources,
        "vector_float32": vector.detach().float().cpu().tolist(),
    }


def evaluate_method(model, proc, episodes, method, *, size, max_new_tokens,
                    layer=None, vector=None, gate=2):
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    inject = method in {"inject", "gate"}
    if inject and (layer is None or vector is None):
        raise ValueError("Inject and the fixed shot cap require a layer and vector")
    records = []
    for episode in episodes:
        if method in {"manyshot", "vaccine"}:
            demos = episode["demos"]
        elif method == "gate":
            demos = episode["demos"][:gate]
        else:
            demos = []
        raw = _generate(
            model, proc, episode_messages(episode, demos, size),
            hook_layer=layer if inject else None, add_vec=vector if inject else None,
            max_new_tokens=max_new_tokens)
        info = episode["evidence"]
        parsed = parse_answer(raw, info["label_space"])
        shown = [c for _, c in demos]
        records.append({
            **info, "condition": "remap", "method": method,
            "k": len(demos), "support_k": len(episode["demos"]),
            "img_size": size, "max_new_tokens": max_new_tokens,
            "prompt_format": "single", "raw_output": raw,
            "prediction": parsed.prediction, "generation_prediction": parsed.prediction,
            "correct": parsed.prediction == info["gold"],
            "demo_source_ids": [image_id(image) for image, _ in demos],
            "demo_source_classes": shown, "demo_label_classes": shown,
            "shuffle_map": None, "parse_status": parsed.status,
            "parse_matches": list(parsed.matched_classes),
            "unparseable": parsed.prediction is None,
            "likelihood_prediction": None, "likelihood_sum_prediction": None,
            "likelihood_scores": None, "likelihood_mean_scores": None,
            "likelihood_token_counts": None, "follows_corrupted_mapping": False,
            "generation_follows_corrupted_mapping": False,
            "likelihood_follows_corrupted_mapping": False,
            **shortcut_diagnostics(parsed.prediction, shown),
        })
    return {"method": method, "score_mode": "generation",
            "summary": summarize_records(records), "records": records}


def saved_adapter_config(root, seeds, model, train_task, eval_task):
    """Require recorded task provenance instead of labeling unknown adapters transfer."""
    revision = model_revision(model)
    if not revision:
        raise ValueError("comparison requires a pinned model revision")
    checkpoints = checkpoint_manifest({"vaccine": Path(root)}, seeds)["vaccine"]
    for checkpoint in checkpoints.values():
        if checkpoint["base_model_name_or_path"] != model:
            raise ValueError("saved adapter base model does not match --model")
        source = checkpoint["training_source"]
        if source is None:
            raise ValueError("saved adapter requires its source-result training provenance")
        run = json.loads(Path(source["path"]).read_text())["run_config"]
        recorded_task = run.get("train_task", run.get("task"))
        if recorded_task != train_task or recorded_task == eval_task:
            raise ValueError("saved adapter training task must match --train-task and differ from evaluation")
        if run.get("model") != model or run.get("model_revision") != revision:
            raise ValueError("saved adapter model/revision differs from evaluation")
        if run.get("format") != "single":
            raise ValueError("saved adapter requires explicit single-turn training format metadata")
        checkpoint["training_run_config"] = run
    return {"mode": "saved_checkpoint", "checkpoints": checkpoints}


def source_hashes():
    directory = Path(__file__).parent
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("compare.py", "inject.py", "core.py", "protocol.py", "scoring.py",
                         "evaluation.py", "models.py", "provenance.py", "results.py",
                         "circuit.py", "option_order_eval.py")}


def summarize_cells(cells, seeds):
    return {
        method: mean_t_ci([cells[f"{seed}:{method}"]["summary"]["accuracy"]
                           for seed in seeds if f"{seed}:{method}" in cells])
        for method in METHODS
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--train-task", choices=TASKS, default="shapes4")
    ap.add_argument("--eval-task", choices=TASKS, default="cifar4")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--gate", type=int, default=2, help="fixed raw-demo prefix, not a copy-rate policy")
    ap.add_argument("--locus", choices=("early", "mid", "late"), default="mid")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--max-new-tokens", type=int, default=6)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--train-k", type=int, default=8)
    ap.add_argument("--regions", default="connector")
    ap.add_argument("--objective", choices=("episode_remap", "natural"), default="episode_remap")
    ap.add_argument("--adapter-root", help="saved seed_N adapter directories; no training performed")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--option-order", choices=("random", "class"), default="random")
    ap.add_argument("--out", default="data/figdata/compare_matched_fixedmap_v3.json")
    ap.add_argument("--preflight", action="store_true")
    a = ap.parse_args()
    revision = model_revision(a.model)
    if not revision:
        ap.error("comparison requires a pinned model revision")
    seeds = [int(seed) for seed in a.seeds.split(",")]
    if not seeds or len(set(seeds)) != len(seeds):
        ap.error("seeds must be nonempty and unique")
    if a.train_task == a.eval_task:
        ap.error("transfer evaluation requires different training and evaluation tasks")
    if not 0 < a.n <= 120 or a.n % 4:
        ap.error("n must be a positive multiple of four <=120")
    if min(a.k, a.gate, a.train_k) < 0 or min(a.size, a.max_new_tokens, a.rank, a.steps) < 1:
        ap.error("shot counts must be nonnegative; size, generation budget, rank and steps positive")
    if a.gate > a.k:
        ap.error("gate prefix cannot exceed K")
    protocol = ProtocolConfig(mapping_mode="fixed", remap_token_set="legacy",
                              option_order=a.option_order)
    training = (saved_adapter_config(a.adapter_root, seeds, a.model, a.train_task, a.eval_task)
                if a.adapter_root else {
                    "mode": "fresh_training", "task": a.train_task, "rank": a.rank,
                    "steps": a.steps, "regions": a.regions, "k": a.train_k,
                    "objective": a.objective, "learning_rate": 1e-4,
                    "img_size": a.size, "option_order": a.option_order,
                    "demo_class_policy": protocol.demo_class_policy,
                })
    config = {
        "schema_version": 3, "experiment": "exploratory_matched_fixedmap_comparison",
        "model": a.model, "model_revision": revision,
        "train_task": a.train_task, "eval_task": a.eval_task,
        "k": a.k, "gate": a.gate, "locus": a.locus, "n": a.n,
        "size": a.size, "max_new_tokens": a.max_new_tokens, "seeds": seeds,
        "protocol": protocol.metadata(), "prompt_format": "single",
        "score_mode": "generation", "dtype": "bfloat16", "training": training,
        "vector": {"transductive": True, "n_queries": min(8, a.n),
                   "estimator": "mean_raw_last_prefill_state", "gain": 1.0},
        "n_train_per_class": 96, "n_test_pool": 120,
        "software_versions": software_versions(), "source_sha256": source_hashes(),
    }
    result = load_result(a.out, config, {"cells": {}, "vectors": {}, "training_runs": {},
                                       "summary": {}, "seeds_done": []})
    remaining = [f"{seed}:{method}" for seed in seeds for method in METHODS
                 if f"{seed}:{method}" not in result["cells"]]
    print(json.dumps({"remaining_cells": len(remaining), "out": a.out,
                      "training_mode": training["mode"]}), flush=True)
    if a.preflight or not remaining:
        return
    names, pool, test = TASKS[a.eval_task](96, 120)
    for seed in seeds:
        pending = [method for method in METHODS if f"{seed}:{method}" in remaining]
        if not pending:
            continue
        episodes = comparison_episodes(names, pool, test, k=a.k, n=a.n, seed=seed,
                                       protocol=protocol)
        model, proc = load_model(a.model, dtype=torch.bfloat16)
        adapted = layer = layers = vector = active = None
        try:
            if any(method in pending for method in ("inject", "gate")):
                layers = get_layers(model)
                layer = layers[locus_idx(len(layers), a.locus)]
                vector, evidence = extract_matched_vector(
                    model, proc, episodes, layer, size=a.size, max_new_tokens=a.max_new_tokens)
                previous_vector = result["vectors"].get(str(seed))
                if previous_vector is not None and previous_vector != evidence:
                    raise RuntimeError(
                        "resumed task-vector extraction differs from saved evidence; "
                        "choose a new output path")
                result["vectors"][str(seed)] = evidence
            for method in pending:
                active = model
                if method == "vaccine":
                    if a.adapter_root:
                        checkpoint = training["checkpoints"][str(seed)]
                        adapted = PeftModel.from_pretrained(model, checkpoint["path"]).eval()
                        result["training_runs"][str(seed)] = checkpoint
                    else:
                        tn, tp, tt = TASKS[a.train_task](96, 120)
                        targets = sorted({target for region in a.regions.split(",")
                                          for target in region_targets(model, region)})
                        adapted = train_region(
                            model, proc, tn, tp, tt, targets, "remove", k=a.train_k,
                            size=a.size, steps=a.steps, rank=a.rank, seed=seed,
                            objective=a.objective, option_order=a.option_order,
                            demo_class_policy=protocol.demo_class_policy)
                        result["training_runs"][str(seed)] = adapted.revision_training_metadata
                    active = adapted
                cell = evaluate_method(
                    active, proc, episodes, method, size=a.size,
                    max_new_tokens=a.max_new_tokens, layer=layer, vector=vector, gate=a.gate)
                cell.update(seed=seed, protocol=protocol.metadata())
                result["cells"][f"{seed}:{method}"] = cell
                result["summary"] = summarize_cells(result["cells"], seeds)
                if all(f"{seed}:{arm}" in result["cells"] for arm in METHODS):
                    result["seeds_done"] = sorted(set(result["seeds_done"]) | {seed})
                write_json_atomic(a.out, result)
                print(f"seed {seed} {method}: {cell['summary']['accuracy']:.3f}", flush=True)
        finally:
            del active, adapted, model, proc, layer, layers, vector
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
    print("DONE", result["summary"], flush=True)


if __name__ == "__main__":
    main()
