"""Paired option-order sensitivity and transfer evaluation of saved adapters.

No optimization is performed. Each frozen/checkpoint arm receives identical
queries, demonstrations, mappings, and option permutations. Completed cells are
persisted individually so interrupted evaluations can resume. Checkpoint byte
hashes and available source-result training protocols are recorded; changed
weights or source provenance make resume fail instead of combining checkpoints.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
from peft import PeftModel

from .core import TASKS, build_messages, pick_device, predict
from .evaluation import evaluate_classification_cell, mean_t_ci
from .models import load_model
from .protocol import validate_evaluation_grid
from .protocol import ProtocolConfig
from .provenance import model_revision, software_versions
from .results import load_result, write_json_atomic


EPISODE_FIELDS = ("query_index", "query_id", "gold", "label_space",
                  "demo_source_classes", "demo_source_ids", "demo_label_classes",
                  "shuffle_map")


def paired_option_effect(class_cell: dict, random_cell: dict) -> dict:
    """Validate paired episode identities before measuring output sensitivity."""
    left, right = class_cell["records"], random_cell["records"]
    if not left or len(left) != len(right):
        raise ValueError("paired option cells must contain equal nonzero query counts")
    for old, new in zip(left, right):
        for field in EPISODE_FIELDS:
            if old[field] != new[field]:
                raise ValueError(f"option-order comparison changed {field}")
    return {
        "n": len(left),
        "random_minus_class_accuracy": sum(
            int(new["correct"]) - int(old["correct"])
            for old, new in zip(left, right)) / len(left),
        "prediction_change_rate": sum(
            old["prediction"] != new["prediction"]
            for old, new in zip(left, right)) / len(left),
        "random_wins": sum(new["correct"] and not old["correct"]
                           for old, new in zip(left, right)),
        "class_wins": sum(old["correct"] and not new["correct"]
                          for old, new in zip(left, right)),
    }


def checkpoint_manifest(adapters: dict[str, Path], seeds: list[int]) -> dict:
    """Bind a resumed run to the actual saved checkpoint bytes, not just paths."""
    manifest = {}
    for name, root in adapters.items():
        manifest[name] = {}
        training_source = None
        source_directory = root.parent
        if source_directory.name.endswith("_adapters"):
            source_path = source_directory.with_name(
                source_directory.name.removesuffix("_adapters") + ".json")
            if source_path.is_file():
                source_bytes = source_path.read_bytes()
                source = json.loads(source_bytes)
                run = source.get("run_config", {})
                training_source = {
                    "path": str(source_path),
                    "sha256": hashlib.sha256(source_bytes).hexdigest(),
                    "protocol": run.get("protocol"),
                    "objective": run.get("objective"),
                    "steps": run.get("steps"), "rank": run.get("rank"),
                }
        for seed in seeds:
            directory = root / f"seed_{seed}"
            config_path = directory / "adapter_config.json"
            weights = sorted(directory.glob("adapter_model*.safetensors"))
            weights += sorted(directory.glob("adapter_model*.bin"))
            if not config_path.is_file() or not weights:
                raise FileNotFoundError(f"incomplete adapter checkpoint: {directory}")
            files = {}
            for path in [config_path, *weights]:
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                files[path.name] = digest
            config = json.loads(config_path.read_text())
            manifest[name][str(seed)] = {
                "path": str(directory), "sha256": files,
                "base_model_name_or_path": config.get("base_model_name_or_path"),
                "training_source": training_source,
            }
    return manifest



def validate_checkpoint_models(manifest: dict, model_id: str) -> None:
    """Require checkpoint provenance to identify the requested base model."""
    for arm, checkpoints in manifest.items():
        for seed, checkpoint in checkpoints.items():
            base = checkpoint.get("base_model_name_or_path")
            if base != model_id:
                raise ValueError(
                    f"checkpoint {arm}/seed_{seed} base {base!r} differs from {model_id!r}")

def summarize_cells(cells, tasks, arms, conditions, ks, seeds, orders):
    summary = {}
    for task in tasks:
        for arm in arms:
            for condition in conditions:
                for k in ks:
                    group = f"{task}:{arm}:{condition}:{k}"
                    metrics = {}
                    for order in orders:
                        available = [cells[f"{seed}:{group}:{order}"] for seed in seeds
                                     if f"{seed}:{group}:{order}" in cells]
                        if available:
                            metrics[order] = {
                                metric: mean_t_ci([cell["summary"][metric] for cell in available])
                                for metric in ("accuracy", "unparseable_rate",
                                               "follows_corrupted_mapping_rate")
                            }
                    if "class" in orders and "random" in orders:
                        pairs = [paired_option_effect(cells[f"{seed}:{group}:class"],
                                                      cells[f"{seed}:{group}:random"])
                                 for seed in seeds
                                 if all(f"{seed}:{group}:{order}" in cells
                                        for order in ("class", "random"))]
                        if pairs:
                            metrics["paired"] = {
                                metric: mean_t_ci([pair[metric] for pair in pairs])
                                for metric in ("random_minus_class_accuracy",
                                               "prediction_change_rate")
                            }
                            metrics["paired"]["seed_effects"] = pairs
                    if metrics:
                        summary[group] = metrics
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", action="append", default=[], metavar="NAME=DIRECTORY",
                        help="repeatable; directory contains seed_0, seed_1, etc.")
    parser.add_argument("--tasks", default="shapes4")
    parser.add_argument("--conditions", default="standard,shuffled,remap,remap_shuffled")
    parser.add_argument("--ks", default="0,1,4,8,16")
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--option-orders", default="class,random")
    parser.add_argument("--img-size", type=int, default=112)
    parser.add_argument("--max-new-tokens", type=int, default=6)
    parser.add_argument("--out", required=True)
    parser.add_argument("--preflight", action="store_true",
                        help="check checkpoints and report planned cells without loading models")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("max-new-tokens must be positive")
    adapters = {}
    for item in args.adapter:
        name, separator, directory = item.partition("=")
        if not separator or not name or name == "frozen" or ":" in name or name in adapters:
            parser.error("adapters require unique NAME=DIRECTORY names other than frozen")
        adapters[name] = Path(directory)
    tasks = args.tasks.split(",")
    conditions = args.conditions.split(",")
    ks = [int(value) for value in args.ks.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]
    validate_evaluation_grid(seeds, ks, args.n, min_k=0)
    orders = args.option_orders.split(",")
    if any(task not in TASKS for task in tasks):
        parser.error("unknown task")
    if any(condition not in {"standard", "shuffled", "remap", "remap_shuffled"}
           for condition in conditions):
        parser.error("unknown condition")
    if any(order not in {"class", "random"} for order in orders):
        parser.error("option orders must be class and/or random")
    if not 0 < args.n <= 120 or args.n % 4 or any(k < 0 for k in ks):
        parser.error("n must be a positive multiple of four <=120 and K nonnegative")
    for label, values in (("tasks", tasks), ("conditions", conditions), ("ks", ks),
                          ("seeds", seeds), ("option orders", orders)):
        if len(set(values)) != len(values):
            parser.error(f"duplicate {label}")
    checkpoints = checkpoint_manifest(adapters, seeds)
    try:
        validate_checkpoint_models(checkpoints, args.model)
    except ValueError as error:
        parser.error(str(error))
    arms = ["frozen", *adapters]
    config = {
        "schema_version": 2, "experiment": "paired_option_order_saved_adapters",
        "model": args.model, "model_revision": model_revision(args.model),
        "checkpoints": checkpoints, "tasks": tasks, "arms": arms,
        "conditions": conditions, "ks": ks, "n": args.n, "seeds": seeds,
        "option_orders": orders, "img_size": args.img_size,
        "max_new_tokens": args.max_new_tokens, "score_mode": "generation",
        "generation_policy": "explicit_greedy_single_output_v1",
        "protocols": {order: ProtocolConfig(option_order=order).metadata() for order in orders},
        "dtype": "bfloat16", "software_versions": software_versions(),
        "n_train_per_class": 96, "n_test_pool": 120,
    }
    result = load_result(args.out, config, {"cells": {}, "summary": {}})
    plan = [(seed, task, arm, condition, k, order) for seed in seeds for task in tasks
            for arm in arms for condition in conditions for k in ks for order in orders]
    key = lambda values: ":".join(str(value) for value in values)
    remaining = [cell for cell in plan if key(cell) not in result["cells"]]
    print(json.dumps({"planned_cells": len(plan), "remaining_cells": len(remaining),
                      "generations_remaining": len(remaining) * args.n,
                      "arms": arms, "out": args.out}), flush=True)
    if args.preflight or not remaining:
        return
    task_data = {task: TASKS[task](96, 120) for task in tasks}
    device = pick_device()

    def generate(model, processor, messages):
        return predict(model, processor, messages, max_new_tokens=args.max_new_tokens)

    for seed in seeds:
        for arm in arms:
            arm_cells = [cell for cell in remaining if cell[0] == seed and cell[2] == arm]
            if not arm_cells:
                continue
            base, processor = load_model(args.model, device=device, dtype=torch.bfloat16)
            model = base if arm == "frozen" else PeftModel.from_pretrained(
                base, adapters[arm] / f"seed_{seed}").eval()
            for cell in arm_cells:
                _, task, _, condition, k, order = cell
                names, pool, test = task_data[task]
                result["cells"][key(cell)] = evaluate_classification_cell(
                    model, processor, names, pool, test, condition, k, args.n,
                    args.img_size, seed, build_messages, generate,
                    protocol=ProtocolConfig(option_order=order), score_mode="generation")
                result["summary"] = summarize_cells(
                    result["cells"], tasks, arms, conditions, ks, seeds, orders)
                write_json_atomic(args.out, result)
                print(f"cell {key(cell)} done", flush=True)
            del model, base, processor
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            elif device == "mps":
                torch.mps.empty_cache()


if __name__ == "__main__":
    main()
