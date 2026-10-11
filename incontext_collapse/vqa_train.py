"""Fit VQA answers on separate question prefixes; images may repeat across prefixes.

Image-disjoint scientific evaluation uses vqa_diverse/vqa_confirmatory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import math
from pathlib import Path
import random

import torch
from peft import LoraConfig, get_peft_model

from .core import region_targets
from .evaluation import mean_t_ci
from .models import load_model
from .results import load_result, write_json_atomic
from .protocol import seed_training
from .provenance import DATASET_REVISIONS, model_revision, software_versions
from .scoring import encode_supervised_answer
from .vqa import (DATASETS, build_vqa_messages, load_task, vqa_norm,
                  validate_evaluation_request, implementation_manifest)
from .vqa_repair import evaluate_cell, validate_evidence_cell


def representative_answer(answers):
    """Return the most frequent normalized VQA answer with stable tie-breaking."""
    normalized = [vqa_norm(answer) for answer in answers if vqa_norm(answer)]
    if not normalized:
        raise ValueError("VQA example has no non-empty answer")
    counts = Counter(normalized)
    return max(counts, key=lambda answer: (counts[answer], -normalized.index(answer)))


def train_connector(model, processor, pool, targets, seed, steps, rank, k, size):
    if (len(pool) < 2 or any(type(v) is not int or v < 1 for v in (steps, rank, size))
            or type(k) is not int or not 0 <= k < len(pool) or not targets):
        raise ValueError("training requires positive steps/rank/size, targets, and K non-query supports")
    seed_training(seed)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    repaired = get_peft_model(
        model,
        LoraConfig(r=rank, lora_alpha=2 * rank, target_modules=targets,
                   lora_dropout=0.05),
    )
    repaired.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in repaired.parameters() if parameter.requires_grad],
        lr=1e-4,
    )
    rng = random.Random(seed)
    losses = []
    parameters_with_grad = 0
    for _ in range(steps):
        query_index = rng.randrange(len(pool))
        query_image, question, answers = pool[query_index]
        candidates = pool[:query_index] + pool[query_index + 1:]
        demos = rng.sample(candidates, k)
        messages = build_vqa_messages(demos, query_image, question, size)
        inputs, labels, _ = encode_supervised_answer(
            processor, messages, representative_answer(answers), repaired.device)
        loss = repaired(**inputs, labels=labels).loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite VQA training loss: {float(loss.detach().cpu())}")
        loss.backward()
        parameters_with_grad = sum(
            parameter.grad is not None
            for parameter in repaired.parameters() if parameter.requires_grad)
        if parameters_with_grad == 0:
            raise RuntimeError("no VQA adapter parameter received a gradient")
        losses.append(float(loss.detach().cpu()))
        optimizer.step()
        optimizer.zero_grad()
    repaired.eval()
    model.config.use_cache = True
    metadata = {
        "objective": "vqa_context_answer",
        "torch_seed": seed,
        "steps": steps,
        "k": k,
        "rank": rank,
        "learning_rate": 1e-4,
        "n_targets": len(targets),
        "targets": targets,
        "trainable_parameters": sum(
            parameter.numel() for parameter in repaired.parameters()
            if parameter.requires_grad),
        "parameters_with_grad_last_step": parameters_with_grad,
        "losses": losses,
    }
    return repaired, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="vqav2")
    parser.add_argument("--ks", default="0,4,8")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--n-train", type=int, default=96)
    parser.add_argument("--train-k", type=int, default=8)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--img-size", type=int, default=112)
    parser.add_argument("--adapters-dir", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ks = [int(value) for value in args.ks.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]
    adapters_dir = Path(
        args.adapters_dir or str(Path(args.out).with_suffix("")) + "_adapters")
    if (args.n < 1 or args.n_train < 2 or args.img_size < 1 or args.steps < 1 or args.rank < 1
            or args.train_k < 0 or args.train_k >= args.n_train or not ks or min(ks) < 0
            or max(ks) > args.n_train or len(ks) != len(set(ks)) or not seeds
            or min(seeds) < 0 or len(seeds) != len(set(seeds)) or args.dataset not in DATASETS):
        parser.error("invalid VQA training/evaluation grid; K must fit its support pool")
    run_config = {
        "schema_version": 3,
        "source_sha256": implementation_manifest(("vqa_train", "vqa", "vqa_repair", "models", "core", "protocol", "scoring", "provenance", "results")),
        "model": args.model,
        "dataset": args.dataset,
        "ks": ks,
        "n": args.n,
        "n_train": args.n_train,
        "train_k": args.train_k,
        "steps": args.steps,
        "rank": args.rank,
        "seeds": seeds,
        "img_size": args.img_size,
        "adapters_dir": str(adapters_dir),
        "metric": "vqa_soft_accuracy",
        "data_partition": "prefix_train_then_prefix_test_v2",
        "model_revision": model_revision(args.model),
        "dataset_revision": DATASET_REVISIONS.get(DATASETS[args.dataset][0]),
        "dtype": "bfloat16",
        "software_versions": software_versions(),
    }
    result = load_result(args.out, run_config, {
        "seeds_done": [], "evidence": {}, "training_runs": {}})
    pool, test = load_task(args.dataset, args.n_train, args.n, seed=0)

    if (set(result["evidence"]) != set(map(str, result["seeds_done"]))
            or set(result["training_runs"]) != set(result["evidence"])
            or len(set(result["seeds_done"])) != len(result["seeds_done"])
            or not set(result["seeds_done"]).issubset(seeds)):
        raise ValueError("resumed VQA training seed membership differs")
    for seed in result["seeds_done"]:
        metadata = result["training_runs"][str(seed)]
        checkpoint = Path(metadata['adapter_path'])
        hashes = metadata.get('checkpoint_sha256')
        if not hashes:
            raise ValueError("resumed training lacks checkpoint byte provenance")
        for filename, expected in hashes.items():
            if Path(filename).name != filename or hashlib.sha256((checkpoint/filename).read_bytes()).hexdigest() != expected:
                raise ValueError("resumed trained checkpoint changed")
        saved = result["evidence"][str(seed)]
        if set(saved) != {"baseline", "repaired"}:
            raise ValueError("resumed training arm grid differs")
        for arm in saved.values():
            if set(arm) != set(map(str, ks)):
                raise ValueError("resumed training K grid differs")
            for k in ks:
                validate_evidence_cell(arm[str(k)], pool, test, k, args.n, seed)
    for seed in seeds:
        if seed in result["seeds_done"]:
            continue
        adapter_path = adapters_dir / "connector_both" / f"seed_{seed}"
        if adapter_path.exists():
            raise ValueError(f"uncommitted checkpoint already exists; preserve it and use a new output directory: {adapter_path}")
        base, processor = load_model(args.model, dtype=torch.bfloat16)
        baseline = {
            str(k): evaluate_cell(
                base, processor, pool, test, k, args.n, args.img_size, seed)
            for k in ks
        }
        targets = region_targets(base, "connector", "both")
        repaired, metadata = train_connector(
            base, processor, pool, targets, seed, args.steps, args.rank,
            args.train_k, args.img_size)
        adapter_path = adapters_dir / "connector_both" / f"seed_{seed}"
        repaired.save_pretrained(adapter_path)
        metadata["adapter_path"] = str(adapter_path)
        metadata["checkpoint_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [adapter_path/"adapter_config.json", *sorted(adapter_path.glob("adapter_model*"))]
            if path.is_file()}
        if not all(math.isfinite(loss) for loss in metadata["losses"]):
            raise RuntimeError("non-finite loss persisted after VQA training")
        repaired_evidence = {
            str(k): evaluate_cell(
                repaired, processor, pool, test, k, args.n, args.img_size, seed)
            for k in ks
        }
        result["evidence"][str(seed)] = {
            "baseline": baseline,
            "repaired": repaired_evidence,
        }
        result["training_runs"][str(seed)] = metadata
        result["seeds_done"].append(seed)
        write_json_atomic(args.out, result)
        print(f"seed {seed} done", flush=True)
        del repaired, base, processor
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    result["summary"] = {
        arm: {
            str(k): mean_t_ci([
                result["evidence"][str(seed)][arm][str(k)]["summary"]["accuracy"]
                for seed in seeds
            ])
            for k in ks
        }
        for arm in ("baseline", "repaired")
    }
    write_json_atomic(args.out, result)
    print("DONE", result["summary"], flush=True)


if __name__ == "__main__":
    main()
