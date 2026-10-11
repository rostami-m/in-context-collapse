"""Evaluate frozen and repaired checkpoints on aligned open-ended VQA prompts."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
from pathlib import Path

import torch
from peft import PeftModel

from .evaluation import mean_t_ci
from .models import load_model
from .protocol import image_id
from .results import load_result, write_json_atomic
from .vqa import (DATASETS, build_vqa_messages, load_task, vqa_score, _generate,
                  validate_evaluation_request, implementation_manifest)
from .option_order_eval import checkpoint_manifest
from .provenance import DATASET_REVISIONS, model_revision, software_versions


def vqa_query_id(image, question):
    """Identify a VQA item even when several questions share one image."""
    question_digest = hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]
    return f"{image_id(image)}:{question_digest}"


def evaluate_cell(model, processor, pool, test, k, n, size, seed):
    """Evaluate one aligned VQA cell and retain every prediction."""
    validate_evaluation_request(pool, test, [k], n, [seed])
    rng = random.Random(1000 * seed + k)
    records = []
    for query_index, (query_image, question, answers) in enumerate(test[:n]):
        demos = rng.sample(pool, k)
        prediction = _generate(
            model, processor,
            build_vqa_messages(demos, query_image, question, size),
        ).strip()
        records.append({
            "query_index": query_index,
            "query_id": vqa_query_id(query_image, question),
            "question": question,
            "answers": answers,
            "prediction": prediction,
            "score": vqa_score(prediction, answers),
            "demo_ids": [vqa_query_id(image, demo_question)
                         for image, demo_question, _ in demos],
            "demo_questions": [demo_question for _, demo_question, _ in demos],
        })
    return {
        "k": k,
        "seed": seed,
        "summary": {
            "n": len(records),
            "accuracy": sum(record["score"] for record in records) / len(records),
        },
        "records": records,
    }


def validate_evidence_cell(cell, pool, test, k, n, seed):
    """Replay identities and scoring before skipping a completed/resumed cell."""
    validate_evaluation_request(pool, test, [k], n, [seed])
    if cell.get("k") != k or cell.get("seed") != seed or len(cell.get("records", [])) != n:
        raise ValueError("resumed VQA cell grid/count differs")
    rng = random.Random(1000 * seed + k)
    scores = []
    for index, (row, query) in enumerate(zip(cell["records"], test[:n])):
        image, question, answers = query
        demos = rng.sample(pool, k)
        expected = {"query_index": index, "query_id": vqa_query_id(image, question),
                    "question": question, "answers": answers,
                    "demo_ids": [vqa_query_id(image, q) for image, q, _ in demos],
                    "demo_questions": [q for _, q, _ in demos]}
        if any(row.get(key) != value for key, value in expected.items()) or not isinstance(row.get("prediction"), str):
            raise ValueError("resumed VQA query/support identities differ")
        score = vqa_score(row["prediction"], answers)
        if row.get("score") != score:
            raise ValueError("resumed VQA score differs from raw answer")
        scores.append(score)
    if cell.get("summary") != {"n": n, "accuracy": sum(scores)/n}:
        raise ValueError("resumed VQA summary differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapters-dir", required=True)
    parser.add_argument("--adapter-key", default="connector_both")
    parser.add_argument("--datasets", default="vqav2")
    parser.add_argument("--ks", default="0,4,8")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--n-pool", type=int, default=32,
                        help="number of examples before the held-out test partition")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--img-size", type=int, default=112)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    datasets = args.datasets.split(",")
    ks = [int(value) for value in args.ks.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]
    if (args.n_pool < 1 or args.n < 1 or args.img_size < 1
            or not ks or min(ks) < 0 or max(ks) > args.n_pool or len(set(ks)) != len(ks)
            or not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds)
            or not datasets or len(set(datasets)) != len(datasets) or any(name not in DATASETS for name in datasets)):
        parser.error("invalid VQA dimensions/seeds/datasets; each K must fit the declared pool")
    checkpoints = checkpoint_manifest({args.adapter_key: Path(args.adapters_dir) / args.adapter_key}, seeds)
    for checkpoint in checkpoints[args.adapter_key].values():
        if checkpoint['base_model_name_or_path'] != args.model or checkpoint['training_source'] is None:
            raise ValueError("checkpoint base model or source provenance differs")
        training = json.loads(Path(checkpoint['training_source']['path']).read_text())['run_config']
        if training.get('model_revision') != model_revision(args.model):
            raise ValueError("checkpoint base revision differs")
    run_config = {
        "schema_version": 3,
        "checkpoints": checkpoints,
        "source_sha256": implementation_manifest(),
        "model": args.model,
        "adapters_dir": args.adapters_dir,
        "adapter_key": args.adapter_key,
        "datasets": datasets,
        "ks": ks,
        "n": args.n,
        "n_pool": args.n_pool,
        "seeds": seeds,
        "img_size": args.img_size,
        "metric": "vqa_soft_accuracy",
        "data_partition": "prefix_train_then_prefix_test_v2",
        "model_revision": model_revision(args.model),
        "dataset_revisions": {
            name: DATASET_REVISIONS.get(DATASETS[name][0]) for name in datasets
        },
        "dtype": "bfloat16",
        "software_versions": software_versions(),
    }
    result = load_result(args.out, run_config, {"seeds_done": [], "evidence": {}})
    tasks = {
        name: load_task(name, args.n_pool, args.n)
        for name in datasets
    }

    if (set(result["evidence"]) != set(map(str, result["seeds_done"]))
            or len(set(result["seeds_done"])) != len(result["seeds_done"])
            or not set(result["seeds_done"]).issubset(seeds)):
        raise ValueError("resumed VQA seed membership differs")
    for seed in result["seeds_done"]:
        saved = result["evidence"][str(seed)]
        if set(saved) != set(datasets):
            raise ValueError("resumed VQA dataset grid differs")
        for name, (pool, test) in tasks.items():
            if set(saved[name]) != {"baseline", "repaired"}:
                raise ValueError("resumed VQA arm grid differs")
            for arm in saved[name].values():
                if set(arm) != set(map(str, ks)):
                    raise ValueError("resumed VQA K grid differs")
                for k in ks:
                    validate_evidence_cell(arm[str(k)], pool, test, k, args.n, seed)
    for seed in seeds:
        if seed in result["seeds_done"]:
            continue
        adapter_path = Path(args.adapters_dir) / args.adapter_key / f"seed_{seed}"
        if not adapter_path.exists():
            raise FileNotFoundError(f"missing adapter checkpoint: {adapter_path}")
        base, processor = load_model(args.model, dtype=torch.bfloat16)
        seed_evidence = {name: {"baseline": {}, "repaired": {}} for name in datasets}
        for name, (pool, test) in tasks.items():
            for k in ks:
                seed_evidence[name]["baseline"][str(k)] = evaluate_cell(
                    base, processor, pool, test, k, args.n, args.img_size, seed)
        repaired = PeftModel.from_pretrained(base, adapter_path).eval()
        for name, (pool, test) in tasks.items():
            for k in ks:
                seed_evidence[name]["repaired"][str(k)] = evaluate_cell(
                    repaired, processor, pool, test, k, args.n, args.img_size, seed)
        result["evidence"][str(seed)] = seed_evidence
        result["seeds_done"].append(seed)
        write_json_atomic(args.out, result)
        print(f"seed {seed} done", flush=True)
        del repaired, base, processor
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    result["summary"] = {}
    for name in datasets:
        result["summary"][name] = {}
        for arm in ("baseline", "repaired"):
            result["summary"][name][arm] = {
                str(k): mean_t_ci([
                    result["evidence"][str(seed)][name][arm][str(k)]["summary"]["accuracy"]
                    for seed in seeds
                ])
                for k in ks
            }
    write_json_atomic(args.out, result)
    print("DONE", result["summary"], flush=True)


if __name__ == "__main__":
    main()
