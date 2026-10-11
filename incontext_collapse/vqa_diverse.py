"""Image-disjoint VQA follow-up for existing, frozen adapter checkpoints."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from peft import PeftModel

from .evaluation import mean_t_ci
from .models import load_model
from .option_order_eval import checkpoint_manifest
from .protocol import image_id
from .provenance import DATASET_REVISIONS, load_pinned_dataset, model_revision, software_versions
from .results import load_result, write_json_atomic
from .vqa import DATASETS, adapt, implementation_manifest
from .vqa_repair import evaluate_cell, validate_evidence_cell, vqa_query_id


def select_image_disjoint(items, n_pool, exclude_prefix, n, seed):
    """Uniformly sample image groups, then one question within each chosen group."""
    if n_pool < 1 or exclude_prefix < n_pool or len(items) <= exclude_prefix:
        raise ValueError("invalid source pool or excluded-prefix size")
    pool = list(items[:n_pool])
    excluded = {image_id(item[0]) for item in items[:exclude_prefix]}
    groups = defaultdict(list)
    seen = set()
    for item in items[exclude_prefix:]:
        key = image_id(item[0])
        query_id = vqa_query_id(item[0], item[1])
        if key not in excluded and query_id not in seen:
            groups[key].append(item)
            seen.add(query_id)
    if n < 1 or len(groups) < n:
        raise ValueError(f"need {n} eligible distinct images; found {len(groups)}")
    rng = random.Random(seed)
    selected = rng.sample(sorted(groups), n)
    test = [rng.choice(groups[key]) for key in selected]
    return pool, test, {
        "excluded_image_ids": sorted(excluded),
        "n_training_images": len({image_id(item[0]) for item in pool}),
        "n_eligible_images": len(groups),
        "n_eligible_questions": sum(len(group) for group in groups.values()),
        "n_evaluation_images": n,
    }


def prepare_data(directory, dataset, scan, n_pool, exclude_prefix, n, seed):
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    config = {
        "dataset": dataset,
        "dataset_revision": DATASET_REVISIONS.get(DATASETS[dataset][0]),
        "scan_valid_questions": scan,
        "n_pool": n_pool,
        "exclude_prefix": exclude_prefix,
        "n": n,
        "sampling_seed": seed,
        "sampling": "uniform_images_then_uniform_question_fixed_prefix_v1",
    }
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text())
        if payload["config"] != config:
            raise ValueError("cached VQA manifest configuration differs")
        return manifest_path
    hf_id, subset, split = DATASETS[dataset]
    stream = load_pinned_dataset(hf_id, subset, split=split, streaming=True)
    items = []
    for example in stream:
        item = adapt(dataset, example)
        if item[0] is not None and item[1] and item[2]:
            items.append(item)
        if len(items) >= scan:
            break
    if len(items) != scan:
        raise ValueError(f"requested a {scan}-question frame, found {len(items)}")
    pool, test, diversity = select_image_disjoint(
        items, n_pool, exclude_prefix, n, seed)
    (directory / "images").mkdir(parents=True, exist_ok=True)

    def serialize(item):
        image, question, answers = item
        key = image_id(image)
        path = directory / "images" / f"{key}.png"
        if not path.exists():
            image.save(path, format="PNG")
        with Image.open(path) as saved:
            if image_id(saved) != key:
                raise ValueError("lossless VQA cache changed the image identity")
        return {"image": f"images/{key}.png", "image_id": key,
                "query_id": vqa_query_id(image, question),
                "question": question, "answers": answers}

    write_json_atomic(manifest_path, {
        "config": config, "diversity": diversity,
        "pool": [serialize(item) for item in pool],
        "test": [serialize(item) for item in test],
    })
    return manifest_path


def load_manifest(path):
    path = Path(path)
    payload = json.loads(path.read_text())

    def restore(record):
        with Image.open(path.parent / record["image"]) as image:
            image = image.copy()
        if image_id(image) != record["image_id"]:
            raise ValueError("VQA cached-image hash mismatch")
        if vqa_query_id(image, record["question"]) != record["query_id"]:
            raise ValueError("VQA cached-question identity mismatch")
        return image, record["question"], record["answers"]

    pool = [restore(record) for record in payload["pool"]]
    test = [restore(record) for record in payload["test"]]
    test_images = [image_id(item[0]) for item in test]
    excluded = set(payload["diversity"]["excluded_image_ids"])
    pool_images = {image_id(item[0]) for item in pool}
    if not pool_images.issubset(excluded):
        raise ValueError("VQA pool images are absent from excluded-image provenance")
    if len(test_images) != len(set(test_images)) or excluded.intersection(test_images):
        raise ValueError("VQA evaluation images are repeated or overlap the excluded pool")
    if len(pool) != payload["config"]["n_pool"] or len(test) != payload["config"]["n"]:
        raise ValueError("incomplete VQA manifest")
    if (len(pool_images) != payload["diversity"]["n_training_images"]
            or len(test_images) != payload["diversity"]["n_evaluation_images"]):
        raise ValueError("VQA diversity counts disagree with cached images")
    return pool, test, payload


def validate_training_partition(training, data_config):
    if training.get("data_partition") != "prefix_train_then_prefix_test_v2":
        raise ValueError("unsupported source VQA training partition")
    if training.get("n_train") != data_config["n_pool"]:
        raise ValueError("VQA demonstration pool differs from the actual training prefix")
    if data_config["exclude_prefix"] < training["n_train"] + training["n"]:
        raise ValueError("VQA excluded prefix omits source training or original test questions")
    if (training.get("dataset") != data_config["dataset"]
            or training.get("dataset_revision") != data_config["dataset_revision"]):
        raise ValueError("source and follow-up VQA dataset revisions differ")


def paired_image_interval(baseline_cells, repaired_cells, draws=10000, seed=90210):
    """Bootstrap image-level paired changes after averaging matched training seeds."""
    if len(baseline_cells) != len(repaired_cells) or not baseline_cells:
        raise ValueError("paired VQA cells must have identical nonempty seed counts")
    if type(draws) is not int or draws < 2:
        raise ValueError("at least two bootstrap draws required")
    differences = defaultdict(list)
    for baseline, repaired in zip(baseline_cells, repaired_cells):
        for cell in (baseline, repaired):
            ids = [record["query_id"] for record in cell["records"]]
            if not ids or len(ids) != len(set(ids)):
                raise ValueError("paired VQA cells have empty or duplicate queries")
        left = {record["query_id"]: record for record in baseline["records"]}
        right = {record["query_id"]: record for record in repaired["records"]}
        if left.keys() != right.keys():
            raise ValueError("unpaired VQA questions")
        for key, record in left.items():
            if record["demo_ids"] != right[key]["demo_ids"]:
                raise ValueError("unpaired VQA demonstrations")
            differences[key].append(right[key]["score"] - record["score"])
    if any(len(values) != len(baseline_cells) for values in differences.values()):
        raise ValueError("VQA seed panels do not share their evaluation questions")
    image_keys = [key.split(":")[0] for key in differences]
    if len(set(image_keys)) != len(image_keys):
        raise ValueError("bootstrap expects one question per image")
    values = np.array([np.mean(differences[key]) for key in sorted(differences)])
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("bootstrap requires finite nonempty paired image effects")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "ci95_low": float(np.quantile(means, .025)),
            "ci95_high": float(np.quantile(means, .975)), "n_images": len(values),
            "bootstrap_draws": draws, "bootstrap_seed": seed,
            "scope": "image sampling conditional on fixed frame and fitted checkpoints"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--dataset", default="vqav2")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--scan", type=int, default=4096)
    parser.add_argument("--n-pool", type=int, default=96)
    parser.add_argument("--exclude-prefix", type=int, default=136)
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--sampling-seed", type=int, default=20261004)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--adapters-dir")
    parser.add_argument("--shapes-adapters-dir")
    parser.add_argument("--ks", default="0,4,8")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--img-size", type=int, default=112)
    parser.add_argument("--out")
    args = parser.parse_args()
    ks = [int(value) for value in args.ks.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]
    if (len(set(ks)) != len(ks) or len(set(seeds)) != len(seeds)
            or any(k < 0 or k > args.n_pool for k in ks)):
        parser.error("seeds and K values must be unique, and 0 <= K <= n_pool")
    manifest_path = prepare_data(args.data_dir, args.dataset, args.scan, args.n_pool,
                                 args.exclude_prefix, args.n, args.sampling_seed)
    pool, test, manifest = load_manifest(manifest_path)
    print("VQA image sampling", {key: value for key, value in manifest["diversity"].items()
                                  if key != "excluded_image_ids"}, flush=True)
    if args.prepare_only:
        return
    if not args.out or not args.adapters_dir:
        parser.error("evaluation requires --out and --adapters-dir")
    adapters = {"vqa_connector": args.adapters_dir}
    if args.shapes_adapters_dir:
        adapters["shapes_connector"] = args.shapes_adapters_dir
    checkpoints = checkpoint_manifest(
        {arm: Path(directory) / "connector_both" for arm, directory in adapters.items()}, seeds)
    for arm, seed_checkpoints in checkpoints.items():
        for checkpoint in seed_checkpoints.values():
            if checkpoint["base_model_name_or_path"] != args.model:
                raise ValueError("VQA evaluation checkpoint belongs to a different base model")
            if checkpoint['training_source'] is None:
                raise ValueError("VQA evaluation requires checkpoint source training provenance")
            actual_training = json.loads(Path(checkpoint['training_source']['path']).read_text())['run_config']
            if actual_training.get('model_revision') != model_revision(args.model):
                raise ValueError("checkpoint training and VQA base-model revisions differ")
            if arm == "vqa_connector":
                source = checkpoint["training_source"]
                if source is None:
                    raise ValueError("VQA checkpoint requires its source training artifact")
                training = json.loads(Path(source["path"]).read_text())["run_config"]
                validate_training_partition(training, manifest["config"])
                if training.get("model_revision") != model_revision(args.model):
                    raise ValueError("VQA source and evaluation base-model revisions differ")
    arms = ["frozen", *adapters]
    config = {
        "schema_version": 3, "method": "image-disjoint VQA checkpoint evaluation",
        "source_sha256": implementation_manifest(("vqa_diverse", "vqa", "vqa_repair", "models", "protocol", "provenance", "results", "option_order_eval")),
        "model": args.model, "model_revision": model_revision(args.model),
        "dataset": args.dataset, "dataset_revision": manifest["config"]["dataset_revision"],
        "data_partition": manifest["config"], "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "n": args.n, "ks": ks, "seeds": seeds, "arms": arms,
        "adapters": adapters, "img_size": args.img_size, "dtype": "bfloat16",
        "checkpoints": checkpoints,
        "software_versions": software_versions(), "metric": "vqa_soft_accuracy",
    }
    result = load_result(args.out, config, {
        "evidence": {}, "seeds_done": [], "diversity": manifest["diversity"]})
    if not set(result["evidence"]).issubset(map(str, seeds)) or not set(result["seeds_done"]).issubset(seeds):
        raise ValueError("resumed VQA seed grid differs")
    for seed_id, by_arm in result["evidence"].items():
        if not set(by_arm).issubset(arms):
            raise ValueError("resumed VQA arm grid differs")
        for cells in by_arm.values():
            if not set(cells).issubset(map(str, ks)):
                raise ValueError("resumed VQA K grid differs")
            for k, cell in cells.items():
                validate_evidence_cell(cell, pool, test, int(k), args.n, int(seed_id))
    for seed in result["seeds_done"]:
        saved = result["evidence"].get(str(seed), {})
        if set(saved) != set(arms) or any(set(cells) != set(map(str, ks)) for cells in saved.values()):
            raise ValueError("completed VQA seed is incomplete")
    for seed in seeds:
        seed_result = result["evidence"].setdefault(str(seed), {})
        for arm in arms:
            cells = seed_result.setdefault(arm, {})
            if all(str(k) in cells for k in ks):
                continue
            base, processor = load_model(args.model, dtype=torch.bfloat16)
            model = base
            if arm != "frozen":
                checkpoint = Path(adapters[arm]) / "connector_both" / f"seed_{seed}"
                model = PeftModel.from_pretrained(base, checkpoint).eval()
            for k in ks:
                if str(k) not in cells:
                    cells[str(k)] = evaluate_cell(
                        model, processor, pool, test, k, args.n, args.img_size, seed)
                    write_json_atomic(args.out, result)
                    print(f"seed={seed} arm={arm} K={k}: {cells[str(k)]['summary']}", flush=True)
            del model, base, processor
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        if seed not in result["seeds_done"]:
            result["seeds_done"].append(seed)
            write_json_atomic(args.out, result)
    result["summary"] = {
        arm: {str(k): mean_t_ci([
            result["evidence"][str(seed)][arm][str(k)]["summary"]["accuracy"]
            for seed in seeds]) for k in ks} for arm in arms}
    result["paired_image_intervals"] = {
        arm: {str(k): paired_image_interval(
            [result["evidence"][str(seed)]["frozen"][str(k)] for seed in seeds],
            [result["evidence"][str(seed)][arm][str(k)] for seed in seeds])
            for k in ks} for arm in adapters}
    write_json_atomic(args.out, result)
    print("DONE", result["summary"], flush=True)


if __name__ == "__main__":
    main()
