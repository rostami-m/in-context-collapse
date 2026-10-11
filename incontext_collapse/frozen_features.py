"""Frozen vision-feature nearest-neighbor baseline for classification ICL."""
from __future__ import annotations

import argparse
import random

import torch

from .core import TASKS
from .evaluation import mean_t_ci
from .models import load_model
from .protocol import validate_evaluation_grid, validate_named_axis
from .protocol import (ProtocolConfig, corrupt_demo_labels, sample_balanced_demos,
                       image_id, sample_balanced_queries, sample_label_space)
from .provenance import model_revision, software_versions
from .results import load_result, write_json_atomic


def nearest_demo_label(query_embedding, demo_embeddings, demo_labels):
    """Return the label of the cosine-nearest demonstration."""
    if len(demo_embeddings) != len(demo_labels):
        raise ValueError("demo embeddings and labels must have equal length")
    if not demo_labels:
        raise ValueError("nearest-neighbor prediction requires a demonstration")
    query = torch.nn.functional.normalize(query_embedding.float(), dim=-1)
    demos = torch.nn.functional.normalize(torch.stack(demo_embeddings).float(), dim=-1)
    return demo_labels[int((demos @ query).argmax().item())]


@torch.no_grad()
def image_embedding(model, processor, image, size):
    """Mean-pool the frozen Qwen2-VL connector output for one image."""
    if not hasattr(model, "visual"):
        raise TypeError("frozen-feature baseline requires a model.visual encoder")
    prepared = image.convert("RGB").resize((size, size))
    inputs = processor.image_processor(images=[prepared], return_tensors="pt").to(model.device)
    features = model.visual(
        inputs["pixel_values"].type(model.visual.get_dtype()),
        grid_thw=inputs["image_grid_thw"],
    )
    return features.float().mean(dim=0).cpu()


def evaluate_cell(embeddings, names, pool, test, condition, k, n, seed, protocol):
    rng = random.Random(1000 * seed + k)
    items = sample_balanced_queries(test, n, rng)
    records = []
    for query_index, (query_image, gold) in enumerate(items):
        source_demos = sample_balanced_demos(
            pool, k, rng, class_policy=protocol.demo_class_policy)
        shown_demos = list(source_demos)
        shuffle_map = None
        if condition == "remap_shuffled":
            shown_demos, shuffle_map = corrupt_demo_labels(
                shown_demos, range(len(names)), rng)
        label_space = sample_label_space(
            names, "remap", rng, mapping_mode=protocol.mapping_mode,
            token_set=protocol.remap_token_set)
        prediction = nearest_demo_label(
            embeddings[id(query_image)],
            [embeddings[id(image)] for image, _ in shown_demos],
            [label for _, label in shown_demos],
        )
        records.append({
            "query_index": query_index,
            "query_id": image_id(query_image),
            "gold": gold,
            "prediction": prediction,
            "correct": prediction == gold,
            "condition": condition,
            "k": k,
            "label_space": label_space,
            "demo_source_classes": [label for _, label in source_demos],
            "demo_source_ids": [image_id(image) for image, _ in source_demos],
            "demo_label_classes": [label for _, label in shown_demos],
            "shuffle_map": shuffle_map,
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
            "accuracy": sum(record["correct"] for record in records) / count,
            "follows_corrupted_mapping_rate": sum(
                record["follows_corrupted_mapping"] for record in records) / count,
        },
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", default="shapes4")
    parser.add_argument("--conditions", default="remap,remap_shuffled")
    parser.add_argument("--ks", default="1,4,8,16")
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--img-size", type=int, default=112)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    conditions = args.conditions.split(",")
    validate_named_axis(conditions, "conditions", {"remap", "remap_shuffled"})
    if any(condition not in {"remap", "remap_shuffled"} for condition in conditions):
        parser.error("conditions must be remap and/or remap_shuffled")
    ks = [int(value) for value in args.ks.split(",")]
    if any(k < 1 for k in ks):
        parser.error("frozen-feature nearest neighbor requires K >= 1")
    seeds = [int(value) for value in args.seeds.split(",")]
    validate_evaluation_grid(seeds, ks, args.n, min_k=1)
    # This non-generative baseline has no displayed candidate list.
    protocol = ProtocolConfig(option_order="class")
    run_config = {
        "schema_version": 2,
        "baseline": "frozen_connector_feature_1nn",
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
    result = load_result(args.out, run_config, {"seeds_done": [], "evidence": {}})
    names, pool, test = TASKS[args.task](96, 120)
    model, processor = load_model(args.model, dtype=torch.bfloat16)
    images = [image for class_images in pool.values() for image in class_images]
    images.extend(image for image, _ in test)
    embeddings = {}
    for index, image in enumerate(images, start=1):
        embeddings[id(image)] = image_embedding(model, processor, image, args.img_size)
        if index % 100 == 0:
            print(f"embedded {index}/{len(images)} images", flush=True)

    for seed in seeds:
        if seed in result["seeds_done"]:
            continue
        result["evidence"][str(seed)] = {
            condition: {
                str(k): evaluate_cell(
                    embeddings, names, pool, test, condition, k, args.n, seed, protocol)
                for k in ks
            }
            for condition in conditions
        }
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
                for metric in ("accuracy", "follows_corrupted_mapping_rate")
            }
    write_json_atomic(args.out, result)
    print("DONE", result["summary"], flush=True)


if __name__ == "__main__":
    main()
