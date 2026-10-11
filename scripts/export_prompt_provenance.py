#!/usr/bin/env python3
"""Reconstruct processor inputs offline on CPU; never load model weights.

These are new measurements of pinned cached processors and the CURRENT prompt
protocol, not recovered measurements of historical panel runs. One seed-0 query
is exported for each recorded model/task/condition/K combination.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCOPE = (
    "CPU-only reconstruction with pinned cached processors and the current "
    "2.5-options protocol; not historical measurements, performance evaluation, "
    "or a claim that the historical processor revision is known."
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def slug(model):
    return model.replace("/", "--")


def panel_design(directory):
    """Read actual historical cells, without interpreting missing metadata."""
    groups = {}
    sources = []
    for path in sorted(Path(directory).glob("*.json")):
        cell = json.loads(path.read_text())
        if not all(key in cell for key in ("model", "task", "cond", "K", "seed")):
            continue
        key = (cell["model"], cell["task"], cell["cond"], int(cell["K"]))
        groups.setdefault(key, []).append({
            "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "sha256": digest(path), "seed": cell["seed"], "n": cell.get("n"),
            "recorded_protocol": cell.get("protocol"),
        })
        sources.append({"path": str(path.name), "sha256": digest(path)})
    return groups, sources


def cached_snapshot(cache, model, pinned_revision=None):
    directory = Path(cache) / ("models--" + model.replace("/", "--"))
    if pinned_revision:
        snapshot = directory / "snapshots" / pinned_revision
        if not snapshot.is_dir():
            raise FileNotFoundError(f"pinned snapshot is not cached: {model}@{pinned_revision}")
        return snapshot, "repository-pinned current revision"
    candidates = sorted((directory / "snapshots").glob("*"))
    candidates = [path for path in candidates if path.is_dir()]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected one unambiguous cached snapshot for {model}; found {len(candidates)}")
    return candidates[0], "sole cached revision pinned for this reconstruction; historical revision unknown"


def snapshot_sources(snapshot):
    allowed = {".json", ".jinja", ".txt", ".model", ".py"}
    return [{"name": path.name, "sha256": digest(path)}
            for path in sorted(Path(snapshot).iterdir())
            if path.is_file() and path.suffix in allowed
            and not path.name.endswith(".safetensors.index.json")
            and not path.name.endswith(".bin.index.json")]


def decoder_config(config):
    for path, node in (("text_config", config.get("text_config", {})),
                       ("llm_config", config.get("llm_config", {})), ("", config)):
        count = node.get("num_hidden_layers")
        if isinstance(count, int) and count > 0:
            return {"config_field": f"{path + '.' if path else ''}num_hidden_layers",
                    "num_decoder_blocks": count, "zero_based_block_range": [0, count - 1],
                    "region_assignment": "not inferred from generic thirds"}
    return {"num_decoder_blocks": None, "reason": "decoder layer count not found in cached config"}


def saved_target_profiles(paths):
    profiles = []
    for path in paths:
        source = json.loads(Path(path).read_text())
        model = source.get("model", source.get("run_config", {}).get("model"))
        for group, cell in source.items():
            if not isinstance(cell, dict) or not cell.get("training_runs"):
                continue
            for run in cell["training_runs"]:
                targets = run.get("targets", [])
                indices = sorted({int(match.group(1)) for target in targets
                                  if (match := re.search(r"(?:^|\.)layers\.(\d+)\.", target))})
                profiles.append({
                    "model": model, "group": group, "seed": run.get("torch_seed"),
                    "objective": run.get("objective"), "targets": targets,
                    "zero_based_target_layer_indices": indices,
                    "target_span": [min(indices), max(indices)] if indices else None,
                    "span_note": "endpoints do not imply that intermediate blocks were adapted",
                    "trainable_parameters": run.get("trainable_parameters"),
                    "source": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                    "source_sha256": digest(path),
                })
    return profiles


def visual_token_id(config, processor):
    for key in ("image_token_id", "image_token_index"):
        if isinstance(config.get(key), int):
            return config[key]
    value = getattr(processor, "image_token_id", None)
    if isinstance(value, int):
        return value
    token = getattr(processor, "image_token", None)
    if isinstance(token, str):
        value = processor.tokenizer.convert_tokens_to_ids(token)
        if value != getattr(processor.tokenizer, "unk_token_id", None):
            return value
    return None


def count_tokens(ids, config, processor, number_of_images):
    image_id = visual_token_id(config, processor)
    placeholders = sum(value == image_id for value in ids) if image_id is not None else None
    separate = config.get("model_type") == "mllama"
    supported = config.get("model_type") in {
        "qwen2_vl", "qwen2_5_vl", "llava", "llava_next", "llava_onevision", "idefics3", "internvl"}
    expanded = (supported and placeholders is not None and placeholders > number_of_images)
    return {
        "input_tokens_total": len(ids), "image_token_id": image_id,
        "image_placeholder_tokens": placeholders,
        "visual_tokens_in_lm_sequence": placeholders if expanded else None,
        "non_image_tokens_in_lm_sequence": len(ids) - placeholders if placeholders is not None else None,
        "visual_count_status": (
            "actual expanded image-token IDs in processor input_ids" if expanded else
            "visual states use separate cross-attention; image markers are not visual feature tokens"
            if separate else "image-token expansion not verified; no visual feature count asserted"),
        "text_count_note": "non-image count includes chat, image-boundary and other special tokens",
        "model_forward_performed": False,
    }


def serialize_messages(messages, image_directory):
    from incontext_collapse.protocol import image_id
    records = []
    for message in messages:
        content = []
        for item in message["content"]:
            if item["type"] != "image":
                content.append(dict(item))
                continue
            image = item["image"]
            identifier = image_id(image)
            target = image_directory / f"{identifier}.png"
            if not target.exists():
                image.save(target)
            content.append({"type": "image", "image_id": identifier,
                            "path_from_export_root": f"images/{identifier}.png",
                            "size": list(image.size), "mode": image.mode,
                            "file_sha256": digest(target)})
        records.append({"role": message["role"], "content": content})
    return records


def first_episode(task_data, condition, k, seed, n):
    from incontext_collapse.core import build_messages
    from incontext_collapse.protocol import (
        ProtocolConfig, corrupt_demo_labels, image_id, sample_balanced_demos,
        sample_balanced_queries, sample_label_space, sample_option_order)
    names, pool, test = task_data
    protocol = ProtocolConfig()
    rng = random.Random(1000 * seed + k)
    query, gold = sample_balanced_queries(test, n, rng)[0]
    source = sample_balanced_demos(pool, k, rng, class_policy=protocol.demo_class_policy)
    shown, permutation = source, None
    if condition in {"shuffled", "remap_shuffled"} and source:
        shown, permutation = corrupt_demo_labels(source, range(len(names)), rng)
    labels = sample_label_space(
        names, "remap" if condition in {"remap", "remap_shuffled"} else "standard", rng,
        mapping_mode=protocol.mapping_mode, token_set=protocol.remap_token_set)
    order = sample_option_order(len(names), seed, k, 0, protocol.option_order)
    messages = build_messages(shown, query, labels, 112, option_order=order)
    return messages, {
        "protocol": protocol.metadata(), "seed": seed, "query_index": 0,
        "query_sampling_n": n, "query_image_id": image_id(query), "query_class": gold,
        "label_space": labels, "option_order": order, "shuffle_map": permutation,
        "demo_source_classes": [label for _, label in source],
        "demo_shown_classes": [label for _, label in shown],
        "demo_image_ids": [image_id(image) for image, _ in source],
        "input_image_size": [112, 112], "number_of_images": k + 1,
    }


def cached_task_data(task, dataset_cache, revisions):
    """Read pinned Arrow files directly, without creating locks in the shared cache."""
    from datasets import Dataset
    from incontext_collapse.core import _balanced_test_items, task_shapes4
    if task == "shapes4":
        return task_shapes4(96, 120), {"source": "core.task_shapes4(96, 120)"}
    specification = {
        "cifar4": ("uoft-cs/cifar10", "uoft-cs___cifar10", "cifar10", "img", [0, 1, 2, 3]),
        "fashion4": ("zalando-datasets/fashion_mnist", "zalando-datasets___fashion_mnist",
                     "fashion_mnist", "image", [0, 1, 2, 9]),
    }
    dataset_id, cache_name, prefix, image_column, classes = specification[task]
    revision = revisions[dataset_id]
    candidates = sorted((Path(dataset_cache) / cache_name).glob(f"*/*/{revision}"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one pinned Arrow cache for {dataset_id}@{revision}")
    files = {split: candidates[0] / f"{prefix}-{split}.arrow" for split in ("train", "test")}
    datasets = {split: Dataset.from_file(str(path)) for split, path in files.items()}
    if task == "cifar4":
        names = [datasets["train"].features["label"].names[label] for label in classes]
    else:
        names = ["t-shirt", "trouser", "pullover", "ankle boot"]
    selected = {}
    for split, dataset in datasets.items():
        wanted = 96 if split == "train" else 30
        by_class = {index: [] for index in range(4)}
        for row_index, label in enumerate(dataset["label"]):
            if label in classes:
                index = classes.index(label)
                if len(by_class[index]) < wanted:
                    by_class[index].append(dataset[row_index][image_column])
            if all(len(images) == wanted for images in by_class.values()):
                break
        if any(len(images) != wanted for images in by_class.values()):
            raise ValueError(f"insufficient images in pinned {task} {split} cache")
        selected[split] = by_class
    return (names, selected["train"], _balanced_test_items(selected["test"], 120)), {
        "dataset_id": dataset_id, "revision": revision,
        "selection": "first 96 train and first 30 test images per selected class; same ordering as core task builders",
        "read_mode": "Dataset.from_file; read-only pinned Arrow cache; no cache locks or downloads",
        "files": [{"name": path.name, "sha256": digest(path)} for path in files.values()],
    }


def tensor_metadata(inputs):
    result = {}
    for key, value in inputs.items():
        if not hasattr(value, "shape"):
            continue
        if getattr(value, "device", None) is not None and value.device.type != "cpu":
            raise RuntimeError(f"processor unexpectedly created a non-CPU tensor: {key}")
        result[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        if key in {"image_grid_thw", "image_sizes", "aspect_ratio_ids", "aspect_ratio_mask"}:
            result[key]["values"] = value.tolist()
    return result


def package_source(obj):
    try:
        path = Path(inspect.getfile(type(obj)))
        return {"class": type(obj).__name__, "source_name": path.name, "sha256": digest(path)}
    except (TypeError, OSError):
        return {"class": type(obj).__name__, "source_unavailable": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="fresh output directory")
    parser.add_argument("--models", help="comma-separated IDs; default all recorded panel models")
    parser.add_argument("--tasks", default="shapes4,cifar4,fashion4")
    parser.add_argument("--conditions", default="standard,shuffled,remap")
    parser.add_argument("--ks", help="optional subset of recorded shot counts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--max-pixel-mib", type=int, default=256,
                        help="skip multi-image batches whose pixel storage estimate exceeds this CPU guard")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists; choose a fresh directory")
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[key] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    from huggingface_hub.constants import HF_HUB_CACHE
    from datasets.config import HF_DATASETS_CACHE
    from transformers import AutoProcessor
    from incontext_collapse.core import TASKS
    from incontext_collapse.provenance import DATASET_REVISIONS, MODEL_REVISIONS, software_versions
    torch.set_num_threads(1)
    if not 0 < args.n <= 120 or args.n % 4:
        parser.error("n must be a positive multiple of four up to 120")
    groups, panel_sources = panel_design(ROOT / "data/panel_cells")
    models = args.models.split(",") if args.models else sorted({key[0] for key in groups})
    tasks, conditions = args.tasks.split(","), args.conditions.split(",")
    if any(task not in TASKS for task in tasks):
        parser.error("unknown task")
    if any(condition not in {"standard", "shuffled", "remap"} for condition in conditions):
        parser.error("condition must occur in the historical panel grid")
    ks = set(map(int, args.ks.split(","))) if args.ks else None
    args.out.mkdir(parents=True, exist_ok=False)
    image_directory = args.out / "images"
    image_directory.mkdir(exist_ok=True)
    source_files = [Path(__file__).resolve(), ROOT / "requirements.lock"] + [
        ROOT / "incontext_collapse" / f"{name}.py"
        for name in ("core", "protocol", "evaluation", "provenance", "models")]
    target_sources = [
        ROOT / "data/revision_v24/qwen2vl2b_shapes_depth_q_up_v_episode_remap150_n40_generation_v24.json",
        ROOT / "data/revision_v24/qwen2vl2b_shapes_connector_episode_remap150_n40_generation_v24.json",
        ROOT / "data/revision_v24_completion/qwen25vl3b_shapes_connector_episode_remap150_n40_generation_v24.json",
    ]
    manifest = {
        "schema_version": 1, "scope": SCOPE, "historical_measurements": False,
        "device": "cpu", "model_weights_loaded": False, "network_access": "disabled",
        "software_versions": software_versions(), "dataset_revisions": DATASET_REVISIONS,
        "representative_sampling": {"seed": args.seed, "query_index": 0, "n": args.n},
        "source_files": [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)}
                         for path in source_files],
        "recorded_panel_sources": panel_sources,
        "saved_target_profiles": saved_target_profiles([path for path in target_sources if path.exists()]),
        "models": [], "records": [], "unavailable": [], "task_sources": {},
    }
    data = {}
    for task in tasks:
        try:
            data[task], manifest["task_sources"][task] = cached_task_data(
                task, HF_DATASETS_CACHE, DATASET_REVISIONS)
        except Exception as error:
            manifest["unavailable"].append({"task": task, "stage": "offline dataset load",
                                            "error": f"{type(error).__name__}: {error}"})
    for model in models:
        model_record = {"model": model, "status": "unavailable"}
        manifest["models"].append(model_record)
        combinations = sorted(key for key in groups if key[0] == model and key[1] in data
                              and key[2] in conditions and (ks is None or key[3] in ks))
        try:
            snapshot, revision_basis = cached_snapshot(HF_HUB_CACHE, model, MODEL_REVISIONS.get(model))
            config = json.loads((snapshot / "config.json").read_text())
            model_record.update(revision=snapshot.name, revision_basis=revision_basis,
                                snapshot_sources=snapshot_sources(snapshot),
                                config=config, decoder=decoder_config(config))
            processor = AutoProcessor.from_pretrained(
                str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=False)
            processor.image_processor.do_image_splitting = False
            model_record.update(status="processor loaded", processor_source=package_source(processor),
                                image_processor_source=package_source(processor.image_processor),
                                image_processor_config=processor.image_processor.to_dict(),
                                override={"do_image_splitting": False})
            pixel_per_image = None
            for _, task, condition, k in combinations:
                record = {"model": model, "task": task, "condition": condition, "K": k,
                          "revision": snapshot.name, "status": "unavailable",
                          "historical_grid_sources": groups[(model, task, condition, k)]}
                manifest["records"].append(record)
                relative = Path("prompts") / slug(model) / task / condition / f"K{k}"
                directory = args.out / relative
                directory.mkdir(parents=True, exist_ok=True)
                try:
                    messages, episode = first_episode(data[task], condition, k, args.seed, args.n)
                    record["episode"] = episode
                    write_json(directory / "messages.json", serialize_messages(messages, image_directory))
                    rendered = processor.apply_chat_template(messages, add_generation_prompt=True,
                                                              tokenize=False)
                    (directory / "rendered.txt").write_text(rendered)
                    record["prompt_directory"] = str(relative)
                    if pixel_per_image and pixel_per_image * (k + 1) > args.max_pixel_mib * 1024 ** 2:
                        raise MemoryError("estimated CPU pixel storage exceeds --max-pixel-mib; "
                                          "rendered prompt saved, expanded count not measured")
                    inputs = processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=True,
                        return_dict=True, return_tensors="pt")
                    record["processor_tensors"] = tensor_metadata(inputs)
                    pixel_bytes = sum(value.numel() * value.element_size() for name, value in inputs.items()
                                      if name.startswith("pixel") and hasattr(value, "numel"))
                    pixel_per_image = max(pixel_per_image or 0, pixel_bytes / (k + 1))
                    ids = inputs["input_ids"][0].tolist()
                    write_json(directory / "input_ids.json", ids)
                    decoded = processor.tokenizer.decode(ids, skip_special_tokens=False,
                                                          clean_up_tokenization_spaces=False)
                    (directory / "expanded_token_decode.txt").write_text(decoded)
                    record.update(status="measured", **count_tokens(ids, config, processor, k + 1))
                    record["files"] = [{"name": path.name, "sha256": digest(path)}
                                       for path in sorted(directory.iterdir()) if path.is_file()]
                    del inputs
                except Exception as error:
                    record["error"] = f"{type(error).__name__}: {error}"
                print(f"{model} {task} {condition} K={k}: {record['status']}", flush=True)
            model_record["status"] = "processed"
        except Exception as error:
            model_record["error"] = f"{type(error).__name__}: {error}"
            for _, task, condition, k in combinations:
                manifest["records"].append({
                    "model": model, "task": task, "condition": condition, "K": k,
                    "revision": model_record.get("revision"), "status": "unavailable",
                    "error": model_record["error"],
                    "historical_grid_sources": groups[(model, task, condition, k)],
                })
            print(f"{model}: unavailable: {model_record['error']}", flush=True)
        write_json(args.out / "manifest.json", manifest)
    fields = ["model", "revision", "task", "condition", "K", "status", "input_tokens_total",
              "visual_tokens_in_lm_sequence", "image_placeholder_tokens",
              "non_image_tokens_in_lm_sequence", "prompt_directory", "error"]
    with (args.out / "counts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(manifest["records"])
    measured = sum(record["status"] == "measured" for record in manifest["records"])
    manifest["summary"] = {"measured": measured, "attempted": len(manifest["records"]),
                           "unavailable_models": sum(model["status"] == "unavailable"
                                                     for model in manifest["models"])}
    write_json(args.out / "manifest.json", manifest)
    print(json.dumps(manifest["summary"]), flush=True)


if __name__ == "__main__":
    main()
