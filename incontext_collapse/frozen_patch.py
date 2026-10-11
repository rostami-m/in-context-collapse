"""Frozen bidirectional activation patching for remapped-label ICL.

For paired prompts with identical images and balanced demonstrations, cache the
final query-position residual from a correct-remap prompt and a prompt whose
demonstrated mapping is deranged. Swap that one residual vector at selected LM
depths in both directions. No model parameter is trainable or modified.
"""
from __future__ import annotations

import argparse
import random
from contextlib import ExitStack

import torch

from .core import TASKS, build_messages
from .evaluation import mean_t_ci
from .inject import get_layers, locus_idx
from .models import load_model
from .protocol import validate_evaluation_grid, validate_named_axis
from .protocol import (ProtocolConfig, corrupt_demo_labels, sample_balanced_demos,
                       image_id, sample_balanced_queries, sample_label_space, sample_option_order)
from .provenance import model_revision, software_versions
from .results import load_result, write_json_atomic
from .scoring import parse_answer


def _hidden(output):
    return output[0] if isinstance(output, tuple) else output


def _replace_hidden(output, hidden):
    return (hidden, *output[1:]) if isinstance(output, tuple) else hidden


def query_residual_patch(vector):
    """Build a hook that replaces only the last position during prefill."""
    applied = False

    def hook(_module, _inputs, output):
        nonlocal applied
        hidden = _hidden(output)
        if applied or hidden.ndim != 3 or hidden.shape[1] <= 1:
            return output
        if hidden.shape[0] != vector.shape[0] or hidden.shape[2] != vector.shape[2]:
            raise ValueError(
                f"patch shape mismatch: hidden={tuple(hidden.shape)}, "
                f"vector={tuple(vector.shape)}")
        applied = True
        patched = hidden.clone()
        patched[:, -1:, :] = vector.to(device=hidden.device, dtype=hidden.dtype)
        return _replace_hidden(output, patched)
    return hook


@torch.no_grad()
def cache_query_residuals(model, processor, messages, selected_layers):
    """Run one prefill and cache the last-position residual at each layer."""
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt").to(model.device)
    cached = {}
    with ExitStack() as stack:
        for name, layer in selected_layers.items():
            def capture(_module, _inputs, output, key=name):
                hidden = _hidden(output)
                if hidden.ndim == 3 and hidden.shape[1] > 1:
                    cached[key] = hidden[:, -1:, :].detach().clone()
                return output
            handle = layer.register_forward_hook(capture)
            stack.callback(handle.remove)
        model(**inputs, use_cache=False, return_dict=True)
    missing = set(selected_layers) - set(cached)
    if missing:
        raise RuntimeError(f"failed to cache query residuals at: {sorted(missing)}")
    return cached


@torch.no_grad()
def generate_with_patch(model, processor, messages, layer=None, vector=None):
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt").to(model.device)
    n_input = inputs["input_ids"].shape[1]
    handle = None
    if layer is not None:
        if vector is None:
            raise ValueError("a patch layer requires a residual vector")
        handle = layer.register_forward_hook(query_residual_patch(vector))
    try:
        output = model.generate(**inputs, max_new_tokens=6, do_sample=False,
                                num_beams=1, num_return_sequences=1)
    finally:
        if handle is not None:
            handle.remove()
    return processor.batch_decode(
        output[:, n_input:], skip_special_tokens=True)[0]


def _summary(records, modes, include_wrong_map):
    result = {}
    for mode in modes:
        predictions = [record["predictions"][mode] for record in records]
        result[mode] = {
            "accuracy": sum(pred == record["gold"]
                            for pred, record in zip(predictions, records)) / len(records),
            "unparseable_rate": sum(pred is None for pred in predictions) / len(records),
        }
        if include_wrong_map:
            result[mode]["wrong_map_adherence"] = sum(
                pred == record["shuffle_map"][record["gold"]]
                for pred, record in zip(predictions, records)) / len(records)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", default="shapes4")
    ap.add_argument("--ks", default="4,8,16")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--loci", default="early,mid,late")
    ap.add_argument("--pair", choices=["standard_remap", "remap_corrupt"],
                    default="standard_remap")
    ap.add_argument("--img-size", type=int, default=112)
    ap.add_argument("--option-order", choices=["random", "class"], default="random")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ks = [int(value) for value in args.ks.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]
    validate_evaluation_grid(seeds, ks, args.n, min_k=0)
    loci = args.loci.split(",")
    validate_named_axis(loci, "loci", {"early", "mid", "late"})
    protocol = ProtocolConfig(option_order=args.option_order)
    config = {
        "schema_version": 2, "model": args.model, "task": args.task,
        "ks": ks, "n": args.n, "seeds": seeds, "loci": loci,
        "img_size": args.img_size, "protocol": protocol.metadata(),
        "intervention": "bidirectional_final_query_residual_patch",
        "patch_policy": "once_per_generation_prefill_v2",
        "generation": {"max_new_tokens": 6, "do_sample": False,
                       "num_beams": 1, "num_return_sequences": 1},
        "pair": args.pair,
        "model_revision": model_revision(args.model),
        "dtype": "bfloat16",
        "software_versions": software_versions(),
    }
    result = load_result(args.out, config, {
        "seeds_done": [], "evidence": {}, "per_seed": {},
    })

    model, processor = load_model(args.model, dtype=torch.bfloat16)
    model.requires_grad_(False).eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("frozen intervention has trainable parameters")
    layers = get_layers(model)
    selected = {locus: layers[locus_idx(len(layers), locus)] for locus in loci}
    result["layer_indices"] = {
        locus: locus_idx(len(layers), locus) for locus in loci}
    result["n_model_layers"] = len(layers)
    result["trainable_parameters"] = 0
    names, pool, test = TASKS[args.task](96, 120)
    modes = ["source", "target"]
    for locus in loci:
        modes.extend((f"source_to_target_{locus}", f"target_to_source_{locus}"))

    for seed in seeds:
        if seed in result["seeds_done"]:
            continue
        seed_cells = {}
        seed_summary = {}
        for k in ks:
            rng = random.Random(1000 * seed + k)
            items = sample_balanced_queries(test, args.n, rng)
            records = []
            for query_index, (query_image, gold) in enumerate(items):
                source_demos = sample_balanced_demos(
                    pool, k, rng, class_policy=protocol.demo_class_policy)
                remap_space = sample_label_space(
                    names, "remap", rng, mapping_mode=protocol.mapping_mode,
                    token_set=protocol.remap_token_set)
                if args.pair == "standard_remap":
                    target_demos = source_demos
                    source_space = list(names)
                    target_space = remap_space
                    shuffle_map = None
                else:
                    target_demos, shuffle_map = corrupt_demo_labels(
                        source_demos, range(len(names)), rng)
                    source_space = remap_space
                    target_space = remap_space
                option_order = sample_option_order(len(names), seed, k, query_index,
                                                   protocol.option_order)
                source_messages = build_messages(
                    source_demos, query_image, source_space, args.img_size,
                    option_order=option_order)
                target_messages = build_messages(
                    target_demos, query_image, target_space, args.img_size,
                    option_order=option_order)
                source_vectors = cache_query_residuals(
                    model, processor, source_messages, selected)
                target_vectors = cache_query_residuals(
                    model, processor, target_messages, selected)
                raw = {
                    "source": generate_with_patch(model, processor, source_messages),
                    "target": generate_with_patch(model, processor, target_messages),
                }
                for locus, layer in selected.items():
                    raw[f"source_to_target_{locus}"] = generate_with_patch(
                        model, processor, target_messages, layer, source_vectors[locus])
                    raw[f"target_to_source_{locus}"] = generate_with_patch(
                        model, processor, source_messages, layer, target_vectors[locus])
                predictions = {}
                for mode, text in raw.items():
                    output_space = (target_space if mode == "target" or
                                    mode.startswith("source_to_target_") else source_space)
                    predictions[mode] = parse_answer(text, output_space).prediction
                records.append({
                    "query_index": query_index, "gold": gold, "k": k,
                    "query_id": image_id(query_image),
                    "source_label_space": source_space,
                    "target_label_space": target_space,
                    "option_order": option_order,
                    "source_displayed_options": [source_space[i] for i in option_order],
                    "target_displayed_options": [target_space[i] for i in option_order],
                    "shuffle_map": shuffle_map,
                    "demo_source_classes": [label for _, label in source_demos],
                    "demo_source_ids": [image_id(image) for image, _ in source_demos],
                    "demo_target_classes": [label for _, label in target_demos],
                    "raw_outputs": raw, "predictions": predictions,
                })
            seed_cells[str(k)] = records
            seed_summary[str(k)] = _summary(
                records, modes, include_wrong_map=args.pair == "remap_corrupt")
        result["evidence"][str(seed)] = seed_cells
        result["per_seed"][str(seed)] = seed_summary
        result["seeds_done"].append(seed)
        write_json_atomic(args.out, result)
        print(f"seed {seed} done", flush=True)

    metrics = ["accuracy", "unparseable_rate"]
    if args.pair == "remap_corrupt":
        metrics.append("wrong_map_adherence")
    result["summary"] = {
        str(k): {
            mode: {
                metric: mean_t_ci([
                    result["per_seed"][str(seed)][str(k)][mode][metric]
                    for seed in seeds])
                for metric in metrics
            }
            for mode in modes
        }
        for k in ks
    }
    write_json_atomic(args.out, result)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
