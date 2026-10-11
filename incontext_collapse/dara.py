"""Versioned common-protocol evaluation of the DARA attention baseline.

DARA (Chen et al., 2025) trains a small matrix of per-head, per-image
amplification factors in the first Qwen2-VL decoder layer. The upstream release
uses Unsloth and CUDA. This module independently implements the same released
logit-scaling rule on the Transformers eager-attention path so it can run on
MPS/CPU while sharing this project's sampling, objectives, and evaluator.

This is a compatibility port, not the authors' untouched environment. Metadata
records the upstream commit and every compatibility choice.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
from types import MethodType

import torch
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
)

from .circuit import aggregate_stored_curves, eval_kcurve
from .core import TASKS, build_messages
from .inject import get_layers
from .models import load_model
from .protocol import (ProtocolConfig, sample_label_space, sample_option_order,
                       sample_training_episode, seed_training)
from .provenance import model_revision, software_versions
from .results import load_result, write_json_atomic
from .scoring import encode_supervised_answer

UPSTREAM_REPOSITORY = "https://github.com/chenxshuo/true-micl"
UPSTREAM_COMMIT = "36e8a1910b3eb2aa6e339ca19cca32bbb27e8edc"


def explicit_causal_mask(attention_logits, attention_mask, cache_position=None):
    """Restore causality when Transformers omits a mask for native SDPA."""
    if attention_mask is not None:
        if attention_mask.ndim != 4:
            raise ValueError("custom eager attention requires a four-dimensional mask")
        return attention_mask[:, :, :, :attention_logits.shape[-1]]
    query_length, key_length = attention_logits.shape[-2:]
    if cache_position is None:
        cache_position = torch.arange(key_length - query_length, key_length,
                                      device=attention_logits.device)
    future = torch.arange(key_length, device=attention_logits.device)[None, :] > cache_position[:, None]
    return torch.zeros((query_length, key_length), dtype=attention_logits.dtype,
                       device=attention_logits.device).masked_fill(
                           future, torch.finfo(attention_logits.dtype).min)[None, None]


def image_token_ranges(input_ids: torch.Tensor, image_token_id: int) -> list[tuple[int, int]]:
    """Return inclusive contiguous image-token ranges for one tokenized prompt."""
    if input_ids.ndim != 1:
        raise ValueError("image_token_ranges expects one unbatched token sequence")
    positions = torch.nonzero(input_ids == image_token_id, as_tuple=False).flatten().tolist()
    if not positions:
        return []
    ranges = []
    start = previous = positions[0]
    for position in positions[1:]:
        if position != previous + 1:
            ranges.append((start, previous))
            start = position
        previous = position
    ranges.append((start, previous))
    return ranges


def amplify_attention_logits(attention_logits: torch.Tensor,
                             factors: torch.Tensor,
                             ranges: list[tuple[int, int]]) -> torch.Tensor:
    """Apply the released DARA per-head, per-image logit multiplication rule."""
    if attention_logits.ndim != 4:
        raise ValueError("attention logits must have shape [batch, heads, query, key]")
    if factors.ndim != 2 or factors.shape[0] != attention_logits.shape[1]:
        raise ValueError("factors must have shape [heads, image_slots]")
    if attention_logits.shape[0] != 1:
        raise ValueError("the compatibility baseline currently requires batch size one")
    key_length = attention_logits.shape[-1]
    scale = torch.ones(
        (1, attention_logits.shape[1], 1, key_length),
        device=attention_logits.device,
        dtype=attention_logits.dtype,
    )
    for slot, (start, end) in enumerate(ranges):
        if start < 0 or end < start or start >= key_length:
            continue
        end = min(end, key_length - 1)
        column = slot % factors.shape[1]
        scale[:, :, :, start:end + 1] = factors[:, column].to(
            device=attention_logits.device, dtype=attention_logits.dtype
        ).view(1, -1, 1, 1)
    return attention_logits * scale


def _dara_attention_forward(self, hidden_states, attention_mask=None,
                            position_ids=None, past_key_value=None,
                            output_attentions=False, use_cache=False,
                            cache_position=None, position_embeddings=None, **_kwargs):
    """Qwen2-VL eager attention with DARA applied before mask and softmax."""
    batch_size, query_length, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        batch_size, query_length, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(
        batch_size, query_length, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(
        batch_size, query_length, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"])
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    attention_logits = torch.matmul(
        query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    attention_logits = amplify_attention_logits(
        attention_logits, self.dara_factors, self.dara_image_ranges)

    attention_logits = attention_logits + explicit_causal_mask(
        attention_logits, attention_mask, cache_position)
    if query_states.dtype == torch.float16:
        attention_logits = torch.where(
            torch.isinf(attention_logits), torch.zeros_like(attention_logits), attention_logits)
    attention_weights = torch.nn.functional.softmax(
        attention_logits, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attention_weights = torch.nn.functional.dropout(
        attention_weights, p=self.attention_dropout, training=self.training)
    attention_output = torch.matmul(attention_weights, value_states)
    expected_shape = (batch_size, self.num_heads, query_length, self.head_dim)
    if attention_output.size() != expected_shape:
        raise ValueError(
            f"attention output should have shape {expected_shape}, got {attention_output.size()}")
    attention_output = attention_output.transpose(1, 2).contiguous().reshape(
        batch_size, query_length, -1)
    attention_output = self.o_proj(attention_output)
    if not output_attentions:
        attention_weights = None
    return attention_output, attention_weights, past_key_value


def install_dara(model, *, max_images: int = 8):
    """Freeze ``model`` and install trainable DARA factors in decoder layer zero."""
    if getattr(model.config, "model_type", "").lower() != "qwen2_vl":
        raise ValueError("DARA compatibility baseline currently supports Qwen2-VL only")
    if not isinstance(max_images, int) or isinstance(max_images, bool) or max_images < 1:
        raise ValueError("max_images must be a positive integer")
    layers = get_layers(model)
    attention = layers[0].self_attn
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameter_device = next(attention.parameters()).device
    attention.register_parameter(
        "dara_factors",
        torch.nn.Parameter(torch.ones(
            attention.num_heads, max_images, dtype=torch.float32, device=parameter_device)),
    )
    # The official release scales the amplifier gradient by 0.1.
    attention.dara_factors.register_hook(lambda gradient: gradient * 0.1)
    attention.dara_image_ranges = []
    attention.forward = MethodType(_dara_attention_forward, attention)

    image_token_id = model.config.image_token_id

    def update_ranges(_module, args, kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is None or input_ids.ndim != 2 or input_ids.shape[-1] <= 1:
            return
        if input_ids.shape[0] != 1:
            raise ValueError("DARA compatibility baseline requires batch size one")
        attention.dara_image_ranges = image_token_ranges(input_ids[0], image_token_id)

    hook = model.register_forward_pre_hook(update_ranges, with_kwargs=True)
    return attention, hook


def train_dara(model, processor, names, pool, *, steps: int, k: int,
               learning_rate: float, seed: int, objective: str = "episode_remap",
               size: int = 112, option_order: str = "random"):
    """Train only the installed DARA factors on natural or remapped labels."""
    if objective not in {"natural", "episode_remap"}:
        raise ValueError(f"unsupported DARA objective: {objective}")
    seed_training(seed)
    attention = get_layers(model)[0].self_attn
    with torch.no_grad():
        attention.dara_factors.fill_(1.0)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()
    optimizer = torch.optim.AdamW(
        [attention.dara_factors], lr=learning_rate, weight_decay=0.0)
    rng = random.Random(seed)
    losses = []
    best_loss = float("inf")
    best_step = None
    best_factors = None
    for step in range(steps):
        query_image, query_label, demos = sample_training_episode(pool, k, rng)
        condition = "standard" if objective == "natural" else "remap"
        label_space = sample_label_space(
            names, condition, rng, mapping_mode="episode", token_set="train")
        target = label_space[query_label]
        order = sample_option_order(len(names), seed, k, step, option_order,
                                    phase="training")
        messages = build_messages(demos, query_image, label_space, size,
                                  option_order=order)
        inputs, labels, _ = encode_supervised_answer(
            processor, messages, target, model.device)
        loss = model(**inputs, labels=labels).loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite DARA training loss: {float(loss.detach().cpu())}")
        loss.backward()
        if attention.dara_factors.grad is None:
            raise RuntimeError("DARA amplification factors received no gradient")
        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        optimizer.step()
        optimizer.zero_grad()
        # This local policy differs from the release's final amplifier.pt dump.
        if loss_value < best_loss:
            best_loss = loss_value
            best_step = step + 1
            best_factors = attention.dara_factors.detach().clone()
        if (step + 1) % 25 == 0:
            print(f"DARA {objective} seed={seed} step={step + 1}/{steps} "
                  f"loss={loss_value:.4f}", flush=True)
    final_factors = attention.dara_factors.detach().float().cpu().tolist()
    if best_factors is None:
        raise RuntimeError("DARA training completed without a finite checkpoint")
    with torch.no_grad():
        attention.dara_factors.copy_(best_factors)
    model.eval()
    model.config.use_cache = True
    return {
        "objective": objective,
        "torch_seed": seed,
        "steps": steps,
        "k": k,
        "learning_rate": learning_rate,
        "optimizer": "torch.optim.AdamW",
        "weight_decay": 0.0,
        "scheduler": "constant",
        "batch_size": 1,
        "device": str(attention.dara_factors.device),
        "factor_dtype": str(attention.dara_factors.dtype),
        "trainable_parameters": attention.dara_factors.numel(),
        "max_images": attention.dara_factors.shape[1],
        "gradient_scale": 0.1,
        "losses": losses,
        "best_loss": best_loss,
        "best_step": best_step,
        "checkpoint_selection": "minimum per-step training loss (local snapshot policy)",
        "option_order": option_order,
        "final_factors": final_factors,
        "factors": attention.dara_factors.detach().float().cpu().tolist(),
    }


def load_factor_source(path, *, model_id, task, seeds, snapshot="final"):
    """Validate saved factors before loading any model or starting evaluation."""
    if snapshot not in {"selected", "final"}:
        raise ValueError("unknown factor snapshot")
    content = Path(path).read_bytes()
    source = json.loads(content)
    config = source.get("run_config", {})
    expected = {"model": model_id, "task": task,
                "model_revision": model_revision(model_id)}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"factor source mismatch: {key}")
    if config.get("method") != "DARA compatibility port" or config.get("implementation_version", 0) < 2:
        raise ValueError("factor source must be a causally corrected DARA artifact")
    if source.get("invalidated") or source.get("invalidation"):
        raise ValueError("invalidated factors cannot be evaluated")
    runs = {}
    for training in source.get("training_runs", []):
        seed = training["torch_seed"]
        if seed in runs:
            raise ValueError(f"duplicate training seed {seed}")
        selected = training.get("source_selected_factors", training.get("factors"))
        values = training.get("final_factors") if snapshot == "final" else selected
        if values is None:
            raise ValueError(f"missing {snapshot} factors for seed {seed}")
        factors = torch.as_tensor(values, dtype=torch.float32)
        if factors.ndim != 2 or factors.shape[1] != 8 or not torch.isfinite(factors).all():
            raise ValueError(f"invalid {snapshot} factors for seed {seed}")
        if factors.numel() != training.get("trainable_parameters"):
            raise ValueError(f"factor parameter count mismatch for seed {seed}")
        record = deepcopy(training)
        record["source_selected_factors"] = deepcopy(selected)
        record["factors"] = factors.tolist()
        record["evaluation_snapshot"] = snapshot
        record["checkpoint_selection"] = (
            "final training step, loaded from saved factors without retraining"
            if snapshot == "final" else "minimum per-step training loss (local snapshot policy)")
        record["option_order"] = config["protocol"].get("option_order", "class")
        runs[seed] = record
    if not set(seeds).issubset(runs):
        raise ValueError("factor source is missing requested training seeds")
    return source, runs, hashlib.sha256(content).hexdigest()


def restore_factors(attention, values):
    factors = torch.as_tensor(values, dtype=attention.dara_factors.dtype,
                              device=attention.dara_factors.device)
    if factors.shape != attention.dara_factors.shape or not torch.isfinite(factors).all():
        raise ValueError("saved factors are incompatible with the installed attention")
    with torch.no_grad():
        attention.dara_factors.copy_(factors)


def select_training_snapshot(attention, training, snapshot):
    """Honor the chosen evaluation snapshot without discarding either saved state."""
    if snapshot not in {"final", "selected"}:
        raise ValueError("unknown factor snapshot")
    result = deepcopy(training)
    result["source_selected_factors"] = deepcopy(training["factors"])
    values = training["final_factors"] if snapshot == "final" else training["factors"]
    restore_factors(attention, values)
    result["factors"] = deepcopy(values)
    result["evaluation_snapshot"] = snapshot
    result["checkpoint_selection"] = ("final training step" if snapshot == "final" else
        "post-update snapshot at the step with minimum pre-update training loss")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--task", default="shapes4")
    parser.add_argument("--ks", default="4,8,16")
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--train-k", type=int, default=8)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--objective", choices=["natural", "episode_remap"],
                        default="episode_remap")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--option-order", choices=["random", "class"], default="random")
    parser.add_argument("--eval-factors-from", type=Path,
                        help="evaluate saved DARA factors without training")
    parser.add_argument("--factor-snapshot", choices=["final", "selected"], default="final",
                        help="snapshot to evaluate, for fresh training or --eval-factors-from")
    parser.add_argument("--device")
    parser.add_argument("--score-mode", choices=["generation", "likelihood", "both"],
                        default="generation")
    parser.add_argument("--out", default="data/revision_v24_completion/dara_qwen2vl2b_v24.json")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.ks, args.n, args.steps, args.seeds = "4", 4, 1, "0"

    ks = [int(value) for value in args.ks.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]
    if not ks or len(set(ks)) != len(ks) or min(ks) < 0 or not seeds or len(set(seeds)) != len(seeds):
        parser.error("K values must be unique/nonnegative; seeds must be nonempty/unique")
    if args.task not in TASKS or not 0 < args.n <= 120 or args.train_k < 0 or args.steps < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("invalid task, sample count, training K/steps, or learning rate")
    protocol = ProtocolConfig(option_order=args.option_order)
    factor_source, source_runs, source_hash = None, None, None
    if args.eval_factors_from:
        if args.eval_factors_from.resolve() == Path(args.out).resolve():
            parser.error("factor-source and result paths must differ")
        factor_source, source_runs, source_hash = load_factor_source(
            args.eval_factors_from, model_id=args.model, task=args.task,
            seeds=seeds, snapshot=args.factor_snapshot)
        for name in ("objective", "steps", "train_k", "learning_rate"):
            setattr(args, name, factor_source["run_config"][name])
    run_config = {
        "schema_version": 2,
        "implementation_version": 3,
        "evaluation_snapshot": args.factor_snapshot,
        "causal_mask": "explicit, including when native SDPA omits the mask",
        "method": "DARA compatibility port",
        "model": args.model,
        "task": args.task,
        "ks": ks,
        "n": args.n,
        "train_k": args.train_k,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "objective": args.objective,
        "seeds": seeds,
        "score_mode": args.score_mode,
        "protocol": protocol.metadata(),
        "model_revision": model_revision(args.model),
        "dtype": "bfloat16",
        "software_versions": software_versions(),
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "compatibility_notes": {
            "runtime": "Transformers eager attention on MPS/CPU instead of Unsloth CUDA",
            "rule": "first-layer per-head/per-image pre-softmax logit multiplication",
            "max_images": 8,
            "image_slot_overflow": "modulo 8, matching upstream release",
            "gradient_scale": 0.1,
        },
    }
    if factor_source is not None:
        run_config["factor_evaluation"] = {
            "source": str(args.eval_factors_from), "sha256": source_hash,
            "snapshot": args.factor_snapshot, "training_performed": False,
            "source_training_protocol": factor_source["run_config"]["protocol"],
        }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    result = load_result(args.out, run_config, {
        "method": "DARA compatibility port",
        "protocol": protocol.metadata(),
        "seeds_done": [],
        "seed_curves": [],
        "training_runs": [],
        "evidence": [],
    })

    model, processor = load_model(args.model, dtype=torch.bfloat16, device=args.device)
    names, pool, test = TASKS[args.task](96, 120)
    attention, hook = install_dara(model)
    try:
        if "baseline" not in result:
            baseline_curves = result.setdefault("baseline_seed_curves", [])
            baseline_evidence = result.setdefault("baseline_evidence", [])
            for seed in seeds:
                if any(curve["seed"] == seed for curve in baseline_curves):
                    continue
                standard, standard_details = eval_kcurve(
                    model, processor, names, pool, test, ks, False, args.n,
                    seed=seed, protocol=protocol, score_mode=args.score_mode,
                    return_details=True)
                remap, remap_details = eval_kcurve(
                    model, processor, names, pool, test, ks, True, args.n,
                    seed=seed, protocol=protocol, score_mode=args.score_mode,
                    return_details=True)
                baseline_curves.append({"seed": seed, "standard": standard, "remap": remap})
                baseline_evidence.append({
                    "seed": seed, "standard": standard_details, "remap": remap_details})
                write_json_atomic(args.out, result)
                print(f"DARA baseline seed={seed}: standard={standard} remap={remap}",
                      flush=True)
            result["baseline"] = aggregate_stored_curves(baseline_curves, ks)
            result["baseline_evidence"] = baseline_evidence
            write_json_atomic(args.out, result)

        for seed in seeds:
            if seed in result["seeds_done"]:
                continue
            if source_runs is not None:
                training = deepcopy(source_runs[seed])
                restore_factors(attention, training["factors"])
                model.eval()
            else:
                training = train_dara(
                    model, processor, names, pool, steps=args.steps, k=args.train_k,
                    learning_rate=args.learning_rate, seed=seed, objective=args.objective,
                    option_order=args.option_order)
                training = select_training_snapshot(attention, training, args.factor_snapshot)
            standard, standard_details = eval_kcurve(
                model, processor, names, pool, test, ks, False, args.n,
                seed=seed, protocol=protocol, score_mode=args.score_mode,
                return_details=True)
            remap, remap_details = eval_kcurve(
                model, processor, names, pool, test, ks, True, args.n,
                seed=seed, protocol=protocol, score_mode=args.score_mode,
                return_details=True)
            result["training_runs"].append(training)
            result["seed_curves"].append(
                {"seed": seed, "standard": standard, "remap": remap})
            result["evidence"].append(
                {"seed": seed, "standard": standard_details, "remap": remap_details})
            result["seeds_done"].append(seed)
            result.update(aggregate_stored_curves(result["seed_curves"], ks))
            write_json_atomic(args.out, result)
            print(f"seed {seed}: remap={remap}", flush=True)
    finally:
        hook.remove()
    print("DONE", result.get("remap"), flush=True)


if __name__ == "__main__":
    main()
