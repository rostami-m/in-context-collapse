"""Region-restricted adaptation tests for repairing in-context collapse.

For each (region, component) we attach a rank-``r`` LoRA adapter restricted to
that region, train it on REMAP episodes (Eq. for the rescue objective in the
paper), and re-measure the standard/remap K-curves. The central result is a
depth-sensitive profile at matched capacity. These tests identify sites where added capacity is
sufficient to restore behavior; they do not identify the frozen model's unique
causal failure site.

Two arms:
  remove  rescue a collapse-prone model (default; train to use the demonstrations)
  induce  the converse on a robust learner (train to ignore demonstrations)

Resumable: each completed region:component is written to ``--out`` and skipped on
re-run. Multi-seed runs report per-K means and two-sided 95% Student-t intervals.

Example (second-architecture replication, 3 seeds, both tasks)::

    python -m incontext_collapse.circuit --model Qwen/Qwen2.5-VL-3B-Instruct \\
        --task shapes4 --seeds 0,1,2 --out data/figdata/circuit_qwen25_shapes.json
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model

from .adapters import (select_targets_by_budget, select_targets_by_depth_profile,
                       target_parameter_budget)
from .core import (TASKS, build_messages, build_messages_multiturn, region_targets)
from .evaluation import evaluate_classification_cell, mean_t_ci
from .models import load_model
from .protocol import (ProtocolConfig, sample_label_space, sample_training_episode,
                       seed_training, sample_option_order, validate_evaluation_grid, validate_named_axis)
from .provenance import model_revision, software_versions
from .results import load_result, write_json_atomic
from .scoring import encode_supervised_answer

# Selected at runtime by --format; the reported panel uses single-turn prompts.
BUILD = build_messages


class AdapterIsolationError(RuntimeError):
    """The shared base may still contain an adapter and must not be reused."""


def load_compatible_baseline(path, *, model, task, ks, n, seeds, score_mode,
                             protocol, prompt_format, model_revision, dtype,
                             software_versions):
    """Reuse a baseline only under the same pinned inference configuration."""
    if not isinstance(model_revision, str) or not model_revision.strip():
        raise ValueError("baseline import requires a pinned model revision")
    if prompt_format not in {"single", "multi"}:
        raise ValueError("baseline import requires an explicit prompt format")
    if not isinstance(dtype, str) or not dtype.strip():
        raise ValueError("baseline import requires an explicit dtype")
    if (not isinstance(software_versions, dict) or not software_versions
            or any(not isinstance(value, str) or not value.strip()
                   for value in software_versions.values())):
        raise ValueError("baseline import requires explicit software versions")
    source = json.loads(Path(path).read_text())
    config = source.get("run_config", {})
    if not isinstance(config, dict):
        raise ValueError("imported baseline has no valid run configuration")
    expected = {
        "model": model,
        "task": task,
        "n": n,
        "seeds": seeds,
        "score_mode": score_mode,
        "protocol": protocol.metadata(),
        "format": prompt_format,
        "model_revision": model_revision,
        "dtype": dtype,
        "software_versions": software_versions,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if key not in config or config[key] != value
    }
    if not isinstance(config.get("ks"), list):
        raise ValueError("imported baseline has no valid ks metadata")
    source_ks = set(config["ks"])
    if not set(ks).issubset(source_ks):
        mismatches["ks"] = (sorted(source_ks), sorted(ks))
    if mismatches:
        raise ValueError(f"incompatible imported baseline: {mismatches}")
    if "baseline" not in source or "baseline_evidence" not in source:
        raise ValueError("imported result has no baseline evidence")

    def subset_curve(curve):
        subset = {str(k): curve.get(str(k), curve.get(k)) for k in ks}
        if any(value is None for value in subset.values()):
            raise ValueError("imported baseline is missing requested K values")
        return subset

    baseline = {}
    for name in ("standard", "remap"):
        baseline[name] = subset_curve(source["baseline"][name])
        ci_name = name + "_ci95"
        baseline[ci_name] = subset_curve(source["baseline"][ci_name])
    evidence = []
    for seed_record in source["baseline_evidence"]:
        evidence.append({
            "seed": seed_record["seed"],
            "standard": subset_curve(seed_record["standard"]),
            "remap": subset_curve(seed_record["remap"]),
        })
    if sorted(record["seed"] for record in evidence) != sorted(seeds):
        raise ValueError("imported baseline evidence does not contain the requested seeds")
    return baseline, evidence


@torch.no_grad()
def _generate(model, proc, messages, max_new_tokens=6):
    inp = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                   return_dict=True, return_tensors="pt").to(model.device)
    out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                         num_beams=1, num_return_sequences=1)
    return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def eval_kcurve(model, proc, names, pool, test, ks, remap, n, size=112, seed=0,
                protocol=None, score_mode="generation", return_details=False):
    """Accuracy at each K through the same evaluator used by the main panel."""
    protocol = protocol or ProtocolConfig()
    acc = {}
    details = {}
    for k in ks:
        cell = evaluate_classification_cell(
            model, proc, names, pool, test, "remap" if remap else "standard",
            k, n, size, seed, BUILD, _generate, protocol=protocol,
            score_mode=score_mode)
        acc[k] = round(cell["summary"]["accuracy"], 3)
        details[k] = cell
    return (acc, details) if return_details else acc


def train_region(model, proc, names, pool, test, targets, arm,
                 k=8, steps=150, lr=1e-4, size=112, seed=0, rank=8,
                 objective=None, demo_class_policy="random_balanced", option_order="random"):
    """Train a LoRA under an explicit, reproducible supervision objective.

    ``natural`` uses natural demonstrations and targets. ``episode_remap`` samples
    a fresh class-to-nonce map per episode and targets its query nonce.
    ``conflicting_remap_natural`` displays a randomized remap but targets the
    natural class; it is reserved for the collapse-induction arm.
    """
    objective = objective or ("episode_remap" if arm == "remove"
                              else "conflicting_remap_natural")
    if objective not in {"natural", "episode_remap", "conflicting_remap_natural"}:
        raise ValueError(f"unknown training objective: {objective}")
    if type(steps) is not int or steps < 1:
        raise ValueError("training steps must be a positive integer")
    if type(rank) is not int or rank < 1:
        raise ValueError("LoRA rank must be a positive integer")
    if not targets:
        raise ValueError("adapter target list is empty")
    seed_training(seed)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  # cut activation memory
    try:
        model.config.use_cache = False
    except Exception:
        pass
    pm = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank,
                                          target_modules=targets, lora_dropout=0.05))
    pm.train()
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=lr)
    rng = random.Random(seed)
    losses = []
    parameters_with_grad = 0
    for episode_index in range(steps):
        q_img, q_lab, demos = sample_training_episode(
            pool, k, rng, class_policy=demo_class_policy)
        if objective == "natural":
            space = list(names)
            target = names[q_lab]
        else:
            space = sample_label_space(names, "remap", rng, mapping_mode="episode",
                                       token_set="train")
            target = (space if objective == "episode_remap" else names)[q_lab]
        order = sample_option_order(len(names), seed, k, episode_index,
                                    option_order, phase="training")
        m = BUILD(demos, q_img, space, size, option_order=order)
        inp, lab, _ = encode_supervised_answer(proc, m, target, model.device)
        loss = pm(**inp, labels=lab).loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training loss: {float(loss.detach().cpu())}")
        loss.backward()
        parameters_with_grad = sum(
            parameter.grad is not None
            for parameter in pm.parameters() if parameter.requires_grad)
        if parameters_with_grad == 0:
            raise RuntimeError("no trainable adapter parameter received a gradient")
        losses.append(float(loss.detach().cpu()))
        opt.step()
        opt.zero_grad()
    pm.eval()
    try:
        model.config.use_cache = True
    except Exception:
        pass
    pm.revision_training_metadata = {
        "objective": objective,
        "torch_seed": seed,
        "steps": steps,
        "k": k,
        "rank": rank,
        "learning_rate": lr,
        "demo_class_policy": demo_class_policy,
        "option_order": option_order,
        "n_targets": len(targets),
        "targets": list(targets),
        "trainable_parameters": sum(
            parameter.numel() for parameter in pm.parameters() if parameter.requires_grad),
        "parameters_with_grad_last_step": parameters_with_grad,
        "losses": losses,
    }
    return pm


@contextmanager
def isolated_trained_adapter(model, *args, **kwargs):
    """Unload a returned adapter, or stop if its base cannot be safely reused."""
    try:
        pm = train_region(model, *args, **kwargs)
    except Exception as error:
        # PEFT mutates the base during attachment, before a wrapper is returned.
        raise AdapterIsolationError(
            "adapter training failed before returning a recoverable wrapper; "
            "refusing to reuse the shared base model") from error
    try:
        yield pm
    finally:
        try:
            pm.unload()
        except Exception as error:
            raise AdapterIsolationError(
                "adapter cleanup failed; refusing to reuse the shared base model") from error


def aggregate(fn, ks, seeds):
    """fn(seed) -> {curve_name: {k: acc}}; reduce to per-K mean + 95% CI half-width over seeds."""
    acc: dict[str, dict] = {}
    for s in seeds:
        for name, curve in fn(s).items():
            d = acc.setdefault(name, {k: [] for k in ks})
            for k in ks:
                d[k].append(curve[k])
    res = {}
    for name, d in acc.items():
        stats = {k: mean_t_ci(d[k]) for k in ks}
        res[name] = {k: round(stats[k]["mean"], 3) for k in ks}
        res[name + "_ci95"] = {k: {
            "low": (round(stats[k]["ci95_low"], 3) if stats[k]["ci95_low"] is not None else None),
            "high": (round(stats[k]["ci95_high"], 3) if stats[k]["ci95_high"] is not None else None),
            "seed_values": d[k],
        } for k in ks}
    return res


def aggregate_stored_curves(seed_curves, ks):
    """Aggregate persisted ``{seed, standard, remap}`` curve records."""
    acc: dict[str, dict[int, list[float]]] = {}
    for record in seed_curves:
        for name in ("standard", "remap"):
            curve = record[name]
            values = acc.setdefault(name, {k: [] for k in ks})
            for k in ks:
                values[k].append(curve.get(str(k), curve.get(k)))
    res = {}
    for name, values in acc.items():
        stats = {k: mean_t_ci(values[k]) for k in ks}
        res[name] = {k: round(stats[k]["mean"], 3) for k in ks}
        res[name + "_ci95"] = {k: {
            "low": (round(stats[k]["ci95_low"], 3)
                    if stats[k]["ci95_low"] is not None else None),
            "high": (round(stats[k]["ci95_high"], 3)
                     if stats[k]["ci95_high"] is not None else None),
            "seed_values": values[k],
        } for k in ks}
    return res


def main():
    global BUILD
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", choices=["remove", "induce"], default="remove")
    ap.add_argument("--task", default="cifar4")
    ap.add_argument("--regions", default="connector,early,mid,late,all")
    ap.add_argument("--comps", default="both,attn,mlp", help="components per language-model region")
    ap.add_argument("--ks", default="0,1,2,4,8,16")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--parameter-budget", type=int, default=0,
                    help="maximum LoRA parameters per cell; 0 uses every target")
    ap.add_argument("--match-parameter-budget", action="store_true",
                    help="use the smallest candidate budget for every requested cell")
    ap.add_argument("--target-profile", choices=["none", "q_up_v"], default="none",
                    help="component-matched sparse profile for LM depth comparisons")
    ap.add_argument("--objective", choices=["natural", "episode_remap",
                                             "conflicting_remap_natural"],
                    default=None, help="defaults to episode_remap for remove, conflicting for induce")
    ap.add_argument("--score-mode", choices=["generation", "likelihood", "both"],
                    default="both")
    ap.add_argument("--seeds", default="0", help="comma-separated seeds")
    ap.add_argument("--format", choices=["single", "multi"], default="single")
    ap.add_argument("--option-order", choices=["random", "class"], default="random",
                    help="independent displayed-option permutation; class reproduces Protocol 2.4")
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--baseline-from", default=None,
                    help="reuse a compatible baseline result instead of rerunning it")
    ap.add_argument("--adapters-dir", default=None,
                    help="checkpoint directory; defaults beside --out")
    ap.add_argument("--out", default="data/figdata/circuit_v2.json")
    a = ap.parse_args()
    if a.parameter_budget and a.match_parameter_budget:
        ap.error("choose --parameter-budget or --match-parameter-budget, not both")
    BUILD = build_messages_multiturn if a.format == "multi" else build_messages
    ks = [int(x) for x in a.ks.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]
    validate_evaluation_grid(seeds, ks, a.n)
    validate_named_axis(a.regions.split(","), "regions", {"connector", "early", "mid", "late", "all"})
    validate_named_axis(a.comps.split(","), "components", {"both", "attn", "mlp"})
    if a.steps < 1 or a.rank < 1 or a.parameter_budget < 0:
        ap.error("steps/rank must be positive and parameter budget nonnegative")
    objective = a.objective or ("episode_remap" if a.arm == "remove"
                                else "conflicting_remap_natural")
    protocol = ProtocolConfig(option_order=a.option_order)
    adapters_dir = Path(a.adapters_dir or str(Path(a.out).with_suffix("")) + "_adapters")
    run_config = {
        "schema_version": 2, "model": a.model, "arm": a.arm, "task": a.task,
        "regions": a.regions, "components": a.comps, "ks": ks, "n": a.n,
        "steps": a.steps, "rank": a.rank, "parameter_budget": a.parameter_budget,
        "match_parameter_budget": a.match_parameter_budget, "objective": objective,
        "target_profile": a.target_profile,
        "score_mode": a.score_mode, "seeds": seeds, "format": a.format,
        "adapters_dir": str(adapters_dir),
        "model_revision": model_revision(a.model),
        "dtype": "bfloat16",
        "software_versions": software_versions(),
        "protocol": protocol.metadata(),
        "adapter_isolation_policy": "fail_closed_v1",
        "baseline_import_policy": "strict_runtime_v1",
        "completion_policy": "seed_evidence_required_v2",
        "generation_policy": "explicit_greedy_single_output_v1",
    }

    model, proc = load_model(a.model, dtype=torch.bfloat16)
    names, pool, test = TASKS[a.task](96, 120)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    R = load_result(a.out, run_config, {
        "model": a.model, "arm": a.arm, "task": a.task, "ks": ks,
        "objective": objective, "protocol": protocol.metadata(),
        "score_mode": a.score_mode})

    def dump():
        write_json_atomic(a.out, R)

    if "baseline" not in R:
        if a.baseline_from:
            R["baseline"], R["baseline_evidence"] = load_compatible_baseline(
                a.baseline_from, model=a.model, task=a.task, ks=ks, n=a.n,
                seeds=seeds, score_mode=a.score_mode, protocol=protocol,
                prompt_format=run_config["format"],
                model_revision=run_config["model_revision"],
                dtype=run_config["dtype"],
                software_versions=run_config["software_versions"])
            R["baseline_imported_from"] = a.baseline_from
        else:
            baseline_evidence = []

            def baseline_cell(seed):
                standard, standard_details = eval_kcurve(
                    model, proc, names, pool, test, ks, False, a.n, seed=seed,
                    protocol=protocol, score_mode=a.score_mode, return_details=True)
                remap, remap_details = eval_kcurve(
                    model, proc, names, pool, test, ks, True, a.n, seed=seed,
                    protocol=protocol, score_mode=a.score_mode, return_details=True)
                baseline_evidence.append({"seed": seed, "standard": standard_details,
                                          "remap": remap_details})
                return {"standard": standard, "remap": remap}

            R["baseline"] = aggregate(baseline_cell, ks, seeds)
            R["baseline_evidence"] = baseline_evidence
        dump()
        print("baseline:", R["baseline"], flush=True)
    if a.baseline_only:
        print("DONE-baseline")
        return

    comps = [c for c in a.comps.split(",") if c]
    candidates = []
    for region in a.regions.split(","):
        for component in (["both"] if region == "connector" else comps):
            candidate_targets = region_targets(model, region, component)
            candidate_budget = (target_parameter_budget(model, candidate_targets, a.rank)
                                if candidate_targets else 0)
            candidates.append((region, component, candidate_targets, candidate_budget))
    common_budget = (min(budget for _, _, targets, budget in candidates if targets)
                     if a.match_parameter_budget else 0)
    requested_budget = a.parameter_budget or common_budget
    R["budget_rule"] = {
        "rank": a.rank,
        "requested_parameter_budget": requested_budget or None,
        "matched_across_cells": a.match_parameter_budget,
        "whole_linear_modules": True,
    }

    for region, comp, candidate_targets, candidate_budget in candidates:
        key = f"{region}:{comp}"
        # Error-only cells must be retried, not mistaken for completed runs.
        if key in R and "seeds_done" not in R[key]:
            if "error" in R[key]:
                del R[key]
            else:
                raise ValueError(f"{key}: completion lacks seed-level evidence")
        try:
            targets = candidate_targets
            selected_budget = candidate_budget
            if a.target_profile != "none":
                if region == "connector":
                    raise ValueError("target profiles apply only to language-model regions")
                suffixes = {"q_up_v": ("q_proj", "up_proj", "v_proj")}[a.target_profile]
                targets = select_targets_by_depth_profile(targets, suffixes)
                selected_budget = target_parameter_budget(model, targets, a.rank)
                if requested_budget and selected_budget > requested_budget:
                    raise ValueError("target profile exceeds the requested parameter budget")
            elif requested_budget and targets:
                targets, selected_budget = select_targets_by_budget(
                    model, targets, a.rank, requested_budget)
            if not targets:
                R[key] = {"error": "no targets within budget",
                          "candidate_parameter_budget": candidate_budget,
                          "requested_parameter_budget": requested_budget}
                dump()
                continue

            cell_result = R.setdefault(key, {
                "n_candidate_targets": len(candidate_targets),
                "candidate_parameter_budget": candidate_budget,
                "n_targets": len(targets), "targets": targets,
                "selected_parameter_budget": selected_budget,
                "seeds_done": [], "seed_curves": [], "training_runs": [],
                "evidence": [], "seed_errors": {},
            })
            for s in seeds:
                if s in cell_result["seeds_done"]:
                    continue
                with isolated_trained_adapter(
                        model, proc, names, pool, test, targets, a.arm,
                        steps=a.steps, seed=s, rank=a.rank, objective=objective,
                        option_order=protocol.option_order) as pm:
                    adapter_path = adapters_dir / key.replace(":", "_") / f"seed_{s}"
                    pm.save_pretrained(adapter_path)
                    training_metadata = pm.revision_training_metadata
                    training_metadata["adapter_path"] = str(adapter_path)
                    standard, standard_details = eval_kcurve(
                        pm, proc, names, pool, test, ks, False, a.n, seed=s,
                        protocol=protocol, score_mode=a.score_mode, return_details=True)
                    remap, remap_details = eval_kcurve(
                        pm, proc, names, pool, test, ks, True, a.n, seed=s,
                        protocol=protocol, score_mode=a.score_mode, return_details=True)
                cell_result["training_runs"].append(training_metadata)
                cell_result["seed_curves"].append(
                    {"seed": s, "standard": standard, "remap": remap})
                cell_result["evidence"].append(
                    {"seed": s, "standard": standard_details,
                     "remap": remap_details})
                cell_result["seeds_done"].append(s)
                cell_result["seed_errors"].pop(str(s), None)
                cell_result.update(aggregate_stored_curves(
                    cell_result["seed_curves"], ks))
                dump()
        except Exception as e:  # noqa: BLE001 - record and continue to next region
            if key in R and "seeds_done" in R[key]:
                unfinished = next((s for s in seeds if s not in R[key]["seeds_done"]), None)
                R[key]["seed_errors"][str(unfinished)] = repr(e)[:200]
            else:
                R[key] = {"error": repr(e)[:200]}
            if isinstance(e, AdapterIsolationError):
                dump()
                raise
        dump()
        print(key, R[key].get("remap", R[key].get("error")), flush=True)
    incomplete = [f"{region}:{component}" for region, component, _, _ in candidates
                  if sorted(R.get(f"{region}:{component}", {}).get("seeds_done", []))
                  != sorted(seeds)]
    if incomplete:
        print(f"INCOMPLETE: {incomplete}", flush=True)
        return 1
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
