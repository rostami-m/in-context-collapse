"""Adapter target accounting and deterministic parameter-budget selection."""
from __future__ import annotations

import math
import re

import torch


def target_parameter_costs(model, targets, rank: int) -> dict[str, int]:
    """Return the exact LoRA A+B parameter count implied by each Linear target."""
    if type(rank) is not int or rank < 1:
        raise ValueError("LoRA rank must be a positive integer")
    modules = dict(model.named_modules())
    costs = {}
    for name in targets:
        module = modules.get(name)
        if not isinstance(module, torch.nn.Linear):
            raise ValueError(f"LoRA target is not a Linear module: {name}")
        costs[name] = rank * (module.in_features + module.out_features)
    return costs


def target_parameter_budget(model, targets, rank: int) -> int:
    return sum(target_parameter_costs(model, targets, rank).values())


def _evenly_spaced_indices(length: int, count: int) -> list[int]:
    if count >= length:
        return list(range(length))
    if count == 1:
        return [length // 2]
    return [round(i * (length - 1) / (count - 1)) for i in range(count)]


def select_targets_by_budget(model, targets, rank: int, budget: int):
    """Choose depth-spread whole modules without exceeding a LoRA parameter budget.

    Whole Linear modules are the indivisible unit supported by PEFT. Consequently,
    exact equality is not always possible across architectures or module shapes;
    callers must report both the requested and achieved budgets.
    """
    if budget <= 0:
        raise ValueError("parameter budget must be positive")
    targets = list(dict.fromkeys(targets))
    costs = target_parameter_costs(model, targets, rank)
    if sum(costs.values()) <= budget:
        return targets, sum(costs.values())
    if not targets or min(costs.values()) > budget:
        return [], 0

    typical = sorted(costs.values())[len(costs) // 2]
    guess = max(1, min(len(targets), math.ceil(budget / typical)))
    priority = _evenly_spaced_indices(len(targets), guess)
    priority += [index for index in range(len(targets)) if index not in set(priority)]
    selected = []
    used = 0
    for index in priority:
        name = targets[index]
        if used + costs[name] <= budget:
            selected.append(name)
            used += costs[name]
    selected.sort(key=targets.index)
    return selected, used


def select_targets_by_depth_profile(targets, suffixes) -> list[str]:
    """Select the same component profile at evenly spaced depths in a band."""
    targets = list(targets)
    layer_ids = sorted({
        int(match.group(1))
        for target in targets
        if (match := re.search(r"layers\.(\d+)\.", target))
    })
    suffixes = list(suffixes)
    if not suffixes:
        raise ValueError("target profile must contain at least one suffix")
    if len(layer_ids) < len(suffixes):
        raise ValueError("not enough layers for the requested target profile")
    selected = []
    for index, suffix in zip(_evenly_spaced_indices(len(layer_ids), len(suffixes)), suffixes):
        layer_id = layer_ids[index]
        matches = [
            target for target in targets
            if re.search(rf"layers\.{layer_id}\.", target) and target.endswith(suffix)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected one {suffix} target at layer {layer_id}, found {matches}")
        selected.append(matches[0])
    return selected
