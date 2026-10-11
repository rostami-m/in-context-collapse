"""Validate completion and per-query evidence for queued revision experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


_HEAVY_MODEL_MARKERS = ("7b-ov", "7B-Instruct", "mistral-7b", "1.5-7b", "11B-Vision")


def _check_records(cell: dict, n: int, location: str) -> None:
    records = cell.get("records", [])
    if len(records) != n:
        raise ValueError(f"{location}: expected {n} records, found {len(records)}")
    indices = [record.get("query_index") for record in records]
    if (n < 1 or any(type(index) is not int for index in indices)
            or set(indices) != set(range(n))):
        raise ValueError(f"{location}: duplicate or missing query indices")


def validate_paired(result: dict) -> None:
    config = result["run_config"]
    seeds = config["seeds"]
    conditions = config["conditions"]
    ks = config["ks"]
    n = config["n"]
    expected_cells = {
        f"{seed}:{state}:{condition}:{k}"
        for seed in seeds for state in ("frozen", "repaired")
        for condition in conditions for k in ks
    }
    actual_cells = set(result.get("cells_done", []))
    if actual_cells != expected_cells:
        raise ValueError(
            f"paired cells incomplete: {len(actual_cells)}/{len(expected_cells)}")
    if sorted(result.get("seeds_done", [])) != sorted(seeds):
        raise ValueError("paired seeds_done is incomplete")
    for seed in seeds:
        for state in ("frozen", "repaired"):
            for condition in conditions:
                for k in ks:
                    cell = result["evidence"][str(seed)][state][condition][str(k)]
                    _check_records(cell, n, f"{seed}:{state}:{condition}:{k}")


def validate_circuit(result: dict, required_keys: list[str]) -> None:
    config = result["run_config"]
    seeds, ks, n = config["seeds"], config["ks"], config["n"]
    for key in required_keys:
        cell = result.get(key, {})
        if cell.get("error") or cell.get("seed_errors"):
            raise ValueError(f"{key}: recorded error: {cell.get('error') or cell.get('seed_errors')}")
        if sorted(cell.get("seeds_done", [])) != sorted(seeds):
            raise ValueError(f"{key}: seeds incomplete")
        if len(cell.get("training_runs", [])) != len(seeds):
            raise ValueError(f"{key}: training metadata incomplete")
        evidence_by_seed = {entry["seed"]: entry for entry in cell.get("evidence", [])}
        for seed in seeds:
            for condition in ("standard", "remap"):
                for k in ks:
                    detail = evidence_by_seed[seed][condition][str(k)]
                    _check_records(detail, n, f"{key}:{seed}:{condition}:{k}")


def validate_adapter_eval(result: dict) -> None:
    config = result["run_config"]
    seeds, conditions, ks, n = (
        config["seeds"], config["conditions"], config["ks"], config["n"])
    if sorted(result.get("seeds_done", [])) != sorted(seeds):
        raise ValueError("adapter evaluation seeds incomplete")
    for seed in seeds:
        for condition in conditions:
            for k in ks:
                cell = result["evidence"][str(seed)][condition][str(k)]
                _check_records(cell, n, f"{seed}:{condition}:{k}")


def validate_calibration(result: dict) -> None:
    config = result["run_config"]
    seeds, conditions, ks, n = (
        config["seeds"], config["conditions"], config["ks"], config["n"])
    expected = {
        f"{seed}:{condition}:{k}"
        for seed in seeds for condition in conditions for k in ks
    }
    if set(result.get("cells_done", [])) != expected:
        raise ValueError("calibration cells incomplete")
    if sorted(result.get("seeds_done", [])) != sorted(seeds):
        raise ValueError("calibration seeds incomplete")
    for seed in seeds:
        for condition in conditions:
            for k in ks:
                _check_records(
                    result["evidence"][str(seed)][condition][str(k)], n,
                    f"{seed}:{condition}:{k}")


def validate_frozen_features(result: dict) -> None:
    config = result["run_config"]
    seeds, conditions, ks, n = (
        config["seeds"], config["conditions"], config["ks"], config["n"])
    if sorted(result.get("seeds_done", [])) != sorted(seeds):
        raise ValueError("frozen-feature seeds incomplete")
    for seed in seeds:
        for condition in conditions:
            for k in ks:
                _check_records(
                    result["evidence"][str(seed)][condition][str(k)], n,
                    f"{seed}:{condition}:{k}")


def validate_vqa_train(result: dict) -> None:
    config = result["run_config"]
    seeds, ks, n = config["seeds"], config["ks"], config["n"]
    if sorted(result.get("seeds_done", [])) != sorted(seeds):
        raise ValueError("VQA training seeds incomplete")
    runs = result.get("training_runs", {})
    if set(runs) != {str(seed) for seed in seeds}:
        raise ValueError("VQA training metadata incomplete")
    for seed in seeds:
        run = runs[str(seed)]
        if run.get("torch_seed") != seed:
            raise ValueError(f"VQA seed {seed}: training seed mismatch")
        if len(run.get("losses", [])) != config["steps"]:
            raise ValueError(f"VQA seed {seed}: loss trace incomplete")
        adapter_path = Path(run.get("adapter_path", ""))
        if not adapter_path.is_dir() or not (adapter_path / "adapter_config.json").is_file():
            raise ValueError(f"VQA seed {seed}: adapter checkpoint missing")
        for arm in ("baseline", "repaired"):
            for k in ks:
                _check_records(
                    result["evidence"][str(seed)][arm][str(k)], n,
                    f"{seed}:{arm}:{k}")


def validate_vqa_repair(result: dict) -> None:
    config = result["run_config"]
    seeds, datasets, ks, n = (
        config["seeds"], config["datasets"], config["ks"], config["n"])
    if sorted(result.get("seeds_done", [])) != sorted(seeds):
        raise ValueError("VQA repair seeds incomplete")
    for seed in seeds:
        for dataset in datasets:
            for arm in ("baseline", "repaired"):
                for k in ks:
                    _check_records(
                        result["evidence"][str(seed)][dataset][arm][str(k)], n,
                        f"{seed}:{dataset}:{arm}:{k}")


def validate_panel(result: dict, cells_dir: Path) -> None:
    config = result["run_config"]
    models = config["models"]
    tasks = config["tasks"]
    conditions = config["conditions"]
    ks = config["k_grid"]
    seeds = range(config["seeds"])
    n = config["n_test"]
    def model_ks(model):
        if config.get("disable_heavy_k_cap", False):
            return ks
        return ([k for k in ks if k <= 8]
                if any(marker in model for marker in _HEAVY_MODEL_MARKERS) else ks)

    expected_files = {
        f"{model.split('/')[-1]}__{task}__{condition}__K{k}__s{seed}.json"
        for model in models for task in tasks for condition in conditions
        for k in model_ks(model) for seed in seeds
    }
    actual_files = {
        path.name for path in cells_dir.glob("*.json")
        if not path.name.startswith("_")
    }
    if actual_files != expected_files:
        raise ValueError(
            f"panel cells incomplete: {len(actual_files)}/{len(expected_files)}")
    expected_aggregates = {
        f"{model.split('/')[-1]}|{task}|{condition}|K{k}"
        for model in models for task in tasks for condition in conditions
        for k in model_ks(model)
    }
    if set(result.get("cells", {})) != expected_aggregates:
        raise ValueError("panel aggregate cells incomplete")
    for filename in expected_files:
        cell = json.loads((cells_dir / filename).read_text())
        _check_records(cell, n, filename)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "kind",
        choices=["paired", "circuit", "adapter-eval", "calibration",
                 "frozen-features", "vqa-train", "vqa-repair", "panel"])
    parser.add_argument("path")
    parser.add_argument("--keys", default="early:both,mid:both")
    parser.add_argument("--cells-dir")
    args = parser.parse_args()
    result = json.loads(Path(args.path).read_text())
    if args.kind == "paired":
        validate_paired(result)
    elif args.kind == "circuit":
        validate_circuit(result, args.keys.split(","))
    elif args.kind == "adapter-eval":
        validate_adapter_eval(result)
    elif args.kind == "calibration":
        validate_calibration(result)
    elif args.kind == "frozen-features":
        validate_frozen_features(result)
    elif args.kind == "vqa-train":
        validate_vqa_train(result)
    elif args.kind == "vqa-repair":
        validate_vqa_repair(result)
    else:
        cells_dir = Path(args.cells_dir or result["run_config"]["cells_dir"])
        validate_panel(result, cells_dir)
    print(f"VALID {args.kind}: {args.path}")


if __name__ == "__main__":
    main()
