#!/usr/bin/env python3
"""Reduce audited follow-up artifacts without loading models or modifying inputs.

Partial grids receive coverage reports only. Complete configured grids are not
automatically publication-ready, and separate input files are never pooled.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import statistics
import tempfile

try:
    from .audit_revision_artifacts import audit_file, iter_record_cells, official_vqa_score
except ImportError:
    from audit_revision_artifacts import audit_file, iter_record_cells, official_vqa_score


# Two-sided 95% Student-t quantiles, matching the recorded three-decimal convention.
T975 = (None, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306,
        2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110,
        2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
        2.052, 2.048, 2.045, 2.042)
DIMENSIONS = ("task", "condition", "k", "option_order", "arm", "seed")
COMPLETENESS_ERRORS = {
    "root: incomplete or unexpected option-order evaluation grid",
    "root: incomplete image-disjoint VQA grid",
    "root: incorrect VQA seed grid",
    "root: incomplete or unexpected contextual-calibration grid",
    "root: incomplete attention-baseline evaluation grid",
    "root: incorrect attention-baseline seed/condition/K/arm grid",
    "root: incomplete DARA training seeds",
}
SCOPE = (
    "Pointwise descriptive intervals, without multiplicity correction. Seed intervals use "
    "one mean per recorded seed, not one independent replicate per query. A difference "
    "interval containing zero does not establish equivalence. Adapter effects are conditional "
    "on the recorded saved checkpoints and their source training protocol; randomized-option "
    "evaluation does not imply randomized-option training: original v2.4 checkpoints were trained "
    "with class-ordered options. No cross-file pooling or cross-condition "
    "comparisons are performed. CAMA/DARA effects use their own local frozen controls; different "
    "prompt/attention implementations do not support a controlled cross-method ranking."
)


class InvalidArtifact(ValueError):
    pass


def seed_interval(values):
    """Do not substitute a zero-width uncertainty interval for a single seed."""
    values = list(values)
    if not values or len(values) > len(T975):
        raise InvalidArtifact("seed intervals require between 1 and 31 seeds")
    mean = statistics.mean(values)
    half = (T975[len(values) - 1] * statistics.stdev(values) / math.sqrt(len(values))
            if len(values) > 1 else None)
    return {"mean": mean, "ci95_low": mean - half if half is not None else None,
            "ci95_high": mean + half if half is not None else None,
            "n_seeds": len(values), "df": len(values) - 1,
            "interval": "pointwise Student-t across seed means; untruncated"}


def coordinates(key):
    return dict(zip(DIMENSIONS, key))


def extract_grid(payload):
    """Normalize only the explicitly supported follow-up schemas."""
    config = payload["run_config"]
    method = config.get("method")
    if config.get("experiment") == "paired_option_order_saved_adapters":
        kind = "options"
        tasks, conditions = config["tasks"], config["conditions"]
        orders, arms = config["option_orders"], config["arms"]
        entries = []
        for key, cell in payload.get("cells", {}).items():
            seed, task, arm, condition, k, order = key.split(":")
            entries.append(((task, condition, int(k), order, arm, int(seed)), cell))
    else:
        tasks = [config.get("task", config.get("dataset", "unspecified"))]
        orders = [config.get("protocol", {}).get("option_order", "class")]
        entries = []
        if method == "image-disjoint VQA checkpoint evaluation":
            kind, conditions, orders, arms = "vqa", ["vqa"], ["not_applicable"], config["arms"]
            for seed, by_arm in payload.get("evidence", {}).items():
                if int(seed) not in config["seeds"] or not set(by_arm).issubset(arms):
                    raise InvalidArtifact("unexpected VQA seed or arm container")
                for arm, shots in by_arm.items():
                    for k, cell in shots.items():
                        entries.append(((tasks[0], "vqa", int(k), orders[0], arm, int(seed)), cell))
        elif method == "CAMA independent implementation":
            kind, conditions, arms = "cama", ["standard", "remap"], ["frozen", "cama"]
            for cell in payload.get("cells", []):
                entries.append(((tasks[0], cell["condition"], cell["k"], orders[0],
                                 cell["arm"], cell["seed"]), cell))
        elif method == "DARA compatibility port":
            kind, conditions, arms = "dara", ["standard", "remap"], ["frozen", "dara"]
            for container, arm in (("baseline_evidence", "frozen"), ("evidence", "dara")):
                block_seeds = [block["seed"] for block in payload.get(container, [])]
                if len(block_seeds) != len(set(block_seeds)) or not set(block_seeds).issubset(config["seeds"]):
                    raise InvalidArtifact("duplicate or unexpected DARA evidence seeds")
                for block in payload.get(container, []):
                    for condition in conditions:
                        for k, cell in block.get(condition, {}).items():
                            entries.append(((tasks[0], condition, int(k), orders[0], arm, block["seed"]), cell))
        elif config.get("baseline") in {"contextual_calibration_neutral_query", "frozen_connector_feature_1nn"}:
            kind = "features" if config["baseline"] == "frozen_connector_feature_1nn" else "calibration"
            conditions = config["conditions"]
            arms = ["1nn" if kind == "features" else "calibrated"]
            if kind == "features":
                orders = ["not_applicable"]
                if any(k < 1 for k in config["ks"]):
                    raise InvalidArtifact("feature 1-NN requires K >= 1")
            for seed, by_condition in payload.get("evidence", {}).items():
                if int(seed) not in config["seeds"] or not set(by_condition).issubset(conditions):
                    raise InvalidArtifact("unexpected baseline seed or condition container")
                for condition, shots in by_condition.items():
                    for k, cell in shots.items():
                        entries.append(((tasks[0], condition, int(k), orders[0], arms[0], int(seed)), cell))
        else:
            raise InvalidArtifact("unsupported follow-up artifact schema")
    axes = (tasks, conditions, config["ks"], orders, arms, config["seeds"])
    if any(not values or len(values) != len(set(values)) for values in axes):
        raise InvalidArtifact("grid axes must be nonempty and unique")
    if config["n"] < 1 or any(not isinstance(seed, int) for seed in config["seeds"]):
        raise InvalidArtifact("invalid query count or seed identifiers")
    expected = set(itertools.product(*axes))
    keys = [key for key, _ in entries]
    if len(keys) != len(set(keys)) or not set(keys).issubset(expected):
        raise InvalidArtifact("duplicate or unexpected grid coordinates")
    cells = dict(entries)
    if len(list(iter_record_cells(payload))) != len(cells):
        raise InvalidArtifact("record cells outside the declared grid")
    for key, cell in cells.items():
        if (cell.get("seed") != key[5] or cell.get("k") != key[2]
                or (kind != "vqa" and cell.get("condition") != key[1])):
            raise InvalidArtifact(f"cell metadata disagrees with grid coordinates {key}")
        if len(cell["records"]) != config["n"]:
            raise InvalidArtifact(f"incomplete query records within cell {key}")
        indices = [record["query_index"] for record in cell["records"]]
        if indices != list(range(config["n"])):
            raise InvalidArtifact(f"noncanonical query indices in {key}")
        if any(not record.get("query_id") for record in cell["records"]):
            raise InvalidArtifact(f"missing query identity in {key}")
        if kind != "vqa":
            for record in cell["records"]:
                if not isinstance(record.get("correct"), bool):
                    raise InvalidArtifact("missing or non-boolean classification correctness evidence")
                mapping = record.get("shuffle_map")
                if mapping is not None:
                    classes = set(range(len(record.get("label_space", []))))
                    if (not classes or not isinstance(mapping, dict)
                            or set(mapping) != {str(index) for index in classes}
                            or any(type(value) is not int for value in mapping.values())
                            or set(mapping.values()) != classes
                            or any(mapping[str(index)] == index for index in classes)):
                        raise InvalidArtifact("shuffle_map must be a complete class-index derangement bijection")
                if key[3] == "class" and record.get("option_order") is not None:
                    if record["option_order"] != list(range(len(record["label_space"]))):
                        raise InvalidArtifact("class-ordered cell has permuted options")
        if kind == "features":
            if cell.get("protocol") != config.get("protocol"):
                raise InvalidArtifact("feature cell protocol differs from configured protocol")
            for record in cell["records"]:
                if (record.get("k") != key[2] or record.get("condition") != key[1]
                        or record["prediction"] not in record.get("demo_label_classes", [])):
                    raise InvalidArtifact("feature prediction or record disagrees with demonstration/grid")
    if kind == "dara":
        runs = payload.get("training_runs", [])
        recorded = [run.get("torch_seed") for run in runs]
        done = payload.get("seeds_done", [])
        if (len(recorded) != len(set(recorded)) or not set(recorded).issubset(config["seeds"])
                or len(done) != len(set(done)) or not set(done).issubset(config["seeds"])):
            raise InvalidArtifact("duplicate or unexpected DARA source training seeds")
    return kind, cells, expected


def completeness_error(message):
    return (message in COMPLETENESS_ERRORS
            or (message.startswith("root.evidence.") and
                message.endswith((": incorrect VQA arm grid", ": incorrect VQA K grid")))
            or (message.startswith(("root.evidence[", "root.baseline_evidence["))
                and message.endswith(": incorrect DARA K grid")))


def check_pairing(left, right, *, option_change=False, vqa=False):
    fields = (("query_index", "query_id", "question", "answers", "demo_ids") if vqa else
              ("query_index", "query_id", "gold", "label_space", "demo_source_classes",
               "demo_source_ids", "demo_label_classes", "shuffle_map"))
    if not vqa and not option_change:
        fields += ("option_order", "displayed_options")
    left_ids = [[record.get(field) for field in fields] for record in left["records"]]
    right_ids = [[record.get(field) for field in fields] for record in right["records"]]
    if left_ids != right_ids:
        raise InvalidArtifact("comparison episodes are not paired")


def record_metrics(cell, kind, arm):
    records = cell["records"]
    if kind == "vqa":
        return {"accuracy": [official_vqa_score(record["prediction"], record["answers"])
                             for record in records]}
    prediction_key = "raw_prediction" if arm == "raw" else "prediction"
    metrics = {"accuracy": [float(record[prediction_key] == record["gold"]) for record in records]}
    if kind not in {"calibration", "features"}:
        if not all(isinstance(record.get("raw_output"), str)
                   and "generation_prediction" in record and record.get("label_space")
                   for record in records):
            raise InvalidArtifact("generation artifact is missing raw generation evidence")
        if any(record["prediction"] != record["generation_prediction"] for record in records):
            raise InvalidArtifact("follow-up generation prediction differs from scored prediction")
        metrics["unparseable_rate"] = [float(record["generation_prediction"] is None) for record in records]
    if all(record.get("shuffle_map") is not None for record in records):
        metrics["follows_corrupted_mapping_rate"] = [
            float(record[prediction_key] is not None
                  and record[prediction_key] == record["shuffle_map"][str(record["gold"])])
            for record in records]
    return metrics


def image_bootstrap(left, right, draws, seed):
    """One seed-averaged difference per image, resampled in paired image units."""
    differences = defaultdict(list)
    for seed_id in sorted(left):
        check_pairing(left[seed_id], right[seed_id], vqa=True)
        for before, after in zip(left[seed_id]["records"], right[seed_id]["records"]):
            image = before["query_id"].split(":")[0]
            differences[image].append(official_vqa_score(after["prediction"], after["answers"])
                                      - official_vqa_score(before["prediction"], before["answers"]))
    if any(len(values) != len(left) for values in differences.values()):
        raise InvalidArtifact("VQA bootstrap requires one matched question per image per seed")
    values = [statistics.mean(differences[key]) for key in sorted(differences)]
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(draws))

    def quantile(p):
        index = p * (draws - 1)
        lower = int(index)
        return means[lower] + (means[min(lower + 1, draws - 1)] - means[lower]) * (index - lower)

    return {"mean": statistics.mean(values), "ci95_low": quantile(.025),
            "ci95_high": quantile(.975), "n_images": len(values),
            "n_fixed_seeds": len(left), "bootstrap_draws": draws, "bootstrap_seed": seed,
            "unit": "image, after averaging paired differences across fixed seeds",
            "scope": "conditional on fixed fitted checkpoints and eligible image/question frame"}


def reduce_complete(kind, cells, seeds, draws, bootstrap_seed):
    if kind == "calibration":
        cells = {**cells, **{(*key[:4], "raw", key[5]): cell for key, cell in cells.items()}}
    groups = defaultdict(dict)
    for key, cell in cells.items():
        groups[key[:-1]][key[-1]] = cell
    estimates, comparisons = [], []
    for group, by_seed in sorted(groups.items()):
        if set(by_seed) != set(seeds):
            raise InvalidArtifact("incomplete group reached numerical reduction")
        metrics = {seed: record_metrics(by_seed[seed], kind, group[-1]) for seed in seeds}
        if any(set(value) != set(metrics[seeds[0]]) for value in metrics.values()):
            raise InvalidArtifact("metric availability differs across seeds")
        estimates.append({**coordinates(group), "complete": True, "expected_seeds": seeds,
                          "n_queries_per_seed": {str(seed): len(by_seed[seed]["records"]) for seed in seeds},
                          "metrics": {metric: {
                              **seed_interval(statistics.mean(metrics[seed][metric]) for seed in seeds),
                              "seed_values": {str(seed): statistics.mean(metrics[seed][metric]) for seed in seeds}}
                              for metric in metrics[seeds[0]]}})
    for left_group, right_group in itertools.combinations(sorted(groups), 2):
        if left_group[:3] != right_group[:3]:
            continue
        same_order = left_group[3] == right_group[3]
        same_arm = left_group[4] == right_group[4]
        if same_order == same_arm:
            continue
        left_key, right_key = left_group, right_group
        # Give familiar effects a consistent sign: treatment minus frozen/raw,
        # randomized minus class-ordered options, all-layer minus connector.
        if same_order:
            priority = {"frozen": 0, "raw": 0, "connector": 1, "all_layer": 2}
            if (priority.get(left_key[4], 1), left_key[4]) > (priority.get(right_key[4], 1), right_key[4]):
                left_key, right_key = right_key, left_key
        left, right = groups[left_key], groups[right_key]
        per_metric = defaultdict(dict)
        for seed in seeds:
            check_pairing(left[seed], right[seed], option_change=not same_order, vqa=kind == "vqa")
            left_metrics = record_metrics(left[seed], kind, left_key[4])
            right_metrics = record_metrics(right[seed], kind, right_key[4])
            for metric in left_metrics.keys() & right_metrics.keys():
                differences = [after - before for before, after in zip(left_metrics[metric], right_metrics[metric])]
                per_metric[metric][str(seed)] = statistics.mean(differences)
        comparison = {"kind": "arm" if same_order else "option_order",
                      "left": coordinates(left_key), "right": coordinates(right_key),
                      "direction": "right minus left", "complete": True, "expected_seeds": seeds,
                      "metrics": {metric: {**seed_interval(values.values()), "seed_differences": values}
                                  for metric, values in sorted(per_metric.items())}}
        if kind == "vqa":
            comparison["paired_image_bootstrap"] = image_bootstrap(left, right, draws, bootstrap_seed)
        comparisons.append(comparison)
    return estimates, comparisons


def summarize_file(path, *, bootstrap_draws=10000, bootstrap_seed=90210):
    path = Path(path)
    if bootstrap_draws < 2:
        raise InvalidArtifact("bootstrap_draws must be at least 2")
    content = path.read_bytes()
    sha256 = hashlib.sha256(content).hexdigest()
    payload = json.loads(content)
    kind, cells, expected = extract_grid(payload)
    audit = audit_file(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
        raise InvalidArtifact("input changed during audit; retry after the atomic checkpoint write")
    errors = [error for error in audit["errors"] if not completeness_error(error)]
    if errors:
        raise InvalidArtifact("; ".join(errors))
    for key, cell in cells.items():
        record_metrics(cell, kind, key[4])
    config = payload["run_config"]
    missing = expected - cells.keys()
    complete = not missing and not audit["errors"]
    groups = defaultdict(set)
    for key in expected:
        groups[key[:-1]]
    for key in cells:
        groups[key[:-1]].add(key[-1])
    report = {"path": str(path.resolve()), "sha256": sha256, "schema": kind,
              "status": "complete" if complete else "partial", "publication_ready": False,
              "interpretation_scope": SCOPE, "run_config": config,
              "coverage": {"expected_cells": len(expected), "observed_cells": len(cells),
                           "missing_cells": [coordinates(key) for key in sorted(missing)],
                           "groups": [{**coordinates(key), "expected_seeds": config["seeds"],
                                       "observed_seeds": sorted(observed),
                                       "complete": observed == set(config["seeds"])}
                                      for key, observed in sorted(groups.items())]},
              "audit": audit, "estimates": [], "paired_comparisons": [],
              "aggregation_scope": "withheld until the entire configured grid and audit are complete"}
    if complete:
        report["estimates"], report["paired_comparisons"] = reduce_complete(
            kind, cells, config["seeds"], bootstrap_draws, bootstrap_seed)
        report["aggregation_scope"] = "complete configured artifact only; inspect n and seeds, including smoke runs"
    if kind == "features":
        report["verification_scope"] = (
            "Accuracy from stored selected-neighbor classes; embeddings/distances are not stored, "
            "so this reducer cannot independently replay the nearest-neighbor search. "
            "There is no displayed answer list and no generation parse metric.")
    return report


def write_report(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=90210)
    parser.add_argument("--out", type=Path, help="atomically write the JSON report and print only its digest/status")
    args = parser.parse_args()
    if args.out is not None and args.out.resolve() in {path.resolve() for path in args.paths}:
        parser.error("output must not overwrite an input artifact")
    reports = []
    for path in args.paths:
        try:
            reports.append(summarize_file(path, bootstrap_draws=args.bootstrap_draws,
                                          bootstrap_seed=args.bootstrap_seed))
        except (InvalidArtifact, ValueError, KeyError, TypeError, OSError, IndexError) as error:
            reports.append({"path": str(path.resolve()), "status": "rejected", "error": str(error),
                            "publication_ready": False, "estimates": [], "paired_comparisons": []})
            if path.is_file():
                reports[-1]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    sources = [Path(__file__).resolve(), Path(audit_file.__code__.co_filename).resolve()]
    report = {"report_version": 1,
              "analysis_sources": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                                   for path in sources], "artifacts": reports}
    content = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if args.out is None:
        print(content.decode("utf-8"), end="")
    else:
        write_report(args.out, content)
        print(json.dumps({"path": str(args.out.resolve()), "sha256": hashlib.sha256(content).hexdigest(),
                          "artifacts": [{"path": item["path"], "status": item["status"]} for item in reports]}))
    raise SystemExit(1 if any(report["status"] == "rejected" for report in reports)
                     else 2 if any(report["status"] == "partial" for report in reports) else 0)


if __name__ == "__main__":
    main()
