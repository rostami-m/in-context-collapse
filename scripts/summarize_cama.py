#!/usr/bin/env python3
"""Verify paired CAMA episodes and summarize exact per-query outcomes."""
import argparse
import json
import itertools
import math
from pathlib import Path

from scripts.audit_revision_artifacts import parse_generation
from incontext_collapse.evaluation import mean_t_ci
from incontext_collapse.results import write_json_atomic


PAIR_FIELDS = ("query_index", "query_id", "gold", "label_space", "demo_source_ids",
               "demo_source_classes", "demo_label_classes")


def paired_fields(config):
    if config.get("protocol", {}).get("option_order") == "random":
        return PAIR_FIELDS + ("option_order", "displayed_options")
    return PAIR_FIELDS


def verify_reference(payload, reference):
    """Require identical episodes in an existing adapter-evaluation artifact."""
    for field in ("model", "model_revision", "task", "protocol", "n", "seeds"):
        if payload["run_config"][field] != reference["run_config"][field]:
            raise ValueError(f"reference configuration mismatch: {field}")
    checked = 0
    for cell in payload["cells"]:
        if cell["arm"] != "frozen":
            continue
        other = reference["evidence"][str(cell["seed"])][cell["condition"]][str(cell["k"])]
        if len(cell["records"]) != len(other["records"]):
            raise ValueError("reference query count mismatch")
        for left, right in zip(cell["records"], other["records"]):
            for field in paired_fields(payload["run_config"]):
                if field not in left or field not in right or left[field] != right[field]:
                    raise ValueError(f"reference episode mismatch: {field}")
            checked += 1
    return checked


def summarize(payload):
    config = payload["run_config"]
    for axis in ("ks", "seeds"):
        values = config[axis]
        if not values or any(type(v) is not int or v < 0 for v in values) or len(values) != len(set(values)):
            raise ValueError(f"invalid or duplicate {axis} grid")
    if type(config["n"]) is not int or config["n"] < 1:
        raise ValueError("invalid query count")
    indexed = {}
    for cell in payload["cells"]:
        key = (cell["seed"], cell["k"], cell["condition"], cell["arm"])
        if key in indexed:
            raise ValueError(f"duplicate cell {key}")
        indexed[key] = cell
    expected = set(itertools.product(config["seeds"], config["ks"], ("standard", "remap"), ("frozen", "cama")))
    if set(indexed) != expected:
        raise ValueError("incomplete or unexpected CAMA grid")
    report = {"config": config, "comparisons": []}
    for condition in ("standard", "remap"):
        for k in config["ks"]:
            deltas, frozen_scores, cama_scores = [], [], []
            wins = losses = same_correct = same_wrong = changed = 0
            invalid_frozen = invalid_cama = 0
            seconds = {"frozen": 0., "cama": 0.}
            for seed in config["seeds"]:
                frozen = indexed[(seed, k, condition, "frozen")]
                cama = indexed[(seed, k, condition, "cama")]
                if len(frozen["records"]) != config["n"] or len(cama["records"]) != config["n"]:
                    raise ValueError("incomplete query count")
                for left, right in zip(frozen["records"], cama["records"]):
                    for field in paired_fields(config):
                        if field not in left or field not in right or left[field] != right[field]:
                            raise ValueError(f"unpaired {field} at {(seed, k, condition)}")
                    if left.get("shuffle_map") != right.get("shuffle_map"):
                        raise ValueError("unpaired shuffle_map")
                    if (left.get("option_order") != right.get("option_order")
                            or left.get("displayed_options") != right.get("displayed_options")):
                        raise ValueError("unpaired displayed options")
                    for record in (left, right):
                        if not isinstance(record.get("raw_output"), str):
                            raise ValueError("missing raw generation evidence")
                        prediction, status = parse_generation(record["raw_output"], record["label_space"])
                        if (type(record.get("correct")) is not bool or record["correct"] != (prediction == record["gold"])
                                or type(record.get("unparseable")) is not bool or record["unparseable"] != (prediction is None)):
                            raise ValueError("stored outcome differs from raw generation")
                        for field in ("prediction", "generation_prediction"):
                            if field in record and record[field] != prediction:
                                raise ValueError("stored prediction differs from raw generation")
                        if "parse_status" in record and record["parse_status"] != status:
                            raise ValueError("stored parse status differs from raw generation")
                    a, b = left["correct"], right["correct"]
                    wins += bool(b and not a)
                    losses += bool(a and not b)
                    same_correct += bool(a and b)
                    same_wrong += bool(not a and not b)
                    changed += left["raw_output"] != right["raw_output"]
                    invalid_frozen += left["unparseable"]
                    invalid_cama += right["unparseable"]
                frozen_score = sum(r["correct"] for r in frozen["records"]) / config["n"]
                cama_score = sum(r["correct"] for r in cama["records"]) / config["n"]
                frozen_scores.append(frozen_score)
                cama_scores.append(cama_score)
                deltas.append(cama_score - frozen_score)
                for arm, cell in (("frozen", frozen), ("cama", cama)):
                    diagnostics = cell["cama_diagnostics"]
                    if len(diagnostics) != config["n"]:
                        raise ValueError("missing per-query intervention diagnostics")
                    seconds[arm] += sum(d["seconds"] for d in diagnostics)
                    for diagnostic in diagnostics:
                        if (type(diagnostic["seconds"]) not in (int, float)
                                or not math.isfinite(diagnostic["seconds"]) or diagnostic["seconds"] < 0):
                            raise ValueError("invalid runtime diagnostic")
                        if arm == "cama" and k > 0:
                            if sorted(int(layer) for layer in diagnostic["heads"]) != config["cama"]["stage2_layers"]:
                                raise ValueError("CAMA did not execute every Stage II layer")
                            if len(diagnostic["selected_tokens"]) != k + 1:
                                raise ValueError("wrong number of selected image segments")
            total = len(config["seeds"]) * config["n"]
            report["comparisons"].append({
                "condition": condition, "k": k, "n_paired": total,
                "frozen": mean_t_ci(frozen_scores), "cama": mean_t_ci(cama_scores),
                "paired_difference": mean_t_ci(deltas),
                "wins": wins, "losses": losses, "both_correct": same_correct,
                "both_wrong": same_wrong, "changed_generations": changed,
                "unparseable_frozen": invalid_frozen, "unparseable_cama": invalid_cama,
                "seconds": seconds,
            })
    expected = len(config["seeds"]) * len(config["ks"]) * 4
    if len(indexed) != expected:
        raise ValueError(f"expected {expected} cells, got {len(indexed)}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--reference", type=Path,
                        help="verify episode identity against an adapter_eval artifact")
    args = parser.parse_args()
    if args.out and args.out.resolve() in {p.resolve() for p in (args.path, args.reference) if p is not None}:
        parser.error("output must not overwrite an input artifact")
    payload = json.loads(args.path.read_text())
    report = summarize(payload)
    if args.reference:
        count = verify_reference(payload, json.loads(args.reference.read_text()))
        report["reference_episodes_verified"] = count
        print(f"Verified {count} episodes against {args.reference}")
    if args.out:
        write_json_atomic(args.out, report)
    for row in report["comparisons"]:
        print(f"{row['condition']} K={row['k']}: "
              f"frozen={row['frozen']['mean']:.4f} cama={row['cama']['mean']:.4f} "
              f"delta={row['paired_difference']['mean']:+.4f} "
              f"wins/losses={row['wins']}/{row['losses']} "
              f"invalid={row['unparseable_frozen']}/{row['unparseable_cama']}")


if __name__ == "__main__":
    main()
