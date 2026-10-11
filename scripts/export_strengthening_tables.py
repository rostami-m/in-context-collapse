#!/usr/bin/env python3
"""Export manuscript rows from complete, freshly audited strengthening evidence.

Every saved report is checked against an independent re-audit, including its
conditional bootstrap calculations. Table estimates are then recomputed from
raw query records. --check never changes evidence or generated fragments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_strengthening_artifacts import audit_file
from scripts.export_revision_tables import interval_text, number, tex_row
from scripts.summarize_revision_followup import seed_interval, write_report


INPUTS = ("natural_transfer_cifar", "natural_transfer_fashion", "rule_confirmatory")
ARMS = ("frozen", "natural_connector", "episode_connector", "all_layer")
ARM_LABELS = {"frozen": "Frozen", "natural_connector": "Natural connector",
              "episode_connector": "Episodic connector", "all_layer": "All-layer"}
DOMAINS = (("natural_transfer_cifar", "cifar4", "CIFAR-4"),
           ("natural_transfer_fashion", "fashion4", "Fashion-4"))
RULES = ("left_color", "right_color")
FULL_K = (0, 4, 8, 16)
SHORT_K = (4, 8, 16)
AUDIT_DRAWS = 10000
SOURCE_FILES = ("scripts/export_strengthening_tables.py", "scripts/audit_strengthening_artifacts.py",
                "scripts/audit_revision_artifacts.py", "scripts/summarize_revision_followup.py",
                "scripts/export_revision_tables.py")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_data_path(value, root):
    """Interpret recorded paths only under this checkout's data subtree."""
    path = Path(value)
    parts = path.parts
    if ".." in parts:
        raise ValueError("recorded data path traversal is not allowed")
    if path.is_absolute():
        positions = [index for index, part in enumerate(parts) if part == "data"]
        if len(positions) != 1:
            raise ValueError("recorded absolute path must identify one data subtree")
        parts = parts[positions[0]:]
    if len(parts) < 2 or parts[0] != "data":
        raise ValueError("recorded evidence path must lie inside data")
    allowed = root.resolve() / "data"
    local = root.joinpath(*parts).resolve()
    if not local.is_relative_to(allowed):
        raise ValueError("recorded evidence path escapes local data subtree")
    return local.relative_to(root.resolve()).as_posix()


def normalize_report_paths(value, root, key=None):
    if isinstance(value, dict):
        return {name: normalize_report_paths(item, root, name) for name, item in value.items()}
    if isinstance(value, list):
        return [normalize_report_paths(item, root, key) for item in value]
    if key in {"path", "manifest"} and isinstance(value, str):
        return canonical_data_path(value, root)
    return value


def require_equal(actual, expected, location="report"):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"{location}: report fields differ from fresh audit")
        for key, value in expected.items():
            require_equal(actual[key], value, f"{location}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{location}: report rows differ from fresh audit")
        for index, value in enumerate(expected):
            require_equal(actual[index], value, f"{location}[{index}]")
    elif type(expected) in (int, float):
        if (type(actual) not in (int, float) or not math.isfinite(actual)
                or not math.isfinite(expected) or abs(actual - expected) > 1e-12):
            raise ValueError(f"{location}: report metric differs from fresh audit")
    elif actual != expected:
        raise ValueError(f"{location}: report value differs from fresh audit")


def validate_saved_report(saved, fresh, root):
    if saved.get("status") != "complete" or saved.get("errors"):
        raise ValueError("saved report is incomplete or invalid")
    # These queue annotations are not scientific calculations or acceptance gates.
    ignored = {"queue_purpose", "publication_ready"}
    old = normalize_report_paths({key: value for key, value in saved.items() if key not in ignored}, root)
    new = normalize_report_paths({key: value for key, value in fresh.items() if key not in ignored}, root)
    require_equal(old, new)


def expected_config(name):
    common = {"model": "Qwen/Qwen2-VL-2B-Instruct", "arms": list(ARMS),
              "seeds": [0, 1, 2], "n": 40, "score_mode": "generation",
              "img_size": 112, "max_new_tokens": 6}
    if name == "rule_confirmatory":
        return {**common, "experiment": "two_object_rule_transfer", "partition": "confirmatory",
                "rules": list(RULES), "conditions": ["explicit_natural", "implicit_remap"],
                "condition_ks": {"explicit_natural": [0], "implicit_remap": [0, 8, 16]}}
    task = next(task for key, task, _ in DOMAINS if key == name)
    return {**common, "experiment": "paired_option_order_saved_adapters", "tasks": [task],
            "conditions": ["standard", "shuffled", "remap", "remap_shuffled"],
            "ks": list(FULL_K), "option_orders": ["random"]}


class Evidence:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.raw, self.reports, self.provenance = {}, {}, {}
        for name in INPUTS:
            relative = f"data/strengthening_v1/{name}.json"
            report_relative = f"data/strengthening_v1/reports/{name}.json"
            path, report_path = self.root / relative, self.root / report_relative
            if (canonical_data_path(relative, self.root) != relative
                    or canonical_data_path(report_relative, self.root) != report_relative):
                raise ValueError("strengthening evidence file resolves outside its canonical location")
            raw_hash, report_hash = sha256(path), sha256(report_path)
            saved = json.loads(report_path.read_text())
            if saved.get("sha256") != raw_hash:
                raise ValueError(f"artifact bytes changed since saved audit: {name}")
            if canonical_data_path(saved["path"], self.root) != relative:
                raise ValueError(f"saved report refers to another artifact: {name}")
            fresh = audit_file(path, expected_config=expected_config(name), root=self.root,
                               bootstrap_draws=AUDIT_DRAWS)
            if fresh["status"] != "complete" or fresh["errors"]:
                raise ValueError(f"independent audit rejected {name}: {fresh['errors']}")
            validate_saved_report(saved, fresh, self.root)
            payload = json.loads(path.read_text())
            if sha256(path) != raw_hash or sha256(report_path) != report_hash:
                raise ValueError("evidence changed during export")
            self.raw[name], self.reports[name] = payload, fresh
            self.provenance[name] = {"artifact": relative, "artifact_sha256": raw_hash,
                                     "report": report_relative, "report_sha256": report_hash,
                                     "cells": len(payload["cells"]),
                                     "records": sum(len(cell["records"]) for cell in payload["cells"].values()),
                                     "checkpoint_provenance": payload["run_config"]["checkpoints"]}

    def target_rows(self, name, task, arm, condition, k, seed):
        return self.raw[name]["cells"][f"{seed}:{task}:{arm}:{condition}:{k}:random"]["records"]

    def target_seed_values(self, name, task, arm, condition, ks):
        return {str(seed): statistics.mean(
            statistics.mean(row["prediction"] == row["gold"] for row in
                            self.target_rows(name, task, arm, condition, k, seed)) for k in ks)
                for seed in self.raw[name]["run_config"]["seeds"]}

    def target_estimate(self, name, task, arm, condition, ks):
        values = self.target_seed_values(name, task, arm, condition, ks)
        return {**seed_interval(values.values()), "seed_values": values}

    def objective_effect(self, name, task, condition):
        natural = self.target_seed_values(name, task, "natural_connector", condition, SHORT_K)
        episodic = self.target_seed_values(name, task, "episode_connector", condition, SHORT_K)
        differences = {seed: episodic[seed] - natural[seed] for seed in natural}
        result = {**seed_interval(differences.values()), "seed_differences": differences}
        if condition == "remap":
            saved = self.reports[name]["objective_transfer"]["primary"]
            if len(saved) != 1:
                raise ValueError("missing unique primary target contrast")
            for key in ("mean", "ci95_low", "ci95_high", "seed_differences"):
                require_equal(result[key], saved[0][key], f"raw primary.{key}")
        return result

    def rule_rows(self, seed, arm, condition, rule, k):
        return self.raw["rule_confirmatory"]["cells"][f"{seed}:{arm}:{condition}:{rule}:{k}"]["records"]

    def rule_estimate(self, arm, condition, ks, *, rules=RULES, metric="accuracy"):
        seeds = self.raw["rule_confirmatory"]["run_config"]["seeds"]
        values = {}
        for seed in seeds:
            if metric == "both_correct":
                if condition != "implicit_remap":
                    raise ValueError("joint rule correctness is reserved for implicit prompts")
                cells = []
                for k in ks:
                    left = self.rule_rows(seed, arm, condition, RULES[0], k)
                    right = self.rule_rows(seed, arm, condition, RULES[1], k)
                    if len(left) != len(right) or any(a["pair_id"] != b["pair_id"] for a, b in zip(left, right)):
                        raise ValueError("joint rule records are unpaired")
                    cells.append(statistics.mean(a["prediction"] == a["gold"] and b["prediction"] == b["gold"]
                                                 for a, b in zip(left, right)))
                values[str(seed)] = statistics.mean(cells)
            else:
                if metric not in {"accuracy", "invalid"}:
                    raise ValueError("unknown rule metric")
                values[str(seed)] = statistics.mean(
                    statistics.mean(row["prediction"] == row["gold"] if metric == "accuracy"
                                    else row["prediction"] is None
                                    for row in self.rule_rows(seed, arm, condition, rule, k))
                    for rule in rules for k in ks)
        return {**seed_interval(values.values()), "seed_values": values}


def build_fragments(evidence):
    fragments, columns, numerical = {}, {}, {}

    def add(name, description, rows):
        file = f"strengthening_{name}.tex"
        columns[file] = description
        fragments[file] = "% Generated by scripts/export_strengthening_tables.py; do not edit.\n" + "\n".join(rows) + "\n"

    summary, effects, curves = [], [], []
    for name, task, label in DOMAINS:
        for arm, objective in (("natural_connector", "Natural"), ("episode_connector", "Episodic")):
            estimates = [evidence.target_estimate(name, task, arm, condition, SHORT_K)
                         for condition in ("standard", "remap")]
            numerical[f"{task}:{arm}:mean_k4_k8_k16"] = dict(zip(("standard", "remap"), estimates))
            summary.append(tex_row(label, objective, *(number(item["mean"]) for item in estimates)))
        contrasts = [evidence.objective_effect(name, task, condition) for condition in ("remap", "standard")]
        numerical[f"{task}:episodic_minus_natural"] = dict(zip(("remap", "standard"), contrasts))
        effects.append(tex_row(label, *(interval_text(item) for item in contrasts)))
        for arm in ARMS:
            values = ["/".join(number(evidence.target_estimate(name, task, arm, condition, [k])["mean"])
                               for k in FULL_K) for condition in ("standard", "remap")]
            curves.append(tex_row(label, ARM_LABELS[arm], *values))
    add("objective_summary", "Domain & Objective & Standard mean K4/8/16 & Remap mean K4/8/16", summary)
    add("objective_effects", "Domain & Remap episodic-natural mean [95% CI] & Standard episodic-natural mean [95% CI]", effects)
    add("objective_curves", "Domain & Arm & Standard K0/4/8/16 & Remap K0/4/8/16", curves)
    summary, curves = [], []
    for arm in ARMS:
        estimates = [evidence.rule_estimate(arm, "explicit_natural", [0]),
                     evidence.rule_estimate(arm, "implicit_remap", [8, 16]),
                     evidence.rule_estimate(arm, "implicit_remap", [8, 16], metric="both_correct")]
        numerical[f"rule:{arm}:composites"] = dict(zip(("explicit_natural", "implicit", "both_correct"), estimates))
        summary.append(tex_row(ARM_LABELS[arm], *(number(item["mean"]) for item in estimates)))
        for k in (0, 8, 16):
            estimates = [evidence.rule_estimate(arm, "implicit_remap", [k], rules=[rule]) for rule in RULES]
            estimates += [evidence.rule_estimate(arm, "implicit_remap", [k], metric="both_correct"),
                          evidence.rule_estimate(arm, "implicit_remap", [k], metric="invalid")]
            curves.append(tex_row(ARM_LABELS[arm], k, *(number(item["mean"]) for item in estimates)))
    add("rule_summary", "Arm & Explicit natural (K0, both rules) & Implicit mean K8/16 (both rules) & Both-rules-correct mean K8/16", summary)
    add("rule_curves", "Arm & K & Left accuracy & Right accuracy & Both-correct & Invalid (mean over both rules)", curves)
    provenance = {
        "schema_version": 1, "inputs": evidence.provenance,
        "source_sha256": {name: sha256(ROOT / name) for name in SOURCE_FILES},
        "fragment_sha256": {name: hashlib.sha256(content.encode("utf-8")).hexdigest()
                            for name, content in sorted(fragments.items())},
        "columns": columns, "numerical_estimates": numerical,
        "audit": {"status": "complete", "bootstrap_draws_recomputed": AUDIT_DRAWS,
                  "cells": sum(item["cells"] for item in evidence.provenance.values()),
                  "records": sum(item["records"] for item in evidence.provenance.values())},
        "notes": [
            "Rows only; manuscript wrappers own captions and blue revision formatting.",
            "Accuracy values are decimal proportions, not percentages; rounding is half-up to three decimals.",
            "Objective contrasts average paired K4/8/16 differences within each of three seeds.",
            "Intervals are pointwise Student-t intervals across seed means, untruncated and unadjusted for multiplicity.",
            "Rule accuracy averages both rules equally; implicit composites use K8/16, excluding K0.",
            "Both-rules-correct uses one joint indicator per query pair; invalid outputs are failures.",
            "Invalid rates average both rule members, rather than counting either-invalid pairs.",
            "The same forty rule scenes recur across seeds and K; they are not independent extra image samples.",
            "All inputs are re-audited; saved reports including bootstrap estimates are independently verified.",
            "Queue publication_ready annotations are not scientific validation and are not modified.",
            "Saved checkpoints were trained with class-ordered options; these runs use randomized evaluation options.",
            "No cross-artifact comparisons, new training, positive-outcome gating, or equivalence claims are performed.",
        ],
    }
    fragments["strengthening_provenance.json"] = json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n"
    return fragments


def write_fragments(fragments, directory, *, check=False):
    directory = Path(directory)
    changed = []
    for name, content in fragments.items():
        path = directory / name
        if check:
            if not path.exists() or path.read_text() != content:
                changed.append(name)
        else:
            write_report(path, content.encode("utf-8"))
    if changed:
        raise ValueError("Generated strengthening rows differ: " + ", ".join(changed))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="checkout root for relocated evidence")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
        evidence = Evidence(args.root)
        fragments = build_fragments(evidence)
        write_fragments(fragments, args.out_dir or args.root / "paper/revision_results", check=args.check)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, str(error) + "\n")
    print(f"{'Checked' if args.check else 'Wrote'} {len(fragments)} deterministic strengthening fragments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
