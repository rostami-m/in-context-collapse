#!/usr/bin/env python3
"""Generate manuscript row fragments from complete, independently audited evidence.

--check compares deterministic output without editing files. All reported means
and intervals are checked against raw records again, not merely copied from the
combined report. This script imports no model or inference implementation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_revision_artifacts import audit_file, iter_training_runs
from scripts.summarize_revision_followup import check_pairing, extract_grid, record_metrics, seed_interval, write_report

SHAPES = "options_shapes_n40.json"
DEPTH = "options_depth_natural_shapes_n40.json"
QWEN25 = "options_qwen25_shapes_n40.json"
TRANSFER = "options_transfer_n40.json"
SHORT = (4, 8, 16)
FULL = (0, 4, 8, 16)
ARM_LABELS = {"frozen": "Frozen", "connector": "Episode connector", "all_layer": "All-layer",
              "natural_connector": "Natural connector", "early": "Early", "mid": "Middle", "late": "Late"}
CONDITIONS = {"standard": "Standard", "remap": "Remap", "remap_shuffled": "Wrong-map true"}
TRANSFER_LABELS = {**ARM_LABELS, "connector": "Connector"}


def number(value):
    return str(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def estimate_text(estimate, uncertainty=False):
    mean = number(estimate["mean"])
    if not uncertainty:
        return mean
    if estimate["ci95_low"] is None:
        raise ValueError("manuscript uncertainty rows require multiple seeds")
    half = (estimate["ci95_high"] - estimate["ci95_low"]) / 2
    return "$" + mean + r"\pm" + number(half) + "$"


def interval_text(estimate):
    return "$" + number(estimate["mean"]) + r"\,[" + number(estimate["ci95_low"]) + ", " + number(estimate["ci95_high"]) + "]$"


def tex_row(*cells):
    return " & ".join(map(str, cells)) + r" \\"


def resolve_artifact_path(recorded_path, root):
    """Relocate only the pinned artifact subtree; never read an old workspace."""
    path = Path(recorded_path)
    parts = path.parts
    marker = ("data", "revision_v25")
    if ".." in parts:
        raise ValueError("artifact path traversal is not allowed")
    if path.is_absolute():
        matches = [index for index in range(len(parts) - 1) if parts[index:index + 2] == marker]
        if len(matches) != 1:
            raise ValueError("absolute artifact path must unambiguously identify data/revision_v25")
        suffix = parts[matches[0] + 2:]
    else:
        if parts[:2] != marker:
            raise ValueError("artifact path must be inside data/revision_v25")
        suffix = parts[2:]
    if not suffix:
        raise ValueError("artifact path must identify a file")
    allowed = root.resolve() / "data/revision_v25"
    resolved = allowed.joinpath(*suffix).resolve()
    if not resolved.is_relative_to(allowed):
        raise ValueError("artifact path escapes data/revision_v25")
    return resolved


class Evidence:
    def __init__(self, report_path, *, root=ROOT):
        self.root = root
        content = report_path.read_bytes()
        self.report_sha256 = hashlib.sha256(content).hexdigest()
        self.report = json.loads(content)
        self.artifacts, self.raw, self.grids, self.kind = {}, {}, {}, {}
        self.training = {}
        for artifact in self.report["artifacts"]:
            path = resolve_artifact_path(artifact["path"], root)
            name = path.name
            if name in self.artifacts or artifact["status"] != "complete" or artifact["audit"]["errors"]:
                raise ValueError("duplicate, incomplete or rejected artifact in report")
            raw_bytes = path.read_bytes()
            if hashlib.sha256(raw_bytes).hexdigest() != artifact["sha256"]:
                raise ValueError(f"artifact bytes changed: {name}")
            audit = audit_file(path)
            if audit["errors"]:
                raise ValueError(f"independent artifact audit failed: {name}: {audit['errors']}")
            payload = json.loads(raw_bytes)
            kind, cells, expected = extract_grid(payload)
            if cells.keys() != expected:
                raise ValueError(f"artifact has incomplete configured grid: {name}")
            self.artifacts[name], self.raw[name] = artifact, payload
            self.grids[name], self.kind[name] = cells, kind

    def cell(self, name, task, condition, arm, k, seed, order="random"):
        if self.kind[name] == "features":
            order = "not_applicable"
        elif self.kind[name] == "vqa":
            order = "not_applicable"
        stored_arm = "calibrated" if arm == "raw" else arm
        return self.grids[name][task, condition, k, order, stored_arm, seed]

    def estimate(self, name, arm, condition, k, task="shapes4", metric="accuracy", order="random"):
        rows = [row for row in self.artifacts[name]["estimates"] if
                (row["task"], row["condition"], row["k"], row["arm"]) == (task, condition, k, arm)
                and (self.kind[name] in {"features", "vqa"} or row["option_order"] == order)]
        if len(rows) != 1:
            raise ValueError(f"ambiguous or missing estimate: {name}/{arm}/{condition}/{k}")
        estimate = rows[0]["metrics"][metric]
        seeds = self.raw[name]["run_config"]["seeds"]
        means = {}
        for seed in seeds:
            values = record_metrics(self.cell(name, task, condition, arm, k, seed, order), self.kind[name], arm)[metric]
            means[str(seed)] = sum(values) / len(values)
        recomputed = seed_interval(means.values())
        for field in ("mean", "ci95_low", "ci95_high", "n_seeds"):
            if abs(estimate[field] - recomputed[field]) > 1e-9:
                raise ValueError(f"report metric disagrees with raw records: {name}/{field}")
        if set(estimate["seed_values"]) != set(means) or any(
                abs(estimate["seed_values"][seed] - value) > 1e-9 for seed, value in means.items()):
            raise ValueError("report seed means disagree with raw evidence")
        return recomputed

    def curve(self, name, arm, condition, ks=SHORT, *, uncertainty=False, task="shapes4", metric="accuracy"):
        return [estimate_text(self.estimate(name, arm, condition, k, task, metric), uncertainty) for k in ks]

    def parameters(self, name, arm):
        if arm == "frozen":
            return "0"
        counts = []
        for checkpoint in self.raw[name]["run_config"]["checkpoints"][arm].values():
            source = checkpoint["training_source"]
            path = self.root / source["path"]
            if str(path) not in self.training:
                content = path.read_bytes()
                if hashlib.sha256(content).hexdigest() != source["sha256"]:
                    raise ValueError("source-training artifact changed")
                self.training[str(path)] = json.loads(content)
            runs = [run for _, run in iter_training_runs(self.training[str(path)])
                    if (self.root / run.get("adapter_path", "")).resolve() == (self.root / checkpoint["path"]).resolve()]
            if len(runs) != 1:
                raise ValueError("source checkpoint lacks unambiguous training parameter metadata")
            counts.append(runs[0]["trainable_parameters"])
        if len(set(counts)) != 1:
            raise ValueError("adapter parameter count differs across seeds")
        return f"{counts[0]:,}".replace(",", r"{,}")

    def paired(self, name, task, condition, k, left="frozen", right="cama"):
        differences, wins, losses = [], 0, 0
        for seed in self.raw[name]["run_config"]["seeds"]:
            before = self.cell(name, task, condition, left, k, seed)
            after = self.cell(name, task, condition, right, k, seed)
            check_pairing(before, after)
            delta = [int(b["prediction"] == b["gold"]) - int(a["prediction"] == a["gold"])
                     for a, b in zip(before["records"], after["records"])]
            differences.append(sum(delta) / len(delta))
            wins += delta.count(1)
            losses += delta.count(-1)
        return seed_interval(differences), wins, losses


def cross_file_comparisons(evidence):
    left = evidence.raw[SHAPES]["run_config"]
    right = evidence.raw[DEPTH]["run_config"]
    fields = ("model", "model_revision", "dtype", "img_size", "max_new_tokens", "score_mode",
              "n", "n_train_per_class", "n_test_pool", "seeds")
    if any(left[field] != right[field] for field in fields) or left["protocols"]["random"] != right["protocols"]["random"]:
        raise ValueError("cross-file model/prompt/data configurations are not matched")
    matched, flips = 0, 0
    for seed in left["seeds"]:
        for condition in ("standard", "remap", "remap_shuffled"):
            for k in FULL:
                a = evidence.cell(SHAPES, "shapes4", condition, "frozen", k, seed)
                b = evidence.cell(DEPTH, "shapes4", condition, "frozen", k, seed)
                check_pairing(a, b)
                matched += len(a["records"])
                flips += sum(x["prediction"] != y["prediction"] for x, y in zip(a["records"], b["records"]))
    contrasts = []
    for arm in ("early", "mid", "late", "natural_connector"):
        for k in SHORT:
            differences = []
            for seed in left["seeds"]:
                a = evidence.cell(DEPTH, "shapes4", "remap", arm, k, seed)
                b = evidence.cell(SHAPES, "shapes4", "remap", "connector", k, seed)
                check_pairing(a, b)
                differences.append(sum(int(y["prediction"] == y["gold"]) - int(x["prediction"] == x["gold"])
                                       for x, y in zip(a["records"], b["records"])) / len(a["records"]))
            contrasts.append({"direction": "episode connector minus " + arm, "k": k,
                              "condition": "remap", "seed_differences": dict(zip(map(str, left["seeds"]), differences)),
                              **seed_interval(differences)})
    targets = []
    for task in ("cifar4", "fashion4"):
        feature_name, calibration_name = f"frozen_features_{task}_n40.json", f"calibration_{task}_n40.json"
        config_fields = ("model", "model_revision", "dtype", "img_size", "n", "seeds")
        for name in (feature_name, calibration_name):
            if any(evidence.raw[name]["run_config"][field] != evidence.raw[TRANSFER]["run_config"][field]
                   for field in config_fields):
                raise ValueError("target baseline model/data configurations are not matched")
        feature_pairs, calibration_pairs = 0, 0
        target_effects = []
        for seed in left["seeds"]:
            for k in SHORT:
                for condition in ("remap", "remap_shuffled"):
                    a = evidence.cell(feature_name, task, condition, "1nn", k, seed)
                    b = evidence.cell(TRANSFER, task, condition, "connector", k, seed)
                    check_pairing(a, b, option_change=True)
                    feature_pairs += len(a["records"])
            for k in FULL:
                for condition in ("standard", "remap"):
                    a = evidence.cell(calibration_name, task, condition, "calibrated", k, seed)
                    b = evidence.cell(TRANSFER, task, condition, "frozen", k, seed)
                    check_pairing(a, b)
                    calibration_pairs += len(a["records"])
        for k in SHORT:
            differences = []
            for seed in left["seeds"]:
                a = evidence.cell(TRANSFER, task, "remap", "connector", k, seed)["records"]
                b = evidence.cell(feature_name, task, "remap", "1nn", k, seed)["records"]
                differences.append(sum(int(y["prediction"] == y["gold"]) - int(x["prediction"] == x["gold"])
                                       for x, y in zip(a, b)) / len(a))
            target_effects.append({"direction": "frozen-feature 1-NN minus connector generation", "k": k,
                                   "seed_differences": dict(zip(map(str, left["seeds"]), differences)),
                                   **seed_interval(differences)})
        targets.append({"task": task, "sources": {name: evidence.artifacts[name]["sha256"]
                                                   for name in (TRANSFER, feature_name, calibration_name)},
                        "checked_config_fields": list(config_fields),
                        "matched_feature_semantic_episodes": feature_pairs,
                        "matched_calibration_prompt_episodes": calibration_pairs,
                        "feature_contrasts": target_effects,
                        "scope": "paired end-to-end accuracy only; feature lookup has no displayed options or generation decoder; calibration uses restricted likelihood"})
    return {"sources": {name: evidence.artifacts[name]["sha256"] for name in (SHAPES, DEPTH)},
            "checked_config_fields": list(fields) + ["protocols.random"], "matched_frozen_records": matched,
            "frozen_prediction_differences": flips, "contrasts": contrasts,
            "scope": "paired matched episodes across files; fixed class-ordered-trained checkpoints; pointwise seed intervals",
            "target_baselines": targets}


def validate_dara_frozen(e):
    names = [f"dara_{objective}_{snapshot}_n40.json" for objective in ("natural", "episode_remap")
             for snapshot in ("selected", "final")]
    reference = names[0]
    for name in names[1:]:
        for seed in e.raw[reference]["run_config"]["seeds"]:
            for condition in ("standard", "remap"):
                for k in SHORT:
                    left = e.cell(reference, "shapes4", condition, "frozen", k, seed)
                    right = e.cell(name, "shapes4", condition, "frozen", k, seed)
                    check_pairing(left, right)
                    if any(a["prediction"] != b["prediction"] or a["raw_output"] != b["raw_output"]
                           for a, b in zip(left["records"], right["records"])):
                        raise ValueError("DARA frozen controls differ across snapshots/objectives")
    return reference


def build_fragments(e):
    fragments, columns = {}, {}

    def add(name, header, rows):
        columns[name] = header
        fragments[name + ".tex"] = "% Generated by scripts/export_revision_tables.py; do not edit.\n% Columns: " + header + "\n" + "\n".join(rows) + "\n"

    repair = [(SHAPES, "frozen"), (DEPTH, "early"), (DEPTH, "mid"), (DEPTH, "late"), (SHAPES, "connector")]
    add("repair_confirmatory", "Region & trainable parameters & remap K4 & K8 & K16",
        [tex_row(ARM_LABELS[arm], e.parameters(name, arm), *e.curve(name, arm, "remap")) for name, arm in repair])
    add("repair_uncertainty", "Region & condition & K0 & K4 & K8 & K16",
        [tex_row(ARM_LABELS[arm], CONDITIONS[c], *e.curve(name, arm, c, FULL, uncertainty=True))
         for name, arm in repair for c in ("standard", "remap")])
    controls = [(SHAPES, "frozen"), (SHAPES, "connector"), (DEPTH, "natural_connector"), (SHAPES, "all_layer")]
    add("source_controls", "Method & trainable parameters & remap K4/K8/K16 (native generation only)",
        [tex_row(ARM_LABELS[arm], e.parameters(name, arm), "/".join(e.curve(name, arm, "remap")))
         for name, arm in controls])
    add("baseline_uncertainty", "Method & condition & K4 & K8 & K16 (native generation only)",
        [tex_row(label, CONDITIONS[c], *e.curve(name, arm, c, uncertainty=True))
         for name, arm, label, c in ((DEPTH, "natural_connector", "Natural connector", "remap"),
                                    (SHAPES, "all_layer", "All-layer", "remap"),
                                    (QWEN25, "connector", "Qwen2.5 connector", "standard"),
                                    (QWEN25, "connector", "Qwen2.5 connector", "remap"))])
    replication = []
    for name, label in ((SHAPES, "Qwen2-VL-2B"), (QWEN25, "Qwen2.5-VL-3B")):
        replication.extend(tex_row(label, CONDITIONS[c], *e.curve(name, "connector", c))
                           for c in ("standard", "remap", "remap_shuffled"))
        replication.append(tex_row(label, "Wrong-map following", *e.curve(
            name, "connector", "remap_shuffled", metric="follows_corrupted_mapping_rate")))
    add("connector_replication", "Model & condition/metric & K4 & K8 & K16 (episode connectors)", replication)
    add("transfer_confirmatory", "Domain & state & standard K4/K8/K16 & remap K4/K8/K16 & wrong-map true K4/K8/K16",
        [tex_row(task, TRANSFER_LABELS[arm], *("/".join(e.curve(TRANSFER, arm, c, task=key))
                                       for c in ("standard", "remap", "remap_shuffled")))
         for task, key in (("CIFAR-4", "cifar4"), ("Fashion-4", "fashion4"))
         for arm in ("frozen", "connector", "all_layer")])
    add("transfer_retention", "Domain & state & condition & K0 & K4 & K8 & K16",
        [tex_row(task, TRANSFER_LABELS[arm], CONDITIONS[c], *e.curve(TRANSFER, arm, c, FULL, task=key, uncertainty=True))
         for task, key in (("CIFAR-4", "cifar4"), ("Fashion-4", "fashion4"))
         for arm in ("frozen", "connector", "all_layer") for c in ("standard", "remap")])
    rows = []
    for label, task in (("Shapes-4", "shapes4"), ("CIFAR-4", "cifar4"), ("Fashion-4", "fashion4")):
        name = "standard_calibration_shapes_n40.json" if task == "shapes4" else f"calibration_{task}_n40.json"
        for condition in (("standard",) if task == "shapes4" else ("standard", "remap")):
            for arm, rule in (("raw", "Raw"), ("calibrated", "Calibrated")):
                rows.append(tex_row(label, rule, CONDITIONS[condition], *e.curve(name, arm, condition, FULL, task=task, uncertainty=True)))
        if task != "shapes4":
            rows.append(tex_row(label, "1-NN", "Remap", "--",
                                *e.curve(f"frozen_features_{task}_n40.json", "1nn", "remap", task=task, uncertainty=True)))
    add("target_baselines", "Domain & scoring rule & condition & K0 & K4 & K8 & K16", rows)
    for fragment, tasks, ks in (("cama_paired", ("shapes4",), SHORT),
                               ("cama_domains", ("cifar4", "fashion4"), (16,))):
        rows = []
        for task in tasks:
            name = f"cama_classification_{task}_n40.json"
            for condition in ("standard", "remap"):
                for k in ks:
                    effect, wins, losses = e.paired(name, task, condition, k)
                    leading = (CONDITIONS[condition], k) if fragment == "cama_paired" else (task.replace("4", "-4").title(), CONDITIONS[condition])
                    rows.append(tex_row(*leading, estimate_text(e.estimate(name, "frozen", condition, k, task)),
                                        estimate_text(e.estimate(name, "cama", condition, k, task)),
                                        interval_text(effect), wins, losses))
        add(fragment, ("Condition & K" if fragment == "cama_paired" else "Domain & condition (K16)")
            + " & local frozen & CAMA & paired delta [95% CI] & wins & losses (pooled descriptive counts)", rows)
    frozen = validate_dara_frozen(e)
    rows = [tex_row("Local frozen", "--", "/".join(e.curve(frozen, "frozen", "standard")),
                    "/".join(e.curve(frozen, "frozen", "remap")),
                    "/".join(e.curve(frozen, "frozen", "remap", metric="unparseable_rate")))]
    for objective in ("natural", "episode_remap"):
        for snapshot in ("selected", "final"):
            name = f"dara_{objective}_{snapshot}_n40.json"
            rows.append(tex_row(objective.replace("_", " ").title(), snapshot.title(),
                                "/".join(e.curve(name, "dara", "standard")),
                                "/".join(e.curve(name, "dara", "remap")),
                                "/".join(e.curve(name, "dara", "remap", metric="unparseable_rate"))))
    add("dara_details", "Objective & snapshot & standard K4/K8/K16 & remap K4/K8/K16 & remap invalid K4/K8/K16", rows)
    add("qwen7", "Condition & K0 & K1 & K2 & K4 & K8 & K16",
        [tex_row(CONDITIONS[c], *e.curve("options_qwen7_shapes_n40.json", "frozen", c, (0, 1, 2, 4, 8, 16), uncertainty=True))
         for c in ("standard", "remap")])
    add("vqa", "State & K0 & K4 & K8 (seed intervals; fixed 80 distinct images)",
        [tex_row(label, *e.curve("vqav2_distinct80.json", arm, "vqa", (0, 4, 8), task="vqav2", uncertainty=True))
         for arm, label in (("frozen", "Frozen"), ("vqa_connector", "VQA connector"), ("shapes_connector", "Shapes connector"))])
    cross = cross_file_comparisons(e)
    fragments["cross_file_comparisons.json"] = json.dumps(cross, indent=2, sort_keys=True) + "\n"
    manifest = {"report_sha256": e.report_sha256,
                "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "columns": columns,
                "artifact_sha256": {name: artifact["sha256"] for name, artifact in sorted(e.artifacts.items())},
                "notes": ["Fragments contain rows only; wrappers/captions remain manuscript-owned.",
                          "Intervals are pointwise, across three seed means, untruncated, without multiplicity adjustment.",
                          "Wrong-map true denotes accuracy against the original class, not the corrupted demonstration association.",
                          "CAMA wins/losses are pooled descriptive counts, not independent inferential replicates.",
                          "The feature baseline has no displayed option list or generation decoder."]}
    fragments["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    return fragments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "data/revision_v25/completed_revision_summary.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "paper/revision_results")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    fragments = build_fragments(Evidence(args.report))
    mismatches = []
    for name, content in fragments.items():
        path = args.out_dir / name
        if args.check:
            if not path.exists() or path.read_text() != content:
                mismatches.append(name)
        else:
            write_report(path, content.encode("utf-8"))
    if mismatches:
        parser.exit(1, "Generated revision rows differ: " + ", ".join(mismatches) + "\n")
    print(f"{'Checked' if args.check else 'Wrote'} {len(fragments)} deterministic revision fragments")


if __name__ == "__main__":
    main()
