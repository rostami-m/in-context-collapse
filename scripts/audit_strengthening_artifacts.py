#!/usr/bin/env python3
"""Fail-closed, CPU-only audit and reduction of strengthening experiment evidence.

Scientific outcomes are never acceptance criteria. Incomplete or inconsistent
artifacts have no aggregate estimates, including when their observed scores are
high. This module deliberately does not import the rule experiment evaluator.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import re
import statistics

from PIL import Image, ImageDraw

try:
    from .audit_revision_artifacts import audit_file as audit_existing, parse_generation
    from .summarize_revision_followup import seed_interval, summarize_file
except ImportError:
    from audit_revision_artifacts import audit_file as audit_existing, parse_generation
    from summarize_revision_followup import seed_interval, summarize_file


ROOT = Path(__file__).resolve().parents[1]
FAMILY = "two_object_color_selection"
RULES = ["left_color", "right_color"]
COLORS = ["red", "green", "blue", "yellow"]
PALETTE = [[220, 45, 45], [25, 150, 70], [40, 80, 215], [230, 185, 25]]
SCOPE = (
    "All configured outcomes are retained; audit success is not evidence of a positive effect. "
    "Intervals are pointwise and unadjusted for multiplicity. Seed intervals use seed means. "
    "Case-bootstrap intervals condition on the saved checkpoints and observed sampling frame. "
    "An interval including zero does not demonstrate equivalence. Rule transfer is not collapse "
    "repair unless a frozen standard-label baseline independently exhibits a decline. "
    "No cross-artifact or cross-prompt method rankings are computed."
)


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def equal_number(actual, expected, message):
    require(type(actual) in (int, float) and math.isfinite(actual)
            and abs(actual - expected) <= 1e-9, message)


def unique_axis(config, name, *, integer=False):
    values = config.get(name)
    require(isinstance(values, list) and values, f"missing or empty {name} axis")
    require(all(type(value) is int if integer else isinstance(value, str) and value
                for value in values), f"invalid {name} axis")
    require(len(values) == len(set(values)), f"duplicate {name} axis")
    return values


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resolve_path(value, root=ROOT):
    path = Path(value)
    return path if path.is_absolute() else root / path


def verify_checkpoints(config, root=ROOT):
    """Verify recorded checkpoint/source bytes, without loading any tensor."""
    checkpoints = config.get("checkpoints")
    require(isinstance(checkpoints, dict), "missing checkpoint provenance")
    require(set(checkpoints) == set(config["arms"]) - {"frozen"},
            "checkpoint arm metadata differs from configured arms")
    for arm, by_seed in checkpoints.items():
        require(set(by_seed) == {str(seed) for seed in config["seeds"]},
                f"{arm}: checkpoint seed metadata differs")
        for seed, checkpoint in by_seed.items():
            directory = resolve_path(checkpoint["path"], root)
            hashes = checkpoint.get("sha256", {})
            require("adapter_config.json" in hashes and any(
                name.startswith("adapter_model") for name in hashes),
                f"{arm}/{seed}: missing checkpoint file hashes")
            for name, digest in hashes.items():
                require(Path(name).name == name, "checkpoint file name escapes directory")
                require(file_digest(directory / name) == digest,
                        f"{arm}/{seed}: checkpoint hash mismatch: {name}")
            adapter = json.loads((directory / "adapter_config.json").read_text())
            base = adapter.get("base_model_name_or_path")
            require(base == checkpoint.get("base_model_name_or_path") == config["model"],
                    f"{arm}/{seed}: checkpoint base-model metadata differs")
            source = checkpoint.get("training_source")
            require(isinstance(source, dict), f"{arm}/{seed}: missing training source")
            source_path = resolve_path(source["path"], root)
            require(file_digest(source_path) == source["sha256"],
                    f"{arm}/{seed}: training source hash mismatch")
            training = json.loads(source_path.read_text())["run_config"]
            for field in ("protocol", "objective", "steps", "rank"):
                require(source.get(field) == training.get(field),
                        f"{arm}/{seed}: training {field} metadata differs")


def bootstrap_cases(values, draws=10000, seed=90210):
    """Bootstrap independent case units, never their repeated seed evaluations."""
    values = list(values)
    require(values and draws >= 2, "case bootstrap needs observations and at least two draws")
    require(all(math.isfinite(value) for value in values), "nonfinite bootstrap observation")
    rng = random.Random(seed)
    samples = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(draws))

    def quantile(fraction):
        position = fraction * (len(samples) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        return samples[lower] + (position - lower) * (samples[upper] - samples[lower])

    return {"mean": statistics.mean(values), "ci95_low": quantile(.025),
            "ci95_high": quantile(.975), "n_cases": len(values), "draws": draws,
            "seed": seed, "interval": "conditional paired-case percentile bootstrap"}


def validate_generation(record):
    labels = record.get("label_space")
    require(isinstance(labels, list) and len(labels) == 4
            and all(isinstance(label, str) and label for label in labels)
            and len({label.casefold() for label in labels}) == 4,
            "invalid four-class label space")
    require(isinstance(record.get("raw_output"), str), "missing raw generation")
    prediction, status = parse_generation(record["raw_output"], labels)
    require(all(field in record and (record[field] is None or type(record[field]) is int)
                and record[field] == prediction for field in ("prediction", "generation_prediction")),
            "stored prediction differs from independently parsed generation")
    require(record.get("parse_status") == status, "stored parse status differs")
    require(type(record.get("unparseable")) is bool
            and record["unparseable"] == (prediction is None), "unparseable flag differs")
    gold = record.get("gold")
    require(type(gold) is int and 0 <= gold < 4, "invalid gold class")
    require(type(record.get("correct")) is bool
            and record["correct"] == (prediction == gold), "correctness flag differs")
    order = record.get("option_order")
    require(isinstance(order, list) and all(type(value) is int for value in order)
            and sorted(order) == list(range(4)), "invalid displayed-option permutation")
    require(record.get("displayed_options") == [labels[index] for index in order],
            "displayed options differ from label mapping")
    if "parse_matches" in record:
        normalized = " ".join(record["raw_output"].casefold().strip().split())
        matches = [index for index, label in enumerate(labels)
                   if re.search(r"(?<!\w)" + re.escape(label.casefold()) + r"(?!\w)", normalized)]
        require(record["parse_matches"] == matches, "stored parse matches differ")
    return prediction


def audit_options(path, payload, *, root=ROOT, bootstrap_draws=10000):
    config = payload["run_config"]
    for axis in ("tasks", "conditions", "arms", "option_orders"):
        unique_axis(config, axis)
    for axis in ("seeds", "ks"):
        unique_axis(config, axis, integer=True)
    require(config.get("score_mode") == "generation", "transfer readout must be generation")
    require("frozen" in config["arms"], "transfer artifact lacks own frozen control")
    existing = audit_existing(path)
    require(not existing["errors"], "; ".join(existing["errors"]))
    report = summarize_file(path, bootstrap_draws=bootstrap_draws)
    require(report["status"] == "complete", "incomplete option-order grid")
    verify_checkpoints(config, root)
    queries, demos = defaultdict(set), defaultdict(set)
    for key, cell in payload["cells"].items():
        _, task, _, condition, k, _ = key.split(":")
        require(cell.get("score_mode") == "generation", "cell readout differs from config")
        for record in cell["records"]:
            validate_generation(record)
            require(record.get("k") == int(k) and record.get("condition") == condition,
                    "record metadata differs from cell")
            queries[task].add(record["query_id"])
            demos[task].update(record["demo_source_ids"])
    for task in queries:
        require(not queries[task].intersection(demos[task]),
                f"{task}: observed demonstration/evaluation image overlap")
    report["interpretation_scope"] = SCOPE
    report["verification_scope"] = (
        "Existing independent record/grid auditor plus strict complete-grid reducer; "
        "generation scores, observed train/test image-ID disjointness and checkpoint/source hashes. "
        "Original target images are not embedded, so their semantic labels cannot be replayed here."
    )
    report["objective_transfer"] = summarize_objective_transfer(payload)
    return report


def summarize_objective_transfer(payload):
    """Prespecified same-artifact contrasts, called after the complete-grid audit.

    Query pairing is checked within K. Averaging paired seed-level effects over
    K does not assume that different K cells share demonstration identities.
    """
    config, cells = payload["run_config"], payload["cells"]
    seeds, arms = config["seeds"], config["arms"]
    result = {"primary": [], "secondary": [], "standard_diagnostics": [],
              "primary_status": "withheld: requires remap K4/8/16 and both objective arms",
              "interpretation_scope": "Within-artifact paired seed differences, averaged equally over the "
              "prespecified K values. No cross-K or correct-versus-corrupted demonstration pairing is assumed."}
    require(config["option_orders"] == ["random"], "strengthening transfer requires randomized options only")

    def get(seed, task, arm, condition, k):
        return cells[f"{seed}:{task}:{arm}:{condition}:{k}:random"]

    def contrast(task, left, right, ks, condition="remap"):
        seed_effects = {}
        identity = ("query_index", "query_id", "gold", "label_space", "option_order", "displayed_options",
                    "demo_source_ids", "demo_source_classes", "demo_label_classes", "shuffle_map")
        for seed in seeds:
            changes = []
            for k in ks:
                a, b = get(seed, task, left, condition, k), get(seed, task, right, condition, k)
                require(len(a["records"]) == len(b["records"]) == config["n"], "objective contrast query count differs")
                require(all(all(old.get(field) == new.get(field) for field in identity)
                            for old, new in zip(a["records"], b["records"])), "objective arm contrast is unpaired")
                changes.append(statistics.mean(int(new["correct"]) - int(old["correct"])
                                                for old, new in zip(a["records"], b["records"])))
            seed_effects[str(seed)] = statistics.mean(changes)
        return {"task": task, "condition": condition, "left_arm": left, "right_arm": right,
                "sign": "right minus left", "ks": list(ks), "seed_differences": seed_effects,
                **seed_interval(seed_effects.values())}

    complete_primary_grid = "remap" in config["conditions"] and {4, 8, 16}.issubset(config["ks"])
    objective_arms = {"episode_connector", "natural_connector"}.issubset(arms)
    if complete_primary_grid:
        for task in config["tasks"]:
            if objective_arms:
                result["primary"].append(contrast(task, "natural_connector", "episode_connector", [4, 8, 16]))
            for arm in arms:
                if arm != "frozen":
                    result["secondary"].append(contrast(task, "frozen", arm, [4, 8, 16]))
            if {"episode_connector", "all_layer"}.issubset(arms):
                result["secondary"].append(contrast(task, "episode_connector", "all_layer", [4, 8, 16]))
        if objective_arms:
            result["primary_status"] = "complete"
    if "standard" in config["conditions"] and 0 in config["ks"]:
        for task in config["tasks"]:
            for arm in arms:
                baseline = {str(seed): statistics.mean(row["correct"] for row in
                            get(seed, task, arm, "standard", 0)["records"]) for seed in seeds}
                entry = {"task": task, "arm": arm, "k0_accuracy": {
                    **seed_interval(baseline.values()), "seed_values": baseline}, "changes_from_own_k0": []}
                if arm != "frozen":
                    entry["k0_minus_frozen"] = contrast(task, "frozen", arm, [0], "standard")
                for k in config["ks"]:
                    if not k:
                        continue
                    effects = {str(seed): statistics.mean(row["correct"] for row in
                               get(seed, task, arm, "standard", k)["records"]) - baseline[str(seed)]
                               for seed in seeds}
                    entry["changes_from_own_k0"].append({"k": k, "sign": "K minus own K0",
                        **seed_interval(effects.values()), "seed_differences": effects})
                result["standard_diagnostics"].append(entry)
    return result


def pixel_digest(image):
    digest = hashlib.sha256()
    digest.update(image.mode.encode("ascii"))
    digest.update(str(image.size).encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def audit_manifest(path, expected_hash, partition, *, expected_n=None):
    """Reconstruct pixels and derive semantic truth independently of the runner."""
    path = Path(path)
    require(file_digest(path) == expected_hash, "manifest hash mismatch")
    manifest = json.loads(path.read_text())
    config = manifest["config"]
    require(config.get("family") == FAMILY and config.get("schema_version") == 1,
            "unsupported semantic dataset version/family")
    require(config.get("partition") == partition, "manifest partition metadata differs")
    require(config.get("renderer") == "pil_solid_circles_v1", "unsupported scene renderer")
    require(config.get("colors") == COLORS and config.get("color_rgb") == PALETTE,
            "semantic color palette differs")
    require(config.get("training_used") is False and config.get("scene_policy") ==
            "query_shift_reserved_two_other_shifts_repeated_v1", "semantic dataset policy differs")
    n, size = config.get("n"), config.get("img_size")
    require(type(n) is int and n >= 4 and n % 4 == 0 and (expected_n is None or n == expected_n),
            "manifest query count differs")
    require(type(size) is int and size >= 32, "invalid semantic image size")
    scenes = manifest["scenes"]
    queries = manifest["queries"]
    require(isinstance(scenes, dict) and len(queries) == n, "incomplete semantic manifest")
    image_ids, paths = set(), set()
    for scene_id, scene in scenes.items():
        require(scene.get("scene_id") == scene_id, "scene identity metadata differs")
        require(scene.get("partition") == partition, "scene partition metadata differs")
        require(scene.get("role") in {"query", "demonstration"}, "invalid scene role")
        background = scene.get("background")
        require(isinstance(background, list) and len(background) == 3
                and all(type(value) is int and 0 <= value <= 255 for value in background),
                "invalid scene background")
        objects = scene.get("objects")
        require(isinstance(objects, list) and len(objects) == 2, "scene needs exactly two objects")
        ordered = sorted(objects, key=lambda obj: obj["center"][0])
        require(ordered[0]["center"][0] < size / 2 < ordered[1]["center"][0],
                "scene objects do not occupy unambiguous left/right positions")
        rendered = Image.new("RGB", (size, size), tuple(background))
        draw = ImageDraw.Draw(rendered)
        for side, obj in zip(("left", "right"), ordered):
            center, radius, color = obj.get("center"), obj.get("radius"), obj.get("color")
            require(isinstance(center, list) and len(center) == 2
                    and all(type(value) is int for value in center), "invalid object center")
            require(type(radius) is int and radius > 0
                    and all(radius <= value < size - radius for value in center),
                    "object lies outside canvas")
            require(type(color) is int and 0 <= color < 4, "invalid object color")
            require(obj.get("side") == side and scene.get(side + "_color") == color,
                    "scene truth differs from object color/position")
            cx, cy = center
            draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=tuple(PALETTE[color]))
        require(ordered[0]["center"][0] + ordered[0]["radius"]
                < ordered[1]["center"][0] - ordered[1]["radius"], "scene objects overlap")
        png_relative = Path(scene["png_path"])
        require(not png_relative.is_absolute() and ".." not in png_relative.parts,
                "scene PNG escapes manifest directory")
        png_path = path.parent / png_relative
        require(png_path.resolve().is_relative_to(path.parent.resolve()), "scene PNG symlink escapes directory")
        require(png_path not in paths, "duplicate scene PNG path")
        paths.add(png_path)
        require(file_digest(png_path) == scene["png_sha256"], "scene PNG hash mismatch")
        with Image.open(png_path) as actual:
            require(actual.mode == "RGB" and actual.size == (size, size), "scene PNG format differs")
            require(actual.tobytes() == rendered.tobytes(), "scene PNG differs from semantic rendering")
            digest = pixel_digest(actual)
        require(digest == scene.get("image_id"), "scene pixel hash differs")
        require(digest not in image_ids, "duplicate pixels in semantic dataset")
        image_ids.add(digest)
    used_scenes, pair_ids = set(), set()
    for index, query in enumerate(queries):
        pair_id = query.get("pair_id")
        require(isinstance(pair_id, str) and pair_id and pair_id not in pair_ids,
                "missing or duplicate query pair identity")
        pair_ids.add(pair_id)
        scene_id = query["scene_id"]
        demo_ids = query["demo_scene_ids"]
        require(scene_id in scenes and scenes[scene_id]["role"] == "query",
                "query refers to non-query scene")
        require(isinstance(demo_ids, list) and len(demo_ids) == 16
                and len(set(demo_ids)) == 16, "query needs sixteen unique demonstration scenes")
        require(all(demo in scenes and scenes[demo]["role"] == "demonstration" for demo in demo_ids),
                "demonstration refers to non-training scene")
        require(not used_scenes.intersection([scene_id, *demo_ids]),
                "scene reused across query cases or demonstration/query split")
        used_scenes.update([scene_id, *demo_ids])
        left, right = (scenes[scene_id][rule] for rule in RULES)
        shift = (right - left) % 4
        require(left == index % 4 and shift == 1 + (index // 4) % 3,
                "query color balancing/shift policy differs")
        require(shift != 0, "query rules do not disagree")
        pair_types = {(color, (color + offset) % 4)
                      for color in range(4) for offset in range(1, 4) if offset != shift}
        for start in (0, 8):
            observed = [(scenes[demo]["left_color"], scenes[demo]["right_color"])
                        for demo in demo_ids[start:start+8]]
            require(len(observed) == len(set(observed)) and set(observed) == pair_types,
                    "demonstration color-pair holdout/balance policy differs")
        require((left, right) not in pair_types, "query color pair leaks into demonstrations")
    require(used_scenes == set(scenes), "manifest contains unreferenced or missing scenes")
    return manifest


def rule_truth(scene, rule):
    require(rule in RULES, "unsupported semantic rule")
    side = rule.removesuffix("_color")
    ordered = sorted(scene["objects"], key=lambda obj: obj["center"][0])
    return ordered[0 if side == "left" else 1]["color"]


def independent_shuffle(values, key):
    result = list(values)
    random.Random(int.from_bytes(hashlib.sha256(key.encode("ascii")).digest(), "big")).shuffle(result)
    return result


def compatible_semantic_rules(demos, labels):
    """A side is identifiable only if no alternate nonce bijection explains it."""
    return [rule for rule in RULES if any(
        all(mapping[rule_truth(demo, rule)] == label for demo, label in zip(demos, labels))
        for mapping in itertools.permutations(range(4)))]


def expected_transcript(record):
    content = []
    for scene, digest, label in zip(record["demo_scene_ids"], record["demo_source_ids"],
                                    record["demo_label_classes"]):
        content.extend([{"type": "image", "scene_id": scene, "image_id": digest},
                        {"type": "text", "text": f"This image shows: {record['label_space'][label]}."}])
    instruction = (f"What color is the {record['rule'].removesuffix('_color')} object?"
                   if record["condition"] == "explicit_natural"
                   else "Classify the last image using the rule shown by the examples.")
    content.extend([{"type": "image", "scene_id": record["scene_id"], "image_id": record["query_id"]},
                    {"type": "text", "text": instruction + " Answer with exactly one of ["
                     + ", ".join(record["displayed_options"]) + "] and nothing else."}])
    return content


def metric_intervals(by_seed, *, draws=10000):
    """A case has one contribution per fixed seed; bootstrap cases, not rows."""
    seed_ids = sorted(by_seed)
    cases = list(by_seed[seed_ids[0]])
    require(all(list(by_seed[seed]) == cases for seed in seed_ids),
            "conditional case interval has unmatched case identities")
    seed_values = {str(seed): statistics.mean(by_seed[seed].values()) for seed in seed_ids}
    interval = seed_interval(seed_values.values())
    interval["seed_values"] = seed_values
    conditional = bootstrap_cases(
        [statistics.mean(by_seed[seed][case] for seed in seed_ids) for case in cases], draws)
    conditional["n_fixed_seeds"] = len(seed_ids)
    return {"seed_interval": interval, "conditional_case_interval": conditional}


def audit_rules(path, payload, *, root=ROOT, bootstrap_draws=10000):
    config = payload["run_config"]
    require(config.get("schema_version") == 1 and config.get("family") == FAMILY,
            "unsupported rule artifact version/family")
    require(config.get("rules") == RULES, "configured rule set differs")
    require(config.get("conditions") == ["explicit_natural", "implicit_remap"],
            "configured rule conditions differ")
    grids = config.get("condition_ks", {})
    require(set(grids) == {"explicit_natural", "implicit_remap"} and grids["explicit_natural"] == [0]
            and isinstance(grids["implicit_remap"], list) and grids["implicit_remap"]
            and all(type(k) is int and k in {0, 8, 16} for k in grids["implicit_remap"])
            and len(grids["implicit_remap"]) == len(set(grids["implicit_remap"])),
            "configured rule shot grids differ")
    require(config.get("score_mode") == "generation", "rule readout must be generation")
    require(config.get("option_order") == "random" and config.get("protocol") == {
        "protocol_version": "2.5-rule-transfer-v1", "option_order": "random",
        "mapping_mode": "episode", "remap_token_set": "eval"}, "rule protocol metadata differs")
    require(config.get("mapping_policy") == "per_seed_query_eval_nonce_bijection_shared_across_rules_and_K"
            and config.get("identifiable_implicit_ks") == [8, 16], "rule mapping policy differs")
    seeds = unique_axis(config, "seeds", integer=True)
    arms = unique_axis(config, "arms")
    require("frozen" in arms, "rule artifact lacks own frozen control")
    require(config.get("partition") in {"development", "confirmatory"}, "invalid rule partition")
    require(type(config.get("n")) is int and config["n"] > 0, "invalid query count")
    verify_checkpoints(config, root)
    manifest = audit_manifest(resolve_path(config["manifest"], root), config["manifest_sha256"],
                              config["partition"], expected_n=config["n"])
    require(config.get("img_size") == manifest["config"]["img_size"], "image-size metadata differs")
    require(config.get("source_sha256") == file_digest(root / "incontext_collapse/rule_transfer.py"),
            "rule runtime source fingerprint differs")
    development = config.get("development_manifest")
    if config["partition"] == "confirmatory":
        require(isinstance(development, dict), "confirmatory split lacks development exclusion manifest")
        previous = audit_manifest(resolve_path(development["path"], root), development["sha256"],
                                  "development")
        require(not set(previous["scenes"]).intersection(manifest["scenes"]),
                "development/confirmatory scene identities overlap")
        require(not {scene["image_id"] for scene in previous["scenes"].values()}.intersection(
            scene["image_id"] for scene in manifest["scenes"].values()),
            "development/confirmatory pixels overlap")
    else:
        require(development is None, "development artifact has unexpected exclusion manifest")
    expected = {(seed, arm, condition, rule, k) for seed in seeds for arm in arms
                for condition, ks in config["condition_ks"].items() for rule in RULES for k in ks}
    stored = payload.get("cells", {})
    require(isinstance(stored, dict), "invalid rule cell container")
    cells, metrics = {}, defaultdict(dict)
    for key, cell in stored.items():
        parts = key.split(":")
        require(len(parts) == 5, "invalid rule cell key")
        seed, arm, condition, rule, k = parts
        coordinate = (int(seed), arm, condition, rule, int(k))
        require(coordinate in expected and coordinate not in cells, "unexpected or duplicate rule cell")
        require(all(cell.get(field) == value for field, value in zip(
            ("seed", "arm", "condition", "rule", "k"), coordinate)), "rule cell metadata differs")
        require(cell.get("score_mode") == "generation", "rule cell readout differs")
        records = cell["records"]
        require(len(records) == config["n"], "incomplete rule query records")
        require([record.get("query_index") for record in records] == list(range(config["n"])),
                "noncanonical rule query indices")
        for index, record in enumerate(records):
            query = manifest["queries"][index]
            scene = manifest["scenes"][query["scene_id"]]
            partition = config["partition"]
            demo_ids = independent_shuffle(query["demo_scene_ids"][:int(k)],
                f"rule-transfer-v1:demo-order:{partition}:{seed}:{index}:{k}")
            demos = [manifest["scenes"][demo] for demo in demo_ids]
            require(record.get("pair_id") == query["pair_id"]
                    and record.get("scene_id") == query["scene_id"]
                    and record.get("query_id") == scene["image_id"], "query identity differs from manifest")
            require(record.get("query_colors") == {side: scene[side + "_color"] for side in ("left", "right")},
                    "query color metadata differs")
            require(all(record.get(field) == value for field, value in
                        (("rule", rule), ("condition", condition), ("k", int(k)))),
                    "rule record metadata differs from cell")
            require(record.get("demo_scene_ids") == demo_ids
                    and record.get("demo_source_ids") == [demo["image_id"] for demo in demos],
                    "demonstration identities/order differ from manifest")
            require(record.get("demo_colors") == [
                {side: demo[side + "_color"] for side in ("left", "right")} for demo in demos],
                "demonstration color metadata differs")
            labels = [rule_truth(demo, rule) for demo in demos]
            require(record.get("demo_label_classes") == labels, "demonstration semantic labels differ")
            gold = rule_truth(scene, rule)
            other = rule_truth(scene, RULES[1 - RULES.index(rule)])
            require(record.get("gold") == gold and record.get("other_rule_gold") == other,
                    "semantic query gold differs")
            require(gold != other, "paired rule query is not diagnostic")
            prediction = validate_generation(record)
            require(type(record.get("answers_other_rule")) is bool
                    and record["answers_other_rule"] == (prediction == other), "other-rule score differs")
            compatible = compatible_semantic_rules(demos, labels)
            require(record.get("compatible_rules") == compatible, "compatible-rule enumeration differs")
            require(compatible == ([rule] if int(k) else RULES), "demonstrations do not identify intended rule")
            require(record.get("rule_identifiable") is (condition == "explicit_natural" or compatible == [rule]),
                    "rule-identifiability flag differs")
            expected_labels = COLORS if condition == "explicit_natural" else independent_shuffle(
                ["mip", "zorp", "kiki", "lurn"], f"rule-transfer-v1:mapping:{partition}:{seed}:{index}")
            require(record["label_space"] == expected_labels, "label-space RNG/policy differs")
            expected_order = independent_shuffle(range(4),
                f"option-order-v1:rule-transfer-v1:{partition}:{seed}:0:{index}")
            require(record["option_order"] == expected_order, "option-order RNG/policy differs")
            require(record.get("prompt_transcript") == expected_transcript(record),
                    "prompt transcript differs or leaks implicit rule information")
            for metric, value in (("accuracy", prediction == gold), ("unparseable_rate", prediction is None),
                                  ("answers_other_rule_rate", prediction == other)):
                metrics[(arm, condition, rule, int(k), metric)].setdefault(int(seed), {})[query["pair_id"]] = float(value)
        summary = cell["summary"]
        equal_number(summary.get("n"), config["n"], "rule summary count differs")
        for metric in ("accuracy", "unparseable_rate", "answers_other_rule_rate"):
            values = metrics[(arm, condition, rule, int(k), metric)][int(seed)].values()
            equal_number(summary.get(metric), statistics.mean(values), f"rule summary {metric} differs")
        require(summary.get("parse_status_counts") == dict(Counter(row["parse_status"] for row in records)),
                "rule summary parse counts differ")
        cells[coordinate] = records
    require(set(cells) == expected, f"incomplete rule grid: {len(cells)}/{len(expected)} cells")
    # The natural control names the rule explicitly; only implicit pairs isolate demo labels.
    identities = ("query_index", "pair_id", "query_id", "scene_id", "query_colors", "label_space",
                  "option_order", "displayed_options", "demo_scene_ids", "demo_source_ids", "demo_colors")
    pair_summaries = {}
    for seed in seeds:
        for condition, ks in config["condition_ks"].items():
            for k in ks:
                reference = cells[(seed, "frozen", condition, RULES[0], k)]
                for arm in arms:
                    for rule in RULES:
                        records = cells[(seed, arm, condition, rule, k)]
                        require(all(all(old[field] == new[field] for field in identities)
                                    for old, new in zip(reference, records)),
                                "paired rule/arm episode identities differ")
                if condition == "implicit_remap":
                    for arm in arms:
                        left = cells[(seed, arm, condition, RULES[0], k)]
                        right = cells[(seed, arm, condition, RULES[1], k)]
                        for old, new in zip(left, right):
                            if k == 0:
                                require(old["raw_output"] == new["raw_output"],
                                        "identical implicit K0 prompts have different generations")
                            both = old["correct"] and new["correct"]
                            switch = old["prediction"] is not None and new["prediction"] is not None \
                                and old["prediction"] != new["prediction"]
                            for metric, value in (("both_rules_correct", both), ("parsed_prediction_switch", switch)):
                                metrics[(arm, condition, "paired", k, metric)].setdefault(seed, {})[old["pair_id"]] = float(value)
                        pair_summaries[f"{seed}:{arm}:{condition}:{k}"] = {
                            "n": config["n"],
                            "both_rules_correct_rate": statistics.mean(x["correct"] and y["correct"] for x, y in zip(left, right)),
                            "semantic_switch_rate": statistics.mean(x["prediction"] is not None and y["prediction"] is not None
                                                                     and x["prediction"] != y["prediction"] for x, y in zip(left, right)),
                            "either_unparseable_rate": statistics.mean(x["unparseable"] or y["unparseable"] for x, y in zip(left, right)),
                        }
        for arm in arms:
            for rule in RULES:
                first_k, *other_ks = grids["implicit_remap"]
                base = cells[(seed, arm, "implicit_remap", rule, first_k)]
                for k in other_ks:
                    for old, new in zip(base, cells[(seed, arm, "implicit_remap", rule, k)]):
                        require(all(old[field] == new[field] for field in
                                    ("pair_id", "query_id", "label_space", "option_order", "displayed_options")),
                                "implicit mapping/options change across K")
    require(payload.get("paired_rules") == pair_summaries, "stored paired-rule metrics/grid differs")
    composite = {}
    for arm in arms:
        groups = [("explicit_natural_mean", [(arm, "explicit_natural", rule, 0, "accuracy") for rule in RULES])]
        if {8, 16}.issubset(grids["implicit_remap"]):
            groups.extend([
                ("implicit_remap_mean_k8_k16", [(arm, "implicit_remap", rule, k, "accuracy")
                                                 for rule in RULES for k in (8, 16)]),
                ("both_rules_correct_mean_k8_k16", [(arm, "implicit_remap", "paired", k, "both_rules_correct")
                                                    for k in (8, 16)]),
            ])
        for name, keys in groups:
            composite[(arm, name)] = {seed: {query["pair_id"]: statistics.mean(
                metrics[key][seed][query["pair_id"]] for key in keys) for query in manifest["queries"]}
                for seed in seeds}
    estimates, comparisons = [], []
    for (arm, condition, rule, k, metric), values in sorted(metrics.items()):
        coordinate = {"arm": arm, "condition": condition, "rule": rule, "k": k, "metric": metric}
        estimates.append({**coordinate, **metric_intervals(values, draws=bootstrap_draws)})
        if arm != "frozen":
            baseline = metrics[("frozen", condition, rule, k, metric)]
            differences = {seed: {case: value - baseline[seed][case] for case, value in cases.items()}
                           for seed, cases in values.items()}
            comparisons.append({**coordinate, "left_arm": "frozen", "right_arm": arm,
                                **metric_intervals(differences, draws=bootstrap_draws)})
    for (arm, name), values in composite.items():
        estimates.append({"arm": arm, "metric": name, "aggregation": "prespecified within-artifact mean",
                          **metric_intervals(values, draws=bootstrap_draws)})
        if arm != "frozen":
            baseline = composite[("frozen", name)]
            differences = {seed: {case: value - baseline[seed][case] for case, value in cases.items()}
                           for seed, cases in values.items()}
            comparisons.append({"left_arm": "frozen", "right_arm": arm, "metric": name,
                                "sign": "right minus left",
                                "aggregation": "prespecified within-artifact mean",
                                **metric_intervals(differences, draws=bootstrap_draws)})
    for left, right in (("natural_connector", "episode_connector"), ("episode_connector", "all_layer")):
        if left not in arms or right not in arms:
            continue
        for arm, name in composite:
            if arm != right:
                continue
            a, b = composite[(left, name)], composite[(right, name)]
            differences = {seed: {case: value - a[seed][case] for case, value in cases.items()}
                           for seed, cases in b.items()}
            comparisons.append({"left_arm": left, "right_arm": right, "metric": name,
                                "sign": "right minus left", "aggregation": "prespecified within-artifact mean",
                                **metric_intervals(differences, draws=bootstrap_draws)})
    return {"status": "complete", "schema": "rule_transfer", "run_config": config,
            "coverage": {"expected_cells": len(expected), "observed_cells": len(cells),
                         "records": len(cells) * config["n"]},
            "estimates": estimates, "paired_comparisons": comparisons,
            "interpretation_scope": SCOPE, "verification_scope":
            "Independent semantic pixel reconstruction, source/checkpoint hashes, development/test "
            "and query/demo disjointness, held-out color pairs, grid/pair identity, raw generation parsing. "
            "Both-rule success is separate from prediction changes; explicit-rule prompts are not "
            "treated as label-only interventions."}


def expected_subset(actual, expected, location="run_config"):
    require(isinstance(expected, dict), "expected config must be an object")
    for key, value in expected.items():
        require(key in actual, f"{location}.{key}: expected metadata is missing")
        if isinstance(value, dict):
            require(isinstance(actual[key], dict), f"{location}.{key}: metadata type differs")
            expected_subset(actual[key], value, f"{location}.{key}")
        else:
            require(actual[key] == value, f"{location}.{key}: metadata differs from frozen stage config")


def audit_file(path, *, expected_config=None, root=ROOT, bootstrap_draws=10000):
    path = Path(path)
    result = {"path": str(path.resolve()), "status": "invalid", "errors": [], "warnings": [],
              "estimates": [], "paired_comparisons": [], "interpretation_scope": SCOPE}
    try:
        require(type(bootstrap_draws) is int and bootstrap_draws >= 2, "invalid bootstrap draw count")
        before = file_digest(path)
        payload = json.loads(path.read_text())
        require(isinstance(payload, dict), "artifact root must be an object")
        config = payload["run_config"]
        require(isinstance(config, dict), "run_config must be an object")
        require(type(config.get("n")) is int and config["n"] > 0, "invalid configured query count")
        require(not payload.get("invalidated"), "artifact is explicitly invalidated")
        if expected_config is not None:
            expected_subset(config, expected_config)
        if config.get("experiment") == "two_object_rule_transfer":
            reduced = audit_rules(path, payload, root=root, bootstrap_draws=bootstrap_draws)
        elif config.get("experiment") == "paired_option_order_saved_adapters":
            reduced = audit_options(path, payload, root=root, bootstrap_draws=bootstrap_draws)
        else:
            raise AuditError("unsupported strengthening artifact schema")
        require(file_digest(path) == before, "artifact changed during audit")
        result.update(reduced, sha256=before)
    except (AuditError, ValueError, TypeError, KeyError, IndexError, AttributeError, OSError) as error:
        result["errors"].append(str(error))
        if "incomplete" in str(error):
            result["status"] = "partial"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--expected-config", type=Path,
                        help="frozen expected run_config subset; requires a single input")
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--json", action="store_true", help="emit the full machine-readable report")
    args = parser.parse_args(argv)
    if args.expected_config and len(args.paths) != 1:
        parser.error("--expected-config requires exactly one artifact")
    expected = json.loads(args.expected_config.read_text()) if args.expected_config else None
    reports = [audit_file(path, expected_config=expected, bootstrap_draws=args.bootstrap_draws)
               for path in args.paths]
    if args.json:
        print(json.dumps(reports[0] if len(reports) == 1 else reports, indent=2, allow_nan=False))
    else:
        for report in reports:
            print(f"{report['path']}: {report['status']}; {len(report['errors'])} errors")
            for error in report["errors"]:
                print(f"  ERROR {error}")
    return int(any(report["status"] != "complete" or report["errors"] for report in reports))


if __name__ == "__main__":
    raise SystemExit(main())
