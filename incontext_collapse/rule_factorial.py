"""Fixed 2x2 diagnosis of rule selection and answer-label binding.

All cells share fresh scenes, supports, option permutations and (where used)
nonce mappings. The factors change only the final rule instruction and answer
verbalizers. This is attribute-selection classification, not open-ended transfer.
No training, adaptive stopping, query filtering or checkpoint selection occurs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import time

from PIL import Image

from . import rule_transfer as rt
from .protocol import REMAP_TOKEN_SETS, image_id, sample_option_order
from .results import load_result, write_json_atomic


SCHEMA_VERSION = 1
NAMESPACE = "rule-factorial-2026-10-10-v1"
EXPERIMENT = "two_object_rule_factorial"
MODEL = "Qwen/Qwen2-VL-2B-Instruct"
MODEL_REVISION = "895c3a49bc3fa70a340399125c650a463535e71c"
RULES = rt.RULES
CONDITIONS = ("explicit_natural", "explicit_nonce", "implicit_natural", "implicit_nonce")
ARMS = ("frozen", "natural_connector", "episode_connector", "all_layer")
PARTITIONS = rt.PARTITIONS
CELL_FIELDS = ("seed", "arm", "condition", "rule", "k")
PAIR_FIELDS = rt.PAIR_FIELDS
FACTOR_IMAGE_FIELDS = ("query_index", "pair_id", "query_id", "scene_id", "query_colors",
                       "option_order", "demo_scene_ids", "demo_source_ids", "demo_colors",
                       "demo_label_classes", "gold", "other_rule_gold")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
HEX_ID = re.compile(r"^[0-9a-f]{64}(?::.*)?$")


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _hex_identifiers(value):
    """Conservative superset: image IDs, composite VQA IDs and other SHA256s."""
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _hex_identifiers(key)
            yield from _hex_identifiers(child)
    elif isinstance(value, list):
        for child in value:
            yield from _hex_identifiers(child)
    elif isinstance(value, str) and HEX_ID.fullmatch(value):
        yield value[:64]


def _read_prior_identifiers(path):
    if path.suffix.lower() in IMAGE_SUFFIXES:
        with Image.open(path) as image:
            return {image_id(image.convert("RGB"))}
    if path.suffix.lower() == ".jsonl":
        identifiers = set()
        for line in path.read_text().splitlines():
            if line.strip():
                identifiers.update(_hex_identifiers(json.loads(line)))
        return identifiers
    return set(_hex_identifiers(json.loads(path.read_text())))


def _source_shapes_hashes():
    # This is the exact source training/test pool used by the three saved arms.
    # Procedural rendering is CPU-only and does not access an external dataset.
    from .core import task_shapes4
    _, pool, test = task_shapes4(96, 120)
    return sorted({image_id(image) for images in pool.values() for image in images}
                  | {image_id(image) for image, _ in test})


def relative_exclusion_policy(root, excluded_directories):
    """Compare the effective policy without dereferencing a recorded old root."""
    root = Path(root)
    if not root.is_absolute() or ".." in root.parts:
        raise ValueError("inventory root must be an absolute normalized path")
    relative = set()
    for value in excluded_directories:
        path = Path(value)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("excluded directories must be absolute normalized paths")
        try:
            relative.add(path.relative_to(root))
        except ValueError:
            # An output directory outside the input root cannot exclude inputs.
            continue
    return sorted(str(path) for path in relative
                  if not any(path != parent and path.is_relative_to(parent)
                             for parent in relative))


def validate_inventory_policy(config, prior_root, excluded_directories):
    if (config.get("schema_version") != 1 or config.get("namespace") != NAMESPACE
            or config.get("policy") != "all_recorded_64hex_ids_cached_RGB_pixels_and_shapes96_120_v1"
            or relative_exclusion_policy(config["prior_root"], config["excluded_directories"])
            != relative_exclusion_policy(prior_root, excluded_directories)):
        raise ValueError("existing exclusion inventory configuration differs")


def build_exclusion_inventory(path, prior_root, excluded_directories=()):
    """Freeze recorded/cached prior evidence before preparing either partition.

    The inventory is a dated snapshot, not a claim to reconstruct unavailable
    historical raw images. New files added after the snapshot are not silently
    included on resume. Every captured file is checked for subsequent mutation.
    """
    path, prior_root = Path(path).resolve(), Path(prior_root).resolve()
    excluded = sorted({str(Path(p).resolve()) for p in excluded_directories})
    if not prior_root.is_dir():
        raise ValueError("prior data root must exist")
    config = {"schema_version": 1, "namespace": NAMESPACE, "prior_root": str(prior_root),
              "excluded_directories": excluded,
              "policy": "all_recorded_64hex_ids_cached_RGB_pixels_and_shapes96_120_v1"}
    if path.exists():
        saved = load_exclusion_inventory(path, prior_root=prior_root)
        validate_inventory_policy(saved["config"], prior_root, excluded)
        return path
    files, identifiers = [], set()
    for source in sorted(prior_root.rglob("*")):
        if (not source.is_file() or source.resolve() == path
                or any(source.resolve().is_relative_to(Path(root)) for root in excluded)
                or source.suffix.lower() not in IMAGE_SUFFIXES | {".json", ".jsonl"}):
            continue
        digest = _sha(source)
        found = _read_prior_identifiers(source)
        if _sha(source) != digest:
            raise ValueError(f"prior source changed while inventorying: {source}")
        identifiers.update(found)
        files.append({"path": str(source.resolve()), "sha256": digest,
                      "identifiers": len(found)})
    if not files:
        raise ValueError("prior inventory must include historical files")
    shapes = _source_shapes_hashes()
    identifiers.update(shapes)
    inventory = {"config": config, "created_utc": datetime.now(timezone.utc).isoformat(),
                 "files": files, "source_shapes96_120": shapes,
                 "excluded_identifiers": sorted(identifiers),
                 "interpretation": "conservative superset of recorded hashes and cached pixels; "
                                   "unavailable historical raw data cannot be reconstructed"}
    write_json_atomic(path, inventory)
    return path


def load_exclusion_inventory(path, prior_root=None):
    """Verify unchanged inventory bytes against an explicitly relocated mirror.

    The recorded root is used only to obtain safe relative input paths. When a
    new root is supplied, no fallback or read from the old root is permitted.
    """
    inventory = json.loads(Path(path).read_text())
    config = inventory["config"]
    validate_inventory_policy(config, config["prior_root"], config["excluded_directories"])
    if (not inventory["files"]
            or inventory["source_shapes96_120"] != _source_shapes_hashes()):
        raise ValueError("prior exclusion policy/source Shapes pool differs")
    old_root = Path(config["prior_root"])
    root = Path(prior_root if prior_root is not None else old_root).resolve()
    if not root.is_dir():
        raise ValueError("prior data root must exist")
    excluded = [Path(value) for value in relative_exclusion_policy(
        old_root, config["excluded_directories"])]
    identifiers = set(inventory["source_shapes96_120"])
    seen = set()
    for item in inventory["files"]:
        recorded = Path(item["path"])
        if not recorded.is_absolute() or ".." in recorded.parts:
            raise ValueError("unsafe recorded inventory source path")
        try:
            relative = recorded.relative_to(old_root)
        except ValueError as error:
            raise ValueError("inventory source escapes recorded prior root") from error
        source = (root / relative).resolve()
        if (relative in seen or not source.is_relative_to(root)
                or any(relative.is_relative_to(parent) for parent in excluded)):
            raise ValueError("duplicate, excluded or escaping inventory source")
        seen.add(relative)
        if _sha(source) != item["sha256"]:
            raise ValueError(f"prior source checksum differs: {source}")
        found = _read_prior_identifiers(source)
        if len(found) != item["identifiers"]:
            raise ValueError("prior source identifier count differs")
        identifiers.update(found)
    if inventory["excluded_identifiers"] != sorted(identifiers):
        raise ValueError("prior exclusion identifiers differ from captured evidence")
    return inventory


def _blueprint(partition, n, size, inventory_sha256):
    if partition not in PARTITIONS or n < 4 or n % 4 or size != 112:
        raise ValueError("partition must be development/confirmatory, n a positive multiple of four, size=112")
    config = {"schema_version": SCHEMA_VERSION, "experiment": EXPERIMENT,
              "namespace": NAMESPACE, "partition": partition, "n": n, "img_size": size,
              "renderer": "pil_solid_circles_v1", "colors": list(rt.COLORS),
              "color_rgb": [list(value) for value in rt.COLOR_RGB],
              "training_used": False,
              "scene_policy": "query_shift_reserved_two_other_shifts_repeated_v1",
              "exclusion_inventory_sha256": inventory_sha256}
    scenes, queries = {}, []
    for index in range(n):
        left, reserved_shift = index % 4, 1 + (index // 4) % 3
        query_id = f"{NAMESPACE}/{partition}/query/{index:04d}"
        scenes[query_id] = rt._scene(query_id, partition, "query", left,
                                     (left + reserved_shift) % 4, size)
        allowed = [shift for shift in (1, 2, 3) if shift != reserved_shift]
        demo_ids = []
        for block, shift in enumerate(allowed * 2):
            for color in range(4):
                demo_id = f"{NAMESPACE}/{partition}/demo/{index:04d}/{block * 4 + color:02d}"
                scenes[demo_id] = rt._scene(demo_id, partition, "demonstration", color,
                                            (color + shift) % 4, size)
                demo_ids.append(demo_id)
        queries.append({"pair_id": f"{NAMESPACE}:{partition}:query:{index:04d}",
                        "scene_id": query_id, "demo_scene_ids": demo_ids})
    return {"config": config, "queries": queries, "scenes": scenes}


def validate_exclusions(manifest, excluded_identifiers):
    ids = [scene["image_id"] for scene in manifest["scenes"].values()]
    if len(ids) != len(set(ids)):
        raise ValueError("query/support scene pixels are duplicated")
    overlap = set(ids) & set(excluded_identifiers)
    if overlap:
        raise ValueError(f"new scene pixels overlap prior evidence ({len(overlap)} images)")


def build_manifest(directory, partition, n, size=112, excluded_identifiers=(),
                   inventory_sha256=None):
    directory = Path(directory).resolve()
    path = directory / f"manifest_{partition}.json"
    expected = _blueprint(partition, n, size, inventory_sha256)
    if path.exists():
        manifest, _ = load_manifest(path, excluded_identifiers)
        if manifest["config"] != expected["config"]:
            raise ValueError("existing scene manifest configuration differs")
        return path
    # Check the whole image set before persisting anything; do not resample
    # away a collision or choose a different namespace based on model scores.
    images = {}
    for key, scene in expected["scenes"].items():
        images[key] = rt.render_scene(scene, size)
        scene["image_id"] = image_id(images[key])
    validate_exclusions(expected, excluded_identifiers)
    for key, scene in expected["scenes"].items():
        target = rt._safe_image_path(directory, scene["png_path"])
        if target.exists():
            with Image.open(target) as saved:
                if image_id(saved.convert("RGB")) != scene["image_id"]:
                    raise ValueError("existing image differs from deterministic scene")
        else:
            rt._write_png(target, images[key])
        scene["png_sha256"] = _sha(target)
    write_json_atomic(path, expected)
    return path


def load_manifest(path, excluded_identifiers=()):
    path = Path(path)
    manifest = json.loads(path.read_text())
    config = manifest["config"]
    expected = _blueprint(config["partition"], config["n"], config["img_size"],
                          config["exclusion_inventory_sha256"])
    if (config != expected["config"] or manifest["queries"] != expected["queries"]
            or set(manifest) != set(expected) or set(manifest["scenes"]) != set(expected["scenes"])):
        raise ValueError("scene manifest policy/configuration/membership differs")
    images = {}
    for key, blueprint in expected["scenes"].items():
        actual = manifest["scenes"][key]
        if {k: v for k, v in actual.items() if k not in {"image_id", "png_sha256"}} != blueprint:
            raise ValueError("scene metadata differs from deterministic policy")
        png = rt._safe_image_path(path.parent, actual["png_path"])
        if _sha(png) != actual["png_sha256"]:
            raise ValueError("scene PNG checksum differs")
        with Image.open(png) as saved:
            image = saved.convert("RGB")
        if (image.size != (config["img_size"], config["img_size"])
                or image_id(image) != actual["image_id"]
                or image_id(rt.render_scene(blueprint, config["img_size"])) != actual["image_id"]):
            raise ValueError("scene decoded pixels differ from declared objects")
        images[key] = image
    validate_exclusions(manifest, excluded_identifiers)
    return manifest, images


def episode(manifest, seed, condition, rule, k, query_index):
    if condition not in CONDITIONS or rule not in RULES or k not in (8, 16) or seed < 0:
        raise ValueError("unknown condition/rule, invalid seed or K outside 8/16")
    if query_index not in range(manifest["config"]["n"]):
        raise ValueError("query index out of range")
    query = manifest["queries"][query_index]
    scene = manifest["scenes"][query["scene_id"]]
    partition = manifest["config"]["partition"]
    explicit, labels = condition.split("_")
    label_space = list(rt.COLORS if labels == "natural" else REMAP_TOKEN_SETS["eval"][:4])
    if labels == "nonce":
        rt._rng(f"{NAMESPACE}:mapping:{partition}:{seed}:{query_index}").shuffle(label_space)
    order = sample_option_order(4, seed, 0, query_index, "random", phase=f"{NAMESPACE}:{partition}")
    demo_ids = list(query["demo_scene_ids"][:k])
    rt._rng(f"{NAMESPACE}:demo-order:{partition}:{seed}:{query_index}:{k}").shuffle(demo_ids)
    colors = [{"left": manifest["scenes"][key]["left_color"],
               "right": manifest["scenes"][key]["right_color"]} for key in demo_ids]
    side = rule.removesuffix("_color")
    demo_labels = [item[side] for item in colors]
    fits = rt.compatible_rules(colors, demo_labels)
    query_colors = {"left": scene["left_color"], "right": scene["right_color"]}
    if fits != [rule] or any(item == query_colors for item in colors):
        raise ValueError("demonstrations must identify one rule and exclude the query color pair")
    record = {"query_index": query_index, "pair_id": query["pair_id"], "query_id": scene["image_id"],
              "scene_id": query["scene_id"], "query_colors": query_colors,
              "condition": condition, "rule": rule, "k": k,
              "gold": scene[rule], "other_rule_gold": scene[RULES[1 - RULES.index(rule)]],
              "label_space": label_space, "option_order": order,
              "displayed_options": [label_space[index] for index in order],
              "demo_scene_ids": demo_ids,
              "demo_source_ids": [manifest["scenes"][key]["image_id"] for key in demo_ids],
              "demo_colors": colors, "demo_label_classes": demo_labels,
              "compatible_rules": fits, "rule_identifiable": True,
              "instruction_factor": explicit, "label_factor": labels}
    record["prompt_transcript"] = prompt_transcript(record)
    record["prompt_sha256"] = _json_digest(record["prompt_transcript"])
    return record


def prompt_transcript(record):
    if record["instruction_factor"] == "explicit":
        side = record["rule"].removesuffix("_color")
        instruction = (f"Classify the last image by the color of the {side} object, "
                       "using the labels shown by the examples.")
    else:
        instruction = "Classify the last image using the rule and labels shown by the examples."
    content = []
    for key, digest, label in zip(record["demo_scene_ids"], record["demo_source_ids"],
                                  record["demo_label_classes"]):
        content.extend([{"type": "image", "scene_id": key, "image_id": digest},
                        {"type": "text", "text": f"This image shows: {record['label_space'][label]}."}])
    options = ", ".join(record["displayed_options"])
    content.extend([{"type": "image", "scene_id": record["scene_id"], "image_id": record["query_id"]},
                    {"type": "text", "text": f"{instruction} Answer with exactly one of [{options}] and nothing else."}])
    return content


def paired_rule_metrics(left, right):
    if (left["rule"] != "left_color" or right["rule"] != "right_color"
            or any(left[field] != right[field] for field in ("seed", "arm", "k", "condition"))):
        raise ValueError("rule effects require matched left/right cells")
    a, b = left["records"], right["records"]
    if not a or len(a) != len(b):
        raise ValueError("rule pairs must have equal nonempty records")
    for old, new in zip(a, b):
        if any(old[field] != new[field] for field in PAIR_FIELDS) or old["gold"] == new["gold"]:
            raise ValueError("counterfactual rule pair changed images, map, options or required answer")
    return {"n": len(a),
            "both_rules_correct_rate": sum(x["correct"] and y["correct"] for x, y in zip(a, b)) / len(a),
            "semantic_switch_rate": sum(x["prediction"] is not None and y["prediction"] is not None
                                        and x["prediction"] != y["prediction"]
                                        for x, y in zip(a, b)) / len(a),
            "either_unparseable_rate": sum(x["unparseable"] or y["unparseable"]
                                           for x, y in zip(a, b)) / len(a)}


def evaluate_cell(model, processor, manifest, images, seed, arm, condition, rule, k, generate):
    if arm not in ARMS:
        raise ValueError("unknown study arm")
    records = []
    for index in range(manifest["config"]["n"]):
        base = episode(manifest, seed, condition, rule, k, index)
        raw = generate(model, processor, rt.build_messages(base, images))
        records.append(rt.record_output(base, raw))
    return {"seed": seed, "arm": arm, "condition": condition, "rule": rule, "k": k,
            "score_mode": "generation", "summary": rt.summarize_records(records), "records": records}


def validate_cell(cell, manifest):
    if (cell["arm"] not in ARMS or len(cell["records"]) != manifest["config"]["n"]
            or cell.get("score_mode") != "generation"):
        raise ValueError("saved factorial cell has wrong arm, query count or scoring mode")
    for index, record in enumerate(cell["records"]):
        base = episode(manifest, cell["seed"], cell["condition"], cell["rule"], cell["k"], index)
        if record != rt.record_output(base, record["raw_output"]):
            raise ValueError("saved factorial record differs from scene, prompt or raw output")
    if cell["summary"] != rt.summarize_records(cell["records"]):
        raise ValueError("saved factorial summary differs from records")


def cell_key(values):
    return ":".join(map(str, values))


def make_plan(seeds, ks, arms=ARMS):
    if (not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds)
            or not ks or len(set(ks)) != len(ks) or any(k not in (8, 16) for k in ks)
            or tuple(arms) != ARMS):
        raise ValueError("seeds must be unique/nonnegative, K unique 8/16, and all four canonical arms required")
    return [(seed, arm, condition, rule, k) for seed in seeds for arm in arms
            for condition in CONDITIONS for k in ks for rule in RULES]


def derived_pairs(cells, plan):
    paired = {}
    for seed, arm, condition, rule, k in plan:
        if rule != "left_color":
            continue
        left = cell_key((seed, arm, condition, "left_color", k))
        right = cell_key((seed, arm, condition, "right_color", k))
        if left in cells and right in cells:
            paired[f"{seed}:{arm}:{condition}:{k}"] = paired_rule_metrics(cells[left], cells[right])
    return paired


def validate_result(result, manifest, plan):
    planned = {cell_key(values): values for values in plan}
    for key, cell in result["cells"].items():
        if key not in planned or tuple(cell[field] for field in CELL_FIELDS) != planned[key]:
            raise ValueError("saved cell is outside requested factorial grid")
        validate_cell(cell, manifest)
    if derived_pairs(result["cells"], plan) != result["paired_rules"]:
        raise ValueError("saved paired-rule metrics differ from per-query records")


def paired_accuracy_difference(adapter, frozen):
    if any(adapter[field] != frozen[field] for field in ("seed", "condition", "rule", "k")):
        raise ValueError("adapter/frozen cells must be paired")
    a, b = adapter["records"], frozen["records"]
    if not a or len(a) != len(b):
        raise ValueError("adapter/frozen record counts differ")
    for adapted, base in zip(a, b):
        if any(adapted[field] != base[field] for field in (*PAIR_FIELDS, "gold", "prompt_sha256")):
            raise ValueError("adapter/frozen prompts differ")
    return [int(adapted["correct"]) - int(base["correct"]) for adapted, base in zip(a, b)]


def nonce_instruction_interaction(cells, seeds, ks, arm="episode_connector"):
    """Named-minus-implicit difference in adapter-minus-frozen nonce accuracy.

    Seed effects average both selectors and all prespecified K. Query effects
    retain pairing for a conditional image bootstrap performed by the analyzer;
    they are NOT additional independent training replicates.
    """
    from .evaluation import mean_t_ci
    seed_effects, query_effects, query_frame = [], {}, None
    for seed in seeds:
        by_condition = {}
        for condition in ("explicit_nonce", "implicit_nonce"):
            vectors = []
            for k in ks:
                for rule in RULES:
                    a = cells[cell_key((seed, arm, condition, rule, k))]
                    f = cells[cell_key((seed, "frozen", condition, rule, k))]
                    frame = [(row["pair_id"], row["query_id"], row["query_colors"])
                             for row in f["records"]]
                    if query_frame is None:
                        query_frame = frame
                    elif query_frame != frame:
                        raise ValueError("interaction cells use different query frames")
                    vectors.append(paired_accuracy_difference(a, f))
            if len({len(vector) for vector in vectors}) != 1:
                raise ValueError("interaction query frames differ")
            by_condition[condition] = [statistics.fmean(items) for items in zip(*vectors)]
        if len(by_condition["explicit_nonce"]) != len(by_condition["implicit_nonce"]):
            raise ValueError("factor query frames differ")
        values = [a - b for a, b in zip(by_condition["explicit_nonce"], by_condition["implicit_nonce"])]
        query_effects[str(seed)] = values
        seed_effects.append({"seed": seed, "effect": statistics.fmean(values)})
    return {"contrast": "(adapter-frozen)_explicit_nonce - (adapter-frozen)_implicit_nonce",
            "arm": arm, "ks": list(ks), "seed_effects": seed_effects,
            "pair_ids": [item[0] for item in query_frame], "query_effects_by_seed": query_effects,
            "seed_interval": mean_t_ci([item["effect"] for item in seed_effects]),
            "uncertainty_scope": "fitting/prompt seeds conditional on one fixed scene frame"}


def strict_checkpoint_manifest(adapters, seeds, model=MODEL):
    from .option_order_eval import checkpoint_manifest
    from .provenance import model_revision
    if model != MODEL or model_revision(model) != MODEL_REVISION:
        raise ValueError("factorial study requires the pinned source backbone")
    if set(adapters) != set(ARMS[1:]):
        raise ValueError("all three canonical saved adapter arms are required")
    checkpoints = checkpoint_manifest(adapters, seeds)
    for arm in ARMS[1:]:
        expected_region = "all" if arm == "all_layer" else "connector"
        expected_objective = "natural" if arm == "natural_connector" else "episode_remap"
        root = Path(adapters[arm]).resolve()
        for seed in seeds:
            entry = checkpoints[arm][str(seed)]
            if entry["base_model_name_or_path"] != model or entry["training_source"] is None:
                raise ValueError("saved adapter lacks matching base/source training provenance")
            source_path = Path(entry["training_source"]["path"]).resolve()
            source = json.loads(source_path.read_text())
            run = source["run_config"]
            expected = {"model": model, "model_revision": MODEL_REVISION, "task": "shapes4",
                        "regions": expected_region, "components": "both", "steps": 150,
                        "rank": 8, "objective": expected_objective, "format": "single",
                        "dtype": "bfloat16"}
            if any(run.get(key) != value for key, value in expected.items()) or seed not in run["seeds"]:
                raise ValueError(f"saved {arm} training configuration differs from prespecified arm")
            group = source[f"{expected_region}:both"]
            matching = [item for item in group["training_runs"] if item.get("torch_seed") == seed]
            if len(matching) != 1 or seed not in group["seeds_done"]:
                raise ValueError("source training seed is missing, duplicated or incomplete")
            training = matching[0]
            if any(training.get(key) != value for key, value in
                   {"objective": expected_objective, "steps": 150, "rank": 8}.items()):
                raise ValueError("source per-seed training metadata differs")
            # Stored paths were repository-relative; resolve relative to the
            # ancestor containing incontext_collapse instead of the caller cwd.
            project = next((p for p in source_path.parents if (p / "incontext_collapse").is_dir()), None)
            declared = Path(training["adapter_path"])
            if not declared.is_absolute():
                if project is None:
                    raise ValueError("cannot resolve saved training checkpoint provenance")
                declared = project / declared
            if declared.resolve() != (root / f"seed_{seed}").resolve():
                raise ValueError("adapter path does not match its source training record")
            adapter_config = json.loads((root / f"seed_{seed}" / "adapter_config.json").read_text())
            declared_targets = adapter_config.get("target_modules", [])
            resolved_targets = training["targets"]
            # PEFT serializes the all-layer list as seven terminal module names,
            # whereas the training record enumerates all 196 language modules.
            # Match PEFT's exact-or-dot-suffix semantics in both directions.
            target_match = (isinstance(declared_targets, list) and bool(declared_targets)
                            and all(any(target == suffix or target.endswith("." + suffix)
                                        for suffix in declared_targets) for target in resolved_targets)
                            and all(any(target == suffix or target.endswith("." + suffix)
                                        for target in resolved_targets) for suffix in declared_targets))
            if (adapter_config.get("r") != 8 or adapter_config.get("peft_type") != "LORA"
                    or not target_match):
                raise ValueError("checkpoint adapter targets/rank differ from source training metadata")
            entry["source_training_run"] = training
            entry["source_run_config"] = run
    return checkpoints


def source_hashes():
    from . import core, evaluation, models, option_order_eval, protocol, provenance, results, scoring
    modules = (rt, core, evaluation, models, option_order_eval, protocol, provenance, results, scoring)
    return {"rule_factorial.py": _sha(__file__),
            **{Path(module.__file__).name: _sha(module.__file__) for module in modules}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--prior-data-root", default=str(Path(rt.__file__).resolve().parents[1] / "data"))
    parser.add_argument("--prior-exclude-dir", action="append", default=[],
                        help="repeatable; exclude the entire new campaign output directory from the prior snapshot")
    parser.add_argument("--partition", choices=PARTITIONS, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--img-size", type=int, default=112)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--ks", default="8,16")
    parser.add_argument("--adapter", action="append", default=[], metavar="ARM=ROOT")
    parser.add_argument("--max-new-tokens", type=int, default=6)
    parser.add_argument("--protocol-file")
    parser.add_argument("--out")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    # Set before importing transformers/huggingface_hub through checkpoint/model
    # helpers, whose offline constants are initialized at import time.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    seeds, ks = [int(x) for x in args.seeds.split(",")], [int(x) for x in args.ks.split(",")]
    plan = make_plan(seeds, ks)
    expected_grid = (16, [0], [8]) if args.partition == "development" else (80, [0, 1, 2], [8, 16])
    if (args.n, seeds, ks) != expected_grid or args.max_new_tokens != 6 or args.model != MODEL:
        parser.error("fixed study grid: development n16/seeds0/K8; confirmation n80/seeds0,1,2/K8,16; six output tokens")
    directory = Path(args.data_dir).resolve()
    if not args.prepare_only and (not args.out or not args.protocol_file):
        parser.error("evaluation/preflight requires --out and the frozen --protocol-file")
    inventory_path = build_exclusion_inventory(directory / "prior_image_inventory.json",
                                               args.prior_data_root, [directory, *args.prior_exclude_dir])
    inventory = load_exclusion_inventory(inventory_path, prior_root=args.prior_data_root)
    excluded_ids = inventory["excluded_identifiers"]
    inventory_sha = _sha(inventory_path)
    path = build_manifest(directory, args.partition, args.n, args.img_size,
                          excluded_ids, inventory_sha)
    manifest, images = load_manifest(path, excluded_ids)
    development = None
    if args.partition == "confirmatory":
        development_path = directory / "manifest_development.json"
        development_manifest, _ = load_manifest(development_path, excluded_ids)
        if development_manifest["config"]["exclusion_inventory_sha256"] != inventory_sha:
            raise ValueError("development and confirmatory exclusions differ")
        rt.validate_disjoint(development_manifest, manifest)
        development = {"path": str(development_path), "sha256": _sha(development_path)}
    if args.prepare_only:
        print(json.dumps({"manifest": str(path), "sha256": _sha(path), "n": args.n,
                          "prior_inventory": str(inventory_path), "excluded_identifiers": len(excluded_ids),
                          "planned_cells": len(plan), "planned_generations": len(plan) * args.n}), flush=True)
        return
    adapters = {}
    for spec in args.adapter:
        name, separator, value = spec.partition("=")
        if not separator or name not in ARMS[1:] or not value or name in adapters:
            parser.error("adapter arms require one ARM=ROOT for each canonical saved adapter")
        adapters[name] = Path(value).resolve()
    from .provenance import software_versions
    checkpoints = strict_checkpoint_manifest(adapters, seeds, args.model)
    protocol_path = Path(args.protocol_file).resolve()
    config = {"schema_version": SCHEMA_VERSION, "experiment": EXPERIMENT, "namespace": NAMESPACE,
              "model": args.model, "model_revision": MODEL_REVISION, "checkpoints": checkpoints,
              "arms": list(ARMS), "rules": list(RULES), "conditions": list(CONDITIONS),
              "condition_ks": {condition: ks for condition in CONDITIONS}, "seeds": seeds,
              "n": args.n, "img_size": args.img_size, "partition": args.partition,
              "manifest": str(path), "manifest_sha256": _sha(path), "development_manifest": development,
              "exclusion_inventory": str(inventory_path), "exclusion_inventory_sha256": inventory_sha,
              "prior_data_root": str(Path(args.prior_data_root).resolve()),
              "prior_excluded_directories": sorted({str(Path(p).resolve())
                  for p in [directory, *args.prior_exclude_dir]}),
              "protocol_file": str(protocol_path), "protocol_sha256": _sha(protocol_path),
              "max_new_tokens": args.max_new_tokens, "score_mode": "generation", "dtype": "bfloat16",
              "generation": {"do_sample": False, "num_beams": 1},
              "software_versions": software_versions(), "source_sha256": source_hashes(),
              "option_order": "random", "mapping_policy": "per_seed_query_eval_nonce_bijection_shared_across_rules_conditions_K",
              "stopping_policy": "fixed_complete_grid_no_outcome_driven_early_stop",
              "interpretation": "paired attribute-selection classification; fixed scene frame across seeds"}
    result = load_result(args.out, config, {"cells": {}, "paired_rules": {}})
    validate_result(result, manifest, plan)
    remaining = [values for values in plan if cell_key(values) not in result["cells"]]
    print(json.dumps({"planned_cells": len(plan), "remaining_cells": len(remaining),
                      "generations_remaining": len(remaining) * args.n, "out": args.out}), flush=True)
    if args.preflight or not remaining:
        return
    # Only locally cached model files are allowed by this study's execution scope.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from peft import PeftModel
    import torch
    from .core import pick_device, predict
    from .models import load_model
    device = pick_device()
    generate = lambda model, processor, messages: predict(
        model, processor, messages, max_new_tokens=args.max_new_tokens)
    for seed in seeds:
        for arm in ARMS:
            arm_plan = [values for values in remaining if values[:2] == (seed, arm)]
            if not arm_plan:
                continue
            base, processor = load_model(args.model, device=device, dtype=torch.bfloat16)
            model = base.eval() if arm == "frozen" else PeftModel.from_pretrained(
                base, adapters[arm] / f"seed_{seed}").eval()
            for values in arm_plan:
                _, _, condition, rule, k = values
                started = time.monotonic()
                cell = evaluate_cell(model, processor, manifest, images, seed, arm, condition, rule, k, generate)
                validate_cell(cell, manifest)
                result["cells"][cell_key(values)] = cell
                result["paired_rules"] = derived_pairs(result["cells"], plan)
                write_json_atomic(args.out, result)
                print(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "cell": cell_key(values),
                                  "cells_done": len(result["cells"]), "cells_total": len(plan),
                                  "elapsed_seconds": round(time.monotonic() - started, 3)}), flush=True)
            del model, base, processor
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
