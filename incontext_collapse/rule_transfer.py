"""Counterfactual two-object color-rule evaluation of saved adapters.

The two implicit-rule prompts share scenes, nonce mappings, and option order.
Only demonstration answers change. Eight examples identify the selected side;
four balanced examples cannot do so under an unknown nonce mapping. These are
attribute-selection classification tasks, not evidence of general task transfer.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import tempfile

from PIL import Image, ImageDraw

from .protocol import REMAP_TOKEN_SETS, image_id, sample_option_order
from .results import load_result, write_json_atomic
from .scoring import parse_answer


SCHEMA_VERSION = 1
FAMILY = "two_object_color_selection"
COLORS = ("red", "green", "blue", "yellow")
COLOR_RGB = ((220, 45, 45), (25, 150, 70), (40, 80, 215), (230, 185, 25))
RULES = ("left_color", "right_color")
CONDITIONS = ("explicit_natural", "implicit_remap")
PARTITIONS = ("development", "confirmatory")
PAIR_FIELDS = ("query_index", "pair_id", "query_id", "scene_id", "query_colors",
               "label_space", "option_order", "displayed_options", "demo_scene_ids",
               "demo_source_ids", "demo_colors")


def _rng(namespace):
    digest = hashlib.sha256(f"rule-transfer-v1:{namespace}".encode("ascii")).digest()
    return random.Random(int.from_bytes(digest, "big"))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _scene(scene_id, partition, role, left, right, size):
    rng = _rng(f"layout:{scene_id}")
    objects = []
    for side, color, center in (("left", left, 0.27), ("right", right, 0.73)):
        objects.append({"side": side, "color": color,
                        "center": [round(size * (center + rng.uniform(-0.035, 0.035))),
                                   round(size * rng.uniform(0.42, 0.58))],
                        "radius": round(size * rng.uniform(0.13, 0.17))})
    return {"scene_id": scene_id, "partition": partition, "role": role,
            "left_color": left, "right_color": right,
            "background": [rng.randint(240, 250) for _ in range(3)], "objects": objects,
            "png_path": f"images/{partition}/{hashlib.sha256(scene_id.encode()).hexdigest()}.png"}


def render_scene(scene, size):
    image = Image.new("RGB", (size, size), tuple(scene["background"]))
    draw = ImageDraw.Draw(image)
    for obj in scene["objects"]:
        x, y = obj["center"]
        radius = obj["radius"]
        draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                     fill=COLOR_RGB[obj["color"]])
    return image


def _blueprint(partition, n, size):
    if partition not in PARTITIONS or n < 4 or n % 4 or size < 64:
        raise ValueError("partition must be development/confirmatory, n a positive multiple of four, size>=64")
    config = {
        "schema_version": SCHEMA_VERSION, "family": FAMILY, "partition": partition,
        "n": n, "img_size": size, "renderer": "pil_solid_circles_v1",
        "colors": list(COLORS), "color_rgb": [list(value) for value in COLOR_RGB],
        "training_used": False,
        "scene_policy": "query_shift_reserved_two_other_shifts_repeated_v1",
    }
    scenes, queries = {}, []
    for index in range(n):
        left = index % 4
        reserved_shift = 1 + (index // 4) % 3
        right = (left + reserved_shift) % 4
        query_id = f"{partition}/query/{index:04d}"
        scenes[query_id] = _scene(query_id, partition, "query", left, right, size)
        allowed = [shift for shift in (1, 2, 3) if shift != reserved_shift]
        demo_ids = []
        for block, shift in enumerate(allowed * 2):
            for color in range(4):
                demo_id = f"{partition}/demo/{index:04d}/{block * 4 + color:02d}"
                scenes[demo_id] = _scene(demo_id, partition, "demonstration", color,
                                         (color + shift) % 4, size)
                demo_ids.append(demo_id)
        queries.append({"pair_id": f"{partition}:query:{index:04d}",
                        "scene_id": query_id, "demo_scene_ids": demo_ids})
    return {"config": config, "queries": queries, "scenes": scenes}


def _safe_image_path(directory, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("scene image path escapes the manifest directory")
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("scene image symlink escapes the manifest directory")
    return path


def _write_png(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".png", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        image.save(temporary, format="PNG")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_manifest(directory, partition="confirmatory", n=40, size=112):
    """Persist deterministic scenes atomically; never silently replace a manifest."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"manifest_{partition}.json"
    expected = _blueprint(partition, n, size)
    if path.exists():
        manifest, _ = load_manifest(path)
        if manifest["config"] != expected["config"]:
            raise ValueError("existing scene manifest configuration differs")
        return path
    for scene in expected["scenes"].values():
        image = render_scene(scene, size)
        target = _safe_image_path(directory, scene["png_path"])
        if target.exists():
            with Image.open(target) as saved:
                if image_id(saved.convert("RGB")) != image_id(image):
                    raise ValueError("existing image differs from deterministic scene")
        else:
            _write_png(target, image)
        scene["image_id"] = image_id(image)
        scene["png_sha256"] = _sha(target)
    write_json_atomic(path, expected)
    return path


def load_manifest(path):
    """Validate scene semantics, encoded bytes, and decoded pixels before inference."""
    path = Path(path)
    manifest = json.loads(path.read_text())
    config = manifest["config"]
    expected = _blueprint(config["partition"], config["n"], config["img_size"])
    if config != expected["config"] or manifest["queries"] != expected["queries"]:
        raise ValueError("scene manifest policy/configuration differs")
    if set(manifest) != set(expected) or set(manifest["scenes"]) != set(expected["scenes"]):
        raise ValueError("scene manifest membership differs")
    images = {}
    for key, blueprint in expected["scenes"].items():
        actual = manifest["scenes"][key]
        semantic = {field: value for field, value in actual.items()
                    if field not in {"image_id", "png_sha256"}}
        if semantic != blueprint:
            raise ValueError("scene object metadata differs from deterministic policy")
        png = _safe_image_path(path.parent, actual["png_path"])
        if _sha(png) != actual["png_sha256"]:
            raise ValueError("scene PNG checksum differs")
        with Image.open(png) as saved:
            image = saved.convert("RGB")
        if (image.size != (config["img_size"], config["img_size"])
                or image_id(image) != actual["image_id"]
                or image_id(render_scene(blueprint, config["img_size"])) != actual["image_id"]):
            raise ValueError("scene decoded pixels differ from declared objects")
        images[key] = image
    ids = [scene["image_id"] for scene in manifest["scenes"].values()]
    if len(ids) != len(set(ids)):
        raise ValueError("scene pixels are duplicated")
    return manifest, images


def validate_disjoint(first, second):
    if first["config"]["partition"] == second["config"]["partition"]:
        raise ValueError("development and confirmatory partitions must differ")
    first_ids = {scene["image_id"] for scene in first["scenes"].values()}
    second_ids = {scene["image_id"] for scene in second["scenes"].values()}
    if first_ids & second_ids or set(first["scenes"]) & set(second["scenes"]):
        raise ValueError("development and confirmatory scenes overlap")


def compatible_rules(demo_colors, demo_labels):
    """Enumerate every side/bijection explanation, independently of the intended map."""
    if len(demo_colors) != len(demo_labels):
        raise ValueError("demonstration colors/labels length mismatch")
    if any(label not in range(4) for label in demo_labels):
        raise ValueError("demonstration labels must be four-class indices")
    compatible = []
    for rule in RULES:
        side = rule.removesuffix("_color")
        if any(all(mapping[colors[side]] == label
                   for colors, label in zip(demo_colors, demo_labels))
               for mapping in itertools.permutations(range(4))):
            compatible.append(rule)
    return compatible


def episode(manifest, seed, condition, rule, k, query_index):
    if condition not in CONDITIONS or rule not in RULES:
        raise ValueError("unknown condition/rule")
    if k not in ({0} if condition == "explicit_natural" else {0, 8, 16}):
        raise ValueError("explicit rules use K0; implicit rules use K0/8/16")
    query = manifest["queries"][query_index]
    scene = manifest["scenes"][query["scene_id"]]
    partition = manifest["config"]["partition"]
    label_space = list(COLORS if condition == "explicit_natural" else REMAP_TOKEN_SETS["eval"][:4])
    if condition == "implicit_remap":
        _rng(f"mapping:{partition}:{seed}:{query_index}").shuffle(label_space)
    order = sample_option_order(4, seed, 0, query_index, "random",
                                phase=f"rule-transfer-v1:{partition}")
    demo_ids = query["demo_scene_ids"][:k]
    _rng(f"demo-order:{partition}:{seed}:{query_index}:{k}").shuffle(demo_ids)
    colors = [{"left": manifest["scenes"][key]["left_color"],
               "right": manifest["scenes"][key]["right_color"]} for key in demo_ids]
    side = rule.removesuffix("_color")
    demo_labels = [item[side] for item in colors]
    fits = compatible_rules(colors, demo_labels)
    if k and fits != [rule]:
        raise ValueError("implicit demonstration set does not uniquely identify the rule")
    query_colors = {"left": scene["left_color"], "right": scene["right_color"]}
    if any(item == query_colors for item in colors):
        raise ValueError("query color combination appears among demonstrations")
    record = {
        "query_index": query_index, "pair_id": query["pair_id"], "query_id": scene["image_id"],
        "scene_id": query["scene_id"], "query_colors": query_colors,
        "condition": condition, "rule": rule, "k": k,
        "gold": scene[rule], "other_rule_gold": scene[RULES[1 - RULES.index(rule)]],
        "label_space": label_space, "option_order": order,
        "displayed_options": [label_space[index] for index in order],
        "demo_scene_ids": demo_ids,
        "demo_source_ids": [manifest["scenes"][key]["image_id"] for key in demo_ids],
        "demo_colors": colors, "demo_label_classes": demo_labels,
        "compatible_rules": fits,
        "rule_identifiable": condition == "explicit_natural" or fits == [rule],
    }
    record["prompt_transcript"] = prompt_transcript(record)
    return record


def prompt_transcript(record):
    options = ", ".join(record["displayed_options"])
    if record["condition"] == "explicit_natural":
        side = record["rule"].removesuffix("_color")
        instruction = f"What color is the {side} object?"
    else:
        instruction = "Classify the last image using the rule shown by the examples."
    content = []
    for key, digest, label in zip(record["demo_scene_ids"], record["demo_source_ids"],
                                  record["demo_label_classes"]):
        content += [{"type": "image", "scene_id": key, "image_id": digest},
                    {"type": "text", "text": f"This image shows: {record['label_space'][label]}."}]
    content += [{"type": "image", "scene_id": record["scene_id"], "image_id": record["query_id"]},
                {"type": "text", "text": f"{instruction} Answer with exactly one of [{options}] and nothing else."}]
    return content


def build_messages(record, images):
    content = [{"type": "image", "image": images[block["scene_id"]]}
               if block["type"] == "image" else dict(block)
               for block in record["prompt_transcript"]]
    return [{"role": "user", "content": content}]


def record_output(base, raw):
    if not isinstance(raw, str):
        raise ValueError("generation output must be a string")
    parsed = parse_answer(raw, base["label_space"])
    prediction = parsed.prediction
    return {**base, "raw_output": raw, "prediction": prediction,
            "generation_prediction": prediction, "parse_status": parsed.status,
            "parse_matches": list(parsed.matched_classes), "unparseable": prediction is None,
            "correct": prediction == base["gold"],
            "answers_other_rule": prediction is not None and prediction == base["other_rule_gold"]}


def summarize_records(records):
    n = len(records)
    if not n:
        raise ValueError("a rule cell must contain records")
    return {"n": n, "accuracy": sum(row["correct"] for row in records) / n,
            "unparseable_rate": sum(row["unparseable"] for row in records) / n,
            "answers_other_rule_rate": sum(row["answers_other_rule"] for row in records) / n,
            "parse_status_counts": dict(sorted(Counter(row["parse_status"] for row in records).items()))}


def paired_rule_metrics(left, right):
    if (left["condition"] != "implicit_remap" or right["condition"] != "implicit_remap"
            or left["rule"] != "left_color" or right["rule"] != "right_color"
            or any(left[field] != right[field] for field in ("seed", "arm", "k"))):
        raise ValueError("rule effects require matched implicit left/right cells")
    a, b = left["records"], right["records"]
    if not a or len(a) != len(b):
        raise ValueError("rule pairs must have equal nonempty records")
    for old, new in zip(a, b):
        if any(old[field] != new[field] for field in PAIR_FIELDS):
            raise ValueError("counterfactual rule pair changed images, map, or options")
        if old["gold"] == new["gold"]:
            raise ValueError("counterfactual query must require a changed answer")
        if left["k"] == 0 and old["raw_output"] != new["raw_output"]:
            raise ValueError("identical K0 prompts generated different outputs")
    return {"n": len(a),
            "both_rules_correct_rate": sum(x["correct"] and y["correct"] for x, y in zip(a, b)) / len(a),
            "semantic_switch_rate": sum(x["prediction"] is not None and y["prediction"] is not None
                                        and x["prediction"] != y["prediction"]
                                        for x, y in zip(a, b)) / len(a),
            "either_unparseable_rate": sum(x["unparseable"] or y["unparseable"]
                                           for x, y in zip(a, b)) / len(a)}


def evaluate_cell(model, processor, manifest, images, seed, arm, condition, rule, k, generate):
    records = []
    for index in range(manifest["config"]["n"]):
        base = episode(manifest, seed, condition, rule, k, index)
        raw = generate(model, processor, build_messages(base, images))
        records.append(record_output(base, raw))
    return {"seed": seed, "arm": arm, "condition": condition, "rule": rule, "k": k,
            "score_mode": "generation", "summary": summarize_records(records), "records": records}


def validate_cell(cell, manifest):
    if len(cell["records"]) != manifest["config"]["n"] or cell.get("score_mode") != "generation":
        raise ValueError("saved rule cell has wrong query count/scoring mode")
    for index, record in enumerate(cell["records"]):
        base = episode(manifest, cell["seed"], cell["condition"], cell["rule"], cell["k"], index)
        if record != record_output(base, record["raw_output"]):
            raise ValueError("saved rule record differs from its scene or raw output")
    if cell["summary"] != summarize_records(cell["records"]):
        raise ValueError("saved rule cell summary differs from records")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--partition", choices=PARTITIONS, required=True)
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--img-size", type=int, default=112)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--ks", default="0,8,16", help="implicit-rule K values; explicit rules use K0 only")
    parser.add_argument("--adapter", action="append", default=[], metavar="NAME=DIRECTORY")
    parser.add_argument("--max-new-tokens", type=int, default=6)
    parser.add_argument("--out")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    seeds, ks = [int(x) for x in args.seeds.split(",")], [int(x) for x in args.ks.split(",")]
    if (not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds)
            or not ks or len(set(ks)) != len(ks) or any(k not in {0, 8, 16} for k in ks)
            or args.max_new_tokens < 1):
        parser.error("seeds must be unique nonnegative integers; K must be unique values from0,8,16")
    path = build_manifest(args.data_dir, args.partition, args.n, args.img_size)
    manifest, images = load_manifest(path)
    if args.prepare_only:
        print(json.dumps({"manifest": str(path), "sha256": _sha(path), "n": args.n}), flush=True)
        return
    if not args.out:
        parser.error("evaluation/preflight requires --out")
    development = None
    if args.partition == "confirmatory":
        development_path = path.with_name("manifest_development.json")
        development_manifest, _ = load_manifest(development_path)
        validate_disjoint(development_manifest, manifest)
        development = {"path": str(development_path), "sha256": _sha(development_path)}
    adapters = {}
    for spec in args.adapter:
        name, separator, value = spec.partition("=")
        if not separator or not name or not value or name == "frozen" or ":" in name or name in adapters:
            parser.error("adapter arms require unique NAME=DIRECTORY values other than frozen")
        adapters[name] = Path(value)
    from .option_order_eval import checkpoint_manifest
    from .provenance import model_revision, software_versions
    checkpoints = checkpoint_manifest(adapters, seeds)
    for arm in checkpoints.values():
        for checkpoint in arm.values():
            if checkpoint["base_model_name_or_path"] != args.model:
                raise ValueError("adapter base model differs from requested model")
            if checkpoint["training_source"] is None:
                raise ValueError("saved adapter lacks source training provenance")
    arms = ["frozen", *adapters]
    condition_ks = {"explicit_natural": [0], "implicit_remap": ks}
    config = {
        "schema_version": SCHEMA_VERSION, "experiment": "two_object_rule_transfer", "family": FAMILY,
        "model": args.model, "model_revision": model_revision(args.model),
        "checkpoints": checkpoints, "arms": arms, "rules": list(RULES), "conditions": list(CONDITIONS),
        "condition_ks": condition_ks, "seeds": seeds, "n": args.n, "img_size": args.img_size,
        "partition": args.partition, "manifest": str(path), "manifest_sha256": _sha(path),
        "development_manifest": development, "max_new_tokens": args.max_new_tokens,
        "score_mode": "generation", "dtype": "bfloat16", "software_versions": software_versions(),
        "source_sha256": _sha(Path(__file__)), "option_order": "random",
        "protocol": {"protocol_version": "2.5-rule-transfer-v1", "option_order": "random",
                     "mapping_mode": "episode", "remap_token_set": "eval"},
        "mapping_policy": "per_seed_query_eval_nonce_bijection_shared_across_rules_and_K",
        "identifiable_implicit_ks": [8, 16],
        "interpretation": "counterfactual attribute-selection classification; fixed query frame across seeds",
    }
    result = load_result(args.out, config, {"cells": {}, "paired_rules": {}})
    plan = [(seed, arm, condition, rule, k) for seed in seeds for arm in arms
            for condition in CONDITIONS for k in condition_ks[condition] for rule in RULES]
    key = lambda values: ":".join(map(str, values))
    planned = {key(values): values for values in plan}
    for cell_key, cell in result["cells"].items():
        if cell_key not in planned or tuple(cell[field] for field in ("seed", "arm", "condition", "rule", "k")) != planned[cell_key]:
            raise ValueError("saved cell is outside requested grid")
        validate_cell(cell, manifest)
    paired = {}
    for seed in seeds:
        for arm in arms:
            for k in ks:
                left_key = key((seed, arm, "implicit_remap", "left_color", k))
                right_key = key((seed, arm, "implicit_remap", "right_color", k))
                if left_key in result["cells"] and right_key in result["cells"]:
                    paired[f"{seed}:{arm}:implicit_remap:{k}"] = paired_rule_metrics(
                        result["cells"][left_key], result["cells"][right_key])
    if paired != result["paired_rules"]:
        raise ValueError("saved paired-rule metrics differ from per-query records")
    remaining = [values for values in plan if key(values) not in result["cells"]]
    print(json.dumps({"planned_cells": len(plan), "remaining_cells": len(remaining),
                      "generations_remaining": len(remaining) * args.n, "out": args.out}), flush=True)
    if args.preflight or not remaining:
        return
    from peft import PeftModel
    import torch
    from .core import pick_device, predict
    from .models import load_model
    device = pick_device()
    generate = lambda model, processor, messages: predict(
        model, processor, messages, max_new_tokens=args.max_new_tokens)
    for seed in seeds:
        for arm in arms:
            arm_plan = [values for values in remaining if values[:2] == (seed, arm)]
            if not arm_plan:
                continue
            base, processor = load_model(args.model, device=device, dtype=torch.bfloat16)
            model = base.eval() if arm == "frozen" else PeftModel.from_pretrained(
                base, adapters[arm] / f"seed_{seed}").eval()
            for values in arm_plan:
                _, _, condition, rule, k = values
                cell = evaluate_cell(model, processor, manifest, images, seed, arm, condition, rule, k, generate)
                result["cells"][key(values)] = cell
                if condition == "implicit_remap":
                    left_key = key((seed, arm, condition, "left_color", k))
                    right_key = key((seed, arm, condition, "right_color", k))
                    if left_key in result["cells"] and right_key in result["cells"]:
                        result["paired_rules"][f"{seed}:{arm}:{condition}:{k}"] = paired_rule_metrics(
                            result["cells"][left_key], result["cells"][right_key])
                write_json_atomic(args.out, result)
                print(f"[{datetime.now(timezone.utc).isoformat()}] cells {len(result['cells'])}/{len(plan)} {key(values)} done", flush=True)
            del model, base, processor
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
