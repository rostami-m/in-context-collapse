#!/usr/bin/env python3
"""Independently audit revision JSON artifacts from their raw records.

This checker intentionally does not import the experiment evaluator.  It
recomputes simple metrics and protocol invariants from JSON so a shared bug in
the evaluator cannot make both generation and validation pass.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import re
from typing import Any


TOLERANCE = 1e-9
VQA_ARTICLES = {"a", "an", "the"}
VQA_NUMBERS = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
    "nine": "9", "ten": "10",
}
VQA_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
    "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
    "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't",
    "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've",
    "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
    "id've": "i'd've", "i'dve": "i'd've", "im": "i'm", "ive": "i've",
    "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've",
    "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
    "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've",
    "oclock": "o'clock", "oughtnt": "oughtn't", "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't",
    "shed've": "she'd've", "she'dve": "she'd've", "she's": "she's",
    "shouldve": "should've", "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd", "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've", "somebodyll": "somebody'll",
    "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've",
    "someone'dve": "someone'd've", "someonell": "someone'll", "someones": "someone's",
    "somethingd": "something'd", "somethingd've": "something'd've",
    "something'dve": "something'd've", "somethingll": "something'll",
    "thats": "that's", "thered": "there'd", "thered've": "there'd've",
    "there'dve": "there'd've", "therere": "there're", "theres": "there's",
    "theyd": "they'd", "theyd've": "they'd've", "they'dve": "they'd've",
    "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas",
    "wasnt": "wasn't", "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've",
    "werent": "weren't", "whatll": "what'll", "whatre": "what're", "whats": "what's",
    "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's",
    "whereve": "where've", "whod": "who'd", "whod've": "who'd've",
    "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've",
    "whyll": "why'll", "whyre": "why're", "whys": "why's", "wont": "won't",
    "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll",
    "y'allll": "y'all'll", "yall'd've": "y'all'd've", "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've",
    "you'dve": "you'd've", "youll": "you'll", "youre": "you're", "youve": "you've",
}
VQA_PUNCTUATION = (';', '/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\',
                   '_', '-', '>', '<', '@', '`', ',', '?', '!')
VQA_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
VQA_COMMA_BETWEEN_DIGITS = re.compile(r"\d,\d")


def close(left, right) -> bool:
    return left is not None and right is not None and abs(float(left) - float(right)) <= TOLERANCE


def iter_record_cells(value: Any, location: str = "root"):
    if isinstance(value, dict):
        if isinstance(value.get("records"), list) and isinstance(value.get("summary"), dict):
            yield location, value
            return
        for key, child in value.items():
            yield from iter_record_cells(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from iter_record_cells(child, f"{location}[{index}]")


def classification_summary(records: list[dict]) -> dict:
    parsed = [record for record in records if "correct" in record]
    if len(parsed) != len(records):
        return {}
    return {
        "n": len(records),
        "accuracy": sum(bool(record["correct"]) for record in records) / len(records),
        "unparseable_rate": sum(bool(record.get("unparseable")) for record in records) / len(records),
        "copy_first_rate": sum(bool(record.get("copies_first")) for record in records) / len(records),
        "copy_last_rate": sum(bool(record.get("copies_last")) for record in records) / len(records),
        "copy_any_demo_rate": sum(bool(record.get("copies_any_demo")) for record in records) / len(records),
        "copy_majority_rate": sum(bool(record.get("copies_majority")) for record in records) / len(records),
    }


def parse_generation(answer: str, labels: list[str]):
    normalized = " ".join(str(answer).casefold().strip().split())
    hits = [
        index for index, label in enumerate(labels)
        if re.search(r"(?<!\w)" + re.escape(str(label).casefold()) + r"(?!\w)", normalized)
    ]
    if not hits:
        return None, "no_label"
    if len(hits) > 1:
        return None, "option_echo" if len(hits) == len(labels) else "ambiguous_labels"
    stripped = normalized.strip(" \t\n\r.,;:!?\"'`()[]{}")
    status = "exact" if stripped == str(labels[hits[0]]).casefold() else "single_label_verbose"
    return hits[0], status


def normalize_vqa_answer(answer: str) -> str:
    """Independent implementation of the declared official-style VQA normalization."""
    text = str(answer).lower().replace("\n", " ").replace("\t", " ").strip()
    output = text
    for punctuation in VQA_PUNCTUATION:
        if (f"{punctuation} " in text or f" {punctuation}" in text
                or VQA_COMMA_BETWEEN_DIGITS.search(text)):
            output = output.replace(punctuation, "")
        else:
            output = output.replace(punctuation, " ")
    output = VQA_PERIOD_STRIP.sub("", output)
    words = []
    for word in output.split():
        word = VQA_NUMBERS.get(word, word)
        if word not in VQA_ARTICLES:
            words.append(VQA_CONTRACTIONS.get(word, word))
    return " ".join(words)


def official_vqa_score(prediction: str, answers: list[str]) -> float:
    predicted = normalize_vqa_answer(prediction)
    normalized = [normalize_vqa_answer(answer) for answer in answers]
    if len(normalized) <= 1:
        return float(bool(normalized) and predicted == normalized[0])
    agreement = []
    for index in range(len(normalized)):
        matches = sum(
            predicted == answer
            for other, answer in enumerate(normalized)
            if other != index
        )
        agreement.append(min(1.0, matches / 3.0))
    return sum(agreement) / len(agreement)


def audit_cell(location: str, cell: dict, errors: list[str], warnings: list[str]) -> None:
    records = cell["records"]
    if not records:
        errors.append(f"{location}: empty records")
        return
    indices = [record.get("query_index") for record in records]
    if len(set(indices)) != len(indices):
        errors.append(f"{location}: duplicate query_index values")
    if any("query_id" not in record for record in records):
        warnings.append(f"{location}: no stable query_id; examples cannot be audited from JSON alone")

    expected = classification_summary(records)
    for metric, value in expected.items():
        if metric not in cell["summary"]:
            continue
        reported = cell["summary"].get(metric)
        if not close(value, reported):
            errors.append(f"{location}: {metric} reported={reported!r}, recomputed={value!r}")

    if expected:
        counts = Counter(record["gold"] for record in records)
        if counts and max(counts.values()) - min(counts.values()) > 1:
            warnings.append(f"{location}: unbalanced query counts {dict(sorted(counts.items()))}")
        for index, record in enumerate(records):
            prediction = record.get("prediction")
            if bool(record["correct"]) != (prediction == record["gold"]):
                errors.append(f"{location}.records[{index}]: inconsistent correct flag")
            label_space = record.get("label_space")
            if label_space is not None and len(label_space) != len(set(label_space)):
                errors.append(f"{location}.records[{index}]: duplicate labels in label_space")
            option_order = record.get("option_order")
            protocol = cell.get("protocol", {})
            if protocol.get("protocol_version") == "2.5-options":
                if protocol.get("option_order") != "random" or option_order is None:
                    errors.append(f"{location}.records[{index}]: missing randomized-option provenance")
            if option_order is not None and label_space is not None:
                if (not isinstance(option_order, list)
                        or sorted(option_order) != list(range(len(label_space)))):
                    errors.append(f"{location}.records[{index}]: option order is not a permutation")
                elif record.get("displayed_options") != [label_space[j] for j in option_order]:
                    errors.append(f"{location}.records[{index}]: displayed options disagree with mapping")
                elif protocol.get("option_order") == "random":
                    key = (f"option-order-v1:evaluation:{cell['seed']}:"
                           f"{record['k']}:{record['query_index']}")
                    rng_seed = int.from_bytes(hashlib.sha256(key.encode("ascii")).digest(), "big")
                    expected_order = list(range(len(label_space)))
                    random.Random(rng_seed).shuffle(expected_order)
                    if option_order != expected_order:
                        errors.append(f"{location}.records[{index}]: option RNG provenance mismatch")
            if record.get("query_id") in record.get("demo_source_ids", []):
                errors.append(f"{location}.records[{index}]: query image reused as demonstration")
            if record.get("raw_output") is not None and label_space is not None:
                parsed, status = parse_generation(record["raw_output"], label_space)
                if parsed != record.get("generation_prediction"):
                    errors.append(
                        f"{location}.records[{index}]: generation prediction does not match raw output")
                if status != record.get("parse_status"):
                    errors.append(
                        f"{location}.records[{index}]: parse status does not match raw output")
                if cell.get("score_mode") in {"generation", "both"}:
                    if prediction != parsed:
                        errors.append(f"{location}.records[{index}]: scored prediction differs from raw generation")
                    if record.get("unparseable") is not (parsed is None):
                        errors.append(f"{location}.records[{index}]: unparseable flag differs from raw generation")
            if "demo_label_classes" in record:
                shown = record["demo_label_classes"]
                counts = Counter(shown)
                majority = {label for label, count in counts.items() if count == max(counts.values())} if counts else set()
                flags = {"copies_first": bool(shown) and prediction == shown[0],
                         "copies_last": bool(shown) and prediction == shown[-1],
                         "copies_any_demo": prediction is not None and prediction in shown,
                         "copies_majority": prediction is not None and prediction in majority}
                for field, expected_flag in flags.items():
                    if field in record and record[field] is not expected_flag:
                        errors.append(f"{location}.records[{index}]: {field} differs from shown labels")
            k = record.get("k")
            demos = record.get("demo_source_classes")
            if k is not None and demos is not None and len(demos) != k:
                errors.append(f"{location}.records[{index}]: K={k}, demos={len(demos)}")
            mapping = record.get("shuffle_map")
            if mapping:
                pairs = ((int(source), int(target)) for source, target in mapping.items())
                if any(source == target for source, target in pairs):
                    errors.append(f"{location}.records[{index}]: shuffle map is not a derangement")

    is_vqa = all("score" in record and "answers" in record and "prediction" in record
                 for record in records)
    if is_vqa:
        recomputed_scores = []
        for index, record in enumerate(records):
            query_image = record.get("query_id", "").split(":")[0]
            demo_images = [key.split(":")[0] for key in record.get("demo_ids", [])]
            if query_image and query_image in demo_images:
                errors.append(f"{location}.records[{index}]: VQA query image reused as demonstration")
            score = official_vqa_score(record["prediction"], record["answers"])
            recomputed_scores.append(score)
            if not close(score, record["score"]):
                errors.append(
                    f"{location}.records[{index}]: VQA score reported={record['score']!r}, "
                    f"official={score!r}")
        accuracy = sum(recomputed_scores) / len(recomputed_scores)
        if "accuracy" in cell["summary"] and not close(accuracy, cell["summary"]["accuracy"]):
            errors.append(
                f"{location}: VQA accuracy reported={cell['summary']['accuracy']!r}, "
                f"official={accuracy!r}")


def iter_training_runs(value: Any, location: str = "root"):
    if isinstance(value, dict):
        runs = value.get("training_runs")
        if isinstance(runs, list):
            for index, run in enumerate(runs):
                yield f"{location}.training_runs[{index}]", run
        elif isinstance(runs, dict):
            for key, run in runs.items():
                yield f"{location}.training_runs.{key}", run
        for key, child in value.items():
            yield from iter_training_runs(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from iter_training_runs(child, f"{location}[{index}]")


def audit_attention_grid(payload, config, errors):
    """Check named cells and episode pairing independently of total record counts."""
    entries = []
    if config["method"] == "CAMA independent implementation":
        arms = ("frozen", "cama")
        for index, cell in enumerate(payload.get("cells", [])):
            entries.append((f"root.cells[{index}]", cell.get("seed"),
                            cell.get("condition"), cell.get("k"), cell.get("arm"), cell))
    else:
        arms = ("frozen", "dara")
        for container, arm in (("baseline_evidence", "frozen"), ("evidence", "dara")):
            for index, block in enumerate(payload.get(container, [])):
                for condition in ("standard", "remap"):
                    shots = block.get(condition, {})
                    if set(shots) != {str(k) for k in config["ks"]}:
                        errors.append(f"root.{container}[{index}].{condition}: incorrect DARA K grid")
                    for k, cell in shots.items():
                        location = f"root.{container}[{index}].{condition}.{k}"
                        try:
                            shot_count = int(k)
                        except ValueError:
                            errors.append(f"{location}: invalid DARA K key")
                            continue
                        entries.append((location, block.get("seed"), condition,
                                        shot_count, arm, cell))
    expected = set(itertools.product(config["seeds"], ("standard", "remap"), config["ks"], arms))
    observed = [(seed, condition, k, arm) for _, seed, condition, k, arm, _ in entries]
    if len(observed) != len(set(observed)):
        errors.append("root: duplicate attention-baseline cell identities")
    if set(observed) != expected:
        errors.append("root: incorrect attention-baseline seed/condition/K/arm grid")
    paired = {}
    for location, seed, condition, k, arm, cell in entries:
        if (cell.get("seed") != seed or cell.get("k") != k
                or cell.get("condition") != condition
                or len(cell["records"]) != config["n"]):
            errors.append(f"{location}: attention cell metadata disagrees with grid")
        if config.get("protocol") is not None and cell.get("protocol") != config["protocol"]:
            errors.append(f"{location}: attention cell protocol differs from run configuration")
        identities = [{field: record.get(field) for field in (
            "query_index", "query_id", "gold", "label_space", "demo_source_classes",
            "demo_source_ids", "demo_label_classes", "shuffle_map",
            "option_order", "displayed_options")} for record in cell["records"]]
        group = (seed, condition, k)
        if group in paired and paired[group] != identities:
            errors.append(f"{location}: attention arm episodes are not paired")
        paired[group] = identities


def audit_calibration_cell(location, cell, errors):
    raw_correct, calibrated_correct, follows_mapping = [], [], []
    for index, record in enumerate(cell["records"]):
        entry = f"{location}.records[{index}]"
        n_labels = len(record.get("label_space", []))
        query = record.get("query_mean_scores")
        neutral = record.get("content_free_mean_scores")
        calibrated = record.get("calibrated_scores")
        if not n_labels or any(not isinstance(values, list) or len(values) != n_labels
                               for values in (query, neutral, calibrated)):
            errors.append(f"{entry}: incomplete calibration score vectors")
            continue
        if not all(isinstance(value, (int, float)) and math.isfinite(value)
                   for values in (query, neutral, calibrated) for value in values):
            errors.append(f"{entry}: non-finite or invalid calibration score")
            continue
        expected = [real - bias for real, bias in zip(query, neutral)]
        if not all(close(actual, wanted) for actual, wanted in zip(calibrated, expected)):
            errors.append(f"{entry}: calibration scores do not equal query minus content-free scores")
        raw_prediction = max(range(n_labels), key=query.__getitem__)
        prediction = max(range(n_labels), key=expected.__getitem__)
        if record.get("raw_prediction") != raw_prediction or record.get("prediction") != prediction:
            errors.append(f"{entry}: calibration prediction differs from score argmax")
        raw_correct.append(raw_prediction == record["gold"])
        calibrated_correct.append(prediction == record["gold"])
        mapping = record.get("shuffle_map")
        follows_mapping.append(mapping is not None and prediction == mapping.get(str(record["gold"])))
        if (record.get("raw_correct") != raw_correct[-1]
                or record.get("correct") != calibrated_correct[-1]):
            errors.append(f"{entry}: calibration correctness flag differs from score argmax")
    if len(raw_correct) == len(cell["records"]) and raw_correct:
        for metric, outcomes in (("raw_accuracy", raw_correct),
                                 ("calibrated_accuracy", calibrated_correct),
                                 ("follows_corrupted_mapping_rate", follows_mapping)):
            value = sum(outcomes) / len(outcomes)
            if not close(cell["summary"].get(metric), value):
                errors.append(f"{location}: {metric} disagrees with independent calibration scores")


def audit_file(path: Path) -> dict:
    payload = json.loads(path.read_text())
    errors: list[str] = []
    warnings: list[str] = []
    if payload.get("invalidated"):
        errors.append(f"root: invalidated experiment: {payload['invalidated']}")
    config = payload.get("run_config", {})
    if (isinstance(config, dict) and config.get("method") == "DARA compatibility port"
            and config.get("implementation_version", 1) < 2):
        errors.append("root: DARA implementation predates the causal-mask correction")
    cells = 0
    records = 0
    for location, cell in iter_record_cells(payload):
        cells += 1
        records += len(cell["records"])
        audit_cell(location, cell, errors, warnings)

    if config.get("method") == "image-disjoint VQA checkpoint evaluation":
        expected_cells = len(config["seeds"]) * len(config["ks"]) * len(config["arms"])
        if cells != expected_cells or records != expected_cells * config["n"]:
            errors.append("root: incomplete image-disjoint VQA grid")
        excluded = set(payload.get("diversity", {}).get("excluded_image_ids", []))
        if not excluded:
            errors.append("root: missing VQA excluded-image provenance")
        evidence = payload.get("evidence", {})
        if set(evidence) != {str(seed) for seed in config["seeds"]}:
            errors.append("root: incorrect VQA seed grid")
        reference_queries = None
        for seed, arms in evidence.items():
            if set(arms) != set(config["arms"]):
                errors.append(f"root.evidence.{seed}: incorrect VQA arm grid")
            paired_episodes = {}
            for arm, shots in arms.items():
                if set(shots) != {str(k) for k in config["ks"]}:
                    errors.append(f"root.evidence.{seed}.{arm}: incorrect VQA K grid")
                for k, cell in shots.items():
                    location = f"root.evidence.{seed}.{arm}.{k}"
                    if (cell.get("seed") != int(seed) or cell.get("k") != int(k)
                            or len(cell["records"]) != config["n"]):
                        errors.append(f"{location}: VQA cell metadata disagrees with grid")
                    queries = [{field: record.get(field) for field in
                                ("query_index", "query_id", "question", "answers")}
                               for record in cell["records"]]
                    if reference_queries is not None and queries != reference_queries:
                        errors.append(f"{location}: VQA query identities change across grid")
                    reference_queries = queries
                    episodes = [{"query": query, "demo_ids": record.get("demo_ids")}
                                for query, record in zip(queries, cell["records"])]
                    if k in paired_episodes and paired_episodes[k] != episodes:
                        errors.append(f"{location}: VQA arm episodes are not paired")
                    paired_episodes[k] = episodes
        for location, cell in iter_record_cells(payload):
            images = [record["query_id"].split(":")[0] for record in cell["records"]]
            if len(set(images)) != len(images):
                errors.append(f"{location}: repeated VQA evaluation images")
            if excluded.intersection(images):
                errors.append(f"{location}: VQA evaluation overlaps excluded images")

    if config.get("experiment") == "paired_option_order_saved_adapters":
        expected_keys = {":".join(map(str, combination)) for combination in itertools.product(
            config["seeds"], config["tasks"], config["arms"], config["conditions"],
            config["ks"], config["option_orders"])}
        stored = payload.get("cells", {})
        if stored.keys() != expected_keys:
            errors.append("root: incomplete or unexpected option-order evaluation grid")
        paired = {}
        for key, cell in stored.items():
            seed, task, arm, condition, k, order = key.split(":")
            if (cell.get("seed") != int(seed) or cell.get("k") != int(k)
                    or cell.get("condition") != condition
                    or cell.get("protocol") != config["protocols"][order]):
                errors.append(f"root.cells.{key}: cell metadata disagrees with grid key")
            if len(cell["records"]) != config["n"]:
                errors.append(f"root.cells.{key}: incorrect query count")
            identity = [{field: record.get(field) for field in (
                "query_index", "query_id", "gold", "label_space", "demo_source_classes",
                "demo_source_ids", "demo_label_classes", "shuffle_map")}
                for record in cell["records"]]
            group = (seed, task, condition, k)
            if group in paired and paired[group] != identity:
                errors.append(f"root.cells.{key}: episode identities differ across arms or option orders")
            paired[group] = identity

    if config.get("baseline") == "contextual_calibration_neutral_query":
        expected_keys = set(itertools.product(map(str, config["seeds"]), config["conditions"],
                                             map(str, config["ks"])))
        observed = set()
        for seed, conditions in payload.get("evidence", {}).items():
            for condition, shots in conditions.items():
                for k, cell in shots.items():
                    location = f"root.evidence.{seed}.{condition}.{k}"
                    observed.add((seed, condition, k))
                    if (cell.get("seed") != int(seed) or cell.get("k") != int(k)
                            or cell.get("condition") != condition
                            or len(cell["records"]) != config["n"]
                            or cell.get("protocol") != config.get("protocol")):
                        errors.append(f"{location}: calibration cell metadata disagrees with grid")
                    audit_calibration_cell(location, cell, errors)
        if observed != expected_keys or cells != len(expected_keys):
            errors.append("root: incomplete or unexpected contextual-calibration grid")

    training_runs = 0
    for location, run in iter_training_runs(payload):
        training_runs += 1
        losses = run.get("losses", [])
        if len(losses) != run.get("steps"):
            errors.append(f"{location}: loss count does not equal steps")
        if not all(math.isfinite(float(loss)) for loss in losses):
            errors.append(f"{location}: non-finite training loss")
        if "torch_seed" not in run:
            warnings.append(f"{location}: Torch/LoRA initialization seed was not recorded")

    if (isinstance(config, dict) and config.get("method") in
            {"DARA compatibility port", "CAMA independent implementation"}
            and all(field in config for field in ("seeds", "ks", "n"))):
        expected_cells = 4 * len(config["seeds"]) * len(config["ks"])
        if cells != expected_cells or records != expected_cells * config["n"]:
            errors.append("root: incomplete attention-baseline evaluation grid")
        audit_attention_grid(payload, config, errors)
        if config["method"] == "DARA compatibility port":
            if (sorted(payload.get("seeds_done", [])) != sorted(config["seeds"])
                    or training_runs != len(config["seeds"])
                    or sorted(run.get("torch_seed", -1) for _, run in iter_training_runs(payload))
                    != sorted(config["seeds"])):
                errors.append("root: incomplete DARA training seeds")

    replay = config.get("factor_evaluation", {}).get("training_performed") is False
    config = payload.get("run_config")
    if isinstance(config, dict) and "model" in config:
        for field in ("model_revision", "dtype", "software_versions"):
            if field not in config:
                warnings.append(f"root.run_config: missing {field}")
    return {
        "path": str(path),
        "cells": cells,
        "records": records,
        "training_runs": training_runs,
        "new_training_runs": 0 if replay else training_runs,
        "reused_training_runs": training_runs if replay else 0,
        "errors": errors,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--summary", action="store_true",
                        help="print only aggregate counts and files containing errors")
    args = parser.parse_args()
    paths = []
    for path in args.paths:
        paths.extend(sorted(path.glob("*.json")) if path.is_dir() else [path])
    reports = [audit_file(path) for path in paths]
    if args.as_json:
        print(json.dumps(reports, indent=2))
    elif args.summary:
        print(
            f"{len(reports)} files, {sum(r['cells'] for r in reports)} cells, "
            f"{sum(r['records'] for r in reports)} records, "
            f"{sum(r['new_training_runs'] for r in reports)} new training runs, "
            f"{sum(r['reused_training_runs'] for r in reports)} reused training records, "
            f"{sum(len(r['errors']) for r in reports)} errors, "
            f"{sum(len(r['warnings']) for r in reports)} warnings"
        )
        for report in reports:
            if report["errors"]:
                print(f"  {report['path']}: {len(report['errors'])} errors")
    else:
        for report in reports:
            print(
                f"{report['path']}: {report['cells']} cells, {report['records']} records, "
                f"{len(report['errors'])} errors, {len(report['warnings'])} warnings"
            )
            for message in report["errors"]:
                print(f"  ERROR {message}")
            for message in report["warnings"]:
                print(f"  WARN  {message}")
    if any(report["errors"] for report in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
