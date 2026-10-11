"""Unified per-query evaluation for classification experiments."""
from __future__ import annotations

from collections import Counter
import math
import random
import statistics

from .protocol import (ProtocolConfig, corrupt_demo_labels, sample_balanced_demos,
                       image_id, sample_balanced_queries, sample_label_space,
                       sample_option_order)
from .scoring import parse_answer, restricted_label_scores, shortcut_diagnostics


def _mean(records, key):
    return sum(bool(record[key]) for record in records) / len(records) if records else 0.0


def _accuracy(records, prediction_key, availability_key):
    evaluated = [record for record in records if record[availability_key] is not None]
    if not evaluated:
        return None
    return sum(record[prediction_key] == record["gold"] for record in evaluated) / len(evaluated)


def _confusion(records, prediction_key):
    """Rows are gold classes, columns are predictions; malformed outputs are omitted."""
    n_classes = max((len(record["label_space"]) for record in records), default=0)
    matrix = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
    for record in records:
        prediction = record[prediction_key]
        if prediction is not None:
            matrix[record["gold"]][prediction] += 1
    return matrix


def summarize_records(records: list[dict]) -> dict:
    parse_counts = Counter(record["parse_status"] for record in records)
    return {
        "n": len(records),
        "accuracy": sum(record["correct"] for record in records) / len(records) if records else 0.0,
        "generation_accuracy": _accuracy(records, "generation_prediction", "raw_output"),
        "likelihood_accuracy": _accuracy(records, "likelihood_prediction", "likelihood_scores"),
        "likelihood_sum_accuracy": _accuracy(
            records, "likelihood_sum_prediction", "likelihood_scores"),
        "unparseable_rate": _mean(records, "unparseable"),
        "copy_first_rate": _mean(records, "copies_first"),
        "copy_last_rate": _mean(records, "copies_last"),
        "copy_any_demo_rate": _mean(records, "copies_any_demo"),
        "copy_majority_rate": _mean(records, "copies_majority"),
        "follows_corrupted_mapping_rate": _mean(records, "follows_corrupted_mapping"),
        "generation_follows_corrupted_mapping_rate": _mean(
            records, "generation_follows_corrupted_mapping"),
        "likelihood_follows_corrupted_mapping_rate": _mean(
            records, "likelihood_follows_corrupted_mapping"),
        "parse_status_counts": dict(sorted(parse_counts.items())),
        "generation_confusion": _confusion(records, "generation_prediction"),
        "likelihood_confusion": _confusion(records, "likelihood_prediction"),
        "likelihood_sum_confusion": _confusion(records, "likelihood_sum_prediction"),
    }


_T_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def mean_t_ci(values) -> dict:
    """Mean and two-sided 95% seed-t interval; untruncated and pointwise.

    Independence is an experimental assumption, not something this reducer
    verifies. Retained studies use three seeds. Critical values through df=30
    retain the historical three-decimal table; larger df use the fourth-order
    inverse-t approximation, not a normal critical value.
    """
    values = [float(value) for value in values]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("seed estimates must be finite")
    if not values:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "n_seeds": 0}
    mean = statistics.fmean(values)
    if len(values) == 1:
        return {"mean": mean, "ci95_low": None, "ci95_high": None, "n_seeds": 1}
    df = len(values) - 1
    critical = _T_975.get(df)
    if critical is None:
        z = statistics.NormalDist().inv_cdf(0.975)
        critical = (z + (z**3 + z) / (4 * df)
                    + (5*z**5 + 16*z**3 + 3*z) / (96 * df**2)
                    + (3*z**7 + 19*z**5 + 17*z**3 - 15*z) / (384 * df**3)
                    + (79*z**9 + 776*z**7 + 1482*z**5 - 1920*z**3 - 945*z)
                    / (92160 * df**4))
    half = critical * statistics.stdev(values) / len(values) ** 0.5
    return {"mean": mean, "ci95_low": mean - half, "ci95_high": mean + half,
            "n_seeds": len(values)}


def evaluate_classification_cell(model, processor, names, pool, test, condition: str, k: int,
                                 n: int, size: int, seed: int, build_messages, generate,
                                 protocol: ProtocolConfig | None = None,
                                 score_mode: str = "generation") -> dict:
    """Evaluate one cell and retain enough information to audit every prediction."""
    if condition not in {"standard", "shuffled", "remap", "remap_shuffled"}:
        raise ValueError(f"unknown condition: {condition}")
    if score_mode not in {"generation", "likelihood", "both"}:
        raise ValueError(f"unknown score mode: {score_mode}")
    protocol = protocol or ProtocolConfig()
    rng = random.Random(1000 * seed + k)
    items = sample_balanced_queries(test, n, rng)
    records = []

    for query_index, (query_image, gold) in enumerate(items):
        source_demos = sample_balanced_demos(
            pool, k, rng, class_policy=protocol.demo_class_policy)
        shown_demos = list(source_demos)
        shuffle_map = None
        if condition in {"shuffled", "remap_shuffled"} and shown_demos:
            shown_demos, shuffle_map = corrupt_demo_labels(shown_demos, range(len(names)), rng)
        label_space = sample_label_space(
            names, "remap" if condition in {"remap", "remap_shuffled"} else "standard", rng,
            mapping_mode=protocol.mapping_mode, token_set=protocol.remap_token_set)
        option_order = sample_option_order(len(names), seed, k, query_index,
                                           protocol.option_order)
        if protocol.option_order == "class":
            messages = build_messages(shown_demos, query_image, label_space, size)
        else:
            messages = build_messages(shown_demos, query_image, label_space, size,
                                      option_order=option_order)

        raw = None
        parsed = None
        generation_pred = None
        if score_mode in {"generation", "both"}:
            raw = generate(model, processor, messages)
            parsed = parse_answer(raw, label_space)
            generation_pred = parsed.prediction

        likelihood_scores = None
        likelihood_mean_scores = None
        likelihood_token_counts = None
        likelihood_pred = None
        likelihood_sum_pred = None
        if score_mode in {"likelihood", "both"}:
            score_details = restricted_label_scores(model, processor, messages, label_space)
            likelihood_scores = score_details["sum"]
            likelihood_mean_scores = score_details["mean"]
            likelihood_token_counts = score_details["token_counts"]
            likelihood_sum_pred = max(
                range(len(likelihood_scores)), key=likelihood_scores.__getitem__)
            likelihood_pred = max(
                range(len(likelihood_mean_scores)), key=likelihood_mean_scores.__getitem__)

        prediction = likelihood_pred if score_mode == "likelihood" else generation_pred
        if score_mode == "both":
            prediction = generation_pred
        shown_classes = [c for _, c in shown_demos]
        shortcuts = shortcut_diagnostics(prediction, shown_classes)
        record = {
            "query_index": query_index,
            "query_id": image_id(query_image),
            "gold": gold,
            "prediction": prediction,
            "correct": prediction == gold,
            "condition": condition,
            "k": k,
            "label_space": label_space,
            "option_order": option_order,
            "displayed_options": [label_space[index] for index in option_order],
            "demo_source_classes": [c for _, c in source_demos],
            "demo_source_ids": [image_id(image) for image, _ in source_demos],
            "demo_label_classes": shown_classes,
            "shuffle_map": shuffle_map,
            "raw_output": raw,
            "generation_prediction": generation_pred,
            "likelihood_prediction": likelihood_pred,
            "likelihood_sum_prediction": likelihood_sum_pred,
            "likelihood_scores": likelihood_scores,
            "likelihood_mean_scores": likelihood_mean_scores,
            "likelihood_token_counts": likelihood_token_counts,
            "parse_status": parsed.status if parsed else "not_generated",
            "parse_matches": list(parsed.matched_classes) if parsed else [],
            "unparseable": parsed is not None and parsed.prediction is None,
            "follows_corrupted_mapping": (
                shuffle_map is not None and prediction == shuffle_map[gold]),
            "generation_follows_corrupted_mapping": (
                shuffle_map is not None and generation_pred == shuffle_map[gold]),
            "likelihood_follows_corrupted_mapping": (
                shuffle_map is not None and likelihood_pred == shuffle_map[gold]),
            **shortcuts,
        }
        records.append(record)

    return {
        "protocol": protocol.metadata(),
        "condition": condition,
        "k": k,
        "seed": seed,
        "score_mode": score_mode,
        "summary": summarize_records(records),
        "records": records,
    }
