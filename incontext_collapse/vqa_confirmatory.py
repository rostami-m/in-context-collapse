"""Prospective, gated image-disjoint VQAv2 checkpoint evaluation.

No fitting, task search, prompt search or automatic fallback is performed. Both
partitions are fixed before generation. Confirmation requires the complete
baseline-only development artifact to pass the predeclared collapse gate.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
from fractions import Fraction
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import random
import tempfile

import numpy as np
from PIL import Image

from .paired_bootstrap import exact_estimate, integer_image_interval
from .protocol import image_id
from .provenance import load_pinned_dataset, DATASET_REVISIONS, model_revision, software_versions
from .results import load_result
from .vqa import DATASETS, adapt, build_vqa_messages, vqa_norm, vqa_score
from .vqa_diverse import load_manifest as load_old_manifest
from .vqa_diverse import select_image_disjoint, validate_training_partition
from .vqa_repair import vqa_query_id


MODEL = "Qwen/Qwen2-VL-2B-Instruct"
OLD_MANIFEST_SHA256 = "c1b7b3e26142380e0bbbe8c01e5d69dc34757df886b4e9270dd96d74a3282ccb"
EXPECTED_DATASETS_VERSION = "5.0.1"
KS = (0, 4, 8, 16)
SEEDS = (0, 1, 2)
ADAPTER_ARMS = ("shapes_connector", "vqa_connector")
N_PARTITION = {"development": 80, "confirmatory": 400}
SAMPLING_SEED = 20261010
SUPPORT_NAMESPACE = "vqa-confirmatory-20261010-nested-support-v1"
BOOTSTRAP_SEED = 20261010
BOOTSTRAP_DRAWS = 10000
POLICY = {
    "version": "vqa-confirmatory-20261010-v1", "primary_k": 16,
    "ks": list(KS), "seeds": list(SEEDS), "img_size": 112,
    "max_new_tokens": 10, "do_sample": False, "dtype": "bfloat16",
    "development_n": 80, "confirmatory_n": 400,
    "baseline_drop_margin": .10, "repair_gain_margin": .05,
    "retention_noninferiority_margin": .03,
    "development_gate": "D_mean >= .10 and paired image-bootstrap 95% lower > 0",
    "multiplicity": "two adapter claims: Bonferroni 97.5% two-sided intervals; intersection-union within each repair claim",
    "stop": "failed development gate stops confirmation; no alternative prompt, seed, subset, K or task search",
    "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": BOOTSTRAP_SEED,
}


def digest_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, payload):
    """Unique sibling temp file + fsync + replace; never partially replace JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp",
                                           dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def item_identity(item):
    image, question, answers = item
    return {"image_id": image_id(image), "query_id": vqa_query_id(image, question),
            "question": question, "answers": list(answers)}


def verify_old_frame(items, old_payload):
    """Reproduce the original 4096-question frame and its sampled 80 exactly."""
    config = old_payload["config"]
    if len(items) != 4096 or config != {
        "dataset": "vqav2", "dataset_revision": DATASET_REVISIONS[DATASETS["vqav2"][0]],
        "scan_valid_questions": 4096, "n_pool": 96, "exclude_prefix": 136,
        "n": 80, "sampling_seed": 20261004,
        "sampling": "uniform_images_then_uniform_question_fixed_prefix_v1",
    }:
        raise ValueError("historical VQA frame configuration changed")
    pool, old_test, diversity = select_image_disjoint(items, 96, 136, 80, 20261004)
    if diversity != old_payload["diversity"]:
        raise ValueError("historical VQA frame diversity/exclusion mismatch")
    for key, actual in (("pool", pool), ("test", old_test)):
        expected = [{field: row[field] for field in item_identity(actual[0])}
                    for row in old_payload[key]]
        if [item_identity(item) for item in actual] != expected:
            raise ValueError(f"recovered source does not reproduce historical {key}")
    if diversity["n_eligible_images"] != 687 or diversity["n_training_images"] != 18:
        raise ValueError("historical source image counts changed")
    return pool


def select_partitions(items, excluded, n_dev=80, n_confirm=400, seed=SAMPLING_SEED):
    """Uniform image selection, then uniform question; no model-dependent fields."""
    groups = defaultdict(list)
    seen = set()
    for item in items:
        identity = item_identity(item)
        if identity["image_id"] not in excluded and identity["query_id"] not in seen:
            groups[identity["image_id"]].append(item)
            seen.add(identity["query_id"])
    if len(groups) < n_dev + n_confirm or min(n_dev, n_confirm) < 1:
        raise ValueError("insufficient fresh image groups for fixed partitions")
    rng = random.Random(seed)
    selected = rng.sample(sorted(groups), n_dev + n_confirm)
    questions = [rng.choice(groups[key]) for key in selected]
    return questions[:n_dev], questions[n_dev:], sorted(groups)


def read_source_frame(data_dir, source_parquet=None):
    """Read at most 4096 valid rows from the pinned stream or verified first shard.

    A local parquet is accepted only with --source-parquet-sha256 at the CLI.
    Exact frame reproduction remains mandatory whichever transport is used.
    """
    if version("datasets") != EXPECTED_DATASETS_VERSION:
        raise ValueError("datasets version changed; independent protocol review required")
    from datasets import Dataset
    if source_parquet:
        files = [str(path) for path in source_parquet]
        # Local Arrow-backed iteration avoids torch multiprocessing/shared-memory
        # initialization in IterableDataset, and preserves embedded HF features.
        stream = Dataset.from_parquet(files, cache_dir=str(Path(data_dir) / "hf_cache"))
    else:
        hf_id, subset, split = DATASETS["vqav2"]
        stream = load_pinned_dataset(hf_id, subset, split=split, streaming=True,
                                     cache_dir=str(Path(data_dir) / "hf_cache"))
    items = []
    for example in stream:
        image, question, answers = adapt("vqav2", example)
        if image is not None and question and answers:
            # A detached image prevents later lazy stream access and closes handles.
            items.append((image.copy(), question, list(answers)))
        if len(items) == 4096:
            break
    if len(items) != 4096:
        raise ValueError(f"incomplete fixed source frame: {len(items)} valid questions")
    return items


def prepare_data(data_dir, old_manifest, *, allow_fetch=False, source_parquet=None):
    directory = Path(data_dir)
    manifest_path = directory / "manifest.json"
    if digest_file(old_manifest) != OLD_MANIFEST_SHA256:
        raise ValueError("historical 80-image manifest bytes differ from audited source")
    _, _, old_payload = load_old_manifest(old_manifest)  # Validate all old pixels too.
    if manifest_path.exists():
        load_prepared(manifest_path, old_manifest)
        return manifest_path
    if not allow_fetch and source_parquet is None:
        raise ValueError("data absent: prepare once with --allow-fetch or verified --source-parquet")
    items = read_source_frame(directory, source_parquet)
    pool = verify_old_frame(items, old_payload)
    excluded = set(old_payload["diversity"]["excluded_image_ids"])
    excluded.update(row["image_id"] for row in old_payload["test"])
    development, confirmatory, eligible = select_partitions(items, excluded)
    if len(eligible) != 607:
        raise ValueError(f"expected 607 fresh eligible image groups; found {len(eligible)}")
    (directory / "images").mkdir(parents=True, exist_ok=True)

    def serialize(item):
        record = item_identity(item)
        path = directory / "images" / f"{record['image_id']}.png"
        if not path.exists():
            # A crash may leave an invalid image, which is rejected on resume.
            item[0].save(path, format="PNG")
        with Image.open(path) as restored:
            if image_id(restored) != record["image_id"]:
                raise ValueError("lossless cache pixel identity changed")
        record.update(image=f"images/{record['image_id']}.png", image_sha256=digest_file(path))
        return record

    frame = [item_identity(item) for item in items]
    payload = {
        "schema_version": 1, "policy": POLICY,
        "dataset": "lmms-lab/VQAv2", "split": "validation",
        "dataset_revision": DATASET_REVISIONS[DATASETS["vqav2"][0]],
        "datasets_version": version("datasets"), "sampling_seed": SAMPLING_SEED,
        "sampling": "uniform fresh images then uniform question; development first80, confirmation next400",
        "old_manifest_sha256": OLD_MANIFEST_SHA256,
        "source_frame_sha256": digest_json(frame),
        "source_frame_records": frame, "n_source_valid_questions": 4096,
        "excluded_image_ids": sorted(excluded), "eligible_image_ids": eligible,
        "source_transport": {"local_parquet_sha256": {
            Path(path).name: digest_file(path) for path in source_parquet}
            if source_parquet else None, "pinned_stream": source_parquet is None},
        "pool": [serialize(item) for item in pool],
        "partitions": {"development": [serialize(item) for item in development],
                       "confirmatory": [serialize(item) for item in confirmatory]},
    }
    atomic_json(manifest_path, payload)
    load_prepared(manifest_path, old_manifest)
    return manifest_path


def load_prepared(path, old_manifest):
    path = Path(path)
    payload = json.loads(path.read_text())
    if (digest_file(old_manifest) != OLD_MANIFEST_SHA256
            or payload.get("old_manifest_sha256") != OLD_MANIFEST_SHA256
            or payload.get("policy") != POLICY
            or payload.get("dataset_revision") != DATASET_REVISIONS[DATASETS["vqav2"][0]]
            or payload.get("datasets_version") != EXPECTED_DATASETS_VERSION
            or version("datasets") != EXPECTED_DATASETS_VERSION
            or payload.get("sampling_seed") != SAMPLING_SEED):
        raise ValueError("prepared data protocol/provenance mismatch")
    _, _, old = load_old_manifest(old_manifest)
    excluded = set(old["diversity"]["excluded_image_ids"])
    excluded.update(row["image_id"] for row in old["test"])
    if excluded != set(payload["excluded_image_ids"]):
        raise ValueError("historical image exclusions changed")
    frame = payload["source_frame_records"]
    if len(frame) != 4096 or digest_json(frame) != payload["source_frame_sha256"]:
        raise ValueError("source frame digest/count mismatch")
    groups = defaultdict(list)
    seen = set()
    for row in frame:
        if row["image_id"] not in excluded and row["query_id"] not in seen:
            groups[row["image_id"]].append(row)
            seen.add(row["query_id"])
    if sorted(groups) != payload["eligible_image_ids"] or len(groups) != 607:
        raise ValueError("eligible source frame mismatch")
    rng = random.Random(SAMPLING_SEED)
    selected = rng.sample(sorted(groups), 480)
    expected = [rng.choice(groups[key]) for key in selected]
    fields = ("image_id", "query_id", "question", "answers")
    actual = [row for partition in ("development", "confirmatory")
              for row in payload["partitions"][partition]]
    if [{field: row[field] for field in fields} for row in actual] != expected:
        raise ValueError("fixed random partition assignment changed")
    if [{field: row[field] for field in fields} for row in payload["pool"]] != frame[:96]:
        raise ValueError("historical 96-question support pool changed")
    for label, length in N_PARTITION.items():
        if len(payload["partitions"][label]) != length:
            raise ValueError("fixed partition size changed")

    def restore(row):
        image_path = (path.parent / row["image"]).resolve()
        if not image_path.is_relative_to(path.parent.resolve()):
            raise ValueError("image path escapes prepared data directory")
        if digest_file(image_path) != row["image_sha256"]:
            raise ValueError("prepared image file changed")
        with Image.open(image_path) as image:
            image = image.copy()
        if item_identity((image, row["question"], row["answers"])) != {
                field: row[field] for field in fields}:
            raise ValueError("prepared query pixels/question identity changed")
        if len(row["answers"]) != 10:
            raise ValueError("VQAv2 query must retain all ten references")
        return image, row["question"], row["answers"]

    pool = [restore(row) for row in payload["pool"]]
    partitions = {key: [restore(row) for row in rows]
                  for key, rows in payload["partitions"].items()}
    return pool, partitions, payload


def support_order(pool, query, seed):
    key = vqa_query_id(query[0], query[1])
    # Independent per-query streams; prefixes are shared across K and all arms.
    rng = random.Random(int(digest_json([SUPPORT_NAMESPACE, seed, key]), 16))
    return rng.sample(list(pool), max(KS))


def expected_record(pool, query, seed, arm, k, index, partition):
    demos = support_order(pool, query, seed)[:k]
    query_row = item_identity(query)
    demo_rows = [item_identity(item) for item in demos]
    record = {
        "partition": partition, "seed": seed, "arm": arm, "k": k,
        "query_index": index, **query_row,
        "resized_query_image_id": image_id(query[0].convert("RGB").resize((112, 112))),
        "demo_ids": [row["query_id"] for row in demo_rows],
        "demo_image_ids": [row["image_id"] for row in demo_rows],
        "demo_questions": [row["question"] for row in demo_rows],
        "demo_answers": [row["answers"] for row in demo_rows],
        "resized_demo_image_ids": [image_id(item[0].convert("RGB").resize((112, 112)))
                                   for item in demos],
        "pair_id": digest_json([partition, seed, query_row["query_id"]]),
    }
    record["episode_id"] = digest_json({key: value for key, value in record.items()
                                        if key != "arm"})
    record["prompt_sha256"] = digest_json({
        "query": {"pixels": record["resized_query_image_id"], "question": query[1]},
        "supports": [{"pixels": pixels, "text": f"Question: {q}\nAnswer: {answers[0]}."}
                     for pixels, q, answers in zip(record["resized_demo_image_ids"],
                                                  record["demo_questions"], record["demo_answers"])],
        "query_instruction": "Answer with a short word or phrase.",
        "prompt_builder": "vqa.build_vqa_messages", "generation": POLICY,
    })
    return record, demos


def record_key(seed, arm, k, query_id):
    return f"{seed}:{arm}:{k}:{query_id}"


def validate_generation_metadata(row):
    metadata = row.get("generation", {})
    tokens = metadata.get("token_ids")
    eos_ids = metadata.get("eos_token_ids")
    if (not isinstance(tokens, list) or not 0 < len(tokens) <= 10
            or any(not isinstance(token, int) for token in tokens)
            or not isinstance(eos_ids, list)
            or metadata.get("n_generated_tokens") != len(tokens)
            or metadata.get("max_new_tokens") != 10
            or metadata.get("do_sample") is not False
            or metadata.get("ended_with_eos") != (tokens[-1] in eos_ids)
            or metadata.get("reached_token_budget") != (len(tokens) == 10)):
        raise ValueError("raw generation budget/token provenance mismatch")


def validate_records(result, pool, queries, partition, arms, *, require_complete=False):
    expected_keys = set()
    for seed in SEEDS:
        for arm in arms:
            for k in KS:
                for index, query in enumerate(queries):
                    expected, _ = expected_record(pool, query, seed, arm, k, index, partition)
                    key = record_key(seed, arm, k, expected["query_id"])
                    expected_keys.add(key)
                    if key not in result["records"]:
                        continue
                    row = result["records"][key]
                    if any(row.get(field) != value for field, value in expected.items()):
                        raise ValueError("resumed record is not a paired protocol episode")
                    if (not isinstance(row.get("prediction"), str)
                            or row.get("normalized_prediction") != vqa_norm(row["prediction"])
                            or not isinstance(row.get("score"), (int, float))
                            or not math.isfinite(row["score"])
                            or abs(row.get("score", -1) - vqa_score(row["prediction"], query[2])) > 1e-12):
                        raise ValueError("resumed raw prediction/score mismatch")
                    validate_generation_metadata(row)
    if not set(result["records"]).issubset(expected_keys):
        raise ValueError("unexpected records in resumed result")
    if require_complete and set(result["records"]) != expected_keys:
        raise ValueError("incomplete paired result panel")


def image_interval(values, confidence=.95, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED,
                   *, integer_numerators=None, denominator=None):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("bootstrap needs finite nonempty image-level values")
    numerators = None
    if integer_numerators is not None:
        numerators = np.asarray(integer_numerators)
        if (numerators.shape != values.shape or not np.issubdtype(numerators.dtype, np.integer)
                or not isinstance(denominator, int) or denominator <= 0
                or not np.allclose(numerators / denominator, values, atol=1e-12, rtol=0)):
            raise ValueError("integer bootstrap numerators do not match image means")
    if type(draws) is not int or draws < 2 or not 0 < confidence < 1:
        raise ValueError("valid confidence and >=2 bootstrap draws required")
    if numerators is not None:
        exact = integer_image_interval(list(map(int, numerators)), denominator,
            confidence=confidence, draws=draws, seed=seed, generator="numpy")
        result = {key: exact[key] for key in ("mean", "low", "high", "exact_rationals")}
    else:
        rng = np.random.default_rng(seed)
        means = np.empty(draws)
        for start in range(0, draws, 500):
            count = min(500, draws - start)
            indices = rng.integers(0, len(values), (count, len(values)))
            means[start:start + count] = values[indices].mean(axis=1)
        tail = (1-confidence)/2
        result = {"mean": float(values.mean()), "low": float(np.quantile(means, tail)),
                  "high": float(np.quantile(means, 1-tail))}
    return {**result, "confidence": confidence,
            "n_images": len(values), "bootstrap_draws": draws, "bootstrap_seed": seed,
            "scope": "whole-image sampling, support seeds averaged, conditional on fixed checkpoints and frame"}



def analyze_records(records, query_ids, seeds=SEEDS, arms=("frozen",), draws=BOOTSTRAP_DRAWS):
    """Pair K and arms inside each image; never treat support seeds as images."""
    if len(query_ids) != len(set(query_ids)) or not query_ids:
        raise ValueError("analysis requires unique nonempty image queries")
    if len({key.split(":")[0] for key in query_ids}) != len(query_ids):
        raise ValueError("analysis requires one query per image")
    matrices = {}
    for arm in arms:
        for k in KS:
            matrix = []
            for seed in seeds:
                panel = []
                for query in query_ids:
                    row = records[record_key(seed, arm, k, query)]
                    frozen = records[record_key(seed, "frozen", k, query)]
                    zero = records[record_key(seed, arm, 0, query)]
                    if (row["episode_id"] != frozen["episode_id"]
                            or row["demo_ids"] != frozen["demo_ids"]
                            or row["pair_id"] != zero["pair_id"]
                            or row["demo_ids"] != records[record_key(seed, arm, 16, query)]["demo_ids"][:k]):
                        raise ValueError("unpaired arm/K support episodes")
                    panel.append(row["score"])
                matrix.append(panel)
            matrices[arm, k] = np.asarray(matrix, dtype=float)

    def estimate(matrix, confidence=.95):
        from .evaluation import mean_t_ci
        integer_scores = np.rint(matrix * 10).astype(np.int64)
        if not np.allclose(matrix, integer_scores / 10, atol=1e-12, rtol=0):
            raise ValueError("ten-reference VQA score/effect lies outside the exact tenth grid")
        image_numerators = integer_scores.sum(axis=0)
        denominator = 10 * matrix.shape[0]
        result = image_interval(image_numerators / denominator, confidence, draws,
                                integer_numerators=image_numerators, denominator=denominator)
        result["mean"] = int(integer_scores.sum()) / (10 * matrix.size)
        result["seed_means"] = [int(total) / (10 * matrix.shape[1])
                                for total in integer_scores.sum(axis=1)]
        result["seed_mean_t_interval_95"] = mean_t_ci(result["seed_means"])
        return result

    drop = matrices["frozen", 0] - matrices["frozen", 16]
    drop_estimate = estimate(drop)
    collapse = exact_estimate(drop_estimate, "mean") >= Fraction(1, 10) and exact_estimate(drop_estimate, "low") > 0
    report = {"primary_k": 16, "frozen_drop_D": drop_estimate,
              "baseline_collapse_supported": collapse,
              "accuracies": {arm: {str(k): estimate(matrices[arm, k]) for k in KS}
                             for arm in arms}, "adapter_claims": {}}
    for arm in arms:
        if arm == "frozen":
            continue
        gain = estimate(matrices[arm, 16] - matrices["frozen", 16], .975)
        reduction = estimate((matrices[arm, 16] - matrices[arm, 0]) + drop, .975)
        retention = estimate(matrices[arm, 0] - matrices["frozen", 0], .975)
        remaining = estimate(matrices[arm, 0] - matrices[arm, 16], .975)
        supported = bool(collapse and exact_estimate(gain, "mean") >= Fraction(1, 20)
                         and exact_estimate(gain, "low") > 0 and exact_estimate(reduction, "low") > 0
                         and exact_estimate(retention, "low") > Fraction(-3, 100))
        report["adapter_claims"][arm] = {
            "gain_G": gain, "collapse_reduction_R": reduction,
            "zero_shot_retention": retention, "remaining_drop": remaining,
            "repair_supported": supported,
            "collapse_below_margin_supported": bool(supported and exact_estimate(remaining, "high") < Fraction(1, 10)),
        }
    return report


def assert_gate(development, config, pool, queries):
    dev_config = development.get("run_config", {})
    for field in ("policy", "manifest_sha256", "model", "model_revision", "checkpoints",
                  "software_versions", "implementation_sha256"):
        if dev_config.get(field) != config.get(field):
            raise ValueError(f"development/confirmation mismatch: {field}")
    if (dev_config.get("partition") != "development" or dev_config.get("arms") != ["frozen"]
            or dev_config.get("n") != 80 or dev_config.get("seeds") != list(SEEDS)):
        raise ValueError("gate artifact is not the locked baseline-only development study")
    validate_records(development, pool, queries, "development", ["frozen"], require_complete=True)
    analysis = analyze_records(development["records"], [vqa_query_id(q[0], q[1]) for q in queries])
    if not analysis["baseline_collapse_supported"]:
        raise ValueError("STOP: development baseline-collapse gate failed; confirmation is forbidden")
    return analysis


def checkpoint_provenance(adapters, historical_result):
    from .option_order_eval import checkpoint_manifest
    if set(adapters) != set(ADAPTER_ARMS):
        raise ValueError("pin both shapes_connector and vqa_connector before development")
    checkpoints = checkpoint_manifest(adapters, list(SEEDS))
    historical = json.loads(Path(historical_result).read_text())["run_config"]
    if (historical["manifest_sha256"] != OLD_MANIFEST_SHA256
            or historical["model_revision"] != model_revision(MODEL)):
        raise ValueError("historical checkpoint evaluation provenance changed")
    for arm in ADAPTER_ARMS:
        for seed in SEEDS:
            current = checkpoints[arm][str(seed)]
            old = historical["checkpoints"][arm][str(seed)]
            if (current["sha256"] != old["sha256"] or current["base_model_name_or_path"] != MODEL
                    or current["training_source"] is None
                    or current["training_source"]["sha256"] != old["training_source"]["sha256"]):
                raise ValueError("checkpoint weights/config/source differ from previously tested checkpoint")
            training = json.loads(Path(current["training_source"]["path"]).read_text())["run_config"]
            if training.get("model_revision") != model_revision(MODEL):
                raise ValueError("checkpoint source base revision mismatch")
            if arm == "vqa_connector":
                validate_training_partition(training, {
                    "dataset": "vqav2", "dataset_revision": DATASET_REVISIONS[DATASETS["vqav2"][0]],
                    "n_pool": 96, "exclude_prefix": 136})
    return checkpoints


def implementation_manifest():
    from importlib import import_module
    return {name: digest_file(import_module(f"{__package__}.{Path(name).stem}").__file__) for name in (
        "vqa_confirmatory.py", "vqa.py", "vqa_diverse.py", "vqa_repair.py",
        "models.py", "protocol.py", "provenance.py", "option_order_eval.py",
        "core.py", "evaluation.py", "results.py", "paired_bootstrap.py")}


def generate_with_metadata(model, processor, messages):
    """Historical greedy ten-token generation, retaining budget/EOS evidence."""
    import torch
    with torch.no_grad():
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False,
                                 num_beams=1, num_return_sequences=1)
    continuation = outputs[:, inputs["input_ids"].shape[1]:]
    prediction = processor.batch_decode(continuation, skip_special_tokens=True)[0]
    tokens = continuation[0].tolist()
    eos_ids = model.generation_config.eos_token_id
    eos_ids = [] if eos_ids is None else eos_ids if isinstance(eos_ids, list) else [eos_ids]
    metadata = {"token_ids": tokens, "n_generated_tokens": len(tokens),
                "eos_token_ids": eos_ids, "ended_with_eos": bool(tokens and tokens[-1] in eos_ids),
                "reached_token_budget": len(tokens) == 10,
                "max_new_tokens": 10, "do_sample": False}
    validate_generation_metadata({"generation": metadata})
    return prediction, metadata


def run_evaluation(args, config, pool, queries, adapters):
    import torch
    from peft import PeftModel
    from .models import load_model

    result = load_result(args.out, config, {"records": {}})
    validate_records(result, pool, queries, args.partition, config["arms"])
    atomic_json(args.out, result)
    for seed in SEEDS:
        for arm in config["arms"]:
            missing = [(k, index, query) for k in KS for index, query in enumerate(queries)
                       if record_key(seed, arm, k, vqa_query_id(query[0], query[1])) not in result["records"]]
            if not missing:
                continue
            base, processor = load_model(MODEL, dtype=torch.bfloat16)
            model = base
            if arm != "frozen":
                model = PeftModel.from_pretrained(base, adapters[arm] / f"seed_{seed}").eval()
            for count, (k, index, query) in enumerate(missing, 1):
                record, demos = expected_record(pool, query, seed, arm, k, index, args.partition)
                prediction, generation = generate_with_metadata(
                    model, processor, build_vqa_messages(demos, query[0], query[1], 112))
                record.update(prediction=prediction, normalized_prediction=vqa_norm(prediction),
                              score=vqa_score(prediction, query[2]), generation=generation)
                result["records"][record_key(seed, arm, k, record["query_id"])] = record
                if count % 8 == 0 or count == len(missing):
                    atomic_json(args.out, result)
                if count % 40 == 0 or count == len(missing):
                    print(f"{args.partition} seed={seed} arm={arm} completed={count}/{len(missing)}", flush=True)
            del model, base, processor
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
    validate_records(result, pool, queries, args.partition, config["arms"], require_complete=True)
    result["analysis"] = analyze_records(result["records"], [vqa_query_id(q[0], q[1]) for q in queries],
                                         arms=config["arms"])
    result["status"] = ("gate_pass" if result["analysis"]["baseline_collapse_supported"] else "stop_gate_failed"
                        ) if args.partition == "development" else "complete"
    atomic_json(args.out, result)
    print("DONE", result["status"], result["analysis"]["frozen_drop_D"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--old-manifest", default="data/revision_v25/vqa_image_frame/manifest.json")
    parser.add_argument("--historical-result", default="data/revision_v25/vqav2_distinct80.json")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--allow-fetch", action="store_true", help="permit pinned data stream during data preparation only")
    parser.add_argument("--source-parquet")
    parser.add_argument("--source-parquet-sha256")
    parser.add_argument("--parquet-dir", help="directory with only the checked initial validation shards")
    parser.add_argument("--parquet-manifest", help="JSON mapping shard filename to {size, sha256}")
    parser.add_argument("--partition", choices=list(N_PARTITION), default="development")
    parser.add_argument("--n", type=int, help="must match the locked partition size")
    parser.add_argument("--seeds", default="0,1,2", help="locked to 0,1,2")
    parser.add_argument("--adapter", action="append", default=[], metavar="NAME=DIRECTORY",
                        help="both shapes_connector and vqa_connector; directory contains seed_0/1/2")
    parser.add_argument("--development-result", help="complete passed gate artifact, mandatory for confirmation")
    parser.add_argument("--out")
    parser.add_argument("--preflight", action="store_true", help="validate fixed data/checkpoints/gate without loading a model")
    args = parser.parse_args()
    if [int(value) for value in args.seeds.split(",")] != list(SEEDS):
        parser.error("formal study seeds are fixed at 0,1,2")
    if args.n is not None and args.n != N_PARTITION[args.partition]:
        parser.error("formal partition sample size is fixed; smoke testing must use separate synthetic test fixtures")
    source_parquet = None
    if args.source_parquet and args.parquet_dir:
        parser.error("choose one parquet transport")
    if args.source_parquet:
        if (not args.source_parquet_sha256 or
                digest_file(args.source_parquet) != args.source_parquet_sha256):
            parser.error("a local source parquet requires its verified SHA256")
        source_parquet = [Path(args.source_parquet)]
    if args.parquet_dir:
        if not args.parquet_manifest:
            parser.error("--parquet-dir requires --parquet-manifest with size and SHA256 pins")
        pins = json.loads(Path(args.parquet_manifest).read_text())
        if isinstance(pins, list):
            checked = {}
            for entry in pins:
                remote = entry["remote"]
                if entry["sha256"] != remote["lfs"]["oid"]:
                    parser.error("download digest differs from published LFS SHA256")
                name = Path(remote["path"]).name
                if name in checked or Path(entry["local"]).name != name:
                    parser.error("download manifest has duplicate or renamed shards")
                checked[name] = {"size": remote["size"], "sha256": entry["sha256"]}
            pins = checked
        source_parquet = sorted(Path(args.parquet_dir).glob("*.parquet"))
        if not source_parquet or set(pins) != {path.name for path in source_parquet}:
            parser.error("parquet shard manifest differs from local files")
        for path in source_parquet:
            if path.stat().st_size != pins[path.name]["size"] or digest_file(path) != pins[path.name]["sha256"]:
                parser.error("parquet shard size or SHA256 mismatch")
    manifest_path = prepare_data(args.data_dir, args.old_manifest,
                                 allow_fetch=args.allow_fetch, source_parquet=source_parquet)
    if args.prepare_only:
        print("PREPARED", manifest_path, digest_file(manifest_path), flush=True)
        return
    if not args.out:
        parser.error("evaluation/preflight requires --out")
    adapters = {}
    for value in args.adapter:
        name, separator, directory = value.partition("=")
        if not separator or name in adapters or name not in ADAPTER_ARMS:
            parser.error("adapters must name shapes_connector and vqa_connector exactly once")
        adapters[name] = Path(directory).resolve()
    checkpoints = checkpoint_provenance(adapters, args.historical_result)
    pool, partitions, _ = load_prepared(manifest_path, args.old_manifest)
    arms = ["frozen"] if args.partition == "development" else ["frozen", *ADAPTER_ARMS]
    config = {"schema_version": 1, "policy": POLICY, "partition": args.partition,
              "model": MODEL, "model_revision": model_revision(MODEL),
              "manifest_sha256": digest_file(manifest_path), "n": N_PARTITION[args.partition],
              "arms": arms, "seeds": list(SEEDS), "checkpoints": checkpoints,
              "software_versions": software_versions(), "implementation_sha256": implementation_manifest(),
              "metric": "official-style VQAv2 leave-one-annotator-out soft accuracy"}
    if args.partition == "confirmatory":
        if not args.development_result:
            parser.error("confirmation requires --development-result with a passed gate")
        development = json.loads(Path(args.development_result).read_text())
        assert_gate(development, config, pool, partitions["development"])
        config["development_result_sha256"] = digest_file(args.development_result)
    if args.preflight:
        # Check compatibility with any existing output as part of preflight too.
        result = load_result(args.out, config, {"records": {}})
        validate_records(result, pool, partitions[args.partition], args.partition, arms)
        print("PREFLIGHT", json.dumps({"partition": args.partition,
              "generations": config["n"] * len(SEEDS) * len(KS) * len(arms),
              "manifest_sha256": config["manifest_sha256"], "policy": POLICY}), flush=True)
        return
    run_evaluation(args, config, pool, partitions[args.partition], adapters)


if __name__ == "__main__":
    main()
