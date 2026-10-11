"""Protocol tests use synthetic pixels and never load a model or remote data."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from incontext_collapse.vqa_confirmatory import (
    KS, POLICY, SEEDS, analyze_records, assert_gate, atomic_json,
    expected_record, image_interval, item_identity, record_key,
    select_partitions, support_order, validate_records, validate_generation_metadata, verify_old_frame,
)
from incontext_collapse.vqa import vqa_norm, vqa_score
from incontext_collapse.vqa_repair import vqa_query_id


def item(number, question=0):
    image = Image.new("RGB", (4, 4), (number % 256, number // 256, 0))
    return image, f"Question {question}?", ["yes"] * 10


def panel(pool, queries, arms=("frozen",), partition="development", prediction=None):
    prediction = prediction or (lambda arm, k, seed, index: "yes" if k == 0 else "no")
    rows = {}
    for seed in SEEDS:
        for arm in arms:
            for k in KS:
                for index, query in enumerate(queries):
                    row, _ = expected_record(pool, query, seed, arm, k, index, partition)
                    raw = prediction(arm, k, seed, index)
                    row.update(prediction=raw, normalized_prediction=vqa_norm(raw),
                               score=vqa_score(raw, query[2]), generation={
                                   "token_ids": [1, 2], "n_generated_tokens": 2,
                                   "eos_token_ids": [2], "ended_with_eos": True,
                                   "reached_token_budget": False, "max_new_tokens": 10,
                                   "do_sample": False})
                    rows[record_key(seed, arm, k, row["query_id"])] = row
    return {"records": rows}


class ConfirmatoryVQATests(unittest.TestCase):
    def setUp(self):
        self.pool = [item(index) for index in range(20)]
        self.queries = [item(index) for index in range(100, 104)]

    def test_partition_is_fixed_disjoint_and_excludes_all_questions_of_old_images(self):
        items = [item(index, q) for index in range(40) for q in range(3)]
        excluded = {item_identity(item(index))["image_id"] for index in range(5)}
        dev, confirm, eligible = select_partitions(items, excluded, 8, 12, 19)
        repeated = select_partitions(items, excluded, 8, 12, 19)
        self.assertEqual([item_identity(x) for x in dev], [item_identity(x) for x in repeated[0]])
        left = {item_identity(x)["image_id"] for x in dev}
        right = {item_identity(x)["image_id"] for x in confirm}
        self.assertEqual(len(eligible), 35)
        self.assertEqual(len(left), 8)
        self.assertEqual(len(right), 12)
        self.assertFalse(left & right or left & excluded or right & excluded)
        with self.assertRaises(ValueError):
            select_partitions(items, excluded, 30, 10)

    def test_nested_supports_shared_across_arms_and_independent_of_query_iteration(self):
        query = self.queries[0]
        previous = []
        for k in KS:
            baseline, demos = expected_record(self.pool, query, 0, "frozen", k, 0, "development")
            repaired, _ = expected_record(self.pool, query, 0, "shapes_connector", k, 0, "development")
            self.assertEqual(baseline["episode_id"], repaired["episode_id"])
            self.assertEqual(baseline["prompt_sha256"], repaired["prompt_sha256"])
            self.assertEqual(baseline["demo_ids"][:len(previous)], previous)
            self.assertEqual(len(demos), k)
            previous = baseline["demo_ids"]
        first = [item_identity(x) for x in support_order(self.pool, query, 0)]
        support_order(self.pool, self.queries[1], 0)
        self.assertEqual(first, [item_identity(x) for x in support_order(self.pool, query, 0)])
        self.assertNotEqual(first, [item_identity(x) for x in support_order(self.pool, query, 1)])

    def test_raw_soft_scores_resume_rechecked_and_partial_panels_are_allowed_only_for_resume(self):
        result = panel(self.pool, self.queries)
        validate_records(result, self.pool, self.queries, "development", ["frozen"], require_complete=True)
        first = next(iter(result["records"]))
        del result["records"][first]
        validate_records(result, self.pool, self.queries, "development", ["frozen"])
        with self.assertRaises(ValueError):
            validate_records(result, self.pool, self.queries, "development", ["frozen"], require_complete=True)
        row = next(iter(result["records"].values()))
        row["score"] = .125
        with self.assertRaises(ValueError):
            validate_records(result, self.pool, self.queries, "development", ["frozen"])

    def test_resume_rejects_query_pixels_and_support_identity_changes(self):
        result = panel(self.pool, self.queries)
        edited = deepcopy(result)
        next(iter(edited["records"].values()))["resized_query_image_id"] = "changed"
        with self.assertRaises(ValueError):
            validate_records(edited, self.pool, self.queries, "development", ["frozen"])
        edited = deepcopy(result)
        next(row for row in edited["records"].values() if row["k"] == 16)["demo_ids"].reverse()
        with self.assertRaises(ValueError):
            validate_records(edited, self.pool, self.queries, "development", ["frozen"])

    def test_paired_difference_in_differences_and_no_seed_pseudoreplication(self):
        def predict(arm, k, seed, index):
            return "yes" if k == 0 or arm == "shapes_connector" else "no"
        arms = ("frozen", "shapes_connector", "vqa_connector")
        result = panel(self.pool, self.queries, arms=arms, prediction=predict)
        summary = analyze_records(result["records"], [vqa_query_id(q[0], q[1]) for q in self.queries], arms=arms, draws=100)
        self.assertEqual(summary["frozen_drop_D"]["mean"], 1)
        self.assertEqual(summary["frozen_drop_D"]["n_images"], 4)
        claim = summary["adapter_claims"]["shapes_connector"]
        self.assertEqual(claim["gain_G"]["mean"], 1)
        self.assertEqual(claim["collapse_reduction_R"]["mean"], 1)
        self.assertEqual(claim["zero_shot_retention"]["mean"], 0)
        self.assertEqual(claim["gain_G"]["confidence"], .975)
        self.assertTrue(claim["repair_supported"])
        self.assertFalse(summary["adapter_claims"]["vqa_connector"]["repair_supported"])

    def test_apparent_gain_from_damaged_zero_shot_is_not_repair(self):
        def predict(arm, k, seed, index):
            if arm == "frozen":
                return "yes" if k == 0 else "no"
            return "no" if k == 0 else "yes"
        arms = ("frozen", "shapes_connector")
        result = panel(self.pool, self.queries, arms=arms, prediction=predict)
        summary = analyze_records(result["records"], [vqa_query_id(q[0], q[1]) for q in self.queries], arms=arms, draws=100)
        self.assertEqual(summary["adapter_claims"]["shapes_connector"]["gain_G"]["mean"], 1)
        self.assertFalse(summary["adapter_claims"]["shapes_connector"]["repair_supported"])

    def test_analysis_rejects_misaligned_arms_and_multiple_questions_per_image(self):
        arms = ("frozen", "shapes_connector")
        result = panel(self.pool, self.queries, arms=arms)
        queries = [vqa_query_id(q[0], q[1]) for q in self.queries]
        row = next(row for row in result["records"].values() if row["arm"] == "shapes_connector")
        row["episode_id"] = "different"
        with self.assertRaises(ValueError):
            analyze_records(result["records"], queries, arms=arms, draws=100)
        with self.assertRaises(ValueError):
            analyze_records({}, ["image:q1", "image:q2"], draws=100)

    def test_gate_requires_full_passed_baseline_only_study_not_stored_status(self):
        queries = [item(index) for index in range(100, 180)]
        common = {"policy": POLICY, "manifest_sha256": "fixed", "model": "model",
                  "model_revision": "revision", "checkpoints": {}, "software_versions": {},
                  "implementation_sha256": {}}
        config = {**common, "partition": "development", "arms": ["frozen"], "n": 80,
                  "seeds": list(SEEDS)}
        result = panel(self.pool, queries)
        result["run_config"] = config
        self.assertTrue(assert_gate(result, common, self.pool, queries)["baseline_collapse_supported"])
        missing = deepcopy(result)
        missing["records"].pop(next(iter(missing["records"])))
        with self.assertRaises(ValueError):
            assert_gate(missing, common, self.pool, queries)
        failed = panel(self.pool, queries, prediction=lambda *args: "yes")
        failed.update(run_config=config, status="gate_pass")
        with self.assertRaisesRegex(ValueError, "gate failed"):
            assert_gate(failed, common, self.pool, queries)
        with self.assertRaisesRegex(ValueError, "manifest_sha256"):
            assert_gate(result, {**common, "manifest_sha256": "other"}, self.pool, queries)

    def test_original_frame_cannot_be_replaced_by_partial_or_other_sample(self):
        with self.assertRaises(ValueError):
            verify_old_frame(self.queries, {"config": {}})

    def test_atomic_json_preserves_valid_existing_file_on_invalid_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            atomic_json(path, {"records": [1, 2]})
            with self.assertRaises(ValueError):
                atomic_json(path, {"bad": float("nan")})
            self.assertEqual(json.loads(path.read_text()), {"records": [1, 2]})
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_bootstrap_is_deterministic_and_rejects_nonfinite(self):
        self.assertEqual(image_interval([0, .5, 1], draws=100), image_interval([0, .5, 1], draws=100))
        with self.assertRaises(ValueError):
            image_interval([float("nan")], draws=100)

    def test_generation_budget_and_eos_metadata_cannot_be_silently_changed(self):
        row = next(iter(panel(self.pool, self.queries)["records"].values()))
        validate_generation_metadata(row)
        row["generation"]["max_new_tokens"] = 16
        with self.assertRaises(ValueError):
            validate_generation_metadata(row)
        row["generation"]["max_new_tokens"] = 10
        row["generation"]["ended_with_eos"] = False
        with self.assertRaises(ValueError):
            validate_generation_metadata(row)

    def test_resume_rejects_nonfinite_scores_before_analysis(self):
        result = panel(self.pool, self.queries)
        for value in (float("nan"), float("inf"), -float("inf")):
            row = next(iter(result["records"].values()))
            row["score"] = value
            with self.assertRaisesRegex(ValueError, "prediction/score"):
                validate_records(result, self.pool, self.queries, "development", ["frozen"])

    def test_exact_point_threshold_and_integer_bootstrap_zero(self):
        queries = self.queries[:3]
        result = panel(self.pool, queries)
        for row in result["records"].values():
            row["score"] = .9 if row["k"] == 0 else .6 if row["query_index"] == 0 else .9
        summary = analyze_records(result["records"], [vqa_query_id(q[0], q[1]) for q in queries], draws=100)
        self.assertEqual(summary["frozen_drop_D"]["mean"], .1)
        self.assertEqual(summary["frozen_drop_D"]["seed_means"], [.1, .1, .1])
        # A mathematically zero effect stays exactly zero at a bootstrap bound.
        interval = image_interval([0, 0, 0], draws=100,
                                  integer_numerators=[0, 0, 0], denominator=30)
        self.assertEqual(interval["low"], 0)
        self.assertEqual(interval["high"], 0)

    def test_analysis_rejects_effects_outside_ten_reference_score_grid(self):
        result = panel(self.pool, self.queries)
        next(iter(result["records"].values()))["score"] = .125
        with self.assertRaisesRegex(ValueError, "tenth grid"):
            analyze_records(result["records"], [vqa_query_id(q[0], q[1]) for q in self.queries], draws=100)

    def test_exact_five_percent_adapter_gain(self):
        queries = [item(index) for index in range(100, 106)]
        arms = ("frozen", "shapes_connector")
        result = panel(self.pool, queries, arms=arms)
        for row in result["records"].values():
            row["score"] = .9 if row["k"] == 0 else .6
            if row["arm"] == "shapes_connector" and row["query_index"] == 0:
                row["score"] = .9
        summary = analyze_records(result["records"], [vqa_query_id(q[0], q[1]) for q in queries],
                                  arms=arms, draws=100)
        claim = summary["adapter_claims"]["shapes_connector"]
        self.assertEqual(claim["gain_G"]["mean"], .05)
        self.assertEqual(claim["collapse_reduction_R"]["mean"], .05)


if __name__ == "__main__":
    unittest.main()
