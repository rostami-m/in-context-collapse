import unittest
import copy
import json
from pathlib import Path
import tempfile

from scripts.audit_revision_artifacts import audit_cell, audit_file, official_vqa_score


class ArtifactAuditTests(unittest.TestCase):
    def audit_payload(self, payload):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(payload))
            return audit_file(path)

    def vqa_payload(self):
        evidence = {}
        for seed in (0, 1):
            evidence[str(seed)] = {}
            for arm in ("frozen", "vqa_connector"):
                evidence[str(seed)][arm] = {}
                for k in (0, 4):
                    evidence[str(seed)][arm][str(k)] = {
                        "seed": seed, "k": k, "summary": {"n": 1, "accuracy": 1.0},
                        "records": [{"query_index": 0, "query_id": "eval-image:question",
                                     "question": "Is this red?", "answers": ["yes"] * 10,
                                     "prediction": "yes", "score": 1.0,
                                     "demo_ids": [f"training{i}:question" for i in range(k)]}],
                    }
        return {"run_config": {"method": "image-disjoint VQA checkpoint evaluation",
                               "seeds": [0, 1], "ks": [0, 4],
                               "arms": ["frozen", "vqa_connector"], "n": 1},
                "evidence": evidence, "diversity": {"excluded_image_ids": ["training0"]}}

    def attention_payload(self, method):
        def cell(condition, arm=None):
            result = {"seed": 0, "k": 4, "condition": condition,
                      "summary": {"n": 1, "accuracy": 1.0}, "records": [{
                          "query_index": 0, "query_id": "query-image", "gold": 0,
                          "prediction": 0, "correct": True,
                          "label_space": ["mip", "zorp"], "k": 4,
                          "demo_source_classes": [0, 1, 0, 1],
                          "demo_source_ids": [f"training{i}" for i in range(4)],
                          "demo_label_classes": [0, 1, 0, 1], "shuffle_map": None}]}
            if arm:
                result["arm"] = arm
            return result
        result = {"run_config": {"method": method, "implementation_version": 2,
                                 "seeds": [0], "ks": [4], "n": 1}}
        if method == "CAMA independent implementation":
            result["cells"] = [cell(condition, arm) for condition in ("standard", "remap")
                               for arm in ("frozen", "cama")]
        else:
            block = {"seed": 0, "standard": {"4": cell("standard")},
                     "remap": {"4": cell("remap")}}
            result.update(baseline_evidence=[copy.deepcopy(block)], evidence=[block],
                          seeds_done=[0], training_runs=[{"torch_seed": 0, "steps": 1,
                                                         "losses": [1.0]}])
        return result

    def test_vqa_exact_grid_and_arm_pairing_are_checked(self):
        valid = self.vqa_payload()
        self.assertEqual(self.audit_payload(valid)["errors"], [])
        wrong_arm = copy.deepcopy(valid)
        wrong_arm["evidence"]["0"]["unexpected"] = wrong_arm["evidence"]["0"].pop("vqa_connector")
        self.assertTrue(any("arm grid" in error for error in self.audit_payload(wrong_arm)["errors"]))
        for field, value in (("query_id", "other-image:question"),
                             ("question", "A different question?"), ("answers", ["yes"] * 9)):
            altered = copy.deepcopy(valid)
            altered["evidence"]["0"]["vqa_connector"]["0"]["records"][0][field] = value
            self.assertTrue(any("query identities" in error
                                for error in self.audit_payload(altered)["errors"]))
        altered = copy.deepcopy(valid)
        altered["evidence"]["1"]["vqa_connector"]["4"]["records"][0]["demo_ids"][0] = "other:question"
        self.assertTrue(any("not paired" in error for error in self.audit_payload(altered)["errors"]))

    def test_vqa_cell_metadata_and_query_count_are_checked(self):
        for field, value in (("seed", 99), ("k", 8)):
            altered = self.vqa_payload()
            altered["evidence"]["0"]["frozen"]["0"][field] = value
            self.assertTrue(any("metadata disagrees" in error
                                for error in self.audit_payload(altered)["errors"]))
        altered = self.vqa_payload()
        altered["run_config"]["n"] = 2
        self.assertTrue(any("metadata disagrees" in error
                            for error in self.audit_payload(altered)["errors"]))

    def test_attention_exact_grid_rejects_duplicate_cells_with_same_total(self):
        for method in ("CAMA independent implementation", "DARA compatibility port"):
            valid = self.attention_payload(method)
            self.assertEqual(self.audit_payload(valid)["errors"], [])
            altered = copy.deepcopy(valid)
            if method.startswith("CAMA"):
                altered["cells"][-1] = copy.deepcopy(altered["cells"][0])
            else:
                altered["evidence"][0]["seed"] = 99
            self.assertTrue(any("grid" in error for error in self.audit_payload(altered)["errors"]))

    def test_attention_pairing_checks_demos_and_query_identity(self):
        for method in ("CAMA independent implementation", "DARA compatibility port"):
            for field, value in (("query_id", "another-query"), ("demo_source_ids", ["other"] * 4)):
                altered = self.attention_payload(method)
                cell = (altered["cells"][-1] if method.startswith("CAMA")
                        else altered["evidence"][0]["remap"]["4"])
                cell["records"][0][field] = value
                self.assertTrue(any("not paired" in error
                                    for error in self.audit_payload(altered)["errors"]))

    def test_factor_replay_counts_provenance_without_claiming_new_training(self):
        payload = self.attention_payload("DARA compatibility port")
        payload["run_config"]["factor_evaluation"] = {"training_performed": False}
        report = self.audit_payload(payload)
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["training_runs"], 1)
        self.assertEqual(report["new_training_runs"], 0)
        self.assertEqual(report["reused_training_runs"], 1)

    def calibration_payload(self):
        record = {"query_index": 0, "query_id": "query", "gold": 0,
                  "label_space": ["circle", "square"], "k": 0, "shuffle_map": None,
                  "query_mean_scores": [-2.0, -1.0], "content_free_mean_scores": [-3.0, -1.5],
                  "calibrated_scores": [1.0, .5], "raw_prediction": 1, "prediction": 0,
                  "raw_correct": False, "correct": True}
        cell = {"seed": 0, "k": 0, "condition": "standard", "records": [record],
                "summary": {"n": 1, "raw_accuracy": 0.0, "calibrated_accuracy": 1.0,
                            "follows_corrupted_mapping_rate": 0.0}}
        return {"run_config": {"baseline": "contextual_calibration_neutral_query",
                               "seeds": [0], "ks": [0], "conditions": ["standard"], "n": 1},
                "evidence": {"0": {"standard": {"0": cell}}}}

    def test_calibration_recomputes_scores_argmax_and_both_summaries(self):
        valid = self.calibration_payload()
        self.assertEqual(self.audit_payload(valid)["errors"], [])
        for field, value, fragment in (
                ("calibrated_scores", [.9, .5], "query minus"),
                ("raw_prediction", 0, "argmax"), ("prediction", 1, "argmax"),
                ("raw_correct", True, "correctness flag")):
            altered = copy.deepcopy(valid)
            altered["evidence"]["0"]["standard"]["0"]["records"][0][field] = value
            self.assertTrue(any(fragment in error for error in self.audit_payload(altered)["errors"]))
        for metric in ("raw_accuracy", "calibrated_accuracy"):
            altered = copy.deepcopy(valid)
            altered["evidence"]["0"]["standard"]["0"]["summary"][metric] = .5
            self.assertTrue(any(metric in error for error in self.audit_payload(altered)["errors"]))

    def test_calibration_incomplete_grid_is_rejected(self):
        altered = self.calibration_payload()
        altered["run_config"]["ks"] = [0, 4]
        self.assertTrue(any("contextual-calibration grid" in error
                            for error in self.audit_payload(altered)["errors"]))

    def test_missing_or_inconsistent_option_order_is_rejected(self):
        record = {"query_index": 0, "query_id": "query", "gold": 0,
                  "prediction": 0, "correct": True, "label_space": ["mip", "zorp"]}
        cell = {"records": [record], "summary": {"n": 1, "accuracy": 1.0},
                "protocol": {"protocol_version": "2.5-options", "option_order": "random"}}
        errors, warnings = [], []
        audit_cell("cell", cell, errors, warnings)
        self.assertTrue(any("randomized-option provenance" in error for error in errors))
        record.update(option_order=[1, 0], displayed_options=["mip", "zorp"])
        errors = []
        audit_cell("cell", cell, errors, warnings)
        self.assertTrue(any("displayed options" in error for error in errors))

    def test_demo_query_image_overlap_is_rejected(self):
        record = {"query_index": 0, "query_id": "query", "gold": 0,
                  "prediction": 0, "correct": True, "demo_source_ids": ["query"]}
        errors, warnings = [], []
        audit_cell("cell", {"records": [record], "summary": {}}, errors, warnings)
        self.assertTrue(any("reused as demonstration" in error for error in errors))

    def test_partial_attention_experiments_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            for method in ("DARA compatibility port", "CAMA independent implementation"):
                path.write_text(json.dumps({"run_config": {
                    "method": method, "implementation_version": 2,
                    "seeds": [0, 1, 2], "ks": [4], "n": 40}}))
                errors = audit_file(path)["errors"]
                self.assertTrue(any("incomplete attention" in error for error in errors))

    def test_invalidated_and_old_dara_results_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps({"invalidated": "known defect"}))
            self.assertTrue(audit_file(path)["errors"])
            path.write_text(json.dumps({"run_config": {"method": "DARA compatibility port"}}))
            self.assertTrue(audit_file(path)["errors"])

    def test_recomputed_metric_detects_tampered_summary(self):
        records = [{
            "query_index": index,
            "query_id": str(index),
            "gold": index,
            "prediction": index,
            "correct": True,
            "unparseable": False,
            "copies_first": False,
            "copies_last": False,
            "copies_any_demo": False,
            "copies_majority": False,
        } for index in range(2)]
        cell = {"records": records, "summary": {
            "n": 2,
            "accuracy": 0.5,
            "unparseable_rate": 0.0,
            "copy_first_rate": 0.0,
            "copy_last_rate": 0.0,
            "copy_any_demo_rate": 0.0,
            "copy_majority_rate": 0.0,
        }}
        errors, warnings = [], []
        audit_cell("cell", cell, errors, warnings)
        self.assertTrue(any("accuracy" in error for error in errors))

    def test_unbalanced_queries_are_flagged(self):
        records = []
        for index, gold in enumerate((0, 0, 0, 1)):
            records.append({
                "query_index": index,
                "query_id": str(index),
                "gold": gold,
                "prediction": gold,
                "correct": True,
                "unparseable": False,
                "copies_first": False,
                "copies_last": False,
                "copies_any_demo": False,
                "copies_majority": False,
            })
        cell = {"records": records, "summary": {
            "n": 4,
            "accuracy": 1.0,
            "unparseable_rate": 0.0,
            "copy_first_rate": 0.0,
            "copy_last_rate": 0.0,
            "copy_any_demo_rate": 0.0,
            "copy_majority_rate": 0.0,
        }}
        errors, warnings = [], []
        audit_cell("cell", cell, errors, warnings)
        self.assertFalse(errors)
        self.assertTrue(any("unbalanced" in warning for warning in warnings))

    def test_vqa_score_uses_leave_one_annotator_out_agreement(self):
        answers = ["cat"] * 3 + ["dog"] * 7
        self.assertAlmostEqual(official_vqa_score("cat", answers), 0.9)
        records = [{
            "query_index": 0,
            "query_id": "q0",
            "prediction": "cat",
            "answers": answers,
            "score": 1.0,
        }]
        cell = {"records": records, "summary": {"n": 1, "accuracy": 1.0}}
        errors, warnings = [], []
        audit_cell("cell", cell, errors, warnings)
        self.assertTrue(any("VQA score" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
