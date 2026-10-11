import copy
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from incontext_collapse.core import build_messages, build_messages_multiturn
from incontext_collapse.evaluation import evaluate_classification_cell
from incontext_collapse.option_order_eval import checkpoint_manifest, main, paired_option_effect
from incontext_collapse.protocol import ProtocolConfig, sample_option_order


class OptionOrderTests(unittest.TestCase):
    def setUp(self):
        self.names = ["circle", "square", "triangle", "cross"]
        self.pool = {
            c: [Image.new("RGB", (4, 4), (40 * c, i, 0)) for i in range(20)]
            for c in range(4)
        }
        self.test = [(Image.new("RGB", (4, 4), (40 * c, 100 + i, 0)), c)
                     for c in range(4) for i in range(30)]

    def evaluate(self, condition, order, generate, k=4, seed=0):
        return evaluate_classification_cell(
            None, None, self.names, self.pool, self.test, condition, k, 40, 4,
            seed, build_messages, generate, protocol=ProtocolConfig(option_order=order))

    @staticmethod
    def position_shortcut(model, processor, messages):
        blocks = messages[-1]["content"]
        query_class = blocks[-2]["image"].getpixel((0, 0))[0] // 40
        options = re.search(r"\[([^\]]+)\]", blocks[-1]["text"]).group(1).split(", ")
        return options[query_class]

    @staticmethod
    def follow_demonstrations(model, processor, messages):
        blocks = messages[-1]["content"]
        query_class = blocks[-2]["image"].getpixel((0, 0))[0] // 40
        for i in range(0, len(blocks) - 2, 2):
            if blocks[i]["image"].getpixel((0, 0))[0] // 40 == query_class:
                return blocks[i + 1]["text"].removeprefix("This image shows: ").removesuffix(".")
        raise AssertionError("test requires a demonstration from every class")

    def test_class_metadata_preserves_protocol_24_and_random_is_new(self):
        self.assertEqual(ProtocolConfig(option_order="class").metadata(), {
            "protocol_version": "2.4-audit", "demo_class_policy": "random_balanced",
            "mapping_mode": "episode", "remap_token_set": "eval",
        })
        current = ProtocolConfig().metadata()
        self.assertEqual(current["protocol_version"], "2.5-options")
        self.assertEqual(current["option_order"], "random")
        with self.assertRaises(ValueError):
            ProtocolConfig(option_order="unknown")

    def test_query_and_multiturn_instructions_share_display_order(self):
        demos = [(self.pool[0][0], 0), (self.pool[1][0], 1)]
        labels = ["mip", "zorp", "kiki", "lurn"]
        order = [2, 0, 3, 1]
        for builder in (build_messages, build_messages_multiturn):
            messages = builder(demos, self.test[0][0], labels, 4, option_order=order)
            instructions = [block["text"] for message in messages
                            for block in message["content"]
                            if block["type"] == "text" and "[" in block["text"]]
            self.assertTrue(instructions)
            self.assertTrue(all("[kiki, mip, lurn, zorp]" in text for text in instructions))
        single = build_messages(demos, self.test[0][0], labels, 4, option_order=order)
        self.assertEqual(single[0]["content"][1]["text"], "This image shows: mip.")
        self.assertEqual(single[0]["content"][3]["text"], "This image shows: zorp.")
        with self.assertRaises(ValueError):
            build_messages(demos, self.test[0][0], labels, 4, option_order=[0, 0, 2, 3])

    def test_permutations_are_reproducible_and_not_fixed_across_episodes(self):
        orders = [sample_option_order(4, 3, 8, episode) for episode in range(40)]
        self.assertEqual(orders, [sample_option_order(4, 3, 8, i) for i in range(40)])
        self.assertGreater(len({tuple(order) for order in orders}), 12)
        self.assertTrue(all(sorted(order) == [0, 1, 2, 3] for order in orders))
        self.assertEqual(sample_option_order(4, 3, 8, 0, "class"), [0, 1, 2, 3])

    def test_option_variants_preserve_all_episode_identities(self):
        for condition in ("standard", "shuffled", "remap", "remap_shuffled"):
            with self.subTest(condition=condition):
                old = self.evaluate(condition, "class", self.follow_demonstrations)
                new = self.evaluate(condition, "random", self.follow_demonstrations)
                effect = paired_option_effect(old, new)
                self.assertEqual(effect["prediction_change_rate"], 0)
                self.assertTrue(any(record["option_order"] != [0, 1, 2, 3]
                                    for record in new["records"]))
                for record in new["records"]:
                    self.assertEqual(record["displayed_options"],
                                     [record["label_space"][i] for i in record["option_order"]])

    def test_random_options_break_query_class_position_shortcut(self):
        for k in (0, 1, 4, 8):
            with self.subTest(k=k):
                old = self.evaluate("remap", "class", self.position_shortcut, k=k)
                new = self.evaluate("remap", "random", self.position_shortcut, k=k)
                self.assertEqual(old["summary"]["accuracy"], 1)
                self.assertLess(new["summary"]["accuracy"], .5)
                self.assertLess(paired_option_effect(old, new)["random_minus_class_accuracy"], -.5)

    def test_wrong_natural_and_nonce_controls_follow_displayed_association(self):
        for condition in ("shuffled", "remap_shuffled"):
            cell = self.evaluate(condition, "random", self.follow_demonstrations)
            self.assertEqual(cell["summary"]["accuracy"], 0)
            self.assertEqual(cell["summary"]["follows_corrupted_mapping_rate"], 1)
            for record in cell["records"]:
                self.assertTrue(all(int(a) != b for a, b in record["shuffle_map"].items()))
                if condition == "shuffled":
                    self.assertEqual(record["label_space"], self.names)

    def test_paired_summary_rejects_changed_demonstrations(self):
        original = self.evaluate("remap", "class", self.follow_demonstrations)
        altered = copy.deepcopy(original)
        altered["records"][0]["demo_source_ids"][0] = "different-image"
        with self.assertRaisesRegex(ValueError, "demo_source_ids"):
            paired_option_effect(original, altered)

    def test_checkpoint_manifest_tracks_checkpoint_bytes_and_missing_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "seed_0"
            checkpoint.mkdir()
            (checkpoint / "adapter_config.json").write_text(json.dumps({
                "base_model_name_or_path": "example/model"}))
            weights = checkpoint / "adapter_model.safetensors"
            with self.assertRaises(FileNotFoundError):
                checkpoint_manifest({"connector": root}, [0])
            weights.write_bytes(b"initial checkpoint")
            before = checkpoint_manifest({"connector": root}, [0])
            weights.write_bytes(b"replaced checkpoint")
            after = checkpoint_manifest({"connector": root}, [0])
            self.assertNotEqual(before, after)

    def test_manifest_preserves_recorded_training_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source_adapters" / "connector_both"
            checkpoint = root / "seed_0"
            checkpoint.mkdir(parents=True)
            (checkpoint / "adapter_config.json").write_text("{}")
            (checkpoint / "adapter_model.safetensors").write_bytes(b"checkpoint")
            source = Path(temporary) / "source.json"
            source.write_text(json.dumps({"run_config": {
                "protocol": ProtocolConfig(option_order="class").metadata(),
                "objective": "episode_remap", "steps": 150, "rank": 8}}))
            manifest = checkpoint_manifest({"connector": root}, [0])
            recorded = manifest["connector"]["0"]["training_source"]
            self.assertEqual(recorded["protocol"]["protocol_version"], "2.4-audit")
            self.assertEqual(recorded["objective"], "episode_remap")

    def test_runner_resumes_only_missing_cells_and_keeps_six_token_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result.json"
            argv = ["option_order_eval", "--model", "Qwen/Qwen2-VL-2B-Instruct",
                    "--tasks", "shapes4", "--conditions", "remap", "--ks", "0",
                    "--n", "4", "--seeds", "0", "--out", str(output)]
            with patch("sys.argv", argv), \
                    patch("incontext_collapse.option_order_eval.TASKS", {
                        "shapes4": lambda *_: (self.names, self.pool, self.test)}), \
                    patch("incontext_collapse.option_order_eval.pick_device", return_value="cpu"), \
                    patch("incontext_collapse.option_order_eval.load_model",
                          return_value=(object(), object())) as load, \
                    patch("incontext_collapse.option_order_eval.predict", return_value="mip") as predict:
                main()
                first = json.loads(output.read_text())
                self.assertEqual(first["run_config"]["max_new_tokens"], 6)
                self.assertEqual(predict.call_count, 8)
                self.assertTrue(all(call.kwargs["max_new_tokens"] == 6
                                    for call in predict.call_args_list))
                main()
                self.assertEqual(load.call_count, 1)
                partial = copy.deepcopy(first)
                partial["cells"].pop("0:shapes4:frozen:remap:0:random")
                output.write_text(json.dumps(partial))
                main()
                self.assertEqual(load.call_count, 2)
                self.assertEqual(predict.call_count, 12)
                completed = json.loads(output.read_text())
                self.assertEqual(completed["cells"], first["cells"])


if __name__ == "__main__":
    unittest.main()
