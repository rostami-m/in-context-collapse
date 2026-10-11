import math
import unittest
from unittest.mock import patch
from types import MethodType

import torch
from PIL import Image
from transformers import BatchEncoding
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration

from incontext_collapse.cama import (
    CamaConfig, CamaController, Segment, add_column_bias, anchor_gain,
    build_classification_messages, build_qa_messages, joint_relevance,
    prompt_segments, select_heads, select_top,
)
from incontext_collapse.dara import _dara_attention_forward


class CharacterTokenizer:
    """Exact offsets and distinct special tokens for prompt-boundary tests."""
    specials = {"<|vision_start|>": 1000, "<|vision_end|>": 1001, "<|im_end|>": 1002}

    def convert_tokens_to_ids(self, token):
        return self.specials[token]

    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) for c in text],
                "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def prompt_ids(texts):
    ids = []
    for text in texts:
        ids += [1000, 999, 999, 1001] + [ord(c) for c in text]
    return torch.tensor([ids + [1002, 1003]])


class CamaEquationTests(unittest.TestCase):
    def test_classification_gain_is_one_positive_log_ratio_increment(self):
        logits = torch.zeros(1, 2, 6, 6)
        logits[:, :, 2, :2] = torch.log(torch.tensor([.8, .2]))
        logits[:, :, 4, :2] = torch.log(torch.tensor([.3, .7]))
        logits[:, :, 3, :2] = torch.tensor([100., -100.])
        segment = Segment((0, 1), (2, 3, 4), (2, 4))
        expected = torch.tensor([0., .5 * math.log(.7 / .2)])
        torch.testing.assert_close(anchor_gain(logits, segment), expected)

    def test_classification_prompt_has_answer_only_segments_and_separate_options(self):
        image = Image.new("RGB", (4, 4))
        labels = ["amber", "birch"]
        messages = build_classification_messages([(image, 1)], image, labels, 4,
                                                 option_order=[1, 0])
        content = messages[0]["content"]
        self.assertIn("[birch, amber]", content[0]["text"])
        self.assertEqual(content[2]["text"], "A: birch.")
        self.assertEqual(content[4]["text"], "A:")
        ids = prompt_ids([content[2]["text"], content[4]["text"]])
        segments = prompt_segments(ids, CharacterTokenizer(), 999)
        self.assertEqual([[chr(int(ids[0, i])) for i in s.anchors] for s in segments],
                         [["A", "."], ["A", ":"]])
        self.assertTrue(all(len(s.anchors) == 2 for s in segments))
        with self.assertRaisesRegex(ValueError, "supplied answer"):
            prompt_segments(prompt_ids(["A: amber.", "A: birch"]), CharacterTokenizer(), 999)
        qa = build_qa_messages([(image, 1)], image, labels, 4, option_order=[1, 0])
        self.assertIn("[birch, amber]", qa[0]["content"][-1]["text"])

    def test_gain_matches_scalar_equations_and_is_nonnegative(self):
        torch.manual_seed(1)
        logits = torch.randn(1, 2, 6, 6, dtype=torch.float64)
        segment = Segment((0, 1), (2, 3, 4), (2, 3, 4))
        distributions = []
        for row in segment.anchors:
            raw = [sum(float(logits[0, h, row, j]) for h in range(2)) / 2
                   for j in segment.image]
            z = sum(math.exp(x) for x in raw)
            distributions.append([math.exp(x) / z for x in raw])
        expected = []
        for j in range(2):
            expected.append(sum(max(b[j] - a[j], 0) * math.log(b[j] / a[j])
                for a, b in zip(distributions, distributions[1:])))
        actual = anchor_gain(logits, segment)
        torch.testing.assert_close(actual, torch.tensor(expected), atol=1e-6, rtol=1e-5)
        self.assertTrue(bool((actual >= 0).all()))

    def test_identical_anchor_distributions_have_zero_gain(self):
        result = anchor_gain(torch.ones(1, 3, 5, 5), Segment((0, 1), (2, 3), (2, 3)))
        self.assertTrue(torch.equal(result, torch.zeros(2)))

    def test_stable_top_selection_ceil_and_ties(self):
        self.assertEqual(select_top(torch.ones(16), .2).tolist(), [0, 1, 2, 3])
        self.assertEqual(select_top(torch.tensor([0., 2., 1.]), .2).tolist(), [1])

    def test_column_bias_respects_heads_and_causal_order(self):
        original = torch.zeros(1, 3, 5, 5)
        result = add_column_bias(original, [1, 3], torch.tensor([2., 4.]), torch.tensor([1]))
        expected = original.clone()
        expected[0, 1, 2:, 1] = 2
        expected[0, 1, 4:, 3] = 4
        self.assertTrue(torch.equal(result, expected))
        self.assertEqual(float(original.sum()), 0)

    def test_joint_relevance_uses_normalized_concatenation(self):
        hidden = torch.tensor([[[1., 0.], [1., 0.], [0., 1.], [0., 1.],
                                [1., 0.], [1., 0.]]])
        segments = [Segment((i,), (i + 1,), (i + 1,)) for i in (0, 2, 4)]
        actual = joint_relevance(hidden, segments, [torch.tensor(s.image) for s in segments])
        torch.testing.assert_close(actual, torch.tensor([math.e / (math.e + 1), 1 / (math.e + 1)]))

    def test_head_selection_uses_query_to_context_only(self):
        logits = torch.zeros(1, 3, 6, 6)
        logits[0, 1, 5, :2] = 2
        logits[0, 2, 0, :] = 100
        segments = [Segment((0,), (1,), (1,)), Segment((3,), (5,), (5,))]
        self.assertEqual(select_heads(logits, segments, .2).tolist(), [1])

    def test_native_and_qa_prompt_boundaries_exclude_chat_and_query_answer(self):
        tokenizer = CharacterTokenizer()
        texts = ["This image shows: amber.", "Classify the last image. Answer with [amber, birch]."]
        segments = prompt_segments(prompt_ids(texts), tokenizer, 999)
        ids = prompt_ids(texts)[0]
        self.assertEqual([chr(int(ids[i])) for i in segments[0].anchors], ["T", ":", "."])
        self.assertEqual(len(segments[-1].anchors), 2)
        self.assertLess(max(segments[-1].text), ids.numel() - 2)
        qa = prompt_segments(prompt_ids(["Q: Shape? A: amber.", "Q: Shape? A:"]), tokenizer, 999)
        self.assertEqual(len(qa[0].anchors), 3)
        with self.assertRaisesRegex(ValueError, "supplied answer"):
            prompt_segments(prompt_ids(["Q: Shape? A: amber.", "Q: Shape? A: amber"]), tokenizer, 999)

    def test_invalid_settings_rejected(self):
        with self.assertRaises(ValueError):
            CamaConfig(token_fraction=0)
        with self.assertRaises(ValueError):
            CamaConfig(stage1_layers=(2, 1))
        with self.assertRaises(ValueError):
            CamaConfig(stage2_layers=(1,))


class CamaModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        config = Qwen2VLConfig(
            vocab_size=48, hidden_size=16, intermediate_size=32,
            num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
            rope_scaling={"type": "mrope", "mrope_section": [1, 1, 0]},
            attention_dropout=0., eos_token_id=47, pad_token_id=46,
            image_token_id=44, video_token_id=43, vision_start_token_id=42,
            vision_config={"depth": 1, "embed_dim": 16, "hidden_size": 16,
                           "num_heads": 2, "mlp_ratio": 2},
        )
        config._attn_implementation = "eager"
        self.model = Qwen2VLForConditionalGeneration(config).eval()
        self.inputs = {"input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
                       "attention_mask": torch.ones(1, 8, dtype=torch.long)}
        self.config = CamaConfig(stage1_layers=(0,), stage2_layers=(1,))

    def test_disabled_controller_native_logits_and_cached_generation_equivalence(self):
        with torch.no_grad():
            expected = self.model(**self.inputs).logits
            generated = self.model.generate(**self.inputs, max_new_tokens=3, do_sample=False)
            controller = CamaController(self.model, None, self.config)
            try:
                actual = self.model(**self.inputs).logits
                patched = self.model.generate(**self.inputs, max_new_tokens=3, do_sample=False)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                self.assertTrue(torch.equal(generated, patched))
            finally:
                controller.close()
            torch.testing.assert_close(self.model(**self.inputs).logits, expected, atol=0, rtol=0)

    def test_sdpa_model_with_implicit_mask_matches_disabled_patch(self):
        config = self.model.config
        config._attn_implementation = "sdpa"
        model = Qwen2VLForConditionalGeneration(config).eval()
        with torch.no_grad():
            expected = model(**self.inputs).logits
            controller = CamaController(model, None, self.config)
            try:
                actual = model(**self.inputs).logits
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            finally:
                controller.close()

    def test_dara_unit_factors_match_full_native_sdpa_model(self):
        config = self.model.config
        config._attn_implementation = "sdpa"
        model = Qwen2VLForConditionalGeneration(config).eval()
        with torch.no_grad():
            expected = model(**self.inputs).logits
            attention = model.model.layers[0].self_attn
            attention.dara_factors = torch.nn.Parameter(torch.ones(attention.num_heads, 8))
            attention.dara_image_ranges = [(0, 1), (5, 5)]
            attention.forward = MethodType(_dara_attention_forward, attention)
            actual = model(**self.inputs).logits
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            changed = dict(self.inputs, input_ids=self.inputs["input_ids"].clone())
            changed["input_ids"][0, -1] = 10
            future = model(**changed).logits
            torch.testing.assert_close(actual[:, :-1], future[:, :-1], atol=0, rtol=0)

    def test_both_stages_modify_prefill_and_leave_decode_untouched(self):
        controller = CamaController(self.model, None, self.config)
        controller.prompt_length = 8
        controller.segments = [Segment((0, 1), (2, 3, 4), (2, 3, 4)),
                               Segment((5,), (6, 7), (6, 7))]
        controller.gains, controller.heads = {}, {}
        controller.selected = [torch.tensor([0]), torch.tensor([5])]
        controller.selected_scores = [torch.tensor([1.]), torch.tensor([1.])]
        controller.applied_stage1, controller.relevance = [], None
        try:
            controller.phase = "probe"
            with torch.no_grad():
                self.model(**self.inputs, use_cache=False)
            self.assertEqual(set(controller.gains), {0})
            controller.phase = "apply"
            with torch.no_grad():
                result = self.model(**self.inputs, use_cache=True)
            self.assertEqual(controller.applied_stage1, [0])
            self.assertEqual(set(controller.heads), {1})
            self.assertTrue(bool(torch.isfinite(result.logits).all()))
            torch.testing.assert_close(controller.relevance, torch.tensor([1.]))
            decode = torch.randn(1, 4, 1, 9)
            self.assertIs(controller.modify(1, decode, None), decode)
        finally:
            controller.close()

    def test_classification_prompt_full_model_probe_prefill_and_cache(self):
        class TinyProcessor:
            tokenizer = CharacterTokenizer()

            def apply_chat_template(self, messages, **kwargs):
                ids = []
                for block in messages[0]["content"]:
                    ids.extend([1000, 999, 999, 1001] if block["type"] == "image"
                               else [ord(c) for c in block["text"]])
                tokens = torch.tensor([ids + [1002, 1003]])
                return BatchEncoding({"input_ids": tokens, "attention_mask": torch.ones_like(tokens)})

            def batch_decode(self, tokens, **kwargs):
                return [str(row.tolist()) for row in tokens]

        config = self.model.config
        config.vocab_size = 1104
        config.image_token_id = 999
        config.vision_start_token_id = 1000
        config._attn_implementation = "sdpa"
        model = Qwen2VLForConditionalGeneration(config).eval()
        processor = TinyProcessor()
        image = Image.new("RGB", (4, 4))
        messages = build_classification_messages([(image, 1)], image, ["amber", "birch"], 4,
                                                 option_order=[1, 0])
        inputs = processor.apply_chat_template(messages)
        with torch.no_grad():
            native = model.generate(**inputs, max_new_tokens=2, do_sample=False, use_cache=True)
        expected = processor.batch_decode(native[:, inputs["input_ids"].shape[1]:])[0]
        controller = CamaController(model, processor, self.config)
        try:
            with patch.object(model, "generate", wraps=model.generate) as generate:
                self.assertEqual(controller.generate(model, processor, messages, enabled=False,
                                                     max_new_tokens=2), expected)
                self.assertEqual(generate.call_args.kwargs["num_beams"], 1)
                self.assertEqual(generate.call_args.kwargs["num_return_sequences"], 1)
            first = controller.generate(model, processor, messages, max_new_tokens=2)
            diagnostic = controller.diagnostics[-1]
            self.assertTrue(all(len(anchors) == 2 for anchors in diagnostic["anchors"]))
            self.assertEqual(set(diagnostic["heads"]), {1})
            self.assertEqual(len(diagnostic["selected_tokens"]), 2)
            self.assertTrue(all(math.isfinite(gain) for gain in diagnostic["max_gain"]))
            self.assertEqual(controller.generate(model, processor, messages, max_new_tokens=2), first)
            self.assertEqual(controller.diagnostics[-1]["selected_tokens"], diagnostic["selected_tokens"])
        finally:
            controller.close()


if __name__ == "__main__":
    unittest.main()
