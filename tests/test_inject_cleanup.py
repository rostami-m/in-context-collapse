import unittest
from unittest.mock import patch

import torch

from incontext_collapse.inject import _generate, extract_vector


class _Inputs(dict):
    def to(self, device):
        return self


class _Layer(torch.nn.Module):
    def __init__(self, tuple_output=False):
        super().__init__()
        self.tuple_output = tuple_output

    def forward(self, hidden):
        return (hidden, "auxiliary") if self.tuple_output else hidden


class _Model:
    device = torch.device("cpu")

    def __init__(self, tuple_output=False):
        self.layer = _Layer(tuple_output)
        self.failure = None
        self.calls = []
        self.prefill = None
        self.decode = None

    def generate(self, input_ids, **kwargs):
        self.calls.append(kwargs)
        self.prefill = self.layer(torch.arange(12, dtype=torch.float32).reshape(1, 3, 4))
        self.decode = self.layer(torch.zeros(1, 1, 4))
        if self.failure is not None:
            raise self.failure
        return torch.cat((input_ids, torch.tensor([[9, 10]])), dim=1)


class _Processor:
    def __init__(self, layer):
        self.layer = layer
        self.failure = None
        self.decoded = None
        self.hooks_at_decode = None

    def apply_chat_template(self, messages, **kwargs):
        return _Inputs(input_ids=torch.tensor([[1, 2, 3]]))

    def batch_decode(self, tokens, skip_special_tokens):
        self.decoded = tokens
        self.hooks_at_decode = len(self.layer._forward_hooks)
        if self.failure is not None:
            raise self.failure
        return ["answer"]


def _hidden(output):
    return output[0] if isinstance(output, tuple) else output


class InjectCleanupTests(unittest.TestCase):
    def test_plain_generation_preserves_default_budget_and_answer_slice(self):
        model = _Model()
        proc = _Processor(model.layer)
        self.assertEqual(_generate(model, proc, []), "answer")
        self.assertEqual(model.calls, [{"max_new_tokens": 8, "do_sample": False, "num_beams": 1, "num_return_sequences": 1}])
        self.assertTrue(torch.equal(proc.decoded, torch.tensor([[9, 10]])))
        self.assertEqual(len(model.layer._forward_hooks), 0)

    def test_explicit_generation_budget_is_forwarded(self):
        model = _Model()
        proc = _Processor(model.layer)
        _generate(model, proc, [], max_new_tokens=16)
        self.assertEqual(model.calls, [{"max_new_tokens": 16, "do_sample": False, "num_beams": 1, "num_return_sequences": 1}])

    def test_grab_preserves_states_and_removes_hook(self):
        for tuple_output in (False, True):
            with self.subTest(tuple_output=tuple_output):
                model = _Model(tuple_output)
                proc = _Processor(model.layer)
                text, vector = _generate(model, proc, [], hook_layer=model.layer, grab=True)
                self.assertEqual(text, "answer")
                self.assertTrue(torch.equal(vector, torch.tensor([[8., 9., 10., 11.]])))
                self.assertTrue(torch.equal(
                    _hidden(model.prefill), torch.arange(12).reshape(1, 3, 4)))
                self.assertFalse(vector.requires_grad)
                self.assertEqual(len(model.layer._forward_hooks), 0)
                self.assertEqual(proc.hooks_at_decode, 0)

    def test_add_only_changes_last_prefill_position_and_removes_hook(self):
        for tuple_output in (False, True):
            with self.subTest(tuple_output=tuple_output):
                model = _Model(tuple_output)
                proc = _Processor(model.layer)
                vector = torch.tensor([1., 2., 3., 4.])
                text = _generate(model, proc, [], hook_layer=model.layer, add_vec=vector)
                self.assertEqual(text, "answer")
                expected = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
                expected[:, -1, :] += vector
                self.assertTrue(torch.equal(_hidden(model.prefill), expected))
                self.assertTrue(torch.equal(_hidden(model.decode), torch.zeros(1, 1, 4)))
                if tuple_output:
                    self.assertEqual(model.prefill[1], "auxiliary")
                self.assertEqual(len(model.layer._forward_hooks), 0)
                self.assertEqual(proc.hooks_at_decode, 0)

    def test_combined_grab_and_add_captures_unmodified_vector(self):
        model = _Model()
        proc = _Processor(model.layer)
        _, vector = _generate(model, proc, [], hook_layer=model.layer,
                              add_vec=torch.ones(4), grab=True)
        self.assertTrue(torch.equal(vector, torch.tensor([[8., 9., 10., 11.]])))
        self.assertTrue(torch.equal(model.prefill[:, -1, :], vector + 1))
        self.assertEqual(len(model.layer._forward_hooks), 0)

    def test_generation_failure_removes_grab_and_add_hooks(self):
        for mode in ("grab", "add"):
            with self.subTest(mode=mode):
                model = _Model()
                proc = _Processor(model.layer)
                failure = RuntimeError("generation failed")
                model.failure = failure
                kwargs = {"grab": True} if mode == "grab" else {"add_vec": torch.ones(4)}
                with self.assertRaises(RuntimeError) as caught:
                    _generate(model, proc, [], hook_layer=model.layer, **kwargs)
                self.assertIs(caught.exception, failure)
                self.assertEqual(len(model.layer._forward_hooks), 0)
                self.assertIsNone(proc.decoded)
                model.failure = None
                self.assertEqual(_generate(model, proc, []), "answer")
                self.assertTrue(torch.equal(
                    model.prefill, torch.arange(12).reshape(1, 3, 4)))

    def test_decode_failure_leaves_no_grab_or_add_hooks(self):
        for mode in ("grab", "add"):
            with self.subTest(mode=mode):
                model = _Model()
                proc = _Processor(model.layer)
                failure = ValueError("decode failed")
                proc.failure = failure
                kwargs = {"grab": True} if mode == "grab" else {"add_vec": torch.ones(4)}
                with self.assertRaises(ValueError) as caught:
                    _generate(model, proc, [], hook_layer=model.layer, **kwargs)
                self.assertIs(caught.exception, failure)
                self.assertEqual(proc.hooks_at_decode, 0)
                self.assertEqual(len(model.layer._forward_hooks), 0)

    def test_hook_failure_also_removes_hook(self):
        model = _Model()
        proc = _Processor(model.layer)
        with self.assertRaises(RuntimeError):
            _generate(model, proc, [], hook_layer=model.layer, add_vec=torch.ones(3))
        self.assertEqual(len(model.layer._forward_hooks), 0)
        self.assertIsNone(proc.decoded)

    def test_existing_hooks_are_preserved(self):
        model = _Model()
        proc = _Processor(model.layer)
        existing = model.layer.register_forward_hook(lambda module, inputs, output: output)
        try:
            _generate(model, proc, [], hook_layer=model.layer, grab=True)
            self.assertEqual(list(model.layer._forward_hooks), [existing.id])
        finally:
            existing.remove()


class InjectExtractionTests(unittest.TestCase):
    def test_success_preserves_mean_and_selected_query_count(self):
        vectors = [torch.tensor([[1., 2., 3., 4.]]),
                   torch.tensor([[5., 6., 7., 8.]])]
        queries = [("query0", 0), ("query1", 1), ("unused", 0)]
        layer = object()
        with patch("incontext_collapse.inject.balanced_demos", return_value=[]) as demos, \
                patch("incontext_collapse.inject.build_messages", return_value=[]) as messages, \
                patch("incontext_collapse.inject._generate",
                      side_effect=[("answer", vector) for vector in vectors]) as generate:
            result = extract_vector(None, None, [], queries, ["a", "b"], 112,
                                    layer, 8, 0, n=2)
        self.assertTrue(torch.equal(result, torch.stack(vectors).mean(0)[0]))
        self.assertEqual(demos.call_count, 2)
        self.assertEqual(generate.call_count, 2)
        self.assertEqual([call.args[1] for call in messages.call_args_list],
                         ["query0", "query1"])
        self.assertTrue(all(call.kwargs == {"hook_layer": layer, "grab": True}
                            for call in generate.call_args_list))

    def test_missing_prefill_state_raises_instead_of_skipping(self):
        for missing_index in (0, 1):
            with self.subTest(missing_index=missing_index):
                vectors = [torch.ones(1, 4), torch.ones(1, 4)]
                vectors[missing_index] = None
                with patch("incontext_collapse.inject.balanced_demos", return_value=[]), \
                        patch("incontext_collapse.inject.build_messages", return_value=[]), \
                        patch("incontext_collapse.inject._generate",
                              side_effect=[("answer", vector) for vector in vectors]) as generate:
                    with self.assertRaisesRegex(RuntimeError, f"extraction sample {missing_index}"):
                        extract_vector(None, None, [], [("query0", 0), ("query1", 1)],
                                       ["a", "b"], 112, object(), 8, 0)
                self.assertEqual(generate.call_count, missing_index + 1)

    def test_empty_query_sample_raises_before_generation(self):
        with patch("incontext_collapse.inject._generate") as generate:
            with self.assertRaisesRegex(ValueError, "at least one query image"):
                extract_vector(None, None, [], [], ["a"], 112, object(), 8, 0)
            generate.assert_not_called()

    def test_nonpositive_sample_count_raises_before_generation(self):
        for n in (0, -1):
            with self.subTest(n=n), patch("incontext_collapse.inject._generate") as generate:
                with self.assertRaisesRegex(ValueError, "positive sample count"):
                    extract_vector(None, None, [], [("query0", 0)], ["a"], 112,
                                   object(), 8, 0, n=n)
                generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
