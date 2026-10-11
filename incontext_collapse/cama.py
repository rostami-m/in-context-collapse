"""Independent CAMA evaluation from arXiv:2505.17097v4, equations 3-10.

See docs/cama_implementation.md for the search record, execution ambiguities,
and prompt adaptation. The native/qa prompts adapt the VQA formulas; the
classification prompt uses the paper's A:-only, single-gain variant.
This is not an author-released implementation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import math
import re
import time
from types import MethodType

import torch
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    apply_multimodal_rotary_pos_emb, repeat_kv,
)

from .core import TASKS, build_messages
from .dara import explicit_causal_mask, image_token_ranges
from .evaluation import evaluate_classification_cell, mean_t_ci
from .models import load_model
from .protocol import ProtocolConfig
from .provenance import model_revision, software_versions
from .results import load_result, write_json_atomic


@dataclass(frozen=True)
class CamaConfig:
    stage1_layers: tuple[int, ...] = (1, 2)
    stage2_layers: tuple[int, ...] = (6, 8, 10, 12, 14, 16, 18)
    token_fraction: float = 0.2
    head_fraction: float = 0.2
    head_metric: str = "logits"
    query_weight: float = 0.0
    epsilon: float = 1e-8

    def __post_init__(self):
        if not self.stage1_layers or not self.stage2_layers:
            raise ValueError("both stage layer sets must be nonempty")
        if min(self.stage1_layers) < 0 or max(self.stage1_layers) >= min(self.stage2_layers):
            raise ValueError("Stage I must precede Stage II")
        for layers in (self.stage1_layers, self.stage2_layers):
            if tuple(sorted(set(layers))) != layers:
                raise ValueError("layer indices must be sorted and unique")
        if not 0 < self.token_fraction <= 1 or not 0 < self.head_fraction <= 1:
            raise ValueError("selection fractions must lie in (0, 1]")
        if self.head_metric not in {"logits", "probabilities"}:
            raise ValueError("unknown head metric")
        if not math.isfinite(self.query_weight) or not math.isfinite(self.epsilon) or self.query_weight < 0 or self.epsilon <= 0:
            raise ValueError("invalid query weight or epsilon")


@dataclass(frozen=True)
class Segment:
    image: tuple[int, ...]
    text: tuple[int, ...]
    anchors: tuple[int, ...]


def select_top(values, fraction):
    """Stable top-k, with an explicit ceiling rule for small images/heads."""
    if values.ndim != 1 or not values.numel() or not 0 < fraction <= 1:
        raise ValueError("expected nonempty scores and a fraction in (0, 1]")
    count = max(1, math.ceil(values.numel() * fraction))
    return torch.argsort(values, descending=True, stable=True)[:count]


def anchor_gain(logits, segment, epsilon=1e-8):
    """Equations 3-5: average head logits, then softmax over image tokens."""
    rows = logits[0, :, list(segment.anchors), :][..., list(segment.image)]
    probability = rows.float().mean(0).softmax(-1).clamp_min(epsilon)
    before, after = probability[:-1], probability[1:]
    return ((after - before).clamp_min(0) * (after.log() - before.log())).sum(0)


def joint_relevance(hidden, segments, selected):
    """Equation 9, from final Stage I hidden states; query text has no answer."""
    vectors = []
    for segment, keys in zip(segments, selected):
        visual = hidden[0, keys].float().mean(0)
        textual = hidden[0, list(segment.text)].float().mean(0)
        vectors.append(torch.cat((visual, textual)))
    joint = torch.nn.functional.normalize(torch.stack(vectors), dim=-1)
    return (joint[:-1] @ joint[-1]).softmax(0)


def select_heads(logits, segments, fraction, metric="logits", attention_mask=None):
    """Equation 8 (raw logits); probabilities is a declared sensitivity arm."""
    values = logits.float()
    if metric == "probabilities":
        if attention_mask is None:
            length = values.shape[-1]
            mask = torch.ones(length, length, device=values.device, dtype=torch.bool).triu(1)
            values = values.masked_fill(mask, -torch.inf)
        else:
            values = values + attention_mask[:, :, :, :values.shape[-1]]
        values = values.softmax(-1)
    elif metric != "logits":
        raise ValueError("unknown head metric")
    context = [p for segment in segments[:-1] for p in segment.image + segment.text]
    query = list(segments[-1].text)
    score = values[0, :, query, :][..., context].sum(-1).mean(-1)
    return select_top(score, fraction)


def add_column_bias(logits, columns, weights, heads=None):
    """Add finite biases only at r > j; the native causal mask is still applied."""
    result = logits.clone()
    rows = torch.arange(logits.shape[-2], device=logits.device)
    columns = torch.as_tensor(columns, device=logits.device, dtype=torch.long)
    if heads is None:
        heads = torch.arange(logits.shape[1], device=logits.device)
    bias = (rows[:, None] > columns[None, :]).to(logits.dtype) * weights.to(logits.dtype)
    result[0, heads[:, None, None], rows[None, :, None], columns[None, None, :]] += bias[None]
    return result


def prompt_segments(input_ids, tokenizer, image_token_id):
    """Recover exact boundaries from the prompt, rejecting unsupported formats."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("CAMA supports one prompt at a time")
    ids = input_ids[0].tolist()
    ranges = image_token_ranges(input_ids[0], image_token_id)
    if not ranges:
        raise ValueError("CAMA requires image tokens")
    vision_start = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    segments = []
    for i, (start, end) in enumerate(ranges):
        if end + 1 >= len(ids) or ids[end + 1] != vision_end:
            raise ValueError("missing Qwen vision-end boundary")
        text_start = end + 2
        text_end = text_start
        while text_end < len(ids) and ids[text_end] not in {vision_start, im_end}:
            text_end += 1
        text_ids = ids[text_start:text_end]
        text = tokenizer.decode(text_ids, skip_special_tokens=False)
        if getattr(tokenizer, "is_fast", True):
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        else:
            pieces = [tokenizer.decode([token], skip_special_tokens=False) for token in text_ids]
            if "".join(pieces) != text:
                raise ValueError("slow-tokenizer offsets require independently decodable prompt tokens")
            offset = 0
            offsets = []
            for piece in pieces:
                offsets.append((offset, offset + len(piece)))
                offset += len(piece)
            encoded = {"input_ids": tokenizer(text, add_special_tokens=False)["input_ids"],
                       "offset_mapping": offsets}
        if list(encoded["input_ids"]) != text_ids:
            raise ValueError("text segment did not round-trip; cannot infer anchors safely")
        offsets = encoded["offset_mapping"]

        def at_char(char):
            for offset, (left, right) in enumerate(offsets):
                if left <= char < right:
                    return text_start + offset
            raise ValueError("anchor has no token")

        is_query = i == len(ranges) - 1
        first = len(text) - len(text.lstrip())
        last = len(text.rstrip()) - 1
        if text.lstrip().startswith("Q:"):
            match = re.search(r"\bA:", text)
            if match is None:
                raise ValueError("Q/A prompt has no answer marker")
            anchor_chars = (first, match.start()) if is_query else (first, match.start(), last)
            if is_query and text[match.end():].strip():
                raise ValueError("query must not contain a supplied answer")
        elif text.lstrip().startswith("A:"):
            # With no question span, the sole increment spans the answer text.
            anchor_chars = (first, last)
            if is_query and text[first + 2:].strip():
                raise ValueError("query must not contain a supplied answer")
        elif is_query and text.lstrip().startswith("Classify the last image."):
            anchor_chars = (first, last)
        elif not is_query and text.lstrip().startswith("This image shows:"):
            anchor_chars = (first, text.index(":"), last)
        else:
            raise ValueError(f"unsupported CAMA prompt segment: {text[:45]!r}")
        positions = tuple(range(at_char(first), at_char(last) + 1))
        anchors = tuple(at_char(char) for char in anchor_chars)
        if len(set(anchors)) != len(anchors):
            raise ValueError("anchor tokens must be distinct")
        segments.append(Segment(tuple(range(start, end + 1)), positions, anchors))
    return segments


def build_qa_messages(demos, query_img, label_space, size, option_order=None):
    """Matched-format sensitivity arm; all methods receive the same Q/A prompt."""
    messages = build_messages(demos, query_img, label_space, size, option_order=option_order)
    question = "Q: Classify this image. A:"
    content = messages[0]["content"]
    for i, (_, label) in enumerate(demos):
        content[2 * i + 1]["text"] = f"{question} {label_space[label]}."
    options = label_space if option_order is None else [label_space[i] for i in option_order]
    content[-1]["text"] = (
        "Q: Classify the last image. Answer with exactly one of "
        f"[{', '.join(options)}] and nothing else. A:")
    return messages


def build_classification_messages(demos, query_img, label_space, size, option_order=None):
    """A:-only image/text pairs with one gain, as in CAMA's classification arm.

    The shared answer-option instruction precedes every image. The paper does
    not define non-Q/A endpoints explicitly; we use first/last segment tokens.
    """
    messages = build_messages(demos, query_img, label_space, size, option_order=option_order)
    content = messages[0]["content"]
    for i, (_, label) in enumerate(demos):
        content[2 * i + 1]["text"] = f"A: {label_space[label]}."
    instruction = content[-1]["text"]
    content[-1]["text"] = "A:"
    content.insert(0, {"type": "text", "text": instruction + "\n"})
    return messages


def _attention_forward(self, hidden_states, attention_mask=None, position_ids=None,
                       past_key_value=None, output_attentions=False, use_cache=False,
                       cache_position=None, position_embeddings=None, **kwargs):
    """Native Qwen2-VL eager computation with a pre-mask CAMA intervention."""
    batch, length, _ = hidden_states.size()
    query = self.q_proj(hidden_states).view(batch, length, -1, self.head_dim).transpose(1, 2)
    key = self.k_proj(hidden_states).view(batch, length, -1, self.head_dim).transpose(1, 2)
    value = self.v_proj(hidden_states).view(batch, length, -1, self.head_dim).transpose(1, 2)
    cos, sin = position_embeddings
    query, key = apply_multimodal_rotary_pos_emb(
        query, key, cos, sin, self.rope_scaling["mrope_section"])
    if past_key_value is not None:
        key, value = past_key_value.update(key, value, self.layer_idx,
            {"sin": sin, "cos": cos, "cache_position": cache_position})
    key = repeat_kv(key, self.num_key_value_groups)
    value = repeat_kv(value, self.num_key_value_groups)
    logits = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)
    attention_mask = explicit_causal_mask(logits, attention_mask, cache_position)
    logits = self.cama_controller.modify(self.layer_idx, logits, attention_mask)
    logits = logits + attention_mask
    if query.dtype == torch.float16:
        logits = torch.where(torch.isinf(logits), torch.zeros_like(logits), logits)
    weights = logits.softmax(-1, dtype=torch.float32).to(query.dtype)
    weights = torch.nn.functional.dropout(weights, p=self.attention_dropout, training=self.training)
    output = torch.matmul(weights, value).transpose(1, 2).contiguous().reshape(batch, length, -1)
    return self.o_proj(output), weights if output_attentions else None, past_key_value


class CamaController:
    """Episode-scoped selectors; only prompt data enter either selection stage."""

    def __init__(self, model, processor, config=None):
        if model.config.model_type != "qwen2_vl":
            raise ValueError("CAMA currently supports Qwen2-VL only")
        self.config = config or CamaConfig()
        self.model, self.processor = model, processor
        layers = model.model.layers
        if max(self.config.stage2_layers) >= len(layers):
            raise ValueError("CAMA layer selection exceeds model depth")
        self.originals = []
        self.phase = "off"
        self.segments = []
        self.diagnostics = []
        for index in self.config.stage1_layers + self.config.stage2_layers:
            attention = layers[index].self_attn
            self.originals.append((attention, attention.forward))
            attention.cama_controller = self
            attention.forward = MethodType(_attention_forward, attention)
        self.hidden_hook = layers[max(self.config.stage1_layers)].register_forward_hook(
            self._stage1_hidden)

    def close(self):
        self.phase = "off"
        self.hidden_hook.remove()
        for attention, original in self.originals:
            attention.forward = original
            del attention.cama_controller

    def _stage1_hidden(self, module, args, output):
        if self.phase == "apply" and output[0].shape[1] == self.prompt_length:
            self.relevance = joint_relevance(output[0], self.segments, self.selected)

    def modify(self, layer, logits, attention_mask):
        if self.phase == "off" or logits.shape[-2] != self.prompt_length:
            return logits
        if logits.shape[0] != 1 or logits.shape[-1] != self.prompt_length:
            raise ValueError("CAMA expects an uncached single-prompt prefill")
        if self.phase == "probe":
            if layer in self.config.stage1_layers:
                self.gains[layer] = [anchor_gain(logits, segment, self.config.epsilon)
                                     for segment in self.segments]
            return logits
        n = len(self.segments) - 1
        if layer in self.config.stage1_layers:
            self.applied_stage1.append(layer)
            for i, (keys, scores) in enumerate(zip(self.selected, self.selected_scores)):
                factor = (n - i) / n if i < n else self.config.query_weight
                weights = factor * scores / (scores.max() + self.config.epsilon)
                logits = add_column_bias(logits, keys, weights)
        if layer in self.config.stage2_layers:
            if self.relevance is None:
                raise RuntimeError("Stage II ran before Stage I representations were available")
            heads = select_heads(logits, self.segments, self.config.head_fraction,
                                 self.config.head_metric, attention_mask)
            self.heads[layer] = heads.tolist()
            for i, segment in enumerate(self.segments[:-1]):
                columns = self.selected[i].tolist() + list(segment.text)
                weights = self.relevance[i].expand(len(columns)) * ((n - i) / n)
                logits = add_column_bias(logits, columns, weights, heads)
        return logits

    @torch.no_grad()
    def generate(self, model, processor, messages, enabled=True, max_new_tokens=6):
        if model is not self.model or processor is not self.processor:
            raise ValueError("controller used with another model or processor")
        if model.training:
            raise ValueError("CAMA is inference-only; set model.eval()")
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt").to(model.device)
        self.phase = "off"
        self.prompt_length = inputs["input_ids"].shape[1]
        self.segments = prompt_segments(inputs["input_ids"], processor.tokenizer,
                                        model.config.image_token_id)
        self.gains, self.heads = {}, {}
        self.selected, self.selected_scores, self.applied_stage1 = [], [], []
        self.relevance = None
        start = time.monotonic()
        try:
            if enabled and len(self.segments) > 1:
                self.phase = "probe"
                model(**inputs, use_cache=False)
                if set(self.gains) != set(self.config.stage1_layers):
                    raise RuntimeError("missing Stage I probe layer")
                for i, segment in enumerate(self.segments):
                    score = torch.stack([self.gains[layer][i]
                        for layer in self.config.stage1_layers]).sum(0)
                    indices = select_top(score, self.config.token_fraction)
                    self.selected.append(torch.tensor(segment.image, device=score.device)[indices])
                    self.selected_scores.append(score[indices])
                self.phase = "apply"
            output = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                    do_sample=False, num_beams=1, num_return_sequences=1,
                                    use_cache=True)
            if self.phase == "apply":
                if set(self.heads) != set(self.config.stage2_layers):
                    raise RuntimeError("missing Stage II application")
                if set(self.applied_stage1) != set(self.config.stage1_layers):
                    raise RuntimeError("missing Stage I application")
            self.diagnostics.append({
                "enabled": enabled, "seconds": time.monotonic() - start,
                "prompt_tokens": self.prompt_length,
                "image_token_counts": [len(s.image) for s in self.segments],
                "anchors": [list(s.anchors) for s in self.segments],
                "selected_tokens": [keys.tolist() for keys in self.selected],
                "max_gain": [float(scores.max()) for scores in self.selected_scores],
                "heads": dict(self.heads),
                "relevance": self.relevance.tolist() if self.relevance is not None else [],
            })
            return processor.batch_decode(output[:, self.prompt_length:], skip_special_tokens=True)[0]
        finally:
            self.phase = "off"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--task", choices=list(TASKS), default="shapes4")
    parser.add_argument("--ks", default="4,8,16")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int,
                        help="default: 6 on Shapes; 8 on CIFAR/Fashion, matching existing drivers")
    parser.add_argument("--prompt", choices=["native", "qa", "classification"], default="native")
    parser.add_argument("--option-order", choices=["random", "class"], default="random")
    parser.add_argument("--head-metric", choices=["logits", "probabilities"], default="logits")
    parser.add_argument("--query-weight", type=float, default=0.0)
    parser.add_argument("--device")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.max_new_tokens is None:
        args.max_new_tokens = 6 if args.task == "shapes4" else 8
    config = CamaConfig(head_metric=args.head_metric, query_weight=args.query_weight)
    protocol = ProtocolConfig(option_order=args.option_order)
    ks = [int(x) for x in args.ks.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    if not ks or len(set(ks)) != len(ks) or min(ks) < 0 or not seeds or len(set(seeds)) != len(seeds):
        parser.error("K values must be unique/nonnegative; seeds must be nonempty/unique")
    if not 0 < args.n <= 120 or args.max_new_tokens < 1:
        parser.error("n must be in 1..120 and generation budget positive")
    run_config = {
        "schema_version": 2, "method": "CAMA independent implementation",
        "reference": "https://arxiv.org/html/2505.17097v4",
        "model": args.model, "model_revision": model_revision(args.model),
        "task": args.task, "ks": ks, "seeds": seeds, "n": args.n,
        "score_mode": "generation", "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "cama": {**asdict(config), "stage1_layers": list(config.stage1_layers),
                 "stage2_layers": list(config.stage2_layers)}, "protocol": protocol.metadata(),
        "dtype": "bfloat16", "software_versions": software_versions(),
        "implementation_version": 1,
        "selection": "unmodified prompt-only probe; modified Stage I hidden states",
        "decode": "prefill-only modulation; cached greedy decoding",
    }
    if args.prompt == "classification":
        run_config["classification_variant"] = {
            "reference": "arXiv:2505.17097v4, classification paragraph following Table 2",
            "paired_text": "A: <label>. for demonstrations; A: for query",
            "gain": "one positive log-ratio increment from first to last segment token",
            "anchor_interpretation": "non-Q/A endpoints are not explicit in the paper",
            "option_instruction": "before the first image",
        }
    result = load_result(args.out, run_config, {
        "method": run_config["method"], "protocol": protocol.metadata(),
        "cells": [], "trainable_parameters": 0,
    })
    model, processor = load_model(args.model, dtype=torch.bfloat16, device=args.device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    result["device"] = str(model.device)
    controller = CamaController(model, processor, config)
    names, pool, test = TASKS[args.task](96, 120)
    builder = {"native": build_messages, "qa": build_qa_messages,
               "classification": build_classification_messages}[args.prompt]
    try:
        for seed in seeds:
            for k in ks:
                for condition in ("standard", "remap"):
                    for arm in ("frozen", "cama"):
                        identity = (seed, k, condition, arm)
                        if any((c["seed"], c["k"], c["condition"], c["arm"]) == identity
                               for c in result["cells"]):
                            continue
                        controller.diagnostics = []
                        cell = evaluate_classification_cell(
                            model, processor, names, pool, test, condition, k, args.n,
                            112, seed, builder,
                            lambda m, p, messages: controller.generate(
                                m, p, messages, enabled=arm == "cama",
                                max_new_tokens=args.max_new_tokens),
                            protocol=protocol, score_mode="generation")
                        cell["arm"] = arm
                        cell["cama_diagnostics"] = controller.diagnostics
                        result["cells"].append(cell)
                        write_json_atomic(args.out, result)
                        print(f"{identity}: accuracy={cell['summary']['accuracy']:.3f} "
                              f"unparseable={cell['summary']['unparseable_rate']:.3f}", flush=True)
    finally:
        controller.close()
    result["aggregate"] = {}
    for arm in ("frozen", "cama"):
        result["aggregate"][arm] = {
            condition: {str(k): mean_t_ci([
                cell["summary"]["accuracy"] for cell in result["cells"]
                if cell["arm"] == arm and cell["condition"] == condition and cell["k"] == k
            ]) for k in ks} for condition in ("standard", "remap")}
    write_json_atomic(args.out, result)
    print("DONE", result["aggregate"], flush=True)


if __name__ == "__main__":
    main()
