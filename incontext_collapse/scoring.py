"""Output parsing, shortcut diagnostics, and restricted-label scoring."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import re
from typing import Sequence

import torch


@dataclass(frozen=True)
class ParseResult:
    prediction: int | None
    status: str
    matched_classes: tuple[int, ...]
    normalized_answer: str

    def as_dict(self) -> dict:
        out = asdict(self)
        out["matched_classes"] = list(self.matched_classes)
        return out


def _label_pattern(label: str):
    return re.compile(r"(?<!\w)" + re.escape(label.casefold()) + r"(?!\w)")


def parse_answer(answer: str, label_space: Sequence[str]) -> ParseResult:
    """Parse exactly one distinct label and expose ambiguous/echo outputs.

    The submitted parser selected the earliest substring, so echoing the complete
    option list was scored as class 0. The corrected parser marks any multi-label
    output as unparseable and records whether it appears to be an option-list echo.
    """
    normalized = " ".join(str(answer).casefold().strip().split())
    hits = tuple(i for i, label in enumerate(label_space)
                 if _label_pattern(str(label)).search(normalized))
    if not hits:
        return ParseResult(None, "no_label", (), normalized)
    if len(hits) > 1:
        status = "option_echo" if len(hits) == len(label_space) else "ambiguous_labels"
        return ParseResult(None, status, hits, normalized)
    pred = hits[0]
    stripped = normalized.strip(" \t\n\r.,;:!?\"'`()[]{}")
    exact = stripped == str(label_space[pred]).casefold()
    return ParseResult(pred, "exact" if exact else "single_label_verbose", hits, normalized)


def shortcut_diagnostics(prediction: int | None, demo_label_classes: Sequence[int]) -> dict:
    """Describe positional and frequency shortcuts without treating them as causal."""
    if prediction is None or not demo_label_classes:
        return {
            "copies_first": False,
            "copies_last": False,
            "copies_any_demo": False,
            "copies_majority": False,
            "majority_classes": [],
        }
    counts = Counter(demo_label_classes)
    top = max(counts.values())
    majority = sorted(c for c, n in counts.items() if n == top)
    return {
        "copies_first": prediction == demo_label_classes[0],
        "copies_last": prediction == demo_label_classes[-1],
        "copies_any_demo": prediction in counts,
        "copies_majority": prediction in majority,
        "majority_classes": majority,
    }


def subtract_contextual_bias(query_scores: Sequence[float],
                             content_free_scores: Sequence[float]) -> list[float]:
    """Apply contextual calibration in log space.

    Subtracting content-free label log scores is equivalent to dividing the
    corresponding probabilities before renormalization. Keeping this operation
    separate makes the training-free baseline easy to audit and unit test.
    """
    if len(query_scores) != len(content_free_scores):
        raise ValueError("query and content-free score vectors must have equal length")
    if not query_scores:
        raise ValueError("score vectors must not be empty")
    return [float(query) - float(bias)
            for query, bias in zip(query_scores, content_free_scores)]


def images_from_messages(messages) -> list:
    images = []
    for message in messages:
        content = message.get("content", [])
        if isinstance(content, str):
            continue
        for block in content:
            if block.get("type") == "image" and "image" in block:
                images.append(block["image"])
    return images


def encode_supervised_answer(processor, messages, target: str, device):
    """Encode a prompt/answer pair and mask every token before the answer.

    The submitted training code masked all but the final token. This routine finds
    the token boundary by longest common prefix, so multi-token verbalizers receive
    loss on the complete answer sequence.
    """
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    images = images_from_messages(messages)
    prefix = processor(text=[prompt], images=images, return_tensors="pt")
    full = processor(text=[prompt + " " + target], images=images, return_tensors="pt")
    prefix_ids = prefix["input_ids"][0].tolist()
    full_ids = full["input_ids"][0].tolist()
    boundary = 0
    for a, b in zip(prefix_ids, full_ids):
        if a != b:
            break
        boundary += 1
    if boundary >= len(full_ids):
        raise ValueError(f"target {target!r} produced no answer tokens")
    labels = full["input_ids"].clone()
    labels[:, :boundary] = -100
    return full.to(device), labels.to(device), boundary


@torch.no_grad()
def restricted_label_scores(model, processor, messages, label_space: Sequence[str]) -> dict:
    """Return summed and token-normalized sequence log-likelihoods.

    Summed log-likelihood is the probability of the complete label string, but it
    systematically favors shorter verbalizers. The mean is retained beside it so
    mixed-token label sets cannot silently turn token length into class evidence.
    """
    summed = []
    means = []
    token_counts = []
    for label in label_space:
        inputs, labels, _ = encode_supervised_answer(processor, messages, str(label), model.device)
        output = model(**inputs)
        logits = output.logits[:, :-1].float()
        next_ids = inputs["input_ids"][:, 1:]
        mask = labels[:, 1:] != -100
        token_logp = logits.log_softmax(-1).gather(-1, next_ids.unsqueeze(-1)).squeeze(-1)
        selected = token_logp[mask]
        summed.append(float(selected.sum().item()))
        means.append(float(selected.mean().item()))
        token_counts.append(int(selected.numel()))
    return {"sum": summed, "mean": means, "token_counts": token_counts}
