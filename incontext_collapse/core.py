"""Shared core for the In-Context Collapse experiments.

This module is the single source of truth for everything the experiment drivers share:

* **Tasks** -- three image-classification tasks used throughout the paper:
  ``cifar4`` (strong-prior natural images), ``shapes4`` (procedurally generated instances of familiar visual concepts), and ``fashion4`` (fine-grained).
* **Verbalizers / conditions** -- the ``standard`` (natural labels), ``shuffled``
  (content-vs-format control), and ``remap`` (arbitrary labels; learning-vs-retrieval)
  conditions from Section 3 of the paper.
* **Prompting** -- single-turn and multi-turn interleaved many-shot prompt builders.
* **Inference / parsing** -- a unified ``predict`` path that works across VLM
  families through the ``AutoModelForImageTextToText`` + ``apply_chat_template`` API.
* **Region targeting** -- the integration/readout layer partition used by the
  region-restricted repair and the CIRCA vaccine.

It has no command-line interface and writes no files; import it from a driver.
Everything runs on CUDA, Apple MPS, or CPU with no code change and no cloud
dependency.
"""
from __future__ import annotations

import os
import random
import re

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from PIL import Image, ImageDraw

from .protocol import LEGACY_REMAP, sample_balanced_demos, sample_label_space
from .provenance import load_pinned_dataset
from .scoring import parse_answer

# Arbitrary label words for the remap condition. The strings themselves may have
# appeared in pretraining; the episode-specific class-to-label bijection is novel.
REMAP = list(LEGACY_REMAP)


# ===========================================================================
# Device / dtype selection
# ===========================================================================
def pick_device() -> str:
    """Return the best available device string: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str):
    """Half precision on accelerators, full precision on CPU."""
    return torch.float32 if device == "cpu" else torch.float16


# ===========================================================================
# Tasks: each returns (class_names, train_pool{cls: [PIL]}, test[(PIL, cls)])
# ===========================================================================
def task_cifar4(n_train: int, n_test: int):
    """CIFAR-10 restricted to 4 classes (airplane, automobile, bird, cat)."""
    dataset_id = "uoft-cs/cifar10"
    ds = load_pinned_dataset(dataset_id)
    names = ds["train"].features["label"].names
    cls = [0, 1, 2, 3]
    pool = {c: [r["img"] for r in ds["train"] if r["label"] == c][:n_train] for c in cls}
    test_by_class = {
        cls.index(c): [r["img"] for r in ds["test"] if r["label"] == c]
        for c in cls
    }
    test = _balanced_test_items(test_by_class, n_test)
    return [names[c] for c in cls], {cls.index(c): v for c, v in pool.items()}, test


def task_fashion4(n_train: int, n_test: int):
    """Fashion-MNIST restricted to 4 classes (t-shirt, trouser, pullover, ankle boot)."""
    dataset_id = "zalando-datasets/fashion_mnist"
    ds = load_pinned_dataset(dataset_id)
    raw = {0: "t-shirt", 1: "trouser", 2: "pullover", 9: "ankle boot"}
    cls = list(raw)
    pool = {cls.index(c): [r["image"] for r in ds["train"] if r["label"] == c][:n_train] for c in cls}
    test_by_class = {
        cls.index(c): [r["image"] for r in ds["test"] if r["label"] == c]
        for c in cls
    }
    test = _balanced_test_items(test_by_class, n_test)
    return list(raw.values()), pool, test


def _balanced_test_items(by_class: dict[int, list], n: int):
    """Take a deterministic balanced test pool of exactly ``n`` items."""
    classes = sorted(by_class)
    base, remainder = divmod(n, len(classes)) if classes else (0, 0)
    selected = []
    for offset, label in enumerate(classes):
        count = base + int(offset < remainder)
        if len(by_class[label]) < count:
            raise ValueError(f"class {label} has fewer than {count} test examples")
        selected.extend((image, label) for image in by_class[label][:count])
    return selected


def _draw_shape(kind: int, rng: random.Random) -> Image.Image:
    """Render one procedurally generated shape on a random light background."""
    img = Image.new("RGB", (112, 112), tuple(rng.randint(200, 255) for _ in range(3)))
    d = ImageDraw.Draw(img)
    col = tuple(rng.randint(0, 120) for _ in range(3))
    x, y, s = rng.randint(20, 50), rng.randint(20, 50), rng.randint(35, 55)
    if kind == 0:
        d.ellipse([x, y, x + s, y + s], fill=col)
    elif kind == 1:
        d.rectangle([x, y, x + s, y + s], fill=col)
    elif kind == 2:
        d.polygon([(x, y + s), (x + s, y + s), (x + s / 2, y)], fill=col)
    else:
        d.line([x, y, x + s, y + s], fill=col, width=8)
        d.line([x, y + s, x + s, y], fill=col, width=8)
    return img


def task_shapes4(n_train: int, n_test: int):
    """Reproducible synthetic instances: circle, square, triangle, cross.

    Independent fixed RNG streams separate train and test instance generation.
    Familiar concepts and label strings need not be absent from pretraining;
    procedural rendering does not establish concept-level contamination freedom.
    """
    names = ["circle", "square", "triangle", "cross"]
    train_rng = random.Random(0)
    test_rng = random.Random(1)
    pool = {c: [_draw_shape(c, train_rng) for _ in range(n_train)] for c in range(4)}
    test = [(_draw_shape(c, test_rng), c)
            for _ in range(n_test // 4 + 1) for c in range(4)][:n_test]
    return names, pool, test


TASKS = {"cifar4": task_cifar4, "shapes4": task_shapes4, "fashion4": task_fashion4}


# ===========================================================================
# Prompt construction
# ===========================================================================
def balanced_demos(pool: dict, k: int, rng: random.Random,
                   class_policy: str = "random_balanced"):
    """Compatibility wrapper for the versioned demonstration sampler."""
    return sample_balanced_demos(pool, k, rng, class_policy=class_policy)


def displayed_options(label_space, option_order=None):
    """Render options independently of the class-index-to-label mapping."""
    order = list(range(len(label_space))) if option_order is None else list(option_order)
    if sorted(order) != list(range(len(label_space))):
        raise ValueError("option_order must be a permutation of class indices")
    return [label_space[index] for index in order]


def build_messages(demos, query_img, label_space, size, option_order=None):
    """Single-turn interleaved prompt: all demonstrations in one user message.

    Works for Qwen2-VL / Qwen2.5-VL / LLaVA / SmolVLM / most VLM families.
    """
    prep = lambda im: im.convert("RGB").resize((size, size))
    options = displayed_options(label_space, option_order)
    content = []
    for img, c in demos:
        content += [
            {"type": "image", "image": prep(img)},
            {"type": "text", "text": f"This image shows: {label_space[c]}."},
        ]
    content += [
        {"type": "image", "image": prep(query_img)},
        {"type": "text", "text": "Classify the last image. Answer with exactly one of "
                                 f"[{', '.join(options)}] and nothing else."},
    ]
    return [{"role": "user", "content": content}]


def build_messages_multiturn(demos, query_img, label_space, size, option_order=None):
    """Multi-turn prompt: one user/assistant pair per demonstration.

    Optional alternative to the primary single-turn protocol. Model support
    depends on its native chat template; the reported panel uses single-turn
    interleaving, including for LLaVA-OneVision.
    """
    prep = lambda im: im.convert("RGB").resize((size, size))
    options = displayed_options(label_space, option_order)
    q = f"What is in this image? Answer with one of [{', '.join(options)}]."
    msgs = []
    for im, cl in demos:
        msgs += [
            {"role": "user", "content": [{"type": "image", "image": prep(im)}, {"type": "text", "text": q}]},
            {"role": "assistant", "content": [{"type": "text", "text": label_space[cl]}]},
        ]
    msgs += [{"role": "user", "content": [{"type": "image", "image": prep(query_img)}, {"type": "text", "text": q}]}]
    return msgs


def parse(answer: str, label_space):
    """Compatibility wrapper for strict, ambiguity-aware answer parsing."""
    return parse_answer(answer, label_space).prediction


@torch.no_grad()
def predict(model, processor, messages, max_new_tokens: int = 8) -> str:
    """Greedy-decode the model's answer to a chat-formatted message list."""
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt").to(model.device)
    n_in = inputs["input_ids"].shape[1]
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                         num_beams=1, num_return_sequences=1)
    return processor.batch_decode(out[:, n_in:], skip_special_tokens=True)[0]


def label_space_for(condition: str, names, rng: random.Random | None = None,
                    mapping_mode: str = "fixed", token_set: str = "legacy"):
    """Compatibility wrapper for explicit verbalizer sampling.

    Its defaults reproduce the submitted fixed Greek mapping. Revision runs pass
    an RNG and explicit corrected protocol through :mod:`evaluation`.
    """
    return sample_label_space(names, condition, rng or random.Random(0),
                              mapping_mode=mapping_mode, token_set=token_set)


# ===========================================================================
# Region targeting: the integration / readout layer partition
# ===========================================================================
# Linear sublayer name suffixes, by component.
_ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
_MLP = ("gate_proj", "up_proj", "down_proj")
_ALL_LIN = _ATTN + _MLP
# Substrings that identify connector / projector modules across architectures.
_CONNECTOR = ("merger", "multi_modal_projector", "mm_projector", "connector",
              "perceiver", "resampler", "qformer", "language_projection")


def region_targets(model, region: str, component: str = "both"):
    """Return the names of Linear modules in a given depth region of the LM.

    Regions partition the language-model blocks into contiguous thirds plus the
    connector and an ``all`` band:

    * ``connector`` -- the vision->language projector (integration interface)
    * ``early`` / ``mid`` / ``late`` -- first / middle / last third of LM blocks
    * ``all`` -- every LM block

    ``component`` restricts to attention-only (``attn``), MLP-only (``mlp``), or
    both. These are operational repair-target groups, not identified causal
    circuits. Vision-encoder blocks are excluded even when they use ``layers``
    and the same projection suffixes as the language decoder.
    """
    if region not in {"connector", "early", "mid", "late", "all"}:
        raise ValueError(f"unknown region: {region}")
    if component not in {"attn", "mlp", "both"}:
        raise ValueError(f"unknown component: {component}")
    vision_names = {"visual", "vision_model", "vision_tower", "vision_encoder", "image_encoder"}
    names = [n for n, module in model.named_modules()
             if isinstance(module, torch.nn.Linear)
             and not (set(n.split(".")) & vision_names)]
    if region == "connector":
        return [n for n, m in model.named_modules()
                if any(c in n for c in _CONNECTOR) and isinstance(m, torch.nn.Linear)]
    suffixes = {"attn": _ATTN, "mlp": _MLP, "both": _ALL_LIN}[component]
    prefixes = {n[:match.start()] for n in names
                if n.endswith(suffixes) and (match := re.search(r"layers\.(\d+)\.", n))}
    if len(prefixes) > 1:
        raise ValueError(f"ambiguous language-layer namespaces: {sorted(prefixes)}")
    layer_ids = sorted({int(m.group(1)) for n in names
                        if n.endswith(suffixes) and (m := re.search(r"layers\.(\d+)\.", n))})
    if not layer_ids:
        return []
    first = len(layer_ids) // 3
    second = 2 * len(layer_ids) // 3
    band_ids = {
        "early": set(layer_ids[:first]),
        "mid": set(layer_ids[first:second]),
        "late": set(layer_ids[second:]),
        "all": set(layer_ids),
    }[region]
    return [n for n in names
            if (m := re.search(r"layers\.(\d+)\.", n)) and int(m.group(1)) in band_ids
            and n.endswith(suffixes)]
