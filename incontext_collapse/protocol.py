"""Versioned sampling and verbalizer protocol for the revision experiments.

The original release coupled several choices that must be varied independently:
which classes appear when K<C, whether an arbitrary label mapping is fixed or
episode-randomized, and whether training and evaluation reuse the same tokens.
This module makes those choices explicit and serializable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import random
from typing import Iterable, Sequence

import torch


PROTOCOL_VERSION = "2.5-options"
LEGACY_PROTOCOL_VERSION = "2.4-audit"

# Kept only to reproduce the submitted results. These ordered Greek names must not
# be used as evidence for mapping-independent in-context learning.
LEGACY_REMAP = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta")

# Train/eval pools contain disjoint complete verbalizer strings. Their tokenizer
# subwords can overlap, and reusing this evaluation pool is not a test of
# unrestricted vocabulary generalization. Multi-token labels are scored as sequences.
REMAP_TOKEN_SETS = {
    "legacy": LEGACY_REMAP,
    "train": ("dax", "wug", "blicket", "koba", "zibble", "fep", "tulver", "narp"),
    "eval": ("mip", "zorp", "kiki", "lurn", "vash", "tego", "prax", "soma"),
}


@dataclass(frozen=True)
class ProtocolConfig:
    """Choices that define one classification evaluation protocol."""

    demo_class_policy: str = "random_balanced"
    mapping_mode: str = "episode"
    remap_token_set: str = "eval"
    option_order: str = "random"

    def __post_init__(self):
        if self.demo_class_policy not in {"random_balanced", "legacy_cycle"}:
            raise ValueError(f"unknown demo class policy: {self.demo_class_policy}")
        if self.mapping_mode not in {"episode", "fixed"}:
            raise ValueError(f"unknown mapping mode: {self.mapping_mode}")
        if self.remap_token_set not in REMAP_TOKEN_SETS:
            raise ValueError(f"unknown remap token set: {self.remap_token_set}")
        if self.option_order not in {"random", "class"}:
            raise ValueError(f"unknown option order: {self.option_order}")

    def metadata(self) -> dict:
        if self.option_order == "class":
            fields = asdict(self)
            fields.pop("option_order")
            return {"protocol_version": LEGACY_PROTOCOL_VERSION, **fields}
        return {"protocol_version": PROTOCOL_VERSION, **asdict(self)}


def sample_option_order(n_classes: int, seed: int, k: int, episode_index: int,
                        option_order: str = "random", phase: str = "evaluation") -> list[int]:
    """Permute displayed options without consuming the episode-sampling RNG.

    Keeping the stream independent preserves query, demonstration, and mapping
    identities when comparing class-ordered and randomized-option prompts.
    """
    if n_classes < 1:
        raise ValueError("option list must contain at least one class")
    if option_order not in {"random", "class"}:
        raise ValueError(f"unknown option order: {option_order}")
    order = list(range(n_classes))
    if option_order == "random":
        key = f"option-order-v1:{phase}:{seed}:{k}:{episode_index}"
        independent_seed = int.from_bytes(hashlib.sha256(key.encode("ascii")).digest(), "big")
        random.Random(independent_seed).shuffle(order)
    return order


def sample_balanced_demos(pool: dict, k: int, rng: random.Random,
                          class_policy: str = "random_balanced"):
    """Sample K demonstrations with class counts differing by at most one.

    Under the corrected policy, the remainder classes are sampled uniformly. Thus
    K=1 is not always class 0 and K=2 is not always classes 0 and 1. Sampling is
    without replacement within a class whenever the pool is large enough.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    classes = list(pool)
    if not classes and k:
        raise ValueError("cannot sample demonstrations from an empty pool")
    if class_policy == "legacy_cycle":
        plan = [classes[i % len(classes)] for i in range(k)]
    elif class_policy == "random_balanced":
        base, remainder = divmod(k, len(classes)) if classes else (0, 0)
        counts = {c: base for c in classes}
        for c in rng.sample(classes, remainder):
            counts[c] += 1
        plan = [c for c in classes for _ in range(counts[c])]
    else:
        raise ValueError(f"unknown demo class policy: {class_policy}")

    chosen = []
    by_class = {c: [] for c in classes}
    for c in plan:
        by_class[c].append(c)
    for c, slots in by_class.items():
        n = len(slots)
        if not n:
            continue
        images = pool[c]
        picked = rng.sample(images, n) if n <= len(images) else rng.choices(images, k=n)
        chosen.extend((image, c) for image in picked)
    rng.shuffle(chosen)
    return chosen


def sample_balanced_queries(items: Sequence[tuple], n: int, rng: random.Random):
    """Sample an exactly class-balanced query subset when ``n`` is divisible.

    Classification results are described as balanced accuracy in the paper.  The
    old evaluator shuffled a mixed test list and took its first ``n`` entries,
    which could leave materially unequal class counts.  This sampler makes the
    query contract match the reported metric.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n > len(items):
        raise ValueError(f"requested {n} queries from a test set of {len(items)}")
    by_class: dict[int, list[tuple]] = {}
    for item in items:
        by_class.setdefault(item[-1], []).append(item)
    classes = sorted(by_class)
    if not classes and n:
        raise ValueError("cannot sample queries from an empty test set")
    base, remainder = divmod(n, len(classes)) if classes else (0, 0)
    counts = {label: base for label in classes}
    for label in rng.sample(classes, remainder):
        counts[label] += 1
    selected = []
    for label in classes:
        if counts[label] > len(by_class[label]):
            raise ValueError(
                f"class {label} has {len(by_class[label])} queries, needs {counts[label]}")
        selected.extend(rng.sample(by_class[label], counts[label]))
    rng.shuffle(selected)
    return selected


def image_id(image) -> str:
    """Return a stable identifier for a decoded image without storing its pixels."""
    digest = hashlib.sha256()
    digest.update(str(image.mode).encode("ascii"))
    digest.update(str(image.size).encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def seed_training(seed: int) -> None:
    """Seed stochastic adapter initialization and dropout reproducibly."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_training_episode(pool: dict, k: int, rng: random.Random,
                            class_policy: str = "random_balanced"):
    """Sample a training-pool query and demonstrations that exclude that object."""
    classes = list(pool)
    if not classes:
        raise ValueError("cannot sample a query from an empty pool")
    query_class = rng.choice(classes)
    query_image = rng.choice(pool[query_class])
    demo_pool = {
        c: [image for image in images if image is not query_image]
        for c, images in pool.items()
    }
    if not demo_pool[query_class] and k:
        raise ValueError("query class needs a second image for demonstration sampling")
    demos = sample_balanced_demos(demo_pool, k, rng, class_policy=class_policy)
    return query_image, query_class, demos


def sample_label_space(names: Sequence[str], condition: str, rng: random.Random,
                       mapping_mode: str = "episode", token_set: str = "eval") -> list[str]:
    """Return the class-index-to-output-label map for one episode."""
    if condition != "remap":
        return list(names)
    if token_set not in REMAP_TOKEN_SETS:
        raise ValueError(f"unknown remap token set: {token_set}")
    if mapping_mode not in {"episode", "fixed"}:
        raise ValueError(f"unknown mapping mode: {mapping_mode}")
    labels = list(REMAP_TOKEN_SETS[token_set][:len(names)])
    if len(labels) != len(names):
        raise ValueError(f"token set {token_set!r} is too small for {len(names)} classes")
    if mapping_mode == "episode":
        rng.shuffle(labels)
    return labels


def corrupt_demo_labels(demos: Sequence[tuple], classes: Iterable[int], rng: random.Random):
    """Apply a random derangement to shown demonstration labels.

    The image/class pairing is preserved in the first element of each tuple while
    the displayed label class changes. Unlike shuffling only observed labels, this
    remains a meaningful wrong-label control at K=1.
    """
    classes = list(classes)
    if len(classes) < 2:
        raise ValueError("shuffled-label control requires at least two classes")
    perm = classes[:]
    for _ in range(100):
        rng.shuffle(perm)
        if all(a != b for a, b in zip(classes, perm)):
            break
    else:  # deterministic rotation is a guaranteed derangement.
        perm = classes[1:] + classes[:1]
    mapping = dict(zip(classes, perm))
    return [(image, mapping[c]) for image, c in demos], mapping


def protocol_metadata(config: ProtocolConfig, **extra) -> dict:
    return {**config.metadata(), **extra}


def validate_evaluation_grid(seeds, ks, n: int, *, min_k: int = 0,
                             max_n: int = 120) -> None:
    """Reject empty/duplicate axes that would silently skip or reweight a study."""
    for name, values in (("seeds", seeds), ("ks", ks)):
        if not values or any(type(value) is not int for value in values):
            raise ValueError(f"{name} must contain integers")
        if len(set(values)) != len(values):
            raise ValueError(f"duplicate {name} would reweight the experiment")
    if any(k < min_k for k in ks):
        raise ValueError(f"all K values must be at least {min_k}")
    if type(n) is not int or not 0 < n <= max_n:
        raise ValueError(f"n must be between 1 and {max_n}")


def validate_named_axis(values, name: str, allowed) -> None:
    """Require a nonempty, unique, recognized set of requested cells."""
    if not values or len(set(values)) != len(values) or any(v not in allowed for v in values):
        raise ValueError(f"{name} must be nonempty, unique, and drawn from {sorted(allowed)}")
