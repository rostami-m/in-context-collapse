"""Exploratory single-layer activation injection, not a validated CircA repair.

``inject`` averages raw last-prefill residual states over up to eight prompts,
using the first evaluation-query images without their labels. Each prompt gets
a sampled demonstration block. Extraction is transductive; no empty-prompt state
is subtracted. The resulting vector is added with unit gain at that same block's
last prefill position in a query-only prompt.

The ``gate`` serving mode keeps a fixed number of raw demonstrations alongside
the vector. It is not the paper's proposed copy-rate threshold policy. Neither
mode establishes positive context amortization or repair. ``manyshot`` and
``zeroshot`` are comparison modes under the fixed REMAP label mapping.
Optionally load a saved adapter with ``--vaccine PATH``.

The standalone CLI is retired; its original source is retained under
reproducibility/legacy. This module still exposes ``get_layers``, ``locus_idx``, ``extract_vector`` and
``eval_mode`` for reuse by ``compare``.
"""
from __future__ import annotations

import argparse
import json
import random

import torch

from .core import TASKS, REMAP, balanced_demos, build_messages, parse
from .models import load_model


def get_layers(model):
    """Return the language-model's transformer block list (the longest non-vision ModuleList)."""
    cands = [(n, m) for n, m in model.named_modules()
             if isinstance(m, torch.nn.ModuleList) and len(m) >= 8]
    lm = [c for c in cands if "language" in c[0].lower()]
    pick = lm or [c for c in cands if "vis" not in c[0].lower()] or cands
    pick.sort(key=lambda c: len(c[1]))
    return pick[-1][1]


def locus_idx(n_layers, locus):
    """Map a named locus to a block index."""
    return {"early": n_layers // 6, "mid": n_layers // 2, "late": n_layers - 1 - n_layers // 6}[locus]


@torch.no_grad()
def _generate(model, proc, messages, hook_layer=None, add_vec=None, grab=False,
              *, max_new_tokens=8):
    """Greedy decode, optionally grabbing or adding a residual-stream vector at ``hook_layer``."""
    inp = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                   return_dict=True, return_tensors="pt").to(model.device)
    n = inp["input_ids"].shape[1]
    store = {}
    handle = None
    if hook_layer is not None:
        def hook(mod, i, o):
            hs = o[0] if isinstance(o, tuple) else o
            if hs.dim() == 3 and hs.shape[1] > 1:  # prefill only
                if grab:
                    store["v"] = hs[:, -1, :].detach().clone()
                if add_vec is not None:
                    hs[:, -1, :] = hs[:, -1, :] + add_vec
            return o
        handle = hook_layer.register_forward_hook(hook)
    try:
        out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                             num_beams=1, num_return_sequences=1)
    finally:
        if handle is not None:
            handle.remove()
    txt = proc.batch_decode(out[:, n:], skip_special_tokens=True)[0]
    return (txt, store.get("v")) if grab else txt


def extract_vector(model, proc, pool, test, space, size, layer, k, seed, n=8):
    """Average the integration-locus prefill vector induced by k-shot demo blocks."""
    if n <= 0:
        raise ValueError("Vector extraction requires a positive sample count")
    sample = test[:n]
    if not sample:
        raise ValueError("Vector extraction requires at least one query image")
    rng = random.Random(seed)
    acc = []
    for index, (q_img, _) in enumerate(sample):
        _, v = _generate(model, proc, build_messages(balanced_demos(pool, k, rng), q_img, space, size),
                         hook_layer=layer, grab=True)
        if v is None:
            raise RuntimeError(f"No prefill state captured for extraction sample {index}")
        acc.append(v)
    return torch.stack(acc).mean(0)[0]


def eval_mode(model, proc, pool, test, names, k, size, mode,
              layer=None, vec=None, gate=None, seed=0):
    """Evaluate one serving mode on the REMAP condition."""
    space = [REMAP[c] for c in range(len(names))]
    rng = random.Random(seed)
    cor = 0
    for q_img, q_lab in test:
        if mode == "manyshot":
            msg = build_messages(balanced_demos(pool, k, rng), q_img, space, size)
            pred = parse(_generate(model, proc, msg), space)
        elif mode == "zeroshot":
            pred = parse(_generate(model, proc, build_messages([], q_img, space, size)), space)
        else:  # inject / gate: raw demos up to the gate, the vector for the rest
            kraw = 0 if gate is None else min(gate, k)
            msg = build_messages(balanced_demos(pool, kraw, rng), q_img, space, size)
            pred = parse(_generate(model, proc, msg, hook_layer=layer, add_vec=vec), space)
        cor += (pred == q_lab)
    return round(cor / len(test), 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_known_args()
    parser.error("standalone historical Inject CLI retired: its query-prefix sampling "
                 "and aggregate-only output do not meet current evaluation requirements. "
                 "The matched exploratory comparison remains in incontext_collapse.compare; "
                 "use its --preflight and a fresh output path. Neither is a validated CircA Gate.")


if __name__ == "__main__":
    main()
