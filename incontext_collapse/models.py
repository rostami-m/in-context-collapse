"""Pinned local model loading for architectures supported by this environment.

This AutoModel route covers the locally registered compatible families. It is
not a universal loader: InternVL3 uses its dedicated runner/environment, and
historical hosted-provider observations have a separate retained provenance.
Requested identities are exact; a missing or gated model is never substituted.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from .core import pick_device, pick_dtype
from .provenance import model_revision

def load_model(model_id: str, dtype=None, device: str | None = None):
    """Load a VLM and its processor onto the chosen device.

    Returns ``(model, processor)``. ``do_image_splitting`` is disabled where the
    processor supports it (LLaVA-NeXT / OneVision tiling) to keep the visual-token
    count bounded.
    """
    device = device or pick_device()
    dtype = dtype or pick_dtype(device)
    revision = model_revision(model_id)
    if revision is None:
        raise ValueError(f"{model_id}: register a verified model commit before loading")
    processor = AutoProcessor.from_pretrained(
        model_id, revision=revision, trust_remote_code=True, use_fast=False)
    try:
        processor.image_processor.do_image_splitting = False
    except Exception:
        pass
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, revision=revision, torch_dtype=dtype,
        trust_remote_code=True).to(device).eval()
    return model, processor


def load_model_with_fallback(model_id: str, device: str | None = None):
    """Compatibility name for exact loading; failures propagate without substitution.

    New callers should use ``load_model``. The three-element successful return
    is retained for external callers, but the requested model is the only load.
    """
    model, processor = load_model(model_id, device=device)
    return model, processor, model_id
