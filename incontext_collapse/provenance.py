"""Pinned external revisions and compact runtime provenance."""
from __future__ import annotations

from importlib.metadata import version


MODEL_REVISIONS = {
    "OpenGVLab/InternVL3-2B-hf": "cb57a075cb75a2e6d1b668b128d48bb00ae321d2",
    "Qwen/Qwen2-VL-2B-Instruct": "895c3a49bc3fa70a340399125c650a463535e71c",
    "Qwen/Qwen2-VL-7B-Instruct": "eed13092ef92e448dd6875b2a00151bd3f7db0ac",
    "Qwen/Qwen2.5-VL-3B-Instruct": "66285546d2b821cf421d4f5eb2576359d3770cd3",
    "Qwen/Qwen2.5-VL-7B-Instruct": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
}

DATASET_REVISIONS = {
    "lmms-lab/VQAv2": "32665d35052eb4a6d4414851c3c829a72754915a",
    "lmms-lab/textvqa": "9c0699cd19768ac5ab97568f6b3cbac4c0062884",
    # Verified official HF API 2026-10-11; prospective pin, not historical provenance.
    "uoft-cs/cifar100": "aadb3af77e9048adbea6b47c21a81e47dd092ae5",
    "uoft-cs/cifar10": "0b2714987fa478483af9968de7c934580d0bb9a2",
    "zalando-datasets/fashion_mnist": "531be5e2ccc9dba0c201ad3ae567a4f3d16ecdd2",
}


def model_revision(model_id: str) -> str | None:
    return MODEL_REVISIONS.get(model_id)


def software_versions() -> dict[str, str]:
    return {
        package: version(package)
        for package in ("torch", "transformers", "peft", "datasets", "Pillow")
    }


def validate_dataset_cache_revision(builder, dataset_id: str, revision: str) -> None:
    """Reject unpinned HF offline fallback before any dataset rows are returned.

    In datasets 5.0.1, the Cache builder selects the latest cached configuration
    even when load_dataset_builder receives a revision. Its ``hash`` is the
    selected cache-directory revision. Other builder hashes can describe loading
    code or data transformations, so they are not equated to repository commits.
    This checks cache selection identity, not checksums of the Arrow contents.
    """
    from datasets.packaged_modules.cache.cache import Cache

    if isinstance(builder, Cache):
        resolved = getattr(builder, "hash", None)
        if resolved != revision:
            raise ValueError(
                f"{dataset_id}: cached dataset revision {resolved!r} does not "
                f"match pinned revision {revision!r}; refusing offline fallback")


def load_pinned_dataset(dataset_id: str, name=None, *, split=None,
                        streaming=False, cache_dir=None):
    """Load the repository's pinned dataset through one validated HF builder.

    This intentionally supports only the dataset-loading arguments used by this
    codebase. The preparation/streaming behavior follows datasets 5.0.1's
    load_dataset defaults. An unregistered dataset fails before Hub/cache access.
    """
    import re
    from datasets import DownloadMode, VerificationMode, load_dataset_builder
    from datasets.utils.info_utils import is_small_dataset

    revision = DATASET_REVISIONS.get(dataset_id)
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError(f"{dataset_id}: a verified full dataset revision is required")
    builder = load_dataset_builder(dataset_id, name, revision=revision,
                                   cache_dir=cache_dir)
    validate_dataset_cache_revision(builder, dataset_id, revision)
    if streaming:
        return builder.as_streaming_dataset(split=split)
    builder.download_and_prepare(
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
        verification_mode=VerificationMode.BASIC_CHECKS)
    return builder.as_dataset(split=split,
                              in_memory=is_small_dataset(builder.info.dataset_size))
