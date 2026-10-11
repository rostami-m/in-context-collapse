"""Versioned, configuration-safe JSON result persistence."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def write_json_atomic(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Serialize before opening a temporary file, so nonfinite statistics leave
    # any valid result untouched. A unique file prevents writers sharing .tmp.
    serialized = json.dumps(payload, indent=2, allow_nan=False)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def ensure_run_config(path, config: dict) -> None:
    """Create a manifest or reject an attempt to mix incompatible run settings."""
    path = Path(path)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != config:
            raise RuntimeError(
                f"result configuration mismatch at {path}; choose a new output path")
        return
    write_json_atomic(path, config)


def load_result(path, config: dict, initial: dict) -> dict:
    path = Path(path)
    if not path.exists():
        return {"run_config": config, **initial}
    result = json.loads(path.read_text())
    if result.get("run_config") != config:
        raise RuntimeError(
            f"result configuration mismatch at {path}; choose a new output path")
    return result
