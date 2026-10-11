#!/usr/bin/env python3
"""Local filesystem storage used by the experiment runners."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


def _path(uri: str | Path) -> Path:
    value = str(uri)
    if value.startswith("file://"):
        value = value[len("file://"):]
    if "://" in value:
        raise ValueError("RESULTS_DIR must be a local path")
    return Path(value)


def join(base: str, *parts: str) -> str:
    return str(_path(base).joinpath(*parts))


def exists(uri: str) -> bool:
    return _path(uri).exists()


def makedirs(uri: str) -> None:
    _path(uri).mkdir(parents=True, exist_ok=True)


def open_file(uri: str, mode: str = "rb"):
    path = _path(uri)
    if any(c in mode for c in "wax"):
        path.parent.mkdir(parents=True, exist_ok=True)
    return path.open(mode)


def read_text(uri: str) -> str:
    with open_file(uri, "rb") as fh:
        return fh.read().decode("utf-8")


def write_text(uri: str, text: str) -> None:
    with open_file(uri, "wb") as fh:
        fh.write(text.encode("utf-8"))


def append_jsonl(uri: str, record: dict) -> None:
    with open_file(uri, "a") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def read_jsonl(uri: str) -> list[dict]:
    if not exists(uri):
        return []
    return [json.loads(line) for line in read_text(uri).splitlines() if line.strip()]


def listdir(uri: str, suffix: str = "") -> list[str]:
    base = _path(uri)
    if not base.is_dir():
        return []
    return [str(path) for path in sorted(base.iterdir()) if path.name.endswith(suffix)]


def get(src_uri: str, dst_path: str | Path) -> Path:
    src = _path(src_uri)
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        (shutil.copytree if src.is_dir() else shutil.copy2)(src, dst)
    return dst


def put(src_path: str | Path, dst_uri: str) -> str:
    src = Path(src_path)
    dst = _path(dst_uri)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    return str(dst)


def results_dir() -> str:
    return os.environ.get("RESULTS_DIR", "./results")


def check(uri: str) -> int:
    probe = None
    try:
        makedirs(uri)
        # Never overwrite/remove a caller's .storage_check or another process's probe.
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=".storage_check-",
                                         dir=_path(uri), delete=False) as handle:
            probe = Path(handle.name)
            handle.write("ok\n")
        got = probe.read_text(encoding="utf-8").strip()
        if got != "ok":
            print(f"FAIL {uri}: wrote 'ok', read back {got!r}", file=sys.stderr)
            return 1
    except Exception as exc:
        print(f"FAIL {uri}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if probe is not None:
            probe.unlink(missing_ok=True)
    print(f"OK   {uri}  (local, readable and writable)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", metavar="PATH", default=None)
    args = ap.parse_args()
    raise SystemExit(check(args.check or results_dir()))
