#!/usr/bin/env python3
"""Render audited and labeled historical figures into an explicit new directory.

Usage: python scripts/make_figures.py --output-dir NEW_DIRECTORY
Arguments are forwarded to incontext_collapse.figures. This wrapper neither
changes paper assets nor launches model inference. Temporary cache paths use
the host platform's standard temporary directory.
"""
from __future__ import annotations
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent


def main():
    with tempfile.TemporaryDirectory(prefix="incontext-figure-cache-") as cache:
        environment = dict(os.environ, MPLBACKEND="Agg", MPLCONFIGDIR=cache,
                           XDG_CACHE_HOME=cache)
        # Moving plotting caches must not hide the user's existing pinned datasets.
        environment.setdefault("HF_HOME", str(Path(os.environ.get(
            "XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "huggingface"))
        return subprocess.run([sys.executable, "-m", "incontext_collapse.figures", *sys.argv[1:]],
                              cwd=ROOT, env=environment).returncode


if __name__ == "__main__":
    raise SystemExit(main())
