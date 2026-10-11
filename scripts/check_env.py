#!/usr/bin/env python3
"""Read-only check of the exact primary Python environment and available device."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock', type=Path, default=ROOT / 'requirements.lock')
    args = parser.parse_args()
    errors, versions = [], {}
    for line in args.lock.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        name, required = line.strip().split('==')
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        versions[name] = actual
        if actual != required:
            errors.append(f'{name}: expected {required}, found {actual}')
    if sys.version_info[:2] != (3, 12):
        errors.append('recorded primary environment requires Python 3.12')
    device = 'unavailable'
    try:
        import torch
        device = ('cuda' if torch.cuda.is_available() else 'mps' if
                  torch.backends.mps.is_available() else 'cpu')
    except (ImportError, OSError) as error:
        errors.append(str(error))
    print(json.dumps({'python': platform.python_version(), 'platform': platform.platform(),
                      'device': device, 'packages': versions, 'errors': errors,
                      'status': 'passed' if not errors else 'failed',
                      'scope': 'Package/device check; does not certify model caches or reproduce a study.'}, indent=2))
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
