"""Retired historical hosted-model probe; no API request is made by this module.

The original implementation is preserved at
reproducibility/legacy/b8421382d56e8aa7c12fff247008eae0116fb3fd65128ed511f14434d896a475/incontext_collapse/frontier.py.
Its class-prefix sampling, permissive substring scoring and aggregate-only
outputs cannot serve as a fresh reproduction of the current paper. Existing
hosted-model observations remain explicitly descriptive historical evidence.
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_known_args()
    parser.error("historical hosted probe retired; use python scripts/reproduce.py verify "
                 "to inspect retained scope evidence; no hosted rerun is registered")


if __name__ == "__main__":
    main()
