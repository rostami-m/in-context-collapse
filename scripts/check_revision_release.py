#!/usr/bin/env python3
"""Check local revision packaging after the authoritative LaTeX builds.

This verifies literal source references, evidence fingerprints, and build outputs;
it is not a substitute for scientific or visual review.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_revision_followup_queue import source_fingerprint as first_sources
from scripts.run_revision_coverage_queue import source_fingerprint as coverage_sources
from scripts.run_strengthening_queue import source_fingerprint as strengthening_sources


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_amendments(root):
    """Verify explicit maintenance deltas without rewriting execution provenance."""
    path = root / "reproducibility/runtime_amendments.json"
    if not path.exists():
        return {}
    manifest = json.loads(path.read_text())
    require(manifest.get("schema_version") == 1, "unknown runtime amendment schema")
    entries = {}
    for entry in manifest["amendments"]:
        name = entry["path"]
        require(name not in entries, "duplicate runtime amendment: " + name)
        require(bool(entry.get("reason")), "runtime amendment missing rationale")
        require(bool(entry.get("regression_tests")), "runtime amendment missing tests")
        checks = [(name, entry["current_sha256"]),
                  (entry["archived_source"], entry["recorded_sha256"]),
                  *entry["regression_tests"].items()]
        resolved = set()
        for relative, sha in checks:
            source = (root / relative).resolve()
            require(source.is_relative_to(root.resolve()), "runtime amendment path outside repository")
            require(source not in resolved, "colliding runtime amendment paths")
            resolved.add(source)
            require(source.is_file(), "missing runtime amendment input: " + relative)
            require(digest(source) == sha, "runtime amendment hash mismatch: " + relative)
        require(entry["current_sha256"] != entry["recorded_sha256"], "empty runtime amendment")
        entries[name] = entry
    return entries


def verify_recorded_sources(recorded, current, amendments):
    """Accept only identical sources or exact, separately verified archived deltas."""
    require(recorded.keys() == current.keys(), "runtime source manifest keys changed")
    for name, old in recorded.items():
        new = current[name]
        if isinstance(old, dict):
            require(isinstance(new, dict), "runtime source manifest structure changed")
            verify_recorded_sources(old, new, amendments)
        elif old != new:
            entry = amendments.get(name)
            require(entry is not None and old == entry["recorded_sha256"]
                    and new == entry["current_sha256"], "unrecorded runtime source change: " + name)


def manuscript_sources(paper, path, active=()):
    path = path.resolve()
    require(path.is_relative_to(paper.resolve()), "manuscript input outside paper directory")
    require(path not in active, "recursive manuscript input")
    require(path.is_file(), "missing manuscript input: " + str(path))
    source = path.read_text()
    rows = re.findall(r"\\inputrows\s+([^\s]+)", source)
    inputs = re.findall(r"\\input\{([^}]+)\}", source)
    texts, count = [source], len(rows)
    for name in inputs + rows:
        child = paper / (name if name.endswith(".tex") else name + ".tex")
        child_source, child_count = manuscript_sources(paper, child, (*active, path))
        texts.append(child_source)
        count += child_count
    return "\n".join(texts), count


def main():
    paper = ROOT / "paper"
    build = ROOT / ".smoke/manuscript-final"
    main_source = (paper / "Main.tex").read_text()
    response = (paper / "Response.tex").read_text()
    ledger = (ROOT / "docs/reviewer_claim_ledger.md").read_text()
    points = re.findall(r"\\point\{([^}]+)\}", response)
    ledger_points = re.findall(r"^\| (R[\w.]+) \|", ledger, re.MULTILINE)
    require(len(points) == len(set(points)) == 60, "response point count or uniqueness failed")
    require(set(points) == set(ledger_points), "response and claim ledger differ")

    source, row_count = manuscript_sources(paper, paper / "Main.tex")
    labels = re.findall(r"\\label\{([^}]+)\}", source)
    references = re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", source)
    require(len(labels) == len(set(labels)), "duplicate manuscript labels")
    require(set(references) <= set(labels), "unresolved literal manuscript reference")
    citations = {key for group in re.findall(r"\\cite\{([^}]+)\}", source) for key in group.split(",")}
    bibkeys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", (paper / "references.bib").read_text()))
    require(citations <= bibkeys, "missing bibliography key")

    figures = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", main_source)
    baseline = subprocess.check_output(["git", "-C", str(paper), "show", "3012f94:Main.tex"], text=True)
    previous = set(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", baseline))
    require(previous <= set(figures), "a preserved manuscript figure was removed")
    for name in figures:
        require((paper / name).is_file(), "missing figure: " + name)
        require(digest(paper / name) == digest(ROOT / name), "figure copies differ: " + name)
    frontier = subprocess.check_output(["git", "-C", str(paper), "show", "3012f94:figures/fig3_frontier_cliff.pdf"])
    require(hashlib.sha256(frontier).hexdigest() == digest(paper / "figures/fig3_frontier_cliff.pdf"),
            "unreconstructable hosted figure changed")

    amendments = runtime_amendments(ROOT)
    stages = 0
    for directory, fingerprint in (("queue", first_sources), ("coverage_queue", coverage_sources)):
        state = json.loads((ROOT / "data/revision_v25" / directory / "status.json").read_text())
        require(state["status"] == "complete", "queue incomplete: " + directory)
        verify_recorded_sources(state["source_sha256"], fingerprint(), amendments)
        stages += len(state["stages"])
    require(stages == 18, "unexpected experiment stage count")
    subprocess.run([sys.executable, "scripts/export_revision_tables.py", "--check"], cwd=ROOT, check=True)
    strengthening = json.loads((ROOT / "data/strengthening_v1/queue/status.json").read_text())
    require(strengthening["status"] == "complete", "strengthening queue incomplete")
    verify_recorded_sources(strengthening["source_sha256"], strengthening_sources(), amendments)
    require(len(strengthening["stages"]) == 5, "unexpected strengthening stage count")
    subprocess.run([sys.executable, "scripts/export_strengthening_tables.py", "--check"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "scripts/export_region_aggregate_tables.py", "--check"], cwd=ROOT, check=True)

    import pymupdf
    pdfs = {}
    for name in ("Main", "Response"):
        log = (build / (name + ".log")).read_text(errors="replace")
        require(not re.search(r"(?:Reference|Citation) .*undefined|multiply defined|Undefined control sequence", log),
                name + " has unresolved LaTeX references or commands")
        path = build / (name + ".pdf")
        with pymupdf.open(path) as pdf:
            require(all(page.get_text().strip() for page in pdf), name + " has a blank page")
            pdfs[name] = {"pages": len(pdf), "sha256": digest(path)}
    print(json.dumps({"checks": "passed", "stages": stages, "response_points": len(points),
                      "strengthening_stages": len(strengthening["stages"]),
                      "figures": len(figures), "preserved_figures": len(previous),
                      "post_experiment_runtime_amendments": list(amendments),
                      "table_inputs": row_count, "pdfs": pdfs,
                      "scope": "local evidence/source/build checks; not journal acceptance or remote compile verification"}, indent=2))


if __name__ == "__main__":
    from scripts.reproduce import main as reproduce_main
    sys.argv = [sys.argv[0], "verify", *sys.argv[1:]]
    raise SystemExit(reproduce_main())
