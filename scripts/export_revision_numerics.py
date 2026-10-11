#!/usr/bin/env python3
"""Export deterministic, complete-artifact numerical rows from the audited report."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def export(report, source_sha256):
    artifacts = report["artifacts"]
    if not artifacts or any(artifact["status"] != "complete" for artifact in artifacts):
        raise ValueError("numerical dossier requires complete audited artifacts only")
    lines = ["COMPLETED REVISION NUMERICAL DOSSIER", f"Combined report SHA256: {source_sha256}",
             "Means +/- pointwise 95% Student-t half-widths across three recorded seed means.",
             "Paired contrasts are right minus left within artifact/task/condition/K only.",
             "No cross-file rankings, equivalence claims, or multiplicity adjustment are implied.",
             "All evaluations reuse fitted checkpoints; DARA replay provenance is not new training.",
             "Four decimals are retained here to avoid masking small effects or rounding ties.", ""]
    for artifact in artifacts:
        lines += [Path(artifact["path"]).name, "=" * 78,
                  f"Artifact SHA256: {artifact['sha256']}"]
        shots = sorted({row["k"] for row in artifact["estimates"]})
        lines.append("ESTIMATES: task | condition | option_order | arm | metric | "
                     + " | ".join(f"K={k}" for k in shots))
        rows = defaultdict(dict)
        for row in artifact["estimates"]:
            for metric, estimate in row["metrics"].items():
                key = tuple(row[field] for field in ("task", "condition", "option_order", "arm")) + (metric,)
                half = ((estimate["ci95_high"] - estimate["ci95_low"]) / 2
                        if estimate["ci95_low"] is not None else None)
                rows[key][row["k"]] = (f"{estimate['mean']:.4f} +/- {half:.4f}" if half is not None
                                       else f"{estimate['mean']:.4f} [CI unavailable]")
        for key, values in sorted(rows.items()):
            lines.append(" | ".join((*key, *(values.get(k, "--") for k in shots))))
        lines += ["", "PAIRED ACCURACY CONTRASTS: task | condition | left(order,arm) -> right(order,arm) | "
                  + " | ".join(f"K={k}: delta [CI low, high]" for k in shots)]
        contrasts = defaultdict(dict)
        for row in artifact["paired_comparisons"]:
            left, right = row["left"], row["right"]
            key = (left["task"], left["condition"],
                   f"({left['option_order']},{left['arm']}) -> ({right['option_order']},{right['arm']})")
            estimate = row["metrics"]["accuracy"]
            contrasts[key][left["k"]] = (
                f"{estimate['mean']:+.4f} [{estimate['ci95_low']:+.4f}, {estimate['ci95_high']:+.4f}]"
                if estimate["ci95_low"] is not None else f"{estimate['mean']:+.4f} [CI unavailable]")
        for key, values in sorted(contrasts.items()):
            lines.append(" | ".join((*key, *(values.get(k, "--") for k in shots))))
        for row in artifact["paired_comparisons"]:
            if "paired_image_bootstrap" in row:
                interval = row["paired_image_bootstrap"]
                lines.append(f"IMAGE BOOTSTRAP {row['right']['arm']} minus {row['left']['arm']}, "
                             f"K={row['left']['k']}: {interval['mean']:+.6f} "
                             f"[{interval['ci95_low']:+.6f}, {interval['ci95_high']:+.6f}]; "
                             f"n_images={interval['n_images']}, fixed seeds={interval['n_fixed_seeds']}; "
                             "conditional on checkpoints and sampled frame")
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.report.resolve() == args.out.resolve():
        parser.error("output must differ from input")
    if args.out.exists():
        parser.error("output already exists; choose a fresh path")
    content = args.report.read_bytes()
    dossier = export(json.loads(content), hashlib.sha256(content).hexdigest())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(dossier)
    print(json.dumps({"path": str(args.out.resolve()),
                      "sha256": hashlib.sha256(dossier.encode()).hexdigest()}))


if __name__ == "__main__":
    main()
