#!/usr/bin/env python
"""Render the paper figures from result JSONs produced by the experiment drivers.

Reads the per-cell panel results in ``data/panel_cells`` and the aggregated result
JSONs in ``data/figdata``; writes only to an explicit new output directory.
No model weights or GPU are loaded. The Python runtime still needs the scoring
and audit dependencies; qualitative thumbnails also require cached pinned
CIFAR-10/Fashion-MNIST data (downloads require --allow-downloads).
Run ``python -m incontext_collapse.figures --output-dir NEW_DIRECTORY``.
The current paper preserves historical scope figures; copying an unreconstructable
asset is labeled separately from regenerating an estimate.
"""
import argparse
import hashlib
import os
import re
import json, glob, collections
import shutil
from functools import lru_cache
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .evaluation import mean_t_ci

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "data" / "panel_cells"
FD = ROOT / "data" / "figdata"
OUT = None  # Only main(--output-dir NEW_DIRECTORY) selects a write destination.
GENERATED_PDFS = []

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9, "axes.titleweight": "bold",
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 6.8, "legend.frameon": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5, "grid.color": "#aaaaaa",
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight", "lines.linewidth": 1.8, "lines.markersize": 4,
})
PRONE = ["#D62728", "#E8813A", "#e377c2"]          # warm
IMMUNE = ["#2CA02C", "#1F77B4", "#17BECF", "#9467BD"]  # cool
TEAL, RED, GREY = "#117733", "#D62728", "#999999"


def save_figure(fig, filename, **kwargs):
    """Record only outputs produced in this invocation for publication copying."""
    if OUT is None:
        raise RuntimeError("select an explicit fresh output directory before rendering")
    path = OUT / filename
    if Path(filename).name != filename or path.suffix != ".pdf" or path.is_symlink():
        raise ValueError(f"unexpected figure output: {path}")
    # Exclusive creation: even a file created concurrently is never overwritten.
    with path.open("xb") as handle:
        fig.savefig(handle, format="pdf", **kwargs)
    GENERATED_PDFS.append(path)


def sync_paper_figures():
    raise RuntimeError("implicit paper synchronization is retired; review generated assets separately")


def _load_tasks():
    """Deterministic stimuli for the qualitative figures (no model/GPU needed).
    shapes is pure PIL; CIFAR-4/Fashion-4 are small HF downloads."""
    from incontext_collapse import core as C
    rng = __import__("random").Random(0)
    out = {}
    for key, ttl in [("shapes4", "Synthetic shapes"), ("cifar4", "CIFAR-4"), ("fashion4", "Fashion-4")]:
        names, pool, test = C.TASKS[key](8, 8)
        ex = {c: pool[c][0] for c in pool}            # one canonical image per class
        remap = [C.REMAP[c] for c in range(len(names))]
        out[key] = {"title": ttl, "names": names, "remap": remap, "examples": ex, "pool": pool, "test": test}
    return out

def vcurve(model, task, cond):
    d = collections.defaultdict(list)
    for f in glob.glob(str(V2 / f"{model}__{task}__{cond}__K*__s*.json")):
        j = json.load(open(f)); d[j["K"]].append(j["acc"])
    if not d:
        raise ValueError(f"missing historical panel curve: {model}/{task}/{cond}")
    return {k: float(np.mean(v)) for k, v in sorted(d.items())}

def load(name):
    p = FD / f"{name}.json"
    return json.load(open(p)) if p.exists() else None

# ---------------------------------------------------------------- Fig 1: cliff + dissociation
def fig_cliff():
    # Color and marker identify the model, without assigning a categorical regime.
    models = [("Qwen2-VL-2B-Instruct", "Qwen2-VL-2B"), ("Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-3B"),
             ("llava-v1.6-mistral-7b-hf", "LLaVA-1.6-Mistral-7B"), ("Llama-3.2-11B-Vision-Instruct", "Llama-3.2-11B"),
             ("llava-onevision-qwen2-0.5b-ov-hf", "LLaVA-OV-0.5B"), ("llava-1.5-7b-hf", "LLaVA-1.5-7B"),
             ("SmolVLM-Instruct", "SmolVLM-2B"), ("SmolVLM-500M-Instruct", "SmolVLM-500M"),
              ("Qwen2-VL-7B-Instruct", "Qwen2-VL-7B"), ("Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-7B"),
              ("llava-onevision-qwen2-7b-ov-hf", "LLaVA-OV-7B")]
    import matplotlib.cm as cm
    from matplotlib.patches import ConnectionPatch
    colors = cm.tab20(np.linspace(0.0, 1.0, len(models)))
    markers = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "*")
    ZOOM = 16  # the crowded early-K region to magnify in the bottom row
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.2),
                           gridspec_kw={"height_ratios": [1.0, 0.85]})
    def draw(a, cond, zoom):
        for (m, lab), c, marker in zip(models, colors, markers):
            cu = vcurve(m, "cifar4", cond)
            if not cu: continue
            xs = [k for k in cu if (k <= ZOOM if zoom else True)]
            a.plot(xs, [cu[k] for k in xs], marker=marker, color=c, ms=3, lw=1.3, label=lab)
        a.axhline(0.25, ls=":", c=GREY, lw=1); a.set_ylim(-0.02, 1.02)
    titles = [("standard", "Standard-label accuracy", "Accuracy"),
              ("remap", "Fixed-remap accuracy", "Accuracy")]
    for col, (cond, ttl, ylab) in enumerate(titles):
        top, bot = ax[0][col], ax[1][col]
        draw(top, cond, zoom=False); draw(bot, cond, zoom=True)
        top.set_title(ttl); top.set_ylabel(ylab); bot.set_ylabel(ylab)
        bot.set_xlabel("demonstrations $K$ (zoom: $0 \\leq K \\leq 16$)")
        top.set_xlabel("demonstrations $K$ (full range)")
        bot.set_xlim(-0.5, ZOOM + 0.5)
        top.axvspan(0, ZOOM, color="#cccccc", alpha=0.25, zorder=0)
        top.annotate("magnified below", xy=(ZOOM/2, 0.02), xytext=(ZOOM/2, 0.02),
                     fontsize=6, color="#555555", ha="center", va="bottom")
        for xz in (0, ZOOM):
            con = ConnectionPatch(xyA=(xz, -0.02), coordsA=top.transData,
                                  xyB=(xz, 1.02), coordsB=bot.transData,
                                  color="#999999", lw=0.6, ls="--")
            fig.add_artist(con)
    ax[0][0].text(0.02, 0.27, "chance", color=GREY, fontsize=6.5, transform=ax[0][0].get_yaxis_transform())
    h, l = ax[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=6.0, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=[0, 0.07, 1, 1]); save_figure(fig, "fig1_cliff_regimes.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig 2: region-restricted repair (headline)
def fig_circuit():
    regions = ["connector", "early", "mid", "late", "all"]
    colors = [TEAL, TEAL, TEAL, RED, GREY]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for axi, (name, ttl) in enumerate([("circuit_shapes", "Synthetic shapes (contamination-free)"),
                                       ("circuit_cifar", "CIFAR-4")]):
        d = load(name); a = ax[axi]
        base = d["baseline"]["remap"]["16"]
        vals = [d[f"{r}:both"]["remap"]["16"] for r in regions]
        x = np.arange(len(regions))
        # The preserved breadth summary lacks the per-seed records required
        # to recompute a valid small-sample interval.
        a.bar(x, vals, color=colors, edgecolor="k", linewidth=0.5)
        a.axhline(base, ls="--", c="k", lw=1); a.axhline(0.25, ls=":", c=GREY, lw=1)
        a.text(len(regions)-0.5, base+0.01, "baseline", fontsize=6.5, ha="right")
        a.set_xticks(x); a.set_xticklabels(["conn.", "early", "mid", "late", "all"])
        a.set_title(ttl); a.set_ylabel("Fixed-remap accuracy @ $K{=}16$"); a.set_ylim(0, 1.05)
    fig.suptitle("Exploratory region sweep: natural-label training", fontsize=9)
    fig.text(.5, .01, "all: first 64 targets; unmatched budgets; parse diagnostics unavailable", ha="center", fontsize=6.8)
    fig.tight_layout(rect=[0, .04, 1, .94]); save_figure(fig, "fig2_causal_circuit.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig 3: descriptive hosted-model probe
def fig_frontier():
    # Two descriptive panels: (a) standard robustness, (b) remap learning.
    cl_s = load("frontier_claude")              # standard (flat {k:acc})
    cl_r = load("frontier_claude_remap")        # remap ({mean:{k:acc}})
    nova_s, nova_r = load("frontier_nova"), load("frontier_nova_remap")
    if not all((cl_s, cl_r, nova_s, nova_r)):
        print("fig_frontier: complete source aggregates unavailable; preserving existing four-model PDF")
        return
    cl_s, cl_r = {**nova_s, **cl_s}, {**nova_r, **cl_r}
    series = [("Nova-Lite", "#1F77B4"), ("Nova-Pro", "#9467BD"),
              ("Claude-Haiku-4.5", IMMUNE[2]), ("Claude-Sonnet-4.5", RED)]
    if any(name not in source for name, _ in series for source in (cl_s, cl_r)):
        print("fig_frontier: incomplete four-model source data; preserving existing PDF")
        return
    def curve(src, name):
        if not src or name not in src: return None
        v = src[name]
        if isinstance(v, dict) and "error" in v: return None
        d = v["mean"] if isinstance(v, dict) and "mean" in v else v  # remap vs flat-standard
        ks = sorted(int(k) for k in d)
        return ks, [d[str(k)] for k in ks]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for axi, (cl, ttl) in enumerate([(cl_s, "Standard-label accuracy"),
                                      (cl_r, "Fixed-remap accuracy")]):
        a = ax[axi]
        for name, c in series:
            cu = curve(cl, name)
            if not cu: continue
            sty = "-D" if c == RED else "-o"
            a.plot(cu[0], cu[1], sty, color=c, label=name, lw=2.2 if c == RED else 1.6, ms=4)
        a.axhline(0.25, ls=":", c=GREY, lw=1); a.set_title(ttl)
        a.set_xlabel("demonstrations $K$"); a.set_ylim(0, 1.05)
    ax[0].set_ylabel("Accuracy (CIFAR-4)")
    ax[0].text(0.02, 0.27, "chance", color=GREY, fontsize=6.5, transform=ax[0].get_yaxis_transform())
    ax[1].legend(loc="upper left", fontsize=6.5)
    fig.tight_layout(); save_figure(fig, "fig3_frontier_cliff.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig 4: integration != consolidation
def fig_consol():
    d = load("continual"); regs = ["circuit", "all", "late"]; labs = ["conn.+early/mid", "all", "late"]
    aa = [d[r]["AA"] for r in regs]; aasd = [d[r]["AA_sd"] for r in regs]
    bwt = [d[r]["BWT"] for r in regs]; bsd = [d[r]["BWT_sd"] for r in regs]
    colors = [TEAL, GREY, RED]
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 3.0)); x = np.arange(3)
    ax[0].bar(x, aa, yerr=aasd, capsize=3, color=colors, edgecolor="k", linewidth=0.5)
    ax[0].set_xticks(x); ax[0].set_xticklabels(labs); ax[0].set_ylabel("Average accuracy $\\uparrow$")
    ax[0].set_ylim(0.6, 0.75); ax[0].set_title("Continual-learning accuracy")
    ax[1].bar(x, bwt, yerr=bsd, capsize=3, color=colors, edgecolor="k", linewidth=0.5)
    ax[1].set_xticks(x); ax[1].set_xticklabels(labs); ax[1].set_ylabel("Backward transfer $\\uparrow$")
    ax[1].set_title("Forgetting (less negative = better)")
    fig.suptitle("Sequential four-way tasks with reused output labels", fontsize=9)
    fig.text(.5, .01, "Reported across-seed SD; three seeds; unmatched adapter capacities", ha="center", fontsize=6.8)
    fig.tight_layout(rect=[0, .05, 1, .93]); save_figure(fig, "fig4_integration_vs_consolidation.pdf"); plt.close(fig)

def ccurve(model, task, cond):
    d = collections.defaultdict(list)
    for f in glob.glob(str(V2 / f"{model}__{task}__{cond}__K*__s*.json")):
        j = json.load(open(f)); d[j["K"]].append(j.get("copy_rate", 0.0))
    return {k: float(np.mean(v)) for k, v in sorted(d.items())}

# ---------------------------------------------------------------- Fig 0: concept schematic
def fig_concept():
    import matplotlib.patches as mp
    fig, a = plt.subplots(figsize=(7.2, 2.5)); a.axis("off"); a.set_xlim(0, 10.7); a.set_ylim(0, 3)
    boxes = [("Image", 0.1, "#ececec"), ("Vision\nencoder", 1.5, "#dde6f2"), ("Connector", 3.2, "#cdebd6"),
             ("Early/mid LM\n(candidate bands)", 4.9, "#dde6f2"), ("Late LM\n(candidate band)", 7.1, "#dde6f2"),
             ("Label", 9.2, "#ececec")]
    w, y, h = 1.35, 1.45, 0.95
    for txt, x, col in boxes:
        a.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=col, ec="k", lw=0.8))
        a.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=7.3)
    for i in range(len(boxes) - 1):
        a.annotate("", xy=(boxes[i+1][1], y + h/2), xytext=(boxes[i][1] + w, y + h/2),
                   arrowprops=dict(arrowstyle="->", lw=1))
    a.plot([3.2, 8.45], [1.28, 1.28], color=TEAL, lw=2.5)
    a.text(5.83, 1.12, "Compare connector and language-layer repair", ha="center", va="top", color=TEAL, fontsize=7)
    a.plot([4.9, 8.45], [2.55, 2.55], color="#446688", lw=2.5)
    a.text(6.68, 2.7, "Matched-budget depth comparison", ha="center", color="#446688", fontsize=7)
    a.text(5.2, .45, "Repair sufficiency does not identify the frozen failure site", ha="center", color="#444", fontsize=7)
    fig.tight_layout(); save_figure(fig, "fig0_concept.pdf"); plt.close(fig)

# ---------------------------------------------------------------- ED: attn/MLP component cut
def fig_component():
    d = load("circuit_shapes"); regs = ["early", "mid", "late"]
    attn = [d[f"{r}:attn"]["remap"]["16"] for r in regs]
    mlp = [d[f"{r}:mlp"]["remap"]["16"] for r in regs]
    both = [d[f"{r}:both"]["remap"]["16"] for r in regs]
    x = np.arange(len(regs)); wdt = 0.27
    fig, a = plt.subplots(figsize=(4.2, 3.0))
    a.bar(x - wdt, attn, wdt, label="attention", color="#1F77B4", edgecolor="k", lw=0.4)
    a.bar(x, mlp, wdt, label="MLP", color="#117733", edgecolor="k", lw=0.4)
    a.bar(x + wdt, both, wdt, label="both", color="#999999", edgecolor="k", lw=0.4)
    a.axhline(0.25, ls=":", c=GREY); a.set_xticks(x); a.set_xticklabels(regs)
    a.set_ylabel("Fixed-remap accuracy @ $K{=}16$"); a.set_ylim(0, 1.05)
    a.set_title("Exploratory component sweep (Shapes)"); a.legend()
    fig.text(.5, .01, "Natural-label training; budgets and parsing uncontrolled", ha="center", fontsize=6.3)
    fig.tight_layout(rect=[0, .05, 1, 1]); save_figure(fig, "fig5_component_cut.pdf"); plt.close(fig)

# ---------------------------------------------------------------- ED: cross-arch InternVL3
def fig_crossarch():
    # Exploratory natural-label training and fixed-remap evaluation; budgets differ.
    panels = [("circuit_cifar", "Qwen2-VL-2B (MLP)"),
              ("circuit_cifar_qwen25_3b", "Qwen2.5-VL-3B (MLP)"),
              ("crossarch_internvl3", "InternVL3-2B\n(pixel shuffle + MLP)")]
    regs = ["connector", "early", "mid", "late"]
    colors = [TEAL, TEAL, TEAL, RED]
    loaded = [(load(f), lab) for f, lab in panels]
    if any(d is None for d, _ in loaded):
        print("fig_crossarch: missing lesion data", [f for f, _ in panels]); return
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.8), sharey=True)
    for i, (d, lab) in enumerate(loaded):
        a = ax[i]; x = np.arange(len(regs))
        vals = [d[f"{r}:both"]["remap"]["16"] for r in regs]
        base = d["baseline"]["remap"]["16"]
        # Retained legacy means are descriptive; formal uncertainty is omitted
        # because the shipped summary lacks per-seed records.
        a.bar(x, vals, color=colors, edgecolor="k", lw=0.5)
        a.axhline(base, ls="--", c="k", lw=1); a.axhline(0.25, ls=":", c=GREY)
        a.text(len(regs)-0.5, base+0.012, "baseline", ha="right", fontsize=6.2)
        a.set_xticks(x); a.set_xticklabels(["conn.", "early", "mid", "late"], fontsize=7)
        a.set_title(lab, fontsize=8.5); a.set_ylim(0, 1.05)
    ax[0].set_ylabel("Fixed-remap accuracy @ $K{=}16$")
    from matplotlib.patches import Patch
    ax[2].legend(handles=[Patch(fc=TEAL, ec="k", label="conn./early/mid"),
                          Patch(fc=RED, ec="k", label="late band")],
                 fontsize=6.3, loc="upper right")
    fig.suptitle("Exploratory region sweep: natural-label training", fontsize=9)
    fig.text(.5, .01, "Different adapter budgets; parse diagnostics unavailable", ha="center", fontsize=6.8)
    fig.tight_layout(rect=[0, .05, 1, .92]); save_figure(fig, "fig6_crossarch.pdf"); plt.close(fig)

# ---------------------------------------------------------------- ED: content-vs-format control
def fig_control():
    m = "Qwen2-VL-2B-Instruct"
    std = vcurve(m, "cifar4", "standard"); shuf = vcurve(m, "cifar4", "shuffled"); cr = ccurve(m, "cifar4", "standard")
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0))
    ax[0].plot(list(std), list(std.values()), "-o", color="#1F77B4", label="standard labels")
    ax[0].plot(list(shuf), list(shuf.values()), "-s", color="#E8813A", label="shuffled labels")
    ax[0].set_title("Content vs. format (Qwen2-VL-2B)"); ax[0].set_xlabel("demonstrations $K$")
    ax[0].set_ylabel("Accuracy"); ax[0].set_ylim(0, 1.02); ax[0].legend()
    ax[1].plot(list(cr), list(cr.values()), "-^", color="#D62728")
    ax[1].set_title("Copy rate rises with $K$"); ax[1].set_xlabel("demonstrations $K$")
    ax[1].set_ylabel("P(predict last-demo label)"); ax[1].set_ylim(0, 1.02)
    fig.tight_layout(); save_figure(fig, "fig7_content_format.pdf"); plt.close(fig)

def fig_failuremodes():
    """Two modes of collapse: clean reasoning collapse (parseable, wrong) vs output
    degeneration (unparseable). Defends against the 'it's just formatting' reading."""
    def series(m, t, c):
        acc = collections.defaultdict(list); unp = collections.defaultdict(list); nn = collections.defaultdict(list)
        for f in glob.glob(str(V2 / f"{m}__{t}__{c}__K*__s*.json")):
            j = json.load(open(f)); acc[j["K"]].append(j["acc"])
            unp[j["K"]].append(j.get("unparsed", 0)); nn[j["K"]].append(j["n"])
        ks = sorted(acc)
        return ks, [float(np.mean(acc[k])) for k in ks], [float(np.mean(unp[k]) / np.mean(nn[k])) for k in ks]
    rows = [("Qwen2-VL-2B-Instruct", "Qwen2-VL-2B", RED, "-o"),
            ("llava-onevision-qwen2-0.5b-ov-hf", "LLaVA-OV-0.5B", "#E8813A", "-s"),
            ("SmolVLM-500M-Instruct", "SmolVLM-500M", "#e377c2", "-^")]
    fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.0))
    for m, lab, c, sty in rows:
        ks, acc, unp = series(m, "cifar4", "standard")
        if not ks: continue
        ax[0].plot(ks, acc, sty, color=c, label=lab)
        ax[1].plot(ks, unp, sty, color=c, label=lab)
    ax[0].axhline(0.25, ls=":", c=GREY, lw=1)
    ax[0].text(0.5, 0.27, "chance", color=GREY, fontsize=6.5)
    ax[0].set_title("Accuracy collapses"); ax[0].set_xlabel("demonstrations $K$")
    ax[0].set_ylabel("Accuracy (CIFAR-4)"); ax[0].set_ylim(-0.02, 1.02); ax[0].set_xscale("symlog", linthresh=1)
    ax[1].set_title("Malformed-output fraction"); ax[1].set_xlabel("demonstrations $K$")
    ax[1].set_ylabel("P(output not a valid label)"); ax[1].set_ylim(-0.02, 1.02); ax[1].set_xscale("symlog", linthresh=1)
    ax[1].legend(loc="upper left")
    for a in ax[:2]:
        a.set_xticks([0, 1, 2, 4, 8, 16, 64]); a.set_xticklabels(["0", "1", "2", "4", "8", "16", "64"])
        a.set_xlim(-0.25, 70); a.margins(x=0)
    # Panel 3: timing fingerprint — fraction of the total standard-accuracy drop landing on the
    # first demonstration, split by failure mode (clean-reasoning/copy vs output degeneration).
    clean = [("Qwen2-VL-2B-Instruct", "Qwen2-VL-2B"), ("Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-3B"),
             ("llava-v1.6-mistral-7b-hf", "LLaVA-1.6-7B")]
    degen = [("SmolVLM-Instruct", "SmolVLM-2B"), ("SmolVLM-500M-Instruct", "SmolVLM-500M")]
    def firstshot_frac(m):
        fr = []
        for t in ("cifar4", "shapes4", "fashion4"):
            ks, acc, _ = series(m, t, "standard")
            if not ks or 0 not in ks or 1 not in ks:
                continue
            d = {k: a for k, a in zip(ks, acc)}
            mx = max(k for k in ks if k != 64)
            total = d[0] - d[mx]
            if total > 0.15:
                fr.append(max(0.0, min(1.0, (d[0] - d[1]) / total)))
        return float(np.mean(fr)) if fr else 0.0
    labs = [l for _, l in clean] + [l for _, l in degen]
    vals = [firstshot_frac(m) for m, _ in clean] + [firstshot_frac(m) for m, _ in degen]
    cols = [RED] * len(clean) + ["#e377c2"] * len(degen)
    a = ax[2]; xs = np.arange(len(labs))
    a.bar(xs, vals, color=cols, width=0.7, edgecolor="white", linewidth=0.6)
    a.set_xticks(xs); a.set_xticklabels(labs, rotation=35, ha="right", fontsize=6.2)
    a.set_ylim(0, 1.18); a.set_ylabel("first-shot share of total drop")
    a.set_title("When the damage lands"); a.axhline(0.5, ls=":", c=GREY, lw=1)
    a.axvline(len(clean) - 0.5, ls="-", c="#dddddd", lw=0.8)
    a.text((len(clean) - 1) / 2, 1.12, "clean / copy", color=RED, fontsize=6.4, ha="center", va="center")
    a.text(len(clean) + (len(degen) - 1) / 2, 1.12, "degeneration", color="#b04a9a", fontsize=6.4, ha="center", va="center")
    fig.tight_layout(); save_figure(fig, "fig_failuremodes.pdf"); plt.close(fig)

def fig_vaccine():
    # Composite adapters trained on natural Shapes labels, evaluated with fixed remaps.
    # Three models (rows) by three tasks (columns). Per-K mean over
    # 3 seeds with two-sided 95% Student-t bands; baseline (grey) vs +vaccine (teal).
    models = [("vaccine",          "Qwen2-VL-2B"),
              ("vaccine_qwen25_3b","Qwen2.5-VL-3B"),
              ("vaccine_internvl3","InternVL3-2B")]
    tasks = [("shapes4", "Shapes (trained)"), ("cifar4", "CIFAR-4 (transfer)"), ("fashion4", "Fashion-4 (transfer)")]
    loaded = [(load(f), lab) for f, lab in models]
    if any(R is None for R, _ in loaded):
        print("fig_vaccine: missing one of", [f for f, _ in models], "skipping"); return
    def ms(d, ks):
        m = np.array([float(np.mean(d[str(k)])) for k in ks])
        intervals = [mean_t_ci(d[str(k)]) for k in ks]
        e = np.array([(item["ci95_high"] - item["mean"]
                       if item["ci95_high"] is not None else 0.0)
                      for item in intervals])
        return m, e
    fig, ax = plt.subplots(3, 3, figsize=(7.2, 6.2), sharex=True, sharey=True)
    bounds = [0., 1.]
    for r, (R, mlab) in enumerate(loaded):
        for c, (t, ttl) in enumerate(tasks):
            a = ax[r][c]; b = R["baseline"][t]; v = R["vaccine"][t]; ks = sorted(int(k) for k in b)
            for d, col, lab in [(b, GREY, "frozen"), (v, TEAL, "composite adapter")]:
                m, e = ms(d, ks); a.plot(ks, m, "-o", ms=3, color=col, label=lab)
                a.fill_between(ks, m - e, m + e, color=col, alpha=0.15)
                bounds.extend((float(np.min(m - e)), float(np.max(m + e))))
            a.axhline(0.25, ls=":", c=GREY, lw=1); a.set_ylim(-0.02, 1.02)
            if r == 0: a.set_title(ttl, fontsize=9)
            if r == 2: a.set_xlabel("demonstrations $K$")
            if c == 0: a.set_ylabel(f"{mlab}\nfixed-remap accuracy", fontsize=8)
    ax[0][0].legend(loc="upper left", fontsize=7)
    ax[0][0].set_ylim(min(bounds) - .02, max(bounds) + .02)
    fig.suptitle("Exploratory transfer: natural-label Shapes training", fontsize=9)
    fig.text(.5, .01, "Fixed-remap evaluation; shading: 95% Student-t intervals across three seeds", ha="center", fontsize=7)
    fig.tight_layout(rect=[0, .04, 1, .96]); save_figure(fig, "fig8_vaccine_transfer.pdf"); plt.close(fig)

def _box(ax, x, y, w, h, text, fc, ec, fs=7.5):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05", fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

def fig_proposed_phenomenon():
    x = np.linspace(0, 16, 200)
    prone = 0.10 + 0.80 / (1 + np.exp((x - 4.5) / 1.6))
    resist = 0.80 - 0.004 * x
    learner = 0.28 + 0.60 / (1 + np.exp(-(x - 6) / 2.0))
    fig, ax = plt.subplots(figsize=(4.7, 3.2))
    ax.plot(x, prone, color=RED, lw=2.2, label="standard: decline")
    ax.plot(x, resist, color="#1F77B4", lw=2.2, label="standard: stability")
    ax.plot(x, learner, color=TEAL, lw=2.2, label="remap: gain")
    ax.axhline(0.25, ls=":", c=GREY, lw=1); ax.text(0.3, 0.27, "chance", color=GREY, fontsize=6.5)
    ax.annotate("in-context collapse", xy=(6.2, 0.33), xytext=(7.6, 0.66), fontsize=7.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    ax.set_xlabel("demonstrations $K$"); ax.set_ylabel("accuracy (schematic)")
    ax.set_ylim(0, 1.0); ax.set_xlim(0, 16); ax.legend(loc="center right", fontsize=6.8)
    fig.text(.5, .015, "Standard decline and remap gain may coexist", ha="center", fontsize=7)
    fig.tight_layout(rect=[0, .055, 1, 1]); save_figure(fig, "fig_phenomenon.pdf"); plt.close(fig)

def fig_circa_arch():
    GRN, REDF, NEU = "#DDEEDD", "#F6DCDC", "#EEEEEE"
    fig, ax = plt.subplots(figsize=(7.2, 3.2)); ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    _box(ax, 0.15, 2.25, 0.95, 0.9, "image", NEU, "#888")
    _box(ax, 1.35, 2.25, 1.25, 0.9, "vision\nencoder", NEU, "#888")
    _box(ax, 2.85, 2.25, 1.2, 0.9, "connector", GRN, TEAL)
    _box(ax, 4.3, 2.25, 1.7, 0.9, "early/mid LM", NEU, "#888")
    _box(ax, 6.25, 2.25, 1.5, 0.9, "late LM", NEU, "#888")
    _box(ax, 8.0, 2.25, 1.1, 0.9, "answer", NEU, "#888")
    for x0, x1 in [(1.10, 1.35), (2.60, 2.85), (4.05, 4.3), (6.00, 6.25), (7.75, 8.0)]:
        ax.annotate("", xy=(x1, 2.7), xytext=(x0, 2.7), arrowprops=dict(arrowstyle="->", lw=1.2, color="#555"))
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((2.78, 3.28), 1.34, 0.5, boxstyle="round,pad=0.02", fc="none", ec=TEAL, lw=1.3, ls="--"))
    ax.text(3.45, 4.05, "Validated arm: connector adapter", ha="center", fontsize=7, color=TEAL, fontweight="bold")
    _box(ax, 0.15, 0.45, 2.4, 0.85, "Gate hypothesis (unvalidated):\nselect a stopping point", NEU, "#888", fs=6.5)
    ax.annotate("", xy=(0.62, 2.25), xytext=(0.9, 1.30), arrowprops=dict(arrowstyle="->", lw=1.1, color="#555"))
    _box(ax, 4.05, 0.45, 2.2, 0.85, "Inject hypothesis (unvalidated):\ndemos $\\rightarrow$ task vector", NEU, "#888", fs=6.0)
    ax.annotate("", xy=(5.15, 2.25), xytext=(5.15, 1.30), arrowprops=dict(arrowstyle="->", lw=1.2, color=TEAL))
    ax.set_title("CircA: connector adaptation and proposed serving extensions", fontsize=9, fontweight="bold")
    save_figure(fig, "fig_circa_arch.pdf", bbox_inches="tight"); plt.close(fig)

def fig_tasks():
    """Qualitative stimulus panel: real examples of all three tasks with standard vs remap labels.
    Makes 'contamination-free synthetic concepts' and 'arbitrary tokens' concrete."""
    T = _load_tasks()
    keys = ["shapes4", "cifar4", "fashion4"]
    fig, ax = plt.subplots(3, 4, figsize=(7.2, 5.4))
    for r, key in enumerate(keys):
        d = T[key]
        for c in range(4):
            a = ax[r][c]
            a.imshow(d["examples"][c]); a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values(): sp.set_visible(True); sp.set_color("#bbbbbb"); sp.set_linewidth(0.8)
            a.set_title(f"“{d['names'][c]}”\n“{d['remap'][c]}”", fontsize=7.5, pad=3)
        ax[r][0].set_ylabel(d["title"] + ("\n(contamination-free)" if key == "shapes4" else ""),
                            fontsize=8.5, fontweight="bold", rotation=90, labelpad=8)
        ax[r][0].yaxis.set_visible(True); ax[r][0].set_yticks([])
    fig.text(0.5, 0.995, "Top label = STANDARD (true class)      Bottom label = REMAP (arbitrary token)",
             ha="center", va="top", fontsize=8, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97]); save_figure(fig, "fig_tasks.pdf"); plt.close(fig)


def fig_prompt():
    """The interleaved K-shot ICL prompt as the model sees it, across the three conditions.
    The single most intuition-building figure for an ICL paper."""
    T = _load_tasks(); d = T["shapes4"]
    import random
    rng = random.Random(3)
    # a fixed 4-shot demo sequence + one query (true class = circle, idx 0)
    seq = [1, 0, 2, 3]                                  # demo true classes
    q_cls = 0
    demo_imgs = [d["pool"][c][rng.randrange(8)] for c in seq]
    q_img = d["test"][[t[1] for t in d["test"]].index(q_cls)][0]
    conds = [("STANDARD", d["names"], seq, "true labels"),
             ("SHUFFLED", d["names"], [2, 3, 0, 1], "labels permuted"),
             ("REMAP", d["remap"], seq, "arbitrary tokens")]
    fig, ax = plt.subplots(3, 5, figsize=(7.4, 4.6))
    for r, (cname, space, labseq, sub) in enumerate(conds):
        for c in range(4):
            a = ax[r][c]; a.imshow(demo_imgs[c]); a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values(): sp.set_color("#888"); sp.set_linewidth(0.8)
            a.set_xlabel(f"“{space[labseq[c]]}”", fontsize=7.5, labelpad=2)
            if r == 0: a.set_title(f"demo {c+1}", fontsize=7, color="#555")
        a = ax[r][4]; a.imshow(q_img); a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values(): sp.set_color(RED); sp.set_linewidth(1.6)
        a.set_xlabel("?", fontsize=11, color=RED, labelpad=2)
        if r == 0: a.set_title("query", fontsize=7, color=RED)
        ax[r][0].set_ylabel(f"{cname}\n({sub})", fontsize=8, fontweight="bold")
        ax[r][0].yaxis.set_visible(True); ax[r][0].set_yticks([])
    fig.text(0.5, 0.995, "A $K{=}4$ image–label ICL prompt (synthetic shapes), shown under the three conditions",
             ha="center", va="top", fontsize=8, fontweight="bold")
    fig.tight_layout(rect=[0.0, 0, 1, 0.95]); save_figure(fig, "fig_prompt.pdf"); plt.close(fig)


def fig_gallery():
    """Selected STANDARD predictions, including changed answers and unchanged controls.
    Green = correct, red = wrong. Needs gallery.json
    (per-example predictions from gallery.py); skipped if absent."""
    import io, base64
    from PIL import Image
    d = load("gallery")
    if not d:
        print("fig_gallery: gallery.json absent, skipping"); return
    ks = [str(k) for k in d["ks"]]; k_lo, k_hi = ks[0], ks[-1]
    tasks = [t for t in ["cifar4", "fashion4", "shapes4"] if t in d["tasks"]]
    ncol = max(len(d["tasks"][t]["entries"]) for t in tasks)
    fig, ax = plt.subplots(len(tasks), ncol, figsize=(1.7 * ncol, 2.0 * len(tasks)))
    if len(tasks) == 1: ax = [ax]
    ttl = {"cifar4": "CIFAR-4", "fashion4": "Fashion-4", "shapes4": "Synthetic shapes"}
    for r, t in enumerate(tasks):
        ents = d["tasks"][t]["entries"]
        for c in range(ncol):
            a = ax[r][c]; a.set_xticks([]); a.set_yticks([])
            if c >= len(ents):
                a.axis("off"); continue
            e = ents[c]
            a.imshow(Image.open(io.BytesIO(base64.b64decode(e["img"]))))
            lo, hi = e["preds"][k_lo], e["preds"][k_hi]
            mark = lambda ok: r"$\checkmark$" if ok else r"$\times$"
            hcol = TEAL if hi["correct"] else RED
            a.set_title(f"true: {e['true']}", fontsize=7, pad=2)
            a.set_xlabel(f"$K{{=}}{k_lo}$: {lo['pred']} {mark(lo['correct'])}\n"
                         f"$K{{=}}{k_hi}$: {hi['pred']} {mark(hi['correct'])}",
                         fontsize=6.8, labelpad=2, color="#333")
            for sp in a.spines.values(): sp.set_color(hcol); sp.set_linewidth(1.6)
        ax[r][0].set_ylabel(ttl.get(t, t), fontsize=8.5, fontweight="bold")
        ax[r][0].yaxis.set_visible(True); ax[r][0].set_yticks([])
    fig.text(0.5, 0.998, f"Selected STANDARD predictions (Qwen2-VL-2B): $K{{=}}{k_lo}$ and $K{{=}}{k_hi}$\n"
             "Changed answers and unchanged controls", ha="center", va="top", fontsize=8, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); save_figure(fig, "fig_gallery.pdf"); plt.close(fig)


def fig_failcase():
    """Anatomy of a failure: the full K-shot prompt (demo filmstrip + labels) that led the model to a
    wrong answer, with the query and prediction. Highlights recency-copy when present. Needs
    failcase.json (from failcase.py); skipped if absent."""
    import io, base64
    from PIL import Image
    d = load("failcase")
    if not d:
        print("fig_failcase: failcase.json absent, skipping"); return
    # one representative failure per task (prefer a recency-copy case for the mechanistic point)
    rows = []
    for t, td in d["tasks"].items():
        if not td["cases"]: continue
        case = next((c for c in td["cases"] if c.get("is_copy")), td["cases"][0])
        rows.append((t, case))
    if not rows:
        print("fig_failcase: no failure cases found, skipping"); return
    K = d["k"]; ncol = K + 1                                   # K demos + query
    fig, ax = plt.subplots(len(rows), ncol, figsize=(0.95 * ncol, 1.9 * len(rows)))
    if len(rows) == 1: ax = [ax]
    ttl = {"cifar4": "CIFAR-4", "shapes4": "Synthetic shapes", "fashion4": "Fashion-4"}
    for r, (t, c) in enumerate(rows):
        for j, dm in enumerate(c["demos"]):
            a = ax[r][j]; a.imshow(Image.open(io.BytesIO(base64.b64decode(dm["img"]))))
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values(): sp.set_color("#bbb"); sp.set_linewidth(0.6)
            a.set_xlabel(dm["label"], fontsize=6, labelpad=1)
            if r == 0 and j == 0: a.set_title("demonstrations $\\rightarrow$", fontsize=7, color="#555", loc="left")
        # query cell
        a = ax[r][K]; a.imshow(Image.open(io.BytesIO(base64.b64decode(c["query_img"]))))
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values(): sp.set_color(RED); sp.set_linewidth(1.8)
        a.set_xlabel(f"pred: {c['pred']}", fontsize=6.5, color=RED, labelpad=1)
        copytag = "  copies last demo ($\\rightarrow$ \"%s\")" % c["last_demo_label"] if c.get("is_copy") else ""
        if r == 0: a.set_title("query", fontsize=7, color=RED)
        ax[r][0].set_ylabel(f"{ttl.get(t, t)}\ntrue: {c['true']}{copytag}", fontsize=7.2, fontweight="bold")
        ax[r][0].yaxis.set_visible(True); ax[r][0].set_yticks([])
    fig.text(0.5, 0.998, f"Anatomy of a failure (Qwen2-VL-2B, $K{{=}}{K}$, STANDARD): with true-labelled "
             "demonstrations in context, the model mislabels a query it knows zero-shot",
             ha="center", va="top", fontsize=7.8, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94]); save_figure(fig, "fig_failcase.pdf"); plt.close(fig)


@lru_cache(maxsize=None)
def _audited_followup(filename):
    """Recompute plotted statistics from a complete, independently audited artifact."""
    from scripts.summarize_revision_followup import summarize_file

    path = ROOT / "data" / "revision_v25" / filename
    if not path.exists():
        print(f"follow-up figure source absent, preserving existing PDF: {path.name}")
        return None
    report = summarize_file(path)
    if report["status"] != "complete" or report["audit"]["errors"]:
        raise ValueError(f"figure source is not complete and audited: {path}")
    config = report["run_config"]
    if config["seeds"] != [0, 1, 2] or config["n"] != (80 if report["schema"] == "vqa" else 40):
        raise ValueError(f"unexpected figure source dimensions: {path}")
    if report["schema"] == "options":
        protocol = config["protocols"]["random"]
        if protocol.get("protocol_version") != "2.5-options" or protocol.get("option_order") != "random":
            raise ValueError(f"figure requires randomized answer options: {path}")
    return report


def _interval_arrays(intervals):
    arrays = [np.asarray([item[key] for item in intervals], dtype=float)
              for key in ("mean", "ci95_low", "ci95_high")]
    if not all(np.isfinite(values).all() for values in arrays):
        raise ValueError("figure uncertainty contains missing or nonfinite values")
    return arrays


def _followup_curve(axis, report, task, condition, arm, label, color, marker,
                    *, metric="accuracy", order="random", ks=None, linestyle="-"):
    selected = sorted((entry for entry in report["estimates"]
                       if (entry["task"], entry["condition"], entry["arm"], entry["option_order"])
                       == (task, condition, arm, order)
                       and (ks is None or entry["k"] in ks)), key=lambda entry: entry["k"])
    if not selected or (ks is not None and [entry["k"] for entry in selected] != list(ks)):
        raise ValueError(f"missing plotted cells: {task}/{condition}/{arm}")
    shots = [entry["k"] for entry in selected]
    means, lows, highs = _interval_arrays([entry["metrics"][metric] for entry in selected])
    axis.errorbar(shots, means, yerr=np.vstack((means - lows, highs - means)),
                  color=color, marker=marker, linestyle=linestyle, label=label,
                  capsize=2.5, markersize=3.5, linewidth=1.4)
    return [float(lows.min()), float(highs.max())]


def fig_vqa():
    """Distinct-image VQAv2 scores and paired image-bootstrap intervention effects."""
    report = _audited_followup("vqav2_distinct80.json")
    if report is None:
        return
    ks = report["run_config"]["ks"]
    arms = [("frozen", "Frozen", "#777777", "o"),
            ("vqa_connector", "VQA-trained connector", TEAL, "s"),
            ("shapes_connector", "Shapes-trained connector", "#4477AA", "D")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    for arm, label, color, marker in arms:
        _followup_curve(axes[0], report, "vqav2", "vqa", arm, label, color, marker,
                        order="not_applicable", ks=ks)
        if arm == "frozen":
            continue
        paired = {row["right"]["k"]: row["paired_image_bootstrap"]
                  for row in report["paired_comparisons"]
                  if row["left"]["arm"] == "frozen" and row["right"]["arm"] == arm}
        if set(paired) != set(ks) or any(item["n_images"] != 80 for item in paired.values()):
            raise ValueError("VQA figure requires 80 paired images at every K")
        means, lows, highs = _interval_arrays([paired[k] for k in ks])
        axes[1].errorbar(ks, means, yerr=np.vstack((means - lows, highs - means)),
                         color=color, marker=marker, capsize=3, label=label)
    axes[0].set(title="Scores: descriptive seed intervals", ylabel="VQAv2 soft accuracy",
                ylim=(.53, .70))
    axes[1].set(title="Paired effects: image-bootstrap intervals",
                ylabel="Adapter minus frozen", ylim=(-.09, .075))
    axes[1].axhline(0, ls=":", color=GREY, lw=1)
    for axis in axes:
        axis.set(xlabel="demonstrations $K$", xticks=ks)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(.5, .075), fontsize=6.8)
    fig.suptitle("Open-ended VQAv2: 80 distinct held-out images", fontsize=10)
    fig.text(.5, .045, "95% intervals: left, three-seed Student-t; right, 10,000 paired image-bootstrap draws",
             ha="center", fontsize=6.4)
    fig.text(.5, .012, "Image effects average three fixed checkpoints/seeds; inference is conditional on the eligible image/question frame",
             ha="center", fontsize=6.1)
    fig.tight_layout(rect=[0, .15, 1, .95])
    save_figure(fig, "fig_vqa_curves.pdf")
    plt.close(fig)


def fig_vqa_failcase():
    """Selected VQA answer pairs, without scoring claims from incomplete annotations.
    Each model row draws two examples from each of VQAv2, TextVQA, and ScienceQA.
    Needs per-model vqa_gallery_*.json from vqa.py --gallery."""
    import io, base64
    import textwrap
    from PIL import Image
    rows = [("vqa_gallery_qwen2b_v2", "Qwen2-VL-2B"),
            ("vqa_gallery_qwen25_3b", "Qwen2.5-VL-3B")]
    datasets = [("vqav2", "VQAv2"), ("textvqa", "TextVQA"), ("scienceqa", "ScienceQA")]
    PER_DS = 2
    loaded = [(load(f), lab) for f, lab in rows]
    loaded = [(d, lab) for d, lab in loaded if d]
    if not loaded:
        print("fig_vqa_failcase: no per-model gallery data, skipping"); return
    ncol = len(datasets) * PER_DS
    nrow = len(loaded)
    fig, ax = plt.subplots(nrow, ncol, figsize=(1.35 * ncol, 2.35 * nrow))
    if nrow == 1: ax = [ax]
    khi = loaded[0][0]["k_hi"]
    for r, (d, mlab) in enumerate(loaded):
        klo, khi = str(d["k_lo"]), str(d["k_hi"])
        picks = []
        for t, ttl in datasets:
            picks += [(ttl, e) for e in d["tasks"].get(t, {}).get("flips", [])[:PER_DS]]
        if len(picks) < ncol:
            extra = [(tt, e) for t, tt in datasets for e in d["tasks"].get(t, {}).get("flips", [])[PER_DS:]]
            picks += extra[:ncol - len(picks)]
        for c in range(ncol):
            a = ax[r][c]; a.set_xticks([]); a.set_yticks([])
            if c >= len(picks):
                a.axis("off"); continue
            ttl, e = picks[c]
            a.imshow(Image.open(io.BytesIO(base64.b64decode(e["img"]))), interpolation="nearest")
            q = e["q"]; q = q[:40] + ("..." if len(q) > 40 else "")
            a.set_title(f"{ttl}\n{textwrap.fill(q, width=27)}", fontsize=5.5, pad=2)
            # Retained annotations cannot reproduce the original selection scores.
            # Show the saved strings without assigning correctness to either answer.
            lines = [f"one ref.: {e['gold'][0]}",
                     f"K={klo}: {e['preds'][klo]['pred']}",
                     f"K={khi}: {e['preds'][khi]['pred']}"]
            a.set_xlabel("\n".join(textwrap.fill(line, width=28) for line in lines),
                         fontsize=5.5, labelpad=2, color="#333")
            for sp in a.spines.values(): sp.set_color(GREY); sp.set_linewidth(0.8)
        ax[r][0].set_ylabel(mlab, fontsize=8, fontweight="bold")
        ax[r][0].yaxis.set_visible(True); ax[r][0].set_yticks([])
    fig.text(0.5, 0.995, "Selected open-ended VQA answer pairs\n"
             "Two examples per dataset and model; one saved reference answer is shown",
             ha="center", va="top", fontsize=7.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93]); save_figure(fig, "fig_vqa_failcase.pdf"); plt.close(fig)


def fig_transfer_controls():
    """Random-option transfer with paired frozen, connector, and all-layer arms."""
    report = _audited_followup("options_transfer_n40.json")
    if report is None:
        return
    ks = report["run_config"]["ks"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.1), sharex=True, sharey=True)
    bounds = [0., 1.]
    for row, (task, domain) in enumerate((("cifar4", "CIFAR-4"), ("fashion4", "Fashion-4"))):
        for col, condition in enumerate(("standard", "remap")):
            axis = axes[row, col]
            for arm, label, color, marker in (
                ("frozen", "Frozen", "#777777", "o"),
                ("connector", "Shapes-trained connector", TEAL, "s"),
                ("all_layer", "Shapes-trained all-layer", "#4477AA", "D"),
            ):
                bounds.extend(_followup_curve(axis, report, task, condition, arm,
                                              label, color, marker, ks=ks))
            axis.axhline(.25, ls=":", color=GREY, lw=1)
            axis.set_xticks(ks)
            if row == 0:
                axis.set_title("Standard labels" if condition == "standard" else "Episode remap labels")
            if col == 0:
                axis.set_ylabel(domain + "\naccuracy")
            if row == 1:
                axis.set_xlabel("demonstrations $K$ (0 = zero shot)")
    axes[0, 0].set_ylim(min(bounds) - .025, max(bounds) + .025)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(.5, .065), fontsize=6.8)
    fig.suptitle("Shapes-trained adapter transfer: randomized answer options", fontsize=10)
    fig.text(.5, .04, "Protocol 2.5 evaluation; saved adapters were trained with class-ordered answer options",
             ha="center", fontsize=6.5)
    fig.text(.5, .015, "Pointwise 95% Student-t intervals across three seeds; 40 paired queries per seed",
             ha="center", fontsize=6.5)
    fig.tight_layout(rect=[0, .12, 1, .96])
    save_figure(fig, "fig_transfer_controls.pdf")
    plt.close(fig)


def fig_core_controls():
    """Correct and corrupted mapping use in the two completed core model panels."""
    reports = [_audited_followup(name) for name in
               ("options_shapes_n40.json", "options_qwen25_shapes_n40.json")]
    if any(report is None for report in reports):
        return
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.3), sharey=True)
    bounds = [0., 1.]
    for row, (report, model) in enumerate(zip(reports, ("Qwen2-VL-2B", "Qwen2.5-VL-3B"))):
        for col, (condition, metric, title) in enumerate((
            ("standard", "accuracy", "Standard labels"),
            ("remap", "accuracy", "Episode-remap labels"),
            ("remap_shuffled", "follows_corrupted_mapping_rate", "Corrupted remap:\nmapping adherence"),
        )):
            axis = axes[row, col]
            ks = [k for k in report["run_config"]["ks"] if col != 2 or k > 0]
            for arm, label, color, marker, line in (
                ("frozen", "Frozen", "#777777", "o", "-"),
                ("connector", "Connector", TEAL, "s", "-"),
                ("all_layer", "All-layer (2B only)", "#4477AA", "D", "--"),
            ):
                if arm not in report["run_config"]["arms"]:
                    continue
                bounds.extend(_followup_curve(axis, report, "shapes4", condition, arm,
                                              label, color, marker, metric=metric, ks=ks, linestyle=line))
            axis.axhline(.25, ls=":", color=GREY, lw=1)
            axis.set_xticks(report["run_config"]["ks"])
            axis.set_xlim(-.5, 16.5)
            if row == 0:
                axis.set_title(title, fontsize=8)
            if col == 0:
                axis.set_ylabel(model + "\naccuracy / adherence", fontsize=8)
            if row == 1:
                axis.set_xlabel("demonstrations $K$")
    axes[0, 0].set_ylim(min(bounds) - .025, max(bounds) + .025)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(.5, .069), fontsize=7)
    fig.suptitle("Shapes-4: randomized answer options and mapping-use controls", fontsize=10)
    fig.text(.5, .04, "Pointwise 95% Student-t intervals across three seeds; 40 queries per cell; Protocol 2.5",
             ha="center", fontsize=6.5)
    fig.text(.5, .013, "Saved adapters trained with class-ordered options; corrupted and correct conditions are not demo-identity paired",
             ha="center", fontsize=6.2)
    fig.tight_layout(rect=[0, .13, 1, .95])
    save_figure(fig, "fig_core_controls.pdf")
    plt.close(fig)


def fig_coverage_controls():
    """Depth/objective coverage and the frozen 7B low-shot panel, without pooling files."""
    depth = _audited_followup("coverage/options_depth_natural_shapes_n40.json")
    breadth = _audited_followup("coverage/options_qwen7_shapes_n40.json")
    if depth is None or breadth is None:
        return
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))
    bounds = [0., 1.]
    for col, condition in enumerate(("standard", "remap")):
        for arm, label, color, marker in (
            ("frozen", "Frozen", "#777777", "o"),
            ("early", "Early", "#CC6677", "^"),
            ("mid", "Mid", "#AA4499", "D"),
            ("late", "Late", "#4477AA", "v"),
            ("natural_connector", "Connector (natural)", TEAL, "s"),
        ):
            bounds.extend(_followup_curve(axes[0, col], depth, "shapes4", condition, arm,
                                          label, color, marker, ks=depth["run_config"]["ks"]))
        axes[0, col].set_title("Saved-adapter coverage: " + ("standard" if col == 0 else "remap"), fontsize=8)
        corrupted = "shuffled" if condition == "standard" else "remap_shuffled"
        for variant, label, color, marker, metric, line in (
            (condition, "Correct demos: accuracy", "#4477AA", "o", "accuracy", "-"),
            (corrupted, "Corrupted demos: accuracy", "#AA4499", "s", "accuracy", "-"),
            (corrupted, "Corrupted mapping: adherence", "#AA4499", "^", "follows_corrupted_mapping_rate", "--"),
        ):
            ks = [k for k in breadth["run_config"]["ks"] if metric == "accuracy" or k > 0]
            bounds.extend(_followup_curve(axes[1, col], breadth, "shapes4", variant, "frozen",
                                          label, color, marker, metric=metric, ks=ks, linestyle=line))
        axes[1, col].set_title("Frozen Qwen2-VL-7B: " + ("standard" if col == 0 else "remap"), fontsize=8)
    for row, report in enumerate((depth, breadth)):
        for axis in axes[row]:
            axis.set(xticks=report["run_config"]["ks"], xlim=(-.5, 16.5),
                     ylim=(min(bounds) - .025, max(bounds) + .025), xlabel="demonstrations $K$")
            axis.axhline(.25, ls=":", color=GREY, lw=1)
        axes[row, 0].set_ylabel("accuracy" if row == 0 else "accuracy / adherence")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center", ncol=5, bbox_to_anchor=(.5, .535), fontsize=6.6)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(.5, .079), fontsize=6.3)
    fig.suptitle("Shapes-4 coverage: depth, training objective, and frozen 7B", fontsize=10)
    fig.text(.5, .048, "Top: early/mid/late use episode-remap training; connector uses natural labels; saved class-option checkpoints",
             ha="center", fontsize=6.2)
    fig.text(.5, .025, "Random-option evaluation; pointwise 95% Student-t intervals across three seeds, 40 queries per cell",
             ha="center", fontsize=6.3)
    fig.text(.5, .003, "Correct/corrupted conditions are not demo-identity paired; no cross-file depth-versus-connector contrast",
             ha="center", fontsize=6.2)
    fig.subplots_adjust(left=.09, right=.985, bottom=.18, top=.90, hspace=.65, wspace=.25)
    save_figure(fig, "fig_coverage_controls.pdf")
    plt.close(fig)



def main(argv=None):
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new directory; existing paths are never replaced")
    parser.add_argument("--skip-qualitative", action="store_true",
                        help="omit dataset-backed task/prompt thumbnails; recorded in manifest")
    parser.add_argument("--allow-downloads", action="store_true",
                        help="allow pinned qualitative datasets to download when absent")
    args = parser.parse_args(argv)
    destination = args.output_dir.expanduser().resolve()
    for protected in (ROOT / "paper", ROOT / "data", ROOT / "figures"):
        protected = protected.resolve()
        if destination == protected or destination.is_relative_to(protected):
            parser.error("output must be outside the retained paper/data/figures trees")
    if destination.exists():
        parser.error("output already exists; choose a fresh directory")
    if not args.allow_downloads:
        for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
            os.environ[key] = "1"
    source_paths = sorted(set(FD.glob("*.json")) | set(V2.glob("*.json")) |
                          set((ROOT / "data/revision_v25").rglob("*.json")))
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    input_hashes = {str(path.relative_to(ROOT)): digest(path) for path in source_paths}
    manuscript = ROOT / "paper/Main.tex"
    expected_outputs = set(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{figures/([^}]+)\}",
                                      manuscript.read_text()))
    if not expected_outputs:
        raise ValueError("paper figure inventory is unavailable")
    input_hashes["paper/Main.tex"] = digest(manuscript)
    if args.skip_qualitative:
        expected_outputs -= {"fig_tasks.pdf", "fig_prompt.pdf"}
    destination.mkdir(parents=True, exist_ok=False)
    OUT = destination
    GENERATED_PDFS.clear()
    _audited_followup.cache_clear()
    try:
        fig_concept(); fig_cliff(); fig_circuit(); fig_frontier(); fig_consol()
        fig_component(); fig_crossarch(); fig_control(); fig_vaccine()
        fig_proposed_phenomenon(); fig_circa_arch(); fig_failuremodes()
        if not args.skip_qualitative:
            fig_tasks(); fig_prompt()
        fig_gallery(); fig_failcase(); fig_vqa(); fig_vqa_failcase()
        fig_transfer_controls(); fig_core_controls(); fig_coverage_controls()
        generated = {p.name: {"status": "regenerated", "sha256": digest(p)}
                     for p in GENERATED_PDFS}
        hosted = "fig3_frontier_cliff.pdf"
        if hosted not in generated:
            source = ROOT / "figures" / hosted
            if not source.is_file():
                raise ValueError("historical hosted figure is unavailable; cannot preserve its evidence")
            source_hash = digest(source)
            with (destination / hosted).open("xb") as handle:
                handle.write(source.read_bytes())
            if digest(destination / hosted) != source_hash:
                raise RuntimeError("historical hosted asset changed while being copied")
            input_hashes[str(source.relative_to(ROOT))] = source_hash
            generated[hosted] = {"status": "copied_historical_asset", "sha256": source_hash,
                                 "source": str(source.relative_to(ROOT)),
                                 "limitation": "missing Nova aggregates prevent numerical regeneration"}
        missing_outputs = expected_outputs - generated.keys()
        if missing_outputs:
            raise ValueError(f"paper figures were not produced: {sorted(missing_outputs)}")
        if any(digest(ROOT / name) != checksum for name, checksum in input_hashes.items()):
            raise RuntimeError("figure input changed during rendering")
        manifest = {"status": "complete", "paper_sync": False,
                    "skip_qualitative": args.skip_qualitative,
                    "allow_downloads": args.allow_downloads,
                    "source_sha256": digest(Path(__file__)),
                    "input_sha256": input_hashes, "outputs": generated}
        (destination / "figures_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print("Figure outputs (historical copies labeled in manifest):", destination)
    finally:
        plt.close("all")
        OUT = None


if __name__ == "__main__":
    main()
