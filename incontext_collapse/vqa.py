"""Exploratory many-shot VQA evaluation and shared answer-scoring utilities.

Evaluates frozen models on supported VQAv2, GQA, TextVQA, and ScienceQA data
using a single-user-turn prompt. Multi-seed intervals vary demonstration
sampling; they do not represent new model-training replications. Dataset data
are streamed on demand and may use local caches. Use --probe to inspect the
dataset field mapping before a new run.

Scoring uses answer normalization and leave-one-annotator-out VQA agreement when
multiple references are supplied, with exact normalized matching for a single
reference. The local implementation is an official-style metric, not a claim
of bit-for-bit equivalence with every historical benchmark evaluator.

This module does not establish that classification collapse or model ordering
generalizes to open-ended VQA. The current paper's image-disjoint boundary study
uses vqa_diverse.py on 80 distinct images and vqa_confirmatory.py on separate
80-image development/400-image confirmation partitions. The latter observes a
9.32-point confirmation decline, below its declared 10-point mean margin;
neither study establishes qualifying open-ended repair.
Task-specific fitting is implemented separately in vqa_train.py. See
docs/CURRENT_EVIDENCE_GUIDE_2026_10_09.txt for scope and evidence locations.
"""
from __future__ import annotations

import argparse
import json
import hashlib
import importlib
from pathlib import Path
import os
import random
import re
import statistics as st

import torch

from .core import pick_device
from .results import load_result, write_json_atomic
from .evaluation import mean_t_ci
from .models import load_model
from .provenance import DATASET_REVISIONS, load_pinned_dataset, model_revision, software_versions

# name -> (hf_id, config, split)
DATASETS = {
    "vqav2": ("lmms-lab/VQAv2", None, "validation"),
    "gqa": ("lmms-lab/GQA", "testdev_balanced_images", "testdev"),
    "textvqa": ("lmms-lab/textvqa", None, "validation"),
    "scienceqa": ("derek-thomas/ScienceQA", None, "validation"),
}

# ---------- VQAv2 answer normalization and leave-one-annotator-out scoring ----------
_ARTICLES = {"a", "an", "the"}
_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
    "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
    "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't",
    "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've",
    "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
    "id've": "i'd've", "i'dve": "i'd've", "im": "i'm", "ive": "i've",
    "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've",
    "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
    "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've",
    "oclock": "o'clock", "oughtnt": "oughtn't", "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't",
    "shed've": "she'd've", "she'dve": "she'd've", "she's": "she's",
    "shouldve": "should've", "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd", "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've", "somebodyll": "somebody'll",
    "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've",
    "someone'dve": "someone'd've", "someonell": "someone'll", "someones": "someone's",
    "somethingd": "something'd", "somethingd've": "something'd've",
    "something'dve": "something'd've", "somethingll": "something'll",
    "thats": "that's", "thered": "there'd", "thered've": "there'd've",
    "there'dve": "there'd've", "therere": "there're", "theres": "there's",
    "theyd": "they'd", "theyd've": "they'd've", "they'dve": "they'd've",
    "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas",
    "wasnt": "wasn't", "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've",
    "werent": "weren't", "whatll": "what'll", "whatre": "what're", "whats": "what's",
    "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's",
    "whereve": "where've", "whod": "who'd", "whod've": "who'd've",
    "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've",
    "whyll": "why'll", "whyre": "why're", "whys": "why's", "wont": "won't",
    "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll",
    "y'allll": "y'all'll", "yall'd've": "y'all'd've", "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've",
    "you'dve": "you'd've", "youll": "you'll", "youre": "you're", "youve": "you've",
}
_NUMBERS = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
    "nine": "9", "ten": "10",
}
_VQA_PUNCTUATION = (';', '/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\',
                    '_', '-', '>', '<', '@', '`', ',', '?', '!')
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_BETWEEN_DIGITS = re.compile(r"\d,\d")


def _normalize_vqa_punctuation(text: str) -> str:
    output = text
    for punctuation in _VQA_PUNCTUATION:
        if (f"{punctuation} " in text or f" {punctuation}" in text
                or _COMMA_BETWEEN_DIGITS.search(text)):
            output = output.replace(punctuation, "")
        else:
            output = output.replace(punctuation, " ")
    return _PERIOD_STRIP.sub("", output)


def vqa_norm(s):
    words = _normalize_vqa_punctuation(str(s).lower().replace("\n", " ").replace("\t", " ")).split()
    normalized = []
    for word in words:
        word = _NUMBERS.get(word, word)
        if word not in _ARTICLES:
            normalized.append(_CONTRACTIONS.get(word, word))
    return " ".join(normalized)


def vqa_score(pred, answers):
    """Official-style VQAv2 leave-one-annotator-out soft accuracy."""
    p = vqa_norm(pred)
    gold = [vqa_norm(a) for a in answers]
    if len(gold) <= 1:
        return float(bool(gold) and p == gold[0])
    per_annotator = []
    for index in range(len(gold)):
        matches = sum(p == answer for other, answer in enumerate(gold) if other != index)
        per_annotator.append(min(1.0, matches / 3.0))
    return sum(per_annotator) / len(per_annotator)


def adapt(name, ex):
    """Dataset adapter: example -> (PIL image | None, question str, [answer strs])."""
    img = ex.get("image")
    if name == "scienceqa":
        ch = ex.get("choices") or []
        a = ex.get("answer")
        ans = [ch[a]] if isinstance(a, int) and 0 <= a < len(ch) else []
        q = ex.get("question", "")
        if ch:
            q += "\nOptions: " + "; ".join(f"{chr(65 + i)}) {c}" for i, c in enumerate(ch))
        return img, q, ans
    q = ex.get("question") or ex.get("query") or ""
    raw = ex.get("answers")
    if raw is None:
        raw = ex.get("multiple_choice_answer")
        if raw is None:
            raw = ex.get("answer")
        raw = [raw] if raw is not None else []
    if isinstance(raw, (str, int, float)):
        raw = [raw]
    elif isinstance(raw, dict):
        raw = raw.get("answer", [])
        raw = [raw] if isinstance(raw, (str, int, float)) else raw
    if not isinstance(raw, (list, tuple)):
        raise ValueError("unsupported answer container")
    ans = [a.get("answer") if isinstance(a, dict) else a for a in raw]
    ans = [str(answer) for answer in ans if answer is not None and str(answer)]
    return img, q, [a for a in ans if a]


def load_task(name, n_pool, n_test, seed=0):
    hf_id, cfg, split = DATASETS[name]
    ds = load_pinned_dataset(hf_id, cfg, split=split, streaming=True)
    items = []
    for ex in ds:
        img, q, ans = adapt(name, ex)
        if img is not None and q and ans:
            items.append((img, q, ans))
        if len(items) >= n_pool + n_test:
            break
    return partition_task_items(items, n_pool, n_test, seed)


def partition_task_items(items, n_pool, n_test, seed=0):
    """Split before shuffling so the training pool is invariant to test size."""
    if type(n_pool) is not int or n_pool < 1 or type(n_test) is not int or n_test < 1:
        raise ValueError("positive integer pool and test sizes required")
    if len(items) < n_pool + n_test:
        raise ValueError(
            f"need {n_pool + n_test} valid items, found only {len(items)}")
    pool = list(items[:n_pool])
    test = list(items[n_pool:n_pool + n_test])
    random.Random(seed).shuffle(pool)
    random.Random(seed + 1).shuffle(test)
    return pool, test



def validate_evaluation_request(pool, test, ks, n, seeds):
    """Never silently change a declared K, query count or seed replication."""
    if (type(n) is not int or n < 1 or n > len(test) or not pool
            or not ks or len(set(ks)) != len(ks)
            or any(type(k) is not int or k < 0 or k > len(pool) for k in ks)
            or not seeds or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or seed < 0 for seed in seeds)):
        raise ValueError("invalid VQA grid, insufficient queries/supports, or duplicate seeds")


def implementation_manifest(names=("vqa", "vqa_repair", "models", "protocol", "provenance", "results")):
    return {name + ".py": hashlib.sha256(Path(importlib.import_module(
        f"{__package__}.{name}").__file__).read_bytes()).hexdigest() for name in names}


def build_vqa_messages(demos, q_img, q_text, size):
    """Single-turn interleaved prompt, mirroring core.build_messages."""
    prep = lambda im: im.convert("RGB").resize((size, size))
    content = []
    for img, q, ans in demos:
        content += [{"type": "image", "image": prep(img)},
                    {"type": "text", "text": f"Question: {q}\nAnswer: {ans[0]}."}]
    content += [{"type": "image", "image": prep(q_img)},
                {"type": "text", "text": f"Question: {q_text}\nAnswer with a short word or phrase."}]
    return [{"role": "user", "content": content}]


@torch.no_grad()
def _generate(model, proc, messages):
    inp = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                   return_dict=True, return_tensors="pt").to(model.device)
    out = model.generate(**inp, max_new_tokens=10, do_sample=False,
                         num_beams=1, num_return_sequences=1)
    return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def eval_kcurve(model, proc, pool, test, ks, n, size=112, seed=0):
    validate_evaluation_request(pool, test, ks, n, [seed])
    rng = random.Random(seed)
    acc = {}
    for k in ks:
        tot = 0.0
        for q_img, q_text, answers in test[:n]:
            demos = rng.sample(pool, k)
            pred = _generate(model, proc, build_vqa_messages(demos, q_img, q_text, size))
            tot += vqa_score(pred, answers)
        acc[k] = round(tot / min(n, len(test)), 3)
    return acc


def gallery(model, proc, pool, test, k_lo, k_hi, n, size=112, seed=0):
    """Per-example records for a VQA failure gallery: image (png b64), question, gold answer,
    and the model's answer at K=k_lo vs K=k_hi. Returns the clearest flips (correct at k_lo,
    wrong at k_hi) and all records."""
    import io, base64
    validate_evaluation_request(pool, test, [k_lo, k_hi], n, [seed])
    rng = random.Random(seed)
    out = []
    for q_img, q_text, answers in test[:n]:
        recs = {}
        for k in (k_lo, k_hi):
            demos = rng.sample(pool, k)
            pred = _generate(model, proc, build_vqa_messages(demos, q_img, q_text, size))
            recs[str(k)] = {"pred": pred.strip()[:40], "score": vqa_score(pred, answers)}
        buf = io.BytesIO(); q_img.convert("RGB").resize((size, size)).save(buf, format="PNG")
        out.append({"q": q_text[:80], "gold": answers[:3],
                    "img": base64.b64encode(buf.getvalue()).decode(), "preds": recs})
    flips = [e for e in out if e["preds"][str(k_lo)]["score"] >= 0.5 and e["preds"][str(k_hi)]["score"] < 0.5]
    return flips, out


def agg_kcurve(model, proc, pool, test, ks, n, seeds, size=112):
    """Per-K mean + 95% CI half-width over seeds."""
    validate_evaluation_request(pool, test, ks, n, seeds)
    per = {k: [] for k in ks}
    for s in seeds:
        c = eval_kcurve(model, proc, pool, test, ks, n, size=size, seed=s)
        for k in ks:
            per[k].append(c[k])
    mean = {str(k): round(st.fmean(per[k]), 3) for k in ks}
    ci = {}
    for k in ks:
        interval = mean_t_ci(per[k])
        half_width = (interval["ci95_high"] - interval["mean"]
                      if interval["ci95_high"] is not None else None)
        ci[str(k)] = round(half_width, 3) if half_width is not None else None
    return {"mean": mean, "ci": ci, "per_seed": {str(k): per[k] for k in ks}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--datasets", default="vqav2,textvqa")
    ap.add_argument("--ks", default="0,1,2,4,8,16")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seeds", default="0", help="comma-separated seeds")
    ap.add_argument("--out", default="data/figdata/vqa.json")
    ap.add_argument("--probe", action="store_true", help="print each dataset's fields, no model load")
    ap.add_argument("--gallery", action="store_true",
                    help="capture per-example failure cases (K_lo vs K_hi) -> data/figdata/vqa_gallery.json")
    a = ap.parse_args()
    names = a.datasets.split(",")
    seeds = [int(s) for s in a.seeds.split(",")]

    if a.probe:
        for nm in names:
            try:
                pool, test = load_task(nm, 2, 1)
                img, q, ans = (pool + test)[0]
                print(f"[{nm}] OK img={img.size} q={q[:70]!r} ans={ans[:5]}")
            except Exception as e:  # noqa: BLE001
                print(f"[{nm}] FAIL: {type(e).__name__}: {str(e)[:120]}")
        return

    ks = [int(x) for x in a.ks.split(",")]
    if (a.n < 1 or not ks or len(ks) != len(set(ks)) or min(ks) < 0
            or not seeds or min(seeds) < 0 or len(seeds) != len(set(seeds))
            or len(names) != len(set(names)) or any(name not in DATASETS for name in names)):
        ap.error("positive n, unique nonnegative K/seeds, and supported distinct datasets required")
    if a.gallery:
        if len(ks) < 2 or len(seeds) != 1:
            ap.error("gallery requires two distinct K values and exactly one seed")
        k_lo, k_hi = ks[0], ks[-1]
        out = a.out if a.out != "data/figdata/vqa.json" else "data/figdata/vqa_gallery.json"
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        if Path(out).exists():
            raise ValueError("preserve existing gallery; choose a new output path")
        model, proc = load_model(a.model, dtype=torch.bfloat16)
        R = {"model": a.model, "k_lo": k_lo, "k_hi": k_hi, "seed": seeds[0], "tasks": {}}
        for nm in names:
            pool, test = load_task(nm, max(32, k_hi), a.n)
            flips, allrec = gallery(model, proc, pool, test, k_lo, k_hi, a.n, seed=seeds[0])
            R["tasks"][nm] = {"flips": flips, "n_flips": len(flips), "n_total": len(allrec)}
            write_json_atomic(out, R)
            print(f"{nm}: {len(flips)} flips / {len(allrec)} examples (K{k_lo}-correct -> K{k_hi}-wrong)", flush=True)
        print("DONE")
        return

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    config = {"schema_version": 3, "experiment": "exploratory_vqa_curves", "model": a.model,
              "model_revision": model_revision(a.model), "datasets": names,
              "dataset_revisions": {name: DATASET_REVISIONS.get(DATASETS[name][0]) for name in names},
              "ks": ks, "seeds": seeds, "n": a.n, "n_pool": max(32, max(ks)),
              "img_size": 112, "max_new_tokens": 10, "dtype": "bfloat16",
              "metric": "official-style normalized VQA leave-one-annotator-out",
              "software_versions": software_versions(), "source_sha256": implementation_manifest()}
    R = load_result(a.out, config, {"model": a.model, "ks": ks, "seeds": seeds})
    if not set(R).issubset({"run_config", "model", "ks", "seeds", *names}):
        raise ValueError("unexpected resumed VQA result fields")
    pending = [name for name in names if name not in R]
    if not pending:
        print("DONE", flush=True)
        return
    model, proc = load_model(a.model, dtype=torch.bfloat16)
    for nm in names:
        if nm in R:
            continue
        pool, test = load_task(nm, max(32, max(ks)), a.n)
        R[nm] = agg_kcurve(model, proc, pool, test, ks, a.n, seeds)
        write_json_atomic(a.out, R)
        print(f"{nm}: {R[nm]['mean']} (+/- {R[nm]['ci']})", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
