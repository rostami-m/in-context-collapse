"""Exploratory locus comparison in one continual-learning stream (Fig. 4).

Compares three locations for durable, weight-based adaptation in one sequential
CIFAR-100 stream (T tasks of C classes, reused single-token A/B/C/D labels). We
sequentially LoRA-finetune at each of three loci and, after every task, evaluate
all tasks seen so far -> an accuracy matrix giving Average Accuracy (AA) and
Backward Transfer / forgetting (BWT):

  circuit  connector + early/mid
  all      every language-model block
  late     the readout third

In this unmatched configuration the *late* readout has the highest AA and least
forgetting at the fewest parameters. Capacity, task order, and baselines are
unmatched, so this is not evidence of a general locus dissociation.
The CL-specific 28-layer partition is 0-9 / 10-18 / 19-27; it is separate
from the primary repair experiment's region partition. Future results record
the exact resolved modules and refuse to overwrite existing evidence.

Example::

    python -m incontext_collapse.continual --T 5 --C 4 --seeds 3
    python -m incontext_collapse.continual --quick   # fast smoke test
"""
from __future__ import annotations

import argparse
from pathlib import Path
import random

import torch
from peft import LoraConfig, get_peft_model

from .core import pick_device
from .results import write_json_atomic
from .protocol import seed_training
from .provenance import DATASET_REVISIONS, load_pinned_dataset, model_revision, software_versions

MODEL = "Qwen/Qwen2-VL-2B-Instruct"
LABELS = ["A", "B", "C", "D"]
REGIONS = {"circuit": ["connector", "early", "mid"], "all": ["all"], "late": ["late"]}
CL_PARTITION = "inclusive_decoder_index_thirds_v1"
CL_CONNECTOR_TARGETS = ("visual.merger.mlp.0", "visual.merger.mlp.2")
CL_PROJECTIONS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                  "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def cl_layer_bands(n_layers):
    """Partition decoder indices inclusively: 28 layers give 10/9/9 blocks."""
    if not isinstance(n_layers, int) or n_layers < 3:
        raise ValueError("continual learning requires at least three decoder layers")
    last = n_layers - 1
    first_end, middle_end = last // 3, 2 * last // 3
    return {"early": list(range(first_end + 1)),
            "mid": list(range(first_end + 1, middle_end + 1)),
            "late": list(range(middle_end + 1, n_layers))}


def cl_target_manifest(model, region, rank=8):
    """Resolve this Qwen2-VL CL experiment's modules and exact LoRA capacity."""
    config = model.config
    model_type = config.get("model_type") if isinstance(config, dict) else config.model_type
    if model_type != "qwen2_vl":
        raise ValueError("the continual-learning target policy supports Qwen2-VL only")
    if region not in REGIONS or not isinstance(rank, int) or rank < 1:
        raise ValueError("unknown CL region or invalid rank")
    bands = cl_layer_bands(len(model.model.layers))
    selected = (bands["early"] + bands["mid"] if region == "circuit" else
                bands["late"] if region == "late" else
                list(range(len(model.model.layers))))
    targets = list(CL_CONNECTOR_TARGETS) if region == "circuit" else []
    targets.extend(f"model.layers.{index}.{projection}"
                   for index in selected for projection in CL_PROJECTIONS)
    modules = dict(model.named_modules())
    details = []
    for name in targets:
        module = modules.get(name)
        if not isinstance(module, torch.nn.Linear):
            raise ValueError(f"CL target is missing or is not linear: {name}")
        details.append({"name": name, "in_features": module.in_features,
                        "out_features": module.out_features,
                        "lora_parameters": rank * (module.in_features + module.out_features)})
    return {"partition": CL_PARTITION, "region": region, "rank": rank,
            "n_decoder_layers": len(model.model.layers), "layer_bands": bands,
            "selected_decoder_layers": selected,
            "connector_included": region == "circuit", "target_modules": targets,
            "target_shapes": details, "n_targets": len(targets),
            "trainable_parameters": sum(item["lora_parameters"] for item in details)}


def cl_tasks(T, C, n_train, n_test):
    """Build disjoint CIFAR-100 tasks that reuse the same C output letters."""
    ds = load_pinned_dataset("uoft-cs/cifar100")
    prep = lambda im: im.convert("RGB").resize((112, 112))
    by = {}
    for split in ("train", "test"):
        idx = {}
        for r in ds[split]:
            idx.setdefault(r["fine_label"], []).append(r["img"])
        by[split] = idx
    tasks = []
    for t in range(T):
        cls = list(range(t * C, t * C + C))
        pool = {i: [prep(x) for x in by["train"][c][:n_train]] for i, c in enumerate(cls)}
        test = [(prep(x), i) for i, c in enumerate(cls) for x in by["test"][c][:n_test // C]]
        tasks.append((pool, test))
    return tasks


def label_token_ids(proc):
    tokens = {label: proc.tokenizer.encode(label, add_special_tokens=False) for label in LABELS}
    if any(len(value) != 1 for value in tokens.values()):
        raise ValueError("continual learning requires one-token answer labels")
    result = {label: value[0] for label, value in tokens.items()}
    if len(set(result.values())) != len(result):
        raise ValueError("continual answer labels must have distinct token IDs")
    return result


def _msg(img, C):
    return [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": f"Classify into one letter from {LABELS[:C]}. Answer:"}]}]


@torch.no_grad()
def accuracy(model, proc, test, C, lt):
    cor = 0
    for img, lab in test:
        inp = proc.apply_chat_template(_msg(img, C), add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="pt").to(model.device)
        out = model(**inp).logits[0, -1]
        pred = max(range(C), key=lambda i: out[lt[LABELS[i]]].item())
        cor += (pred == lab)
    return cor / len(test)


def train_task(pm, proc, pool, C, lt, steps, lr=1e-4, seed=0):
    pm.train()
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=lr)
    rng = random.Random(seed)
    for _ in range(steps):
        c = rng.randrange(C)
        img = rng.choice(pool[c])
        text = proc.apply_chat_template(_msg(img, C), add_generation_prompt=True, tokenize=False) + LABELS[c]
        inp = proc(text=[text], images=[img], return_tensors="pt").to(pm.device)
        if int(inp["input_ids"][0, -1]) != lt[LABELS[c]]:
            raise ValueError("answer is not the final single token; refusing incorrect loss masking")
        lab = inp["input_ids"].clone()
        lab[:, :-1] = -100
        loss = pm(**inp, labels=lab).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
    pm.eval()


def main():
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--T", type=int, default=5)
    ap.add_argument("--C", type=int, default=4)
    ap.add_argument("--n-train", type=int, default=48)
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--regions", default="circuit,all,late")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="data/continual/continual.json")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.T, a.steps, a.n_train, a.n_test, a.seeds = 2, 15, 12, 8, 2
    regions = a.regions.split(",")
    if not regions or len(set(regions)) != len(regions) or any(r not in REGIONS for r in regions):
        ap.error("regions must be unique members of circuit,all,late")
    if not (1 <= a.C <= len(LABELS) and 1 <= a.T and a.T * a.C <= 100
            and min(a.steps, a.n_train, a.n_test, a.seeds) > 0 and a.n_test % a.C == 0
            and a.n_train <= 500 and a.n_test <= 100 * a.C):
        ap.error("invalid task dimensions, sample counts, steps, or seeds")
    if Path(a.out).exists():
        ap.error("output already exists; choose a new path to preserve existing evidence")

    device = pick_device()
    revision = model_revision(MODEL)
    proc = AutoProcessor.from_pretrained(MODEL, revision=revision, min_pixels=4 * 28 * 28, max_pixels=64 * 28 * 28)
    lt = label_token_ids(proc)
    tasks = cl_tasks(a.T, a.C, a.n_train, a.n_test)
    R = {"model": MODEL, "T": a.T, "C": a.C,
         "run_config": {"schema_version": 3, "model_revision": revision,
                        "software_versions": software_versions(),
                        "dataset": "uoft-cs/cifar100",
                        "dataset_revision": DATASET_REVISIONS["uoft-cs/cifar100"],
                        "historical_reconstruction": False, "experiment": "continual_four_way_stream",
                        "target_partition": CL_PARTITION, "model": MODEL,
                        "T": a.T, "C": a.C, "n_train_per_class": a.n_train,
                        "n_test_per_task": a.n_test, "steps_per_task": a.steps,
                        "seeds": list(range(a.seeds)), "regions": regions,
                        "rank": 8, "lora_alpha": 16, "lora_dropout": 0.05,
                        "learning_rate": 1e-4, "torch_initialization_seeded": True,
                        "sampling_seed": "task_index + 1000 * seed",
                        "score_mode": "restricted_single_letter_logits",
                        "labels": LABELS[:a.C], "img_size": 112,
                        "device": str(device), "dtype": "bfloat16"}}

    for region in regions:
        per = []
        ntr = 0
        manifest = None
        for S in range(a.seeds):
            seed_training(S)
            base = Qwen2VLForConditionalGeneration.from_pretrained(
                MODEL, revision=revision, torch_dtype=torch.bfloat16).to(device).eval()
            resolved = cl_target_manifest(base, region)
            if manifest is not None and resolved != manifest:
                raise RuntimeError("CL adapter targets changed across seeds")
            manifest = resolved
            targets = manifest["target_modules"]
            pm = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, target_modules=targets, lora_dropout=0.05))
            ntr = sum(p.numel() for p in pm.parameters() if p.requires_grad)
            if ntr != manifest["trainable_parameters"]:
                raise RuntimeError("installed CL adapter capacity differs from its target manifest")
            M = [[None] * a.T for _ in range(a.T)]
            for t in range(a.T):
                train_task(pm, proc, tasks[t][0], a.C, lt, a.steps, seed=t + 1000 * S)
                for j in range(t + 1):
                    M[t][j] = accuracy(pm, proc, tasks[j][1], a.C, lt)
            AA = sum(M[a.T - 1][j] for j in range(a.T)) / a.T
            BWT = sum(M[a.T - 1][j] - M[j][j] for j in range(a.T - 1)) / max(1, a.T - 1)
            per.append({"seed": S, "AA": round(AA, 3), "BWT": round(BWT, 3), "matrix": M})
            del base, pm
            if device == "cuda":
                torch.cuda.empty_cache()
        AAs = [p["AA"] for p in per]
        BWTs = [p["BWT"] for p in per]
        mean = lambda x: round(sum(x) / len(x), 3)
        sd = lambda x: round((sum((v - sum(x) / len(x)) ** 2 for v in x) / len(x)) ** 0.5, 3)
        R[region] = {"AA": mean(AAs), "AA_sd": sd(AAs), "BWT": mean(BWTs), "BWT_sd": sd(BWTs),
                     "trainable_params": ntr, "target_manifest": manifest, "per_seed": per}
        write_json_atomic(a.out, R)
        print(f"{region}: AA={mean(AAs)}+/-{sd(AAs)} BWT={mean(BWTs)}+/-{sd(BWTs)} params={ntr}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
