#!/usr/bin/env python3
"""Cached real-model parity, episode isolation, and finite-generation checks."""
import argparse
import math
import random

import torch

from incontext_collapse.cama import (CamaController, build_classification_messages,
                                    build_qa_messages)
from incontext_collapse.core import TASKS, balanced_demos, build_messages
from incontext_collapse.models import load_model
from incontext_collapse.protocol import sample_option_order
from incontext_collapse.results import write_json_atomic


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps")
    parser.add_argument("--prompt", choices=["all", "native", "qa", "classification"], default="all")
    parser.add_argument("--option-order", choices=["random", "class"], default="random")
    parser.add_argument("--out", default=".smoke/cama_followup_validation.json")
    args = parser.parse_args()
    model, processor = load_model("Qwen/Qwen2-VL-2B-Instruct", dtype=torch.bfloat16,
                                  device=args.device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    versions = [p._version for p in model.parameters()]
    names, pool, test = TASKS["shapes4"](8, 8)
    demos = balanced_demos(pool, 4, random.Random(12))
    order = sample_option_order(len(names), 12, 4, 0, args.option_order)
    builders = {"native": build_messages, "qa": build_qa_messages,
                "classification": build_classification_messages}
    controller = None
    checks = []
    try:
        for name, builder in builders.items():
            if args.prompt != "all" and args.prompt != name:
                continue
            messages = builder(demos, test[0][0], names, 112, option_order=order)
            inputs = processor.apply_chat_template(messages, tokenize=True,
                add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
            native = model.generate(**inputs, max_new_tokens=6, do_sample=False, use_cache=True)
            expected = processor.batch_decode(native[:, inputs["input_ids"].shape[1]:],
                                               skip_special_tokens=True)[0]
            controller = CamaController(model, processor)
            disabled = controller.generate(model, processor, messages, enabled=False)
            if disabled != expected:
                raise AssertionError(f"disabled generation differs: {disabled!r} vs {expected!r}")
            first = controller.generate(model, processor, messages)
            first_diagnostic = controller.diagnostics[-1]
            if any(not math.isfinite(g) for g in first_diagnostic["max_gain"]):
                raise AssertionError("nonfinite selection gain")
            if max(first_diagnostic["max_gain"]) <= 0:
                raise AssertionError("all Stage I gains are zero")
            other = builder(demos, test[1][0], names, 112, option_order=order)
            controller.generate(model, processor, other)
            repeated = controller.generate(model, processor, messages)
            if first != repeated:
                raise AssertionError("episode state leaked into repeated generation")
            if controller.diagnostics[-1]["selected_tokens"] != first_diagnostic["selected_tokens"]:
                raise AssertionError("selected tokens changed after unrelated episode")
            zero = builder([], test[0][0], names, 112, option_order=order)
            zero_off = controller.generate(model, processor, zero, enabled=False)
            zero_on = controller.generate(model, processor, zero, enabled=True)
            if zero_off != zero_on or controller.diagnostics[-1]["heads"]:
                raise AssertionError("zero-shot CAMA was not an identity")
            checks.append({"prompt": builder.__name__, "native": expected,
                           "disabled": disabled, "cama": first,
                           "repeat": repeated, "zero_shot": zero_on,
                           "diagnostics": first_diagnostic})
            controller.close()
            controller = None
            print(f"PASS {builder.__name__}: native={expected!r}, cama={first!r}", flush=True)
        if versions != [p._version for p in model.parameters()]:
            raise AssertionError("model parameters were mutated")
        write_json_atomic(args.out, {"device": str(model.device), "checks": checks,
                                    "option_order": args.option_order,
                                    "parameters_unchanged": True, "passed": True})
    finally:
        if controller is not None:
            controller.close()


if __name__ == "__main__":
    main()
