# In-Context Collapse in Vision-Language Models and How to Mitigate It

This repository contains the implementation used for controlled experiments on
in-context collapse and adapter-based mitigation in vision-language models.

The public repository intentionally contains code only. It excludes the
manuscript, reviewer correspondence, generated figures, raw or derived
experimental results, prepared image frames, model checkpoints, local audit
records, and private revision history.

## Installation

Python 3.12 is the recorded environment.

```bash
uv venv --python 3.12 .venv
uv pip sync --python .venv/bin/python requirements.lock
source .venv/bin/activate
python scripts/check_env.py
```

`requirements.lock` pins the complete primary environment. Model and dataset
identities are pinned in `incontext_collapse/provenance.py`. Large model weights
and datasets are downloaded separately from their original providers and remain
subject to their licenses and access requirements.

## Experiment entry points

Each maintained experiment module provides its own command-line help:

```bash
python -m incontext_collapse.circuit --help
python -m incontext_collapse.option_order_eval --help
python -m incontext_collapse.calibration --help
python -m incontext_collapse.frozen_features --help
python -m incontext_collapse.paired_intervention --help
python -m incontext_collapse.shortcut_controls --help
python -m incontext_collapse.rule_factorial --help
python -m incontext_collapse.vqa_confirmatory --help
python -m incontext_collapse.cama --help
python -m incontext_collapse.dara --help
```

New outputs should always use fresh paths. Resume checks bind configurations,
source versions, model revisions, and checkpoint content so that incompatible
results are not mixed silently.

The primary controlled classification path uses balanced episodes, strict output
parsing, explicit greedy decoding, pinned model/dataset revisions, complete
answer-span supervision, and per-query records. VQA evaluation uses normalized
ten-answer leave-one-out scoring. Statistical intervals and decision criteria
are implemented in the experiment and audit modules.

## Tests

Run the code-only unit suite with:

```bash
make test
```

Some integration tests and table/figure exporters require the retained
experimental artifact bundle, which is deliberately not published in this
code-only repository. Use `make test-all` only in a checkout that also has
those artifacts.

## Scope

The code supports reconstruction and new execution when the required model,
dataset, checkpoint, and prepared-data inputs are supplied. This code-only
release does not by itself reproduce numerical manuscript tables because the
underlying experimental artifacts are excluded by design.

## Citation

See `CITATION.cff`.

## License

MIT. See `LICENSE`.
