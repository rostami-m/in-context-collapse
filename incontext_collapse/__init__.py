"""In-Context Collapse: controlled evaluations and exploratory diagnostics.

The current primary evidence evaluates saved Protocol 2.4-trained adapters with
Protocol 2.5 randomized answer options. Historical aggregate sweeps retain their
original protocols and limitations. README.md maps claims to exact artifacts;
current driver defaults alone do not reproduce every historical experiment.

Drivers and roles:
* panel: classification behavior along robustness and mapping-learning axes.
* circuit: region-restricted training and repair-sufficiency controls.
* option_order_eval: matched classification evaluation of saved checkpoints.
* vaccine: legacy/exploratory natural-label composite-adapter domain transfer.
* compare: matched fixed-map diagnostic; Gate/Inject benefits remain unvalidated.
* inject: standalone exploratory activation injection, with known sampling limits.
* vqa: exploratory frozen-model VQA and shared scoring utilities.
* vqa_diverse: distinct-image VQA saved-checkpoint boundary evaluation.
* rule_transfer: paired left/right attribute-selection boundary evaluation.
* continual: exploratory, unmatched adaptation in one continual-learning stream.
* frontier: descriptive hosted-model probe with unresolved sampling/parser limits.
* cama/dara: local baseline implementations with documented compatibility limits.
* figures: rendering from preserved controlled records and historical aggregates.
"""
__all__ = [
    "core", "models", "panel", "circuit", "vaccine", "inject",
    "compare", "vqa", "continual", "frontier", "dara", "figures", "protocol",
    "scoring", "evaluation",
    "adapters",
    "results",
    "adapter_eval",
]
