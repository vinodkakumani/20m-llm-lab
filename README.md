# 20M Parameter LLM Research Lab

An educational, reproducible project for training and evaluating an approximately 20M-parameter decoder-only Transformer on TinyStories. The first experiment is a baseline; a reversible Transformer will be compared only after that baseline is trustworthy.

## Status

**Phase 10.5 — GPU qualification support** is complete. No full run has been launched.

## Planned layout

`model/` holds readable Transformer components. `tokenizer/`, `training/`, and `evaluation/` contain the data-to-results pipeline. `configs/` holds versioned YAML experiment settings. `tests/` contains CPU-safe unit tests. `notebooks/` contains the Colab workflow. `jev/` and `providers/` are optional, isolated control/provider integrations. Generated run artifacts will live in `runs/<run_id>/` and are not committed.

## Next step

Jev remains deferred until explicit approval after GPU qualification.
