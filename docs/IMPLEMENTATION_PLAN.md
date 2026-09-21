# Implementation Plan

## Foundation and milestones

- **Phase 1:** configuration and a parameter-count utility, validated by configuration and tolerance tests.
- **Phase 2:** readable Transformer modules under `model/`, validated by shape, causal-leakage, forward, loss, and backward tests.
- **Phase 3:** shared deterministic CPU test suite, run with `pytest`.
- **Phases 4–5:** tokenizer and TinyStories preparation/packing, validated by encode/decode and dataset-metadata tests.
- **Phase 6:** local smoke training, checking optimizer behavior, metrics, and a tiny loss-reduction run.
- **Phases 7–9:** Colab notebook, checkpoint/resume, evaluation, generation, and plots, validated from a fresh runtime and through checkpoint round trips.
- **Phase 10:** documented baseline run with saved artifacts and measured metrics.
- **Phase 11:** optional Jev harness with mock-client, policy, and injection tests.
- **Phases 12–14:** reversible model, controlled comparison, and research report, validated through matched-config experiments.

## Planned files and dependencies

Phase 1 creates the configuration and count utility. Later phases add the requested module files under `model/`, `tokenizer/`, `training/`, `evaluation/`, `jev/`, and `providers/`; `experiments/baseline/` and `experiments/reversible/` will hold comparison-specific material.

The minimal planned Python dependencies are pinned PyTorch, PyYAML, pytest, Hugging Face `datasets`, Hugging Face `tokenizers`, and plotting support (matplotlib). Versions will be verified against current official documentation before they are pinned. Jev and any coding-provider SDK remain optional; their official APIs must be verified before implementation.

## Colab workflow and artifacts

The notebook will clone the repository, install pinned dependencies, mount Drive, run diagnostics, prepare artifacts, execute smoke checks, train, resume, evaluate, generate samples, and save plots. Each `runs/<run_id>/` directory will contain copied configuration, model/tokenizer/dataset metadata, metrics JSONL, logs, checkpoints, samples, plots, and system information.

## Risks and approval decisions

Key risks are Colab GPU variability, runtime disconnections, TinyStories revision availability, and tokenizer-training cost. The default decisions requiring approval are: parameter tolerance around 20M, TinyStories revision and train/validation policy, pinned dependency versions, baseline training-token budget, and persistent Drive location. Phase 1 can proceed without committing to a final training budget; later phases cannot.
