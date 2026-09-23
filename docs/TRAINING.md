# Training

Training will use PyTorch, AdamW, cosine learning-rate decay with warmup, gradient clipping, and GPU mixed precision when supported. All hyperparameters will be stored in a YAML configuration, copied into each run directory, and never scattered through source files. Phase 1 establishes the initial architecture file at `configs/baseline_20m.yaml`; it is safely loaded and validated before later training fields are added.

Three configuration-controlled modes are planned: CPU/GPU smoke test, short development run, and baseline training run. Dataset size, seed, tokenizer artifact, precision, batch size, accumulation, and total training-token budget will be recorded per run.

Checkpoint and resume instructions will be added after their implementation phases.

## Tokenizer artifact

The baseline tokenizer is a byte-level BPE tokenizer with a requested vocabulary of 32,000 and explicit `<unk>`, `<bos>`, `<eos>`, and `<pad>` tokens. Train it once from an ordered corpus and retain both `tokenizer.json` and `tokenizer_info.json`. The metadata records the Tokenizers version, requested and achieved vocabulary sizes, special-token IDs, input file hashes, and tokenizer SHA-256.

`python -m tokenizer.train_tokenizer` refuses to write into an existing artifact directory. Training code will receive an existing tokenizer path and must never trigger retraining. EOS insertion belongs to the later dataset-packing phase, so encode/decode helpers do not silently add special tokens.

After Phase 5 has prepared a documented ordered text corpus, train the artifact once:

```bash
python3 -m tokenizer.train_tokenizer \
  --input data/tinystories/train.txt \
  --output-dir artifacts/tokenizers/tinystories_bpe_32k
```

Expected output names `tokenizer.json` and `tokenizer_info.json`. A tiny development corpus may produce fewer than 32,000 actual tokens because it does not contain enough distinct byte sequences; the metadata records the actual size. Use the same saved artifact for every comparable run.

## TinyStories preparation

The pinned source is `roneneldan/TinyStories` revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`. It uses its published `train` and `validation` splits; no custom random split is made. Export is explicit and streams only the selected smoke (128/32), development (10,000/1,000), or full subset:

```bash
python3 -m training.dataset export --mode smoke --output-dir data/raw/tinystories_smoke
```

Train the tokenizer from `data/raw/tinystories_smoke/train.txt`, then pack it with that saved artifact:

```bash
python3 -m training.dataset pack --input data/raw/tinystories_smoke/train.txt --tokenizer artifacts/tokenizers/tinystories_bpe_32k/tokenizer.json --output artifacts/datasets/smoke_train.bin
```

One `<eos>` is appended to every document before concatenation. The binary output contains `uint32` blocks of 513 tokens: training will use the first 512 as inputs and the next 512 shifted positions as labels. Its command output and adjacent `.bin.info.json` record document count, tokens including EOS, packed sequences, dropped tail, source storage path, tokenizer hash, and output SHA-256.

## Current local verification

Run the CPU-safe model checks from the repository root:

```bash
python3 -m unittest discover -s tests -v
```

The current suite validates configuration constraints, the exact parameter count, tensor shapes, RMSNorm, RoPE's position-zero behavior, shifted cross-entropy, backward propagation, one optimizer step, tied embeddings, and causal isolation. It does not download a dataset or start training.

## Local smoke training

After producing a packed smoke dataset, run two real CPU updates:

```bash
python3 -m training.train --dataset artifacts/datasets/smoke_train.bin --steps 2 --metrics smoke_metrics.jsonl
```

Expected output is two `TrainMetric` records with finite loss, learning rate, and tokens per second. This confirms the forward pass, shifted labels, backpropagation, AdamW update, gradient clipping, and warmup/cosine scheduler. It does not create resumable run artifacts; checkpointing begins in Phase 8.

## Checkpoint and resume

Every command now creates a new `runs/<UTC timestamp>_<name>/` directory with configuration, model/system metadata, metric log, samples directory, and checkpoints. Checkpoints are atomically written after each smoke step and include model, optimizer, scheduler, step, and Python/PyTorch RNG states. Resume only checkpoints created by this project and under your control.

```bash
python3 -m training.train --dataset artifacts/datasets/smoke_train.bin --steps 4 --run-name smoke
python3 -m training.train --dataset artifacts/datasets/smoke_train.bin --steps 4 --resume runs/<run_id>/checkpoints/step_00000002.pt
```

## Evaluation, samples, and plots

Phase 9 adds validation loss/perplexity, fixed-prompt generation, and plots. Evaluation consumes a checkpoint and packed validation artifact; generation uses the same tokenizer saved for the run. `evaluation.plot.plot_metrics(run_dir)` saves training loss, learning rate, and tokens/sec PNG files in `runs/<run_id>/plots/`. These outputs are measurements, not evidence that an architecture is better.

## Proposed full baseline

`configs/baseline_full.yaml` defines the first full baseline: 200,000,000 training tokens from pinned TinyStories, 512-token context, micro-batch 4, accumulation 8, effective batch 32, AdamW learning rate `3e-4`, 500 warmup steps, cosine decay, clipping at 1.0, and seed 42. Deterministic planning computes 16,384 tokens per optimizer step and **12,208 total steps** (rounding up the final partial budget).

This is a proposed experiment configuration, not a completed training result. For a T4 with memory pressure, begin with micro-batch 1 or 2 and increase accumulation to preserve effective batch 32; record the actual setting in the run configuration. On an L4, start with micro-batch 4. Do not silently alter the effective batch size.

Before launching, export and pack the full train/validation artifacts, train the final 32k tokenizer once from the selected train corpus, copy artifacts to Drive, and verify sections 00–09 of the Colab notebook. The current loop is CPU smoke-focused; GPU mixed precision and full-run execution must be added before this configuration can be launched.

## GPU qualification and precision

`device: auto` resolves to CUDA when available, otherwise CPU. `precision: auto` resolves to FP32 on CPU, BF16 on CUDA only when PyTorch reports support, otherwise FP16. CUDA FP16 uses `torch.autocast` plus `torch.amp.GradScaler`; BF16 uses autocast without scaling. Accumulated micro-losses are divided by accumulation steps before backward; clipping occurs after unscaling and scheduler steps occur once per optimizer step.

On an Apple Silicon Mac with MPS available, `device: auto` resolves to `mps` after CUDA and before CPU. MPS uses FP32 in this project for predictable local development behavior; it is useful for smoke/development runs but is not a substitute for CUDA AMP qualification.

Run the 100-step qualification configuration before the full baseline. It must show a CUDA device, resolved precision, finite loss, checkpoints, token throughput, peak VRAM, and correct token accounting. On OOM, stop: start a **new** run with `4×8` replaced by `2×16` or `1×32` to retain effective batch 32. Never silently apply this fallback.

For the full baseline: `target_training_tokens=200,000,000`, `planned_optimizer_steps=12,208`, and `planned_training_tokens=200,015,872`. Actual tokens are counted from completed micro-batches and recorded in metrics/checkpoints; they are never substituted with the target.
