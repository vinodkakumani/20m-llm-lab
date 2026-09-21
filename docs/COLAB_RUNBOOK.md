# Google Colab Runbook

Use [the notebook](../notebooks/20m_baseline_colab.ipynb) top to bottom.

### Step 01 — Verify GPU

**What you do:** run the GPU cell.

**What you should see:** CUDA availability, `nvidia-smi` output, GPU name, VRAM, and BF16 capability.

**What can go wrong:** CUDA is unavailable.

**Fix:** select **Runtime → Change runtime type → GPU**, reconnect, then rerun.

### Step 02–03 — Clone and install

**What you do:** set your Git URL, clone it under `/content`, and run the pinned installation cell.

**What you should see:** the repository path and installed versions.

**What can go wrong:** private-repository authentication or dependency conflict.

**Fix:** use an accessible remote; restart the runtime after resolving a conflicting package, then rerun from section 00.

### Step 04–06 — Export, tokenize, and pack

**What you do:** begin with `smoke` mode. Export streams only 128 training and 32 validation stories, train one tokenizer artifact, then pack the training text.

**What you should see:** `source_info.json`, `tokenizer.json`, `tokenizer_info.json`, `smoke_train.bin`, and `smoke_train.bin.info.json`.

**What can go wrong:** rerunning tokenizer training reports that the output exists.

**Fix:** reuse the saved artifact. Do not delete it merely to retry training.

### Step 07–09 — Verify and smoke train

**What you do:** run parameter, forward-pass, test-suite, and two-step smoke-training cells.

**What you should see:** `20,074,560` parameters, logits ending in `32000`, all tests passing, and two finite metrics.

**What can go wrong:** out of memory or a missing packed file.

**Fix:** start with smoke mode and check the previous artifact paths. CPU smoke training is valid if a GPU is unavailable.

### Step 12 — Checkpoint and resume

**What you do:** run training with a Drive-backed `runs/` root, then resume using a checkpoint path.

**What you should see:** a new run directory containing `config.yaml`, model/tokenizer/dataset metadata, metrics, logs, and `checkpoints/step_*.pt`.

**What can go wrong:** a Colab disconnect or an untrusted checkpoint.

**Fix:** reconnect, clone the same Git commit, remount Drive, and resume only your locally-created checkpoint. Do not load checkpoint files from unknown sources.

### Steps 10–11 and 13–16 — Deferred operations

Real GPU training, validation, generation, plots, summary, and final artifact export require Phases 9–10. The notebook marks these cells explicitly.
