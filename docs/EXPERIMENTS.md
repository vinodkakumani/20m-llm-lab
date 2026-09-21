# Experiments

No experiments have been run yet.

## Planned baseline: `baseline_20m_tinystories_200m`

This planned experiment uses 20,074,560 parameters, pinned TinyStories revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, a newly saved 32k BPE artifact, context 512, 200,000,000 training tokens, seed 42, effective batch 32, AdamW, cosine decay, and mixed precision selected by GPU capability. GPU, wall time, final losses, perplexity, and run ID are intentionally blank until measured.

Before it, run `baseline_gpu_qualification`: 100 optimizer steps with the same architecture, data format, tokenizer, optimizer, AMP path, checkpoint system, and context length. Record GPU, resolved precision, peak VRAM, tokens/sec, actual tokens, checkpoint/resume result, validation, and samples before approving the full run.

Each completed experiment will record its ID, date, Git commit, model parameter count, TinyStories revision and split, tokenizer artifact, training-token budget, GPU, batch and accumulation settings, precision, optimizer and scheduler, steps, duration, final losses, perplexity, and notes. Comparative claims require these recorded measurements.
