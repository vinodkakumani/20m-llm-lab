# Architecture

The baseline will be a decoder-only Transformer: token IDs `[B, T]` become embeddings `[B, T, D]`, pass through causal Transformer blocks, and produce vocabulary logits `[B, T, V]`. Blocks will use RoPE attention, RMSNorm, SwiGLU, residual connections, and tied input/output embeddings.

The baseline configuration is `V=32000`, `D=320`, six layers, five heads of dimension 64, FFN dimension 1280, and context length 512. Its deterministic count is **20,074,560 trainable parameters**, calculated from [the YAML configuration](../configs/baseline_20m.yaml) by `model.config.calculate_parameter_breakdown`; the test verifies that result rather than displaying a hard-coded runtime count.

This count assumes bias-free Q/K/V, attention-output, and SwiGLU projections; two RMSNorm scales per block plus a final scale; and a tied LM head, which therefore contributes no additional parameters. The acceptance policy is 20,000,000 ± 500,000 parameters (±2.5%).

## Baseline component flow

`TokenEmbedding` maps token IDs `[B, T]` to a residual stream `[B, T, D]`. Each `TransformerBlock` applies RMSNorm, RoPE causal self-attention, and a residual addition; it then applies another RMSNorm, SwiGLU MLP, and a second residual addition. Attention projects Q/K/V from `[B, T, D]` to `[B, H, T, Dh]`, applies a causal kernel, then returns `[B, T, D]`. A final RMSNorm and the embedding-tied LM head produce logits `[B, T, V]`.

For next-token language modeling, logits at position `t` are compared with the token ID at `t + 1` using cross-entropy. This shift is implemented by `BaselineTransformer.next_token_loss`.

The model code is deliberately split across `embeddings.py`, `rope.py`, `attention.py`, `normalization.py`, `mlp.py`, `transformer_block.py`, and `model.py`. Comprehensive component and causal-leakage tests remain Phase 3.
