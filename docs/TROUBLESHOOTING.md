# Troubleshooting

No executable pipeline exists in Phase 0. As phases are implemented, this guide will collect concrete diagnostics and remedies for dependency installation, tokenizer artifacts, dataset caching, CUDA/precision selection, out-of-memory errors, checkpoint resume, and Colab disconnections.

Do not delete a run directory to recover from a failure; preserve its logs and configuration for diagnosis.

## CUDA out of memory

The run stops with its requested configuration intact. Reduce micro-batch from 4 to 2 and increase accumulation from 8 to 16, or use 1 and 32 respectively; both preserve effective batch 32. Start a new run and record the change. Do not resume an interrupted run with changed batch semantics.

## T4 appears in `nvidia-smi`, but the run resolves to CPU

Run `python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`. A CUDA-enabled PyTorch build must report `True`; `nvidia-smi` alone only proves the driver can see the GPU. Restart the Colab GPU runtime, rerun the notebook dependency cells, and check again before qualifying the model.
