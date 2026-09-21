# Google Colab Setup

## Storage model

`/content` is ephemeral runtime storage: clone the source and prepare temporary data there. Google Drive is persistent storage for later checkpoints and exported artifacts. The Git repository is source code only; do not put generated data, tokens, or checkpoints in it.

## Fresh runtime

1. Open `notebooks/20m_baseline_colab.ipynb` in Colab.
2. Select **Runtime → Change runtime type → GPU**, then reconnect.
3. Set `REPOSITORY_URL` in section 02 to your Git remote.
4. Run sections 00–09 in order. The notebook installs the pinned Tokenizers and Datasets dependencies, reports Python/PyTorch/CUDA/GPU details, clones the repository, and runs the full CPU-safe test suite.

Colab GPU type and availability vary. If `torch.cuda.is_available()` is false, repeat step 2. The notebook recommends a conservative micro-batch: 8 for ≥16 GB, 4 for ≥12 GB, 2 for ≥8 GB, and 1 otherwise. Treat this as a starting point, not a silent experiment change.

## Drive

The Drive mount cell requests authorization because mounted code can access your Drive. Set `DRIVE_PROJECT_DIR` to a dedicated folder such as `/content/drive/MyDrive/20m-llm-lab`. Phase 7 creates it but does not yet write checkpoints. Phase 8 will make checkpoint persistence and resume operational.
