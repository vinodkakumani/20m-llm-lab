"""Train and persist a deterministic Byte-Level BPE tokenizer exactly once."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable, Sequence

import tokenizers
from tokenizers import Tokenizer, decoders, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


SPECIAL_TOKENS = ("<unk>", "<bos>", "<eos>", "<pad>")
ARTIFACT_FILENAME = "tokenizer.json"
INFO_FILENAME = "tokenizer_info.json"


@dataclass(frozen=True)
class TokenizerConfig:
    """Training configuration for the baseline tokenizer artifact."""

    vocab_size: int = 32_000
    min_frequency: int = 2
    special_tokens: tuple[str, ...] = SPECIAL_TOKENS

    def __post_init__(self) -> None:
        if self.vocab_size < len(self.special_tokens):
            raise ValueError("vocab_size must accommodate every special token.")
        if self.min_frequency < 1:
            raise ValueError("min_frequency must be at least 1.")
        if len(set(self.special_tokens)) != len(self.special_tokens):
            raise ValueError("special_tokens must be unique.")
        if self.special_tokens[0] != "<unk>":
            raise ValueError("The first special token must be <unk> for the BPE unknown token.")


def build_tokenizer(config: TokenizerConfig) -> Tokenizer:
    """Create an untrained byte-level BPE tokenizer with fixed processing rules."""
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def train_tokenizer(texts: Sequence[str], config: TokenizerConfig) -> Tokenizer:
    """Train BPE from an ordered, materialized sequence of non-empty documents.

    There is no random seed because BPE merge selection is deterministic for a
    fixed library version, configuration, and input order. The caller owns input
    ordering; dataset preparation will pass the documented TinyStories split.
    """
    if not texts:
        raise ValueError("Tokenizer training requires at least one document.")
    if any(not isinstance(text, str) or not text for text in texts):
        raise ValueError("Tokenizer training documents must be non-empty strings.")
    tokenizer = build_tokenizer(config)
    trainer = BpeTrainer(
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        special_tokens=list(config.special_tokens),
        show_progress=False,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_and_save_tokenizer(
    texts: Sequence[str],
    output_dir: str | Path,
    config: TokenizerConfig = TokenizerConfig(),
    source_files: Iterable[str | Path] = (),
) -> Path:
    """Train once and atomically establish a tokenizer artifact directory.

    The output directory must not already exist. This deliberate guard prevents
    model-training code from silently replacing a tokenizer used by a checkpoint.
    """
    artifact_dir = Path(output_dir)
    if artifact_dir.exists():
        raise FileExistsError(
            f"Tokenizer artifact directory already exists: {artifact_dir}. Reuse it; do not retrain."
        )
    tokenizer = train_tokenizer(texts, config)
    artifact_dir.mkdir(parents=True)
    artifact_path = artifact_dir / ARTIFACT_FILENAME
    tokenizer.save(str(artifact_path))

    source_paths = [Path(path) for path in source_files]
    info = {
        "schema_version": 1,
        "library": {"name": "tokenizers", "version": tokenizers.__version__},
        "algorithm": "byte_level_bpe",
        "config": asdict(config),
        "special_token_ids": {token: tokenizer.token_to_id(token) for token in config.special_tokens},
        "vocab_size_actual": tokenizer.get_vocab_size(),
        "document_count": len(texts),
        "source_files": [
            {"path": str(path), "sha256": _sha256_file(path)} for path in source_paths
        ],
        "tokenizer_sha256": _sha256_file(artifact_path),
    }
    (artifact_dir / INFO_FILENAME).write_text(
        json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return artifact_path


def load_tokenizer(artifact_path: str | Path) -> Tokenizer:
    """Load a previously trained tokenizer JSON artifact without retraining."""
    path = Path(artifact_path)
    if not path.is_file():
        raise FileNotFoundError(f"Tokenizer artifact not found: {path}")
    return Tokenizer.from_file(str(path))


def _read_documents(paths: Sequence[Path]) -> list[str]:
    documents: list[str] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            documents.extend(line.rstrip("\n") for line in handle if line.strip())
    return documents


def main() -> None:
    """Train a tokenizer artifact from one-document-per-line UTF-8 text files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="Ordered UTF-8 corpus files")
    parser.add_argument("--output-dir", type=Path, required=True, help="New artifact directory")
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()
    artifact_path = train_and_save_tokenizer(
        _read_documents(args.input),
        args.output_dir,
        TokenizerConfig(vocab_size=args.vocab_size, min_frequency=args.min_frequency),
        source_files=args.input,
    )
    print(f"Saved tokenizer: {artifact_path}")
    print(f"Saved metadata: {artifact_path.parent / INFO_FILENAME}")


if __name__ == "__main__":
    main()

