"""Encode text with an existing tokenizer artifact; this command never trains."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .train_tokenizer import load_tokenizer


def encode_text(artifact_path: str | Path, text: str) -> list[int]:
    """Return token IDs for text without implicitly adding BOS or EOS tokens."""
    return load_tokenizer(artifact_path).encode(text, add_special_tokens=False).ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    print(json.dumps(encode_text(args.tokenizer, args.text)))


if __name__ == "__main__":
    main()

