"""Decode token IDs with an existing tokenizer artifact; this command never trains."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .train_tokenizer import load_tokenizer


def decode_ids(artifact_path: str | Path, token_ids: Sequence[int]) -> str:
    """Decode IDs while retaining explicit special tokens for inspection."""
    return load_tokenizer(artifact_path).decode(list(token_ids), skip_special_tokens=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--ids", required=True, help="JSON list of integer token IDs")
    args = parser.parse_args()
    token_ids = json.loads(args.ids)
    if not isinstance(token_ids, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in token_ids):
        parser.error("--ids must be a JSON list of integers")
    print(decode_ids(args.tokenizer, token_ids))


if __name__ == "__main__":
    main()

