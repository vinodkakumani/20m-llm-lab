"""Tests for deterministic tokenizer artifacts and basic encode/decode behavior."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tokenizer.decode import decode_ids
from tokenizer.encode import encode_text
from tokenizer.train_tokenizer import (
    INFO_FILENAME,
    SPECIAL_TOKENS,
    TokenizerConfig,
    load_tokenizer,
    train_and_save_tokenizer,
)


CORPUS = [
    "Once upon a time, a tiny dragon smiled.",
    "The little dragon liked stories about the moon.",
    "One morning, the dog wanted a story.",
]


class TokenizerTests(unittest.TestCase):
    def test_special_tokens_have_stable_leading_ids(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            artifact_path = train_and_save_tokenizer(
                CORPUS, Path(temporary_directory) / "tokenizer", TokenizerConfig(vocab_size=64)
            )
            tokenizer = load_tokenizer(artifact_path)

            self.assertEqual([tokenizer.token_to_id(token) for token in SPECIAL_TOKENS], [0, 1, 2, 3])

    def test_encode_decode_round_trip_and_saved_metadata(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "tokenizer"
            artifact_path = train_and_save_tokenizer(CORPUS, output_dir, TokenizerConfig(vocab_size=64))
            text = "Once upon a time, a tiny dragon smiled."

            token_ids = encode_text(artifact_path, text)
            metadata = json.loads((output_dir / INFO_FILENAME).read_text(encoding="utf-8"))

            self.assertEqual(decode_ids(artifact_path, token_ids), text)
            self.assertEqual(metadata["algorithm"], "byte_level_bpe")
            self.assertEqual(metadata["config"]["vocab_size"], 64)
            self.assertEqual(len(metadata["tokenizer_sha256"]), 64)

    def test_existing_artifact_directory_cannot_be_silently_retrained(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "tokenizer"
            train_and_save_tokenizer(CORPUS, output_dir, TokenizerConfig(vocab_size=64))

            with self.assertRaisesRegex(FileExistsError, "Reuse it; do not retrain"):
                train_and_save_tokenizer(CORPUS, output_dir, TokenizerConfig(vocab_size=64))

    def test_invalid_tokenizer_configuration_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            TokenizerConfig(vocab_size=32, special_tokens=("<unk>", "<unk>"))


if __name__ == "__main__":
    unittest.main()

