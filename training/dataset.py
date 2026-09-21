"""Explicit TinyStories export, tokenization, EOS handling, and packing."""
from __future__ import annotations

from array import array
from dataclasses import dataclass
from hashlib import sha256
import argparse, itertools, json
from pathlib import Path
from typing import Iterable, Iterator
import yaml

from tokenizer.train_tokenizer import load_tokenizer

@dataclass(frozen=True)
class DatasetConfig:
    repo_id: str; revision: str; text_field: str; train_split: str; validation_split: str; sequence_length: int; modes: dict[str, dict[str, int | None]]

def load_dataset_config(path: str | Path) -> DatasetConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["dataset"]
    config = DatasetConfig(**data)
    if config.sequence_length < 2 or not config.revision or not config.modes:
        raise ValueError("Dataset config needs a pinned revision, modes, and sequence_length >= 2.")
    return config

def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()

def stream_documents(config: DatasetConfig, split: str, limit: int | None) -> Iterator[str]:
    """Explicitly stream a pinned split; imports Datasets only when invoked."""
    from datasets import load_dataset
    rows = load_dataset(config.repo_id, split=split, revision=config.revision, streaming=True, trust_remote_code=False)
    for row in itertools.islice(rows, limit):
        text = row[config.text_field]
        if isinstance(text, str) and text: yield text

def export_text(config: DatasetConfig, mode: str, output_dir: Path) -> dict:
    """Download only the selected streamed subset into ordered text files."""
    if mode not in config.modes: raise ValueError(f"Unknown mode: {mode}")
    if output_dir.exists(): raise FileExistsError(f"Output exists: {output_dir}")
    output_dir.mkdir(parents=True)
    metadata = {"repo_id": config.repo_id, "revision": config.revision, "mode": mode, "text_field": config.text_field, "splits": {}}
    for label, split in (("train", config.train_split), ("validation", config.validation_split)):
        path = output_dir / f"{label}.txt"; count = 0
        with path.open("w", encoding="utf-8") as f:
            for text in stream_documents(config, split, config.modes[mode][f"{label}_examples"]):
                f.write(text.replace("\n", " ") + "\n"); count += 1
        metadata["splits"][label] = {"source_split": split, "documents": count, "path": path.name, "sha256": _sha(path)}
    (output_dir / "source_info.json").write_text(json.dumps(metadata, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    return metadata

def pack_documents(documents: Iterable[str], tokenizer_path: str | Path, sequence_length: int) -> tuple[list[list[int]], int, int]:
    """Append EOS per document, concatenate, and emit fixed 513-token blocks."""
    tokenizer = load_tokenizer(tokenizer_path); eos = tokenizer.token_to_id("<eos>")
    if eos is None: raise ValueError("Tokenizer artifact lacks <eos>.")
    block_size, buffer, blocks, documents_count, token_count = sequence_length + 1, [], [], 0, 0
    for text in documents:
        ids = tokenizer.encode(text, add_special_tokens=False).ids + [eos]
        buffer.extend(ids); documents_count += 1; token_count += len(ids)
        while len(buffer) >= block_size:
            blocks.append(buffer[:block_size]); del buffer[:block_size]
    return blocks, documents_count, token_count

def prepare_packed_split(documents: Iterable[str], tokenizer_path: str | Path, sequence_length: int, output_path: Path) -> dict:
    blocks, docs, tokens = pack_documents(documents, tokenizer_path, sequence_length)
    if output_path.exists(): raise FileExistsError(f"Output exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        for block in blocks: array("I", block).tofile(f)
    info = {"documents": docs, "tokens_including_eos": tokens, "packed_sequences": len(blocks), "sequence_tokens": sequence_length + 1, "dropped_tail_tokens": tokens - len(blocks)*(sequence_length+1), "path": output_path.name, "sha256": _sha(output_path), "tokenizer_path": str(tokenizer_path), "tokenizer_sha256": _sha(Path(tokenizer_path)), "eos_handling": "one <eos> appended per input document"}
    output_path.with_suffix(output_path.suffix + ".info.json").write_text(json.dumps(info, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    return info

def main() -> None:
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="command", required=True)
    e=sub.add_parser("export"); e.add_argument("--config", type=Path, default=Path("configs/tinystories.yaml")); e.add_argument("--mode", default="smoke"); e.add_argument("--output-dir", type=Path, required=True)
    q=sub.add_parser("pack"); q.add_argument("--input", type=Path, required=True); q.add_argument("--tokenizer", type=Path, required=True); q.add_argument("--output", type=Path, required=True); q.add_argument("--sequence-length", type=int, default=512)
    a=p.parse_args()
    if a.command=="export": print(json.dumps(export_text(load_dataset_config(a.config),a.mode,a.output_dir),indent=2))
    else:
        docs=[line.rstrip("\n") for line in a.input.open(encoding="utf-8") if line.strip()]
        print(json.dumps(prepare_packed_split(docs,a.tokenizer,a.sequence_length,a.output),indent=2))
if __name__ == "__main__": main()
