"""Structured, deterministic metric records."""
from dataclasses import asdict, dataclass
import json
@dataclass(frozen=True)
class TrainMetric: step:int; loss:float; learning_rate:float; tokens_per_second:float; elapsed_seconds:float=0.0; tokens_processed:int=0; peak_vram_allocated_bytes:int|None=None; peak_vram_reserved_bytes:int|None=None
def write_metrics(path, metrics):
    with open(path,"w",encoding="utf-8") as f:
        for metric in metrics: f.write(json.dumps(asdict(metric),sort_keys=True)+"\n")
