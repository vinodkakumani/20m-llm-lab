"""Atomic, trusted-local training checkpoints and run-directory metadata."""
from __future__ import annotations
import json, os, platform, random, shutil, sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import torch

def create_run_directory(root: str|Path, name: str) -> Path:
    run_id=f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{name}"
    path=Path(root)/run_id
    path.mkdir(parents=True, exist_ok=False)
    (path/'checkpoints').mkdir(); (path/'samples').mkdir()
    return path

def initialize_run(run_dir: Path, model_config, training_config, tokenizer_info: Path|None=None, dataset_info: Path|None=None) -> None:
    (run_dir/'model_config.json').write_text(json.dumps(asdict(model_config),indent=2,sort_keys=True)+"\n")
    (run_dir/'config.yaml').write_text("training:\n"+"\n".join(f"  {k}: {v}" for k,v in asdict(training_config).items())+"\n")
    info={"python":sys.version,"pytorch":torch.__version__,"cuda":torch.version.cuda,"platform":platform.platform(),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"gpu_memory_bytes":torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,"git_commit":_git_commit(),"seed":training_config.seed}
    (run_dir/'system_info.json').write_text(json.dumps(info,indent=2,sort_keys=True)+"\n")
    for source,target in ((tokenizer_info,'tokenizer_info.json'),(dataset_info,'dataset_info.json')):
        if source: shutil.copy2(source,run_dir/target)
        else: (run_dir/target).write_text(json.dumps({"status":"not supplied for this run"})+"\n")
    (run_dir/'metrics.jsonl').touch(); (run_dir/'training.log').touch()

def _git_commit() -> str|None:
    import subprocess
    try: return subprocess.check_output(['git','rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
    except Exception: return None

def _rng_state():
    return {"python":random.getstate(),"torch_cpu":torch.get_rng_state(),"torch_cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

def _restore_rng(state):
    random.setstate(state['python']); torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda'] is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state['torch_cuda'])

def save_checkpoint(path: Path, model, optimizer, scheduler, step: int, config: dict, scaler=None, tokens_processed: int=0) -> None:
    payload={"model_state":model.state_dict(),"optimizer_state":optimizer.state_dict(),"scheduler_state":scheduler.state_dict(),"scaler_state":scaler.state_dict() if scaler else None,"step":step,"tokens_processed":tokens_processed,"config":config,"rng_state":_rng_state()}
    temporary=path.with_suffix(path.suffix+'.tmp'); torch.save(payload,temporary); os.replace(temporary,path)

def load_checkpoint(path: Path, model, optimizer, scheduler, device='cpu', scaler=None) -> int:
    """Load only a checkpoint created locally by this project; PyTorch uses pickle."""
    payload=torch.load(path,map_location=device,weights_only=False)
    model.load_state_dict(payload['model_state']); optimizer.load_state_dict(payload['optimizer_state']); scheduler.load_state_dict(payload['scheduler_state']); _restore_rng(payload['rng_state'])
    if scaler and payload.get('scaler_state') is not None: scaler.load_state_dict(payload['scaler_state'])
    return int(payload['step'])
