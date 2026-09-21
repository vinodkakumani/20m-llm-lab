"""Local CPU smoke training; checkpointing is added in Phase 8."""
from __future__ import annotations
import argparse, time
from dataclasses import dataclass
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from model import BaselineTransformer, load_baseline_config
from .metrics import TrainMetric, write_metrics
from .checkpoint import create_run_directory, initialize_run, load_checkpoint, save_checkpoint
from .optimizer import build_adamw
from .scheduler import build_cosine_scheduler
from .runtime import autocast_context, make_scaler, resolve_runtime

class PackedTokenDataset(Dataset):
    def __init__(self,path: str|Path, sequence_length:int):
        self.path=Path(path); self.width=sequence_length+1; size=self.path.stat().st_size
        if size % (4*self.width): raise ValueError("Packed file is not aligned to sequence_length + 1 uint32 tokens.")
        self.tokens=torch.from_file(str(self.path),shared=False,size=size//4,dtype=torch.int32)
    def __len__(self): return self.tokens.numel()//self.width
    def __getitem__(self,index): return self.tokens[index*self.width:(index+1)*self.width].to(torch.long)

@dataclass(frozen=True)
class SmokeTrainingConfig:
    micro_batch_size:int=1; gradient_accumulation_steps:int=1; max_steps:int=2; learning_rate:float=3e-4; weight_decay:float=0.1; warmup_steps:int=0; max_grad_norm:float=1.0; seed:int=42; device:str='auto'; precision:str='auto'
    @property
    def effective_batch_size(self): return self.micro_batch_size*self.gradient_accumulation_steps

def run_smoke_training(model_config_path, dataset_path, config=SmokeTrainingConfig(), run_dir: Path|None=None, resume: Path|None=None) -> list[TrainMetric]:
    torch.manual_seed(config.seed); model_config=load_baseline_config(model_config_path).model; runtime=resolve_runtime(config.device,config.precision)
    model=BaselineTransformer(model_config).to(runtime.device); dataset=PackedTokenDataset(dataset_path,model_config.context_length)
    if len(dataset)==0: raise ValueError("Dataset contains no complete packed sequences.")
    loader=DataLoader(dataset,batch_size=config.micro_batch_size,shuffle=False); optimizer=build_adamw(model,config.learning_rate,config.weight_decay); scheduler=build_cosine_scheduler(optimizer,config.warmup_steps,config.max_steps); scaler=make_scaler(runtime)
    start_step=load_checkpoint(resume,model,optimizer,scheduler,runtime.device,scaler) if resume else 0
    if run_dir and not resume: initialize_run(run_dir,model_config,config)
    metrics=[]; iterator=iter(loader)
    tokens_processed=start_step*config.effective_batch_size*model_config.context_length
    if runtime.device.type=='cuda': torch.cuda.reset_peak_memory_stats(runtime.device)
    for step in range(start_step+1,config.max_steps+1):
        started=time.perf_counter(); optimizer.zero_grad(set_to_none=True); total_loss=0.0; step_tokens=0
        for _ in range(config.gradient_accumulation_steps):
            try: batch=next(iterator)
            except StopIteration: iterator=iter(loader); batch=next(iterator)
            batch=batch.to(runtime.device); step_tokens+=batch[:,1:].numel()
            with autocast_context(runtime): logits=model(batch[:,:-1]); loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),batch[:,1:].reshape(-1))
            total_loss+=loss.detach().item(); scaler.scale(loss/config.gradient_accumulation_steps).backward()
        scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),config.max_grad_norm); scaler.step(optimizer); scaler.update(); scheduler.step(); tokens_processed+=step_tokens
        elapsed=time.perf_counter()-started; alloc=torch.cuda.max_memory_allocated(runtime.device) if runtime.device.type=='cuda' else None; reserved=torch.cuda.max_memory_reserved(runtime.device) if runtime.device.type=='cuda' else None
        metrics.append(TrainMetric(step,total_loss/config.gradient_accumulation_steps,float(optimizer.param_groups[0]["lr"]),step_tokens/elapsed,elapsed,tokens_processed,alloc,reserved))
        if run_dir: save_checkpoint(run_dir/'checkpoints'/f'step_{step:08d}.pt',model,optimizer,scheduler,step,{"model_config":str(model_config_path),"dataset":str(dataset_path),"resolved_precision":runtime.precision},scaler,tokens_processed)
    if run_dir: write_metrics(run_dir/'metrics.jsonl',metrics)
    return metrics

def main():
    p=argparse.ArgumentParser(); p.add_argument("--model-config",type=Path,default=Path("configs/baseline_20m.yaml")); p.add_argument("--dataset",type=Path,required=True); p.add_argument("--steps",type=int,default=2); p.add_argument("--micro-batch",type=int,default=1); p.add_argument("--accumulation",type=int,default=1); p.add_argument("--device",default="auto"); p.add_argument("--precision",default="auto"); p.add_argument("--runs-root",type=Path,default=Path("runs")); p.add_argument("--run-name",default="smoke"); p.add_argument("--resume",type=Path); a=p.parse_args()
    run_dir=a.resume.parents[1] if a.resume else create_run_directory(a.runs_root,a.run_name)
    metrics=run_smoke_training(a.model_config,a.dataset,SmokeTrainingConfig(max_steps=a.steps,micro_batch_size=a.micro_batch,gradient_accumulation_steps=a.accumulation,device=a.device,precision=a.precision),run_dir,a.resume)
    print('Run directory:',run_dir)
    for metric in metrics: print(metric)
if __name__=="__main__": main()
