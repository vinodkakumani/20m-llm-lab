"""Single device/precision policy used by every training path."""
from dataclasses import dataclass
import torch
@dataclass(frozen=True)
class Runtime:
 device: torch.device; precision: str
def resolve_runtime(device: str='auto', precision: str='auto') -> Runtime:
 if device not in ('auto','cpu','cuda','mps'): raise ValueError('device must be auto, cpu, cuda, or mps')
 chosen='cuda' if device=='auto' and torch.cuda.is_available() else ('mps' if device=='auto' and torch.backends.mps.is_available() else ('cpu' if device=='auto' else device))
 if chosen=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
 if chosen=='mps' and not torch.backends.mps.is_available(): raise RuntimeError('MPS requested but unavailable')
 resolved='fp32' if chosen in ('cpu','mps') and precision=='auto' else ('bf16' if chosen=='cuda' and precision=='auto' and torch.cuda.is_bf16_supported() else ('fp16' if chosen=='cuda' and precision=='auto' else precision))
 if chosen in ('cpu','mps') and resolved != 'fp32': raise ValueError('CPU and MPS use fp32 in this training loop')
 if resolved not in ('fp32','fp16','bf16'): raise ValueError('precision must be fp32, fp16, bf16, or auto')
 return Runtime(torch.device(chosen),resolved)
def autocast_context(runtime: Runtime):
 dtype={'fp16':torch.float16,'bf16':torch.bfloat16}.get(runtime.precision,torch.float32)
 return torch.autocast(device_type=runtime.device.type,dtype=dtype,enabled=runtime.device.type=='cuda' and runtime.precision!='fp32')
def make_scaler(runtime: Runtime): return torch.amp.GradScaler('cuda',enabled=runtime.device.type=='cuda' and runtime.precision=='fp16')
