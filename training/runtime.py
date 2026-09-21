"""Single device/precision policy used by every training path."""
from dataclasses import dataclass
import torch
@dataclass(frozen=True)
class Runtime:
 device: torch.device; precision: str
def resolve_runtime(device: str='auto', precision: str='auto') -> Runtime:
 if device not in ('auto','cpu','cuda'): raise ValueError('device must be auto, cpu, or cuda')
 chosen='cuda' if device=='auto' and torch.cuda.is_available() else ('cpu' if device=='auto' else device)
 if chosen=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
 resolved='fp32' if chosen=='cpu' and precision=='auto' else ('bf16' if chosen=='cuda' and precision=='auto' and torch.cuda.is_bf16_supported() else ('fp16' if chosen=='cuda' and precision=='auto' else precision))
 if chosen=='cpu' and resolved != 'fp32': raise ValueError('CPU supports fp32 only in this training loop')
 if resolved not in ('fp32','fp16','bf16'): raise ValueError('precision must be fp32, fp16, bf16, or auto')
 return Runtime(torch.device(chosen),resolved)
def autocast_context(runtime: Runtime):
 dtype={'fp16':torch.float16,'bf16':torch.bfloat16}.get(runtime.precision,torch.float32)
 return torch.autocast(device_type=runtime.device.type,dtype=dtype,enabled=runtime.device.type=='cuda' and runtime.precision!='fp32')
def make_scaler(runtime: Runtime): return torch.amp.GradScaler('cuda',enabled=runtime.device.type=='cuda' and runtime.precision=='fp16')
