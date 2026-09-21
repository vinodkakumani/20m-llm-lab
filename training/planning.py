"""Deterministic calculation of training budget and effective batch size."""
from dataclasses import dataclass
from pathlib import Path
import math, yaml
@dataclass(frozen=True)
class TrainingPlan:
 micro_batch_size:int; gradient_accumulation:int; sequence_length:int; training_tokens:int
 @property
 def effective_batch_size(self): return self.micro_batch_size*self.gradient_accumulation
 @property
 def tokens_per_optimizer_step(self): return self.effective_batch_size*self.sequence_length
 @property
 def total_steps(self): return math.ceil(self.training_tokens/self.tokens_per_optimizer_step)
 @property
 def planned_training_tokens(self): return self.total_steps*self.tokens_per_optimizer_step
def load_training_plan(path: str|Path) -> TrainingPlan:
 data=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
 plan=TrainingPlan(**{key:data[key] for key in TrainingPlan.__dataclass_fields__})
 if min(plan.micro_batch_size,plan.gradient_accumulation,plan.sequence_length,plan.training_tokens)<=0: raise ValueError('Training-plan quantities must be positive.')
 return plan
