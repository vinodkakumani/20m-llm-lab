"""Fixed-prompt deterministic text generation."""
from pathlib import Path
import torch
from model import BaselineTransformer, load_baseline_config
from tokenizer.train_tokenizer import load_tokenizer
FIXED_PROMPTS=("Once upon a time","The little girl","One morning","The dog wanted to")
@torch.inference_mode()
def generate(model_config_path,checkpoint_path,tokenizer_path,prompt,max_new_tokens=32,temperature=1.0,seed=42):
 c=load_baseline_config(model_config_path).model; m=BaselineTransformer(c); m.load_state_dict(torch.load(checkpoint_path,map_location='cpu',weights_only=False)['model_state']); m.eval(); tok=load_tokenizer(tokenizer_path); ids=tok.encode(prompt,add_special_tokens=False).ids; g=torch.Generator().manual_seed(seed)
 for _ in range(max_new_tokens):
  x=torch.tensor([ids[-c.context_length:]]); logits=m(x)[0,-1]/temperature; next_id=torch.multinomial(torch.softmax(logits,dim=-1),1,generator=g).item(); ids.append(next_id)
 return tok.decode(ids,skip_special_tokens=True)
def save_fixed_samples(run_dir,model_config,checkpoint,tokenizer):
 out=Path(run_dir)/'samples'/'fixed_prompts.txt'; out.write_text('\n\n'.join(f'PROMPT: {p}\n{generate(model_config,checkpoint,tokenizer,p)}' for p in FIXED_PROMPTS)+'\n'); return out
