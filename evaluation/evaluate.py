"""Validation loss and perplexity for a trusted project checkpoint."""
import math, torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from model import BaselineTransformer, load_baseline_config
from training.train import PackedTokenDataset

@torch.inference_mode()
def evaluate_checkpoint(model_config_path, checkpoint_path, dataset_path, batch_size=1):
 c=load_baseline_config(model_config_path).model; model=BaselineTransformer(c); payload=torch.load(checkpoint_path,map_location='cpu',weights_only=False); model.load_state_dict(payload['model_state']); model.eval()
 losses=[]; tokens=0
 for batch in DataLoader(PackedTokenDataset(dataset_path,c.context_length),batch_size=batch_size):
  logits=model(batch[:,:-1]); loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),batch[:,1:].reshape(-1),reduction='sum'); losses.append(loss.item()); tokens+=batch[:,1:].numel()
 mean=sum(losses)/tokens
 return {"validation_loss":mean,"perplexity":math.exp(mean),"validation_tokens":tokens}
