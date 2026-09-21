from pathlib import Path
from tempfile import TemporaryDirectory
import unittest, torch
from model.config import ModelConfig
from model.model import BaselineTransformer
from training.checkpoint import create_run_directory, initialize_run, load_checkpoint, save_checkpoint
from training.optimizer import build_adamw
from training.scheduler import build_cosine_scheduler
from training.train import SmokeTrainingConfig

class CheckpointTests(unittest.TestCase):
 def test_save_load_and_run_artifacts(self):
  with TemporaryDirectory() as d:
   c=ModelConfig(32,8,1,2,4,16,4); m=BaselineTransformer(c); o=build_adamw(m,.01,.1); s=build_cosine_scheduler(o,0,3)
   run=create_run_directory(d,'test'); initialize_run(run,c,SmokeTrainingConfig())
   x=torch.randint(0,32,(1,4)); loss=m.next_token_loss(m(x),x); loss.backward(); o.step(); s.step(); save_checkpoint(run/'checkpoints'/'step_00000001.pt',m,o,s,1,{})
   expected={k:v.detach().clone() for k,v in m.state_dict().items()}; restored=BaselineTransformer(c); ro=build_adamw(restored,.01,.1); rs=build_cosine_scheduler(ro,0,3)
   self.assertEqual(load_checkpoint(run/'checkpoints'/'step_00000001.pt',restored,ro,rs),1); self.assertTrue(all(torch.equal(expected[k],v) for k,v in restored.state_dict().items())); self.assertTrue((run/'system_info.json').is_file()); self.assertTrue((run/'tokenizer_info.json').is_file()); self.assertTrue((run/'dataset_info.json').is_file())
