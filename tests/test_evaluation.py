from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from tokenizer.train_tokenizer import TokenizerConfig,train_and_save_tokenizer
from training.dataset import prepare_packed_split
from training.train import SmokeTrainingConfig,run_smoke_training
from training.checkpoint import create_run_directory
from evaluation.evaluate import evaluate_checkpoint
from evaluation.plot import plot_metrics
from evaluation.generate import generate
class EvaluationTests(unittest.TestCase):
 def test_evaluation_and_plot(self):
  with TemporaryDirectory() as d:
   root=Path(d); tok=train_and_save_tokenizer(['one two three four five six']*4,root/'tok',TokenizerConfig(vocab_size=32,min_frequency=1)); prepare_packed_split(['one two three four five six']*4,tok,4,root/'v.bin')
   cfg=root/'m.yaml'; cfg.write_text('model:\n  vocab_size: 32\n  d_model: 8\n  n_layers: 1\n  n_heads: 2\n  head_dim: 4\n  ffn_dim: 16\n  context_length: 4\nparameter_count:\n  target: 1\n  tolerance: 999999\n')
   run=create_run_directory(root,'eval'); run_smoke_training(cfg,root/'v.bin',SmokeTrainingConfig(max_steps=1),run)
   checkpoint = run/'checkpoints'/'step_00000001.pt'
   result=evaluate_checkpoint(cfg,checkpoint,root/'v.bin'); self.assertGreater(result['perplexity'],1); self.assertTrue((plot_metrics(run)/'loss.png').is_file())
   self.assertEqual(generate(cfg,checkpoint,tok,'one',max_new_tokens=2,seed=9),generate(cfg,checkpoint,tok,'one',max_new_tokens=2,seed=9))
