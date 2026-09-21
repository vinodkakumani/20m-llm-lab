from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from tokenizer.train_tokenizer import TokenizerConfig, train_and_save_tokenizer
from training.dataset import prepare_packed_split
from training.train import SmokeTrainingConfig, run_smoke_training

class TrainingTests(unittest.TestCase):
 def test_small_training_updates_and_reports_metrics(self):
  with TemporaryDirectory() as d:
   root=Path(d); tok=train_and_save_tokenizer(["one two three four five six seven"],root/'tok',TokenizerConfig(vocab_size=32,min_frequency=1))
   prepare_packed_split(["one two three four five six seven"]*4,tok,4,root/'tokens.bin')
   # Use a tiny independent model configuration rather than the 20M CPU model.
   config=root/'model.yaml'; config.write_text('model:\n  vocab_size: 32\n  d_model: 8\n  n_layers: 1\n  n_heads: 2\n  head_dim: 4\n  ffn_dim: 16\n  context_length: 4\nparameter_count:\n  target: 1\n  tolerance: 999999\n')
   metrics=run_smoke_training(config,root/'tokens.bin',SmokeTrainingConfig(max_steps=2))
   self.assertEqual(len(metrics),2); self.assertTrue(all(m.loss>0 and m.tokens_per_second>0 for m in metrics))
