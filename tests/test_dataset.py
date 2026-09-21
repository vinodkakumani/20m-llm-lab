from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from tokenizer.train_tokenizer import TokenizerConfig, train_and_save_tokenizer
from training.dataset import load_dataset_config, pack_documents, prepare_packed_split

class DatasetTests(unittest.TestCase):
 def test_pinned_config_and_modes(self):
  c=load_dataset_config(Path(__file__).parents[1]/"configs/tinystories.yaml")
  self.assertEqual(c.revision,"f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"); self.assertEqual(c.modes["smoke"]["train_examples"],128)
 def test_eos_and_fixed_packing(self):
  with TemporaryDirectory() as d:
   root=Path(d); tok=train_and_save_tokenizer(["one two three", "four five six"],root/"tok",TokenizerConfig(vocab_size=32,min_frequency=1))
   blocks, docs, tokens=pack_documents(["one two three","four five six"],tok,3)
   eos=__import__('tokenizer.train_tokenizer',fromlist=['']).load_tokenizer(tok).token_to_id('<eos>')
   self.assertEqual(docs,2); self.assertGreaterEqual(tokens,4); self.assertTrue(any(eos in block for block in blocks)); self.assertTrue(all(len(b)==4 for b in blocks))
   info=prepare_packed_split(["one two three","four five six"],tok,3,root/'train.bin')
   self.assertEqual(info['packed_sequences'],len(blocks)); self.assertTrue((root/'train.bin').is_file()); self.assertTrue((root/'train.bin.info.json').is_file())
