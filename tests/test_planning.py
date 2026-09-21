from pathlib import Path
import unittest
from training.planning import load_training_plan
class PlanningTests(unittest.TestCase):
 def test_baseline_budget_is_deterministic(self):
  p=load_training_plan(Path(__file__).parents[1]/'configs/baseline_full.yaml')
  self.assertEqual(p.effective_batch_size,32); self.assertEqual(p.tokens_per_optimizer_step,16384); self.assertEqual(p.total_steps,12208); self.assertEqual(p.planned_training_tokens,200015872)
