import unittest
from unittest.mock import patch
from training.runtime import resolve_runtime
from training.train import SmokeTrainingConfig
class RuntimeTests(unittest.TestCase):
 def test_cpu_auto_fallback(self):
  with patch('torch.cuda.is_available',return_value=False): self.assertEqual((resolve_runtime().device.type,resolve_runtime().precision),('cpu','fp32'))
 def test_cuda_precision_resolution(self):
  with patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.is_bf16_supported',return_value=False): self.assertEqual(resolve_runtime().precision,'fp16')
 def test_effective_batch_is_derived(self): self.assertEqual(SmokeTrainingConfig(micro_batch_size=4,gradient_accumulation_steps=8).effective_batch_size,32)
 def test_mps_auto_selection(self):
  with patch('torch.cuda.is_available',return_value=False),patch('torch.backends.mps.is_available',return_value=True): self.assertEqual(resolve_runtime().device.type,'mps')
 @unittest.skipUnless(__import__('torch').cuda.is_available(),'CUDA-only qualification')
 def test_cuda_available(self): self.assertEqual(resolve_runtime('cuda').device.type,'cuda')
