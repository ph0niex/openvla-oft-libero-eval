"""Tiny CPU model regression: the OFT fork must expose bidirectional SDPA attention."""
from pathlib import Path
import unittest
import torch
import transformers
from transformers import LlamaConfig, LlamaModel

ROOT = Path(__file__).resolve().parents[1]

class OFTTransformersTests(unittest.TestCase):
    def test_custom_fork_origin(self):
        self.assertTrue(Path(transformers.__file__).resolve().is_relative_to(ROOT/'vendor/transformers-openvla-oft'))

    def test_future_token_affects_first_token(self):
        torch.manual_seed(7)
        config = LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4)
        config._attn_implementation = 'sdpa'
        model = LlamaModel(config).eval()
        with torch.inference_mode():
            a = model(torch.tensor([[1,2,3,4]]), use_cache=False).last_hidden_state[:,0]
            b = model(torch.tensor([[1,2,3,5]]), use_cache=False).last_hidden_state[:,0]
        self.assertGreater((a-b).abs().max().item(), 1e-6,
                           'Causal attention detected: plain PyPI Transformers is not valid for OFT.')
