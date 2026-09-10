import unittest

import torch

from generate.generate import sample
from model.model import GPT, GPTConfig


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.config = GPTConfig(
            vocab_size=32,
            block_size=8,
            n_layer=2,
            n_head=2,
            n_embd=16,
            dropout=0.0,
        )
        self.model = GPT(self.config)
        self.model.eval()

    def test_forward_returns_expected_shape_and_loss(self):
        inputs = torch.randint(0, self.config.vocab_size, (2, 5))
        targets = torch.randint(0, self.config.vocab_size, (2, 5))

        logits, loss = self.model(inputs, targets)

        self.assertEqual(logits.shape, (2, 5, self.config.vocab_size))
        self.assertTrue(torch.isfinite(loss))

    def test_attention_cannot_use_future_tokens(self):
        prefix = torch.tensor([[1, 2, 3]])
        extended = torch.tensor([[1, 2, 3, 4, 5]])

        prefix_logits, _ = self.model(prefix)
        extended_logits, _ = self.model(extended)

        torch.testing.assert_close(prefix_logits, extended_logits[:, :3], atol=1e-6, rtol=1e-6)

    def test_forward_rejects_sequences_longer_than_context(self):
        inputs = torch.zeros((1, self.config.block_size + 1), dtype=torch.long)

        with self.assertRaises(AssertionError):
            self.model(inputs)


class GenerationTests(unittest.TestCase):
    def test_sample_preserves_prompt_and_limits_token_ids(self):
        torch.manual_seed(11)
        config = GPTConfig(
            vocab_size=24,
            block_size=8,
            n_layer=1,
            n_head=2,
            n_embd=8,
            dropout=0.0,
        )
        model = GPT(config).eval()
        prompt = torch.tensor([[2, 4, 6]])

        output = sample(
            model,
            prompt,
            max_new_tokens=4,
            block_size=config.block_size,
            temperature=1.0,
            top_k=5,
            top_p=0.9,
            repetition_penalty=1.1,
        )

        self.assertEqual(output.shape, (1, 7))
        torch.testing.assert_close(output[:, :3], prompt)
        self.assertTrue(torch.all((output >= 0) & (output < config.vocab_size)))

    def test_sample_supports_multiple_sequences(self):
        torch.manual_seed(13)
        config = GPTConfig(
            vocab_size=24,
            block_size=8,
            n_layer=1,
            n_head=2,
            n_embd=8,
            dropout=0.0,
        )
        model = GPT(config).eval()
        prompts = torch.tensor([[1, 2], [3, 4]])

        output = sample(
            model,
            prompts,
            max_new_tokens=2,
            block_size=config.block_size,
            top_k=5,
            top_p=0.9,
            repetition_penalty=1.1,
        )

        self.assertEqual(output.shape, (2, 4))
        torch.testing.assert_close(output[:, :2], prompts)


if __name__ == "__main__":
    unittest.main()
