import os
import tempfile
import unittest

import torch

from model import DeepSeekConfig, DeepSeekForCausalLM, build_rope_cache, apply_rope
from tokenizer import ByteBPETokenizer
from training import PackedTextDataset, lr_at, train

torch.manual_seed(0)

TINY = dict(vocab_size=300, context_length=64, hidden_size=64, num_layers=2,
            num_heads=4, num_kv_heads=2, intermediate_size=128)


class TestTokenizer(unittest.TestCase):
    def test_roundtrip_bytes(self):
        tok = ByteBPETokenizer()
        for s in ["", "hello world", "日本語 🚀", "snake_case_1\t\n", "naïve café"]:
            self.assertEqual(tok.decode(tok.encode(s)), s)

    def test_roundtrip_after_training(self):
        tok = ByteBPETokenizer().train("the cat sat on the mat. " * 100, 400)
        self.assertGreater(len(tok.merges), 0)
        for s in ["the cat sat", "unseen ünïcode 🚀", ""]:
            self.assertEqual(tok.decode(tok.encode(s)), s)

    def test_compression(self):
        tok = ByteBPETokenizer().train("the quick brown fox. " * 200, 500)
        line = "the quick brown fox. "
        self.assertLess(len(tok.encode(line)), len(line.encode()))

    def test_out_of_range_ids_never_raise(self):
        tok = ByteBPETokenizer()
        self.assertIsInstance(tok.decode([0, 5, 99999, -1]), str)

    def test_ids_within_vocab(self):
        tok = ByteBPETokenizer().train("abc abc abc " * 50, 400)
        ids = tok.encode("abc abc xyz")
        self.assertTrue(all(0 <= i < tok.vocab_size for i in ids))

    def test_save_load(self):
        tok = ByteBPETokenizer().train("hello hello world " * 50, 400)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.json")
            tok.save(path)
            other = ByteBPETokenizer.load(path)
        self.assertEqual(other.encode("hello world"), tok.encode("hello world"))
        self.assertEqual(other.vocab_size, tok.vocab_size)


class TestRoPE(unittest.TestCase):
    def test_relative_position_property(self):
        cos, sin = build_rope_cache(16, 64, 10000.0)
        q = torch.randn(1, 1, 1, 16)
        k = torch.randn(1, 1, 1, 16)
        dots = []
        for m, n in [(5, 2), (20, 17), (40, 37)]:
            qa = apply_rope(q, cos[m:m + 1], sin[m:m + 1])
            ka = apply_rope(k, cos[n:n + 1], sin[n:n + 1])
            dots.append((qa * ka).sum().item())
        self.assertLess(max(dots) - min(dots), 1e-4)


class TestModel(unittest.TestCase):
    def setUp(self):
        self.config = DeepSeekConfig(**TINY)
        self.model = DeepSeekForCausalLM(self.config).eval()

    def test_default_config_is_about_100m(self):
        n = DeepSeekForCausalLM(DeepSeekConfig()).num_parameters()
        self.assertAlmostEqual(n / 1e6, 100.09, places=1)

    def test_forward_shapes_and_loss(self):
        ids = torch.randint(0, 300, (2, 16))
        out = self.model(ids, labels=ids)
        self.assertEqual(out["logits"].shape, (2, 16, 300))
        self.assertTrue(torch.isfinite(out["loss"]))
        # an untrained model should sit near ln(vocab_size)
        self.assertLess(abs(out["loss"].item() - torch.log(torch.tensor(300.0)).item()), 1.0)

    def test_causality(self):
        ids = torch.randint(0, 300, (1, 16))
        base = self.model(ids)["logits"]
        changed = ids.clone()
        changed[0, -1] = (changed[0, -1] + 1) % 300
        after = self.model(changed)["logits"]
        # changing the last token must not affect any earlier position
        self.assertTrue(torch.allclose(base[:, :-1], after[:, :-1], atol=1e-5))

    def test_kv_cache_matches_full_forward(self):
        ids = torch.randint(0, 300, (1, 12))
        full = self.model(ids)["logits"][:, -1]
        out = self.model(ids[:, :-1], use_cache=True)
        step = self.model(ids[:, -1:], past_key_values=out["past_key_values"],
                          use_cache=True)["logits"][:, -1]
        self.assertTrue(torch.allclose(full, step, atol=1e-4),
                        f"max diff {(full - step).abs().max().item()}")

    def test_greedy_is_deterministic(self):
        ids = torch.randint(0, 300, (1, 4))
        a = self.model.generate(ids, max_new_tokens=8, temperature=0.0)
        b = self.model.generate(ids, max_new_tokens=8, temperature=0.0)
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(a.shape[1], 12)

    def test_generation_respects_restrict_vocab(self):
        ids = torch.randint(0, 50, (1, 4))
        out = self.model.generate(ids, max_new_tokens=16, restrict_vocab=100)
        self.assertLess(out.max().item(), 100)

    def test_generate_text_is_decodable(self):
        tok = ByteBPETokenizer()
        model = DeepSeekForCausalLM(DeepSeekConfig(**{**TINY, "vocab_size": 32000}))
        text = model.generate_text(tok, "Once upon a time", max_new_tokens=16)
        self.assertIsInstance(text, str)          # the original code raised ValueError here
        self.assertTrue(text.startswith("Once upon a time"))

    def test_context_overflow_raises(self):
        ids = torch.randint(0, 300, (1, self.config.context_length + 1))
        with self.assertRaises(ValueError):
            self.model(ids)

    def test_generate_slides_window(self):
        ids = torch.randint(0, 300, (1, self.config.context_length - 2))
        out = self.model.generate(ids, max_new_tokens=8, temperature=0.0)
        self.assertEqual(out.shape[1], ids.shape[1] + 8)

    def test_empty_input_rejected(self):
        with self.assertRaises(ValueError):
            self.model(torch.zeros(1, 0, dtype=torch.long))

    def test_weight_tying(self):
        self.assertIs(self.model.lm_head.weight, self.model.token_embedding.weight)

    def test_save_load_roundtrip(self):
        ids = torch.randint(0, 300, (1, 8))
        before = self.model(ids)["logits"]
        with tempfile.TemporaryDirectory() as d:
            self.model.save_pretrained(d)
            restored = DeepSeekForCausalLM.from_pretrained(d).eval()
        self.assertTrue(torch.allclose(before, restored(ids)["logits"], atol=1e-6))

    def test_gqa_shapes(self):
        cfg = DeepSeekConfig(**{**TINY, "num_heads": 4, "num_kv_heads": 1})
        out = DeepSeekForCausalLM(cfg)(torch.randint(0, 300, (1, 8)), use_cache=True)
        k, v = out["past_key_values"][0]
        self.assertEqual(k.shape[1], 1)           # cache stores only the KV heads
        self.assertEqual(v.shape[2], 8)

    def test_invalid_configs_rejected(self):
        with self.assertRaises(ValueError):
            DeepSeekConfig(hidden_size=768, num_heads=7)
        with self.assertRaises(ValueError):
            DeepSeekConfig(num_heads=12, num_kv_heads=5)


class TestMoE(unittest.TestCase):
    def test_forward_and_aux_loss(self):
        cfg = DeepSeekConfig(**{**TINY, "use_moe": True, "num_experts": 4,
                                "num_experts_per_tok": 2, "moe_intermediate_size": 64})
        model = DeepSeekForCausalLM(cfg)
        ids = torch.randint(0, 300, (2, 16))
        out = model(ids, labels=ids)
        self.assertEqual(out["logits"].shape, (2, 16, 300))
        self.assertIsNotNone(out["aux_loss"])
        self.assertTrue(torch.isfinite(out["loss"]))
        out["loss"].backward()
        self.assertIsNotNone(model.blocks[0].mlp.gate.weight.grad)

    def test_router_is_balanced_at_init(self):
        cfg = DeepSeekConfig(**{**TINY, "use_moe": True, "num_experts": 4,
                                "num_experts_per_tok": 2, "moe_intermediate_size": 64})
        model = DeepSeekForCausalLM(cfg)
        aux = model(torch.randint(0, 300, (4, 32)))["aux_loss"]
        self.assertLess(aux.item(), 4.0)          # perfectly balanced == 1.0


class TestTraining(unittest.TestCase):
    def test_lr_schedule(self):
        self.assertLess(lr_at(0, 1e-3, 10, 100), lr_at(9, 1e-3, 10, 100))
        self.assertAlmostEqual(lr_at(9, 1e-3, 10, 100), 1e-3, places=6)
        self.assertLess(lr_at(99, 1e-3, 10, 100), lr_at(50, 1e-3, 10, 100))
        self.assertAlmostEqual(lr_at(150, 1e-3, 10, 100), 1e-4, places=6)

    def test_dataset_split_and_batches(self):
        tok = ByteBPETokenizer()
        ds = PackedTextDataset.from_text("hello world. " * 500, tok, 32, val_fraction=0.1)
        batch = ds.get_batch("train", 4, torch.device("cpu"))
        self.assertEqual(batch.shape, (4, 32))
        self.assertEqual(batch.dtype, torch.int64)
        self.assertIsNotNone(ds.val)

    def test_tiny_corpus_rejected(self):
        with self.assertRaises(ValueError):
            PackedTextDataset.from_text("short", ByteBPETokenizer(), 1024)

    def test_loss_decreases(self):
        tok = ByteBPETokenizer()
        cfg = DeepSeekConfig(**{**TINY, "vocab_size": tok.vocab_size})
        model = DeepSeekForCausalLM(cfg)
        ds = PackedTextDataset.from_text("the cat sat on the mat. " * 400, tok, 32)
        device = torch.device("cpu")
        probe = ds.get_batch("train", 4, device)
        before = model(probe, labels=probe)["loss"].item()
        with tempfile.TemporaryDirectory() as d:
            train(model, ds, device, steps=40, batch_size=4, lr=1e-3, warmup=5,
                  eval_every=0, log_every=1000, out_dir=d)
        after = model(probe, labels=probe)["loss"].item()
        self.assertLess(after, before, f"loss did not fall: {before:.3f} -> {after:.3f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
