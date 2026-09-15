# DeepSeekQ From Scratch

A decoder-only transformer built from scratch in PyTorch, with a trainable
byte-level BPE tokenizer and a Flask inference service.

## Architecture

| Component | Choice | Why |
|---|---|---|
| Normalisation | RMSNorm, pre-norm, fp32 internals | cheaper than LayerNorm, stable in low precision |
| Positions | RoPE (no learned table) | extrapolates; context can be raised at inference |
| Attention | Grouped-query (12 Q heads / 4 KV heads) | 3x smaller KV cache than MHA |
| Kernel | `F.scaled_dot_product_attention` | FlashAttention when the hardware supports it |
| Feed-forward | SwiGLU, or optional MoE with shared experts | MoE adds capacity without adding FLOPs per token |
| Decoding | Incremental KV cache | generation is O(n), not O(n²) |
| Head | Weights tied to the embedding | saves 24.5M parameters |

### Parameter budget (default config)

```
vocab 32000 · context 1024 · d_model 768 · 12 layers · 12 heads / 4 KV heads · ffn 2048

embeddings (tied)      24,576,000
12 × transformer block 75,515,904
final norm                    768
                      -----------
total                 100,092,672   (~100M, 75.5M non-embedding)
```

Presets in `training.py`: `tiny` (~1M, CPU smoke test), `small` (~30M),
`base` (~100M, default), `moe` (~170M total / ~60M active per token).

## Install

```bash
pip install -r requirements.txt
python tests.py            # 27 correctness tests, CPU, under a minute
```

## Train

```bash
# train a BPE tokenizer and then the model on a plain-text corpus
python training.py --data corpus.txt --train-tokenizer --vocab-size 8000 \
    --preset small --steps 2000 --batch-size 8 --grad-accum 4

# CPU smoke test
python training.py --data corpus.txt --preset tiny --steps 20 --device cpu

# resume
python training.py --data corpus.txt --resume checkpoints/last.pt --steps 5000
```

The loop packs the corpus into one contiguous token stream (no padding, no
wasted compute), and provides cosine LR decay with warmup, gradient
accumulation, gradient clipping, bf16/fp16 autocast on CUDA, a held-out
validation split, best/last checkpointing and resume. Add
`--grad-checkpointing` to trade compute for memory.

Artifacts land in `checkpoints/`: `config.json`, `model.pt`, `tokenizer.json`.

## Serve

```bash
MODEL_DIR=checkpoints python app.py
# production:
MODEL_DIR=checkpoints gunicorn -w 1 -t 120 -b 0.0.0.0:5000 app:app
```

```bash
curl -X POST http://127.0.0.1:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Once upon a time","max_new_tokens":64,"temperature":0.8,"top_p":0.95}'

# token-by-token server-sent events
curl -N -X POST http://127.0.0.1:5000/generate/stream \
  -H "Content-Type: application/json" -d '{"prompt":"Once upon a time"}'
```

| Route | Purpose |
|---|---|
| `GET /` | service + model metadata |
| `GET /health` | device, parameter count, whether real weights are loaded |
| `POST /generate` | full completion |
| `POST /generate/stream` | SSE stream, one JSON frame per chunk |

Environment: `MODEL_DIR`, `DEVICE`, `PORT`, `API_KEY` (enables
`Authorization: Bearer …`), `MAX_NEW_TOKENS` (server-side ceiling).

Every parameter is validated and range-checked; bad input returns `400` with a
message rather than a stack trace. Generation is serialised behind a lock
because a single model object is not thread-safe.

## Local generation check

```bash
python -c "
from model import DeepSeekConfig, DeepSeekForCausalLM, generate_text
from tokenizer import ByteBPETokenizer
cfg = DeepSeekConfig(hidden_size=128, num_layers=4, num_heads=4, num_kv_heads=2, intermediate_size=384)
tok = ByteBPETokenizer(); m = DeepSeekForCausalLM(cfg)
print(generate_text(m, tok, 'The future of AI is', 32))"
```

An untrained model emits noise — that is expected. It should never crash.

## Sampling controls

`temperature` (0 = greedy), `top_k`, `top_p` (nucleus), `repetition_penalty`,
`eos_token_id`, and `restrict_vocab`, which masks ids the tokenizer cannot
decode.

## What changed from the first version

Fixes for defects in the original code:

1. **`POST /generate` always returned 500.** The model sampled ids up to 31999
   while `decode` did `bytes([...])`, which raises for anything above 255. Every
   single request crashed. Decoding now maps ids through the vocabulary and
   drops unknown ids instead of raising.
2. **The tokenizer was byte-level only.** It advertised a 32000 vocabulary but
   emitted 256 distinct ids, so ~24.5M embedding rows could never receive a
   gradient, and text was ~4x longer than it needed to be. It is now a real
   trainable BPE.
3. **It also dropped characters.** The pre-tokenisation pattern silently lost
   underscores; round-trip is now verified over ASCII and Unicode fuzzing.
4. **The "~100M" config was 137.8M.** Corrected, and the count is now asserted
   in the test suite.
5. **Generation was O(n²).** Every step re-ran the full forward pass over the
   whole sequence. Added a KV cache.
6. **Generation broke on GPU.** Input tensors were built on CPU while the model
   sat on CUDA. Tensors now follow the model's device.
7. **Off-by-one on context length.** `t >= context_length` rejected sequences of
   exactly the allowed length; long prompts now slide the window instead of
   silently truncating output.
8. **Learned position embeddings** replaced with RoPE; **MHA** replaced with GQA.
9. **`torch.load` without `weights_only`** — arbitrary code execution from a
   checkpoint. Now loads weights only, and mismatched checkpoints fail loudly
   rather than loading partially via `strict=False`.
10. **Training was unusable**: batch size fixed at 1, no batching, no LR
    schedule, no gradient clipping, no validation split, no checkpointing, no
    mixed precision, and padding labels trained the model to predict spaces.
11. **Unvalidated API input**: `max_new_tokens` was unbounded (trivial DoS),
    non-numeric values raised uncaught exceptions, and a missing body was
    accepted silently.
12. `CONFIG_PATH` was read from the environment and then never used.

### API changes

* `model.generate(...)` now takes and returns **tensors**. For text, use
  `model.generate_text(tokenizer, prompt, ...)`. The module-level
  `generate_text(model, tokenizer, prompt, max_new_tokens)` helper is unchanged.
* `forward()` returns a dict (`logits`, `loss`, `aux_loss`, `past_key_values`)
  and computes the loss internally when `labels` are passed.
* `SimpleTokenizer` remains as an alias of `ByteBPETokenizer`. Token ids shifted
  by 4 (specials occupy 0–3), so **old checkpoints are not compatible**.
