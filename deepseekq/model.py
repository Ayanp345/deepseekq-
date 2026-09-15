"""DeepSeek-inspired decoder-only transformer.

Architecture
------------
* Pre-norm residual blocks with RMSNorm
* Rotary position embeddings (RoPE) - no learned position table, so the model
  extrapolates and the context length can be raised at inference time
* Grouped-query attention (GQA) - fewer KV heads means a much smaller KV cache
* SwiGLU feed-forward, or an optional DeepSeek-style MoE with shared experts
* Fused scaled-dot-product attention (FlashAttention kernels when available)
* Incremental KV cache, so generation is O(n) instead of O(n^2)
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import ByteBPETokenizer

KVCache = List[Tuple[torch.Tensor, torch.Tensor]]


# --------------------------------------------------------------------- config
@dataclass
class DeepSeekConfig:
    """Default preset is ~100M parameters (verified, see ``num_parameters``)."""

    vocab_size: int = 32000
    context_length: int = 1024
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    num_kv_heads: int = 4          # GQA: 3 query heads share each KV head
    intermediate_size: int = 2048
    dropout: float = 0.0
    bias: bool = False
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02
    # --- optional mixture-of-experts ---
    use_moe: bool = False
    num_experts: int = 8
    num_experts_per_tok: int = 2
    num_shared_experts: int = 1
    moe_intermediate_size: int = 512
    moe_aux_loss_coef: float = 0.01

    def __post_init__(self):
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        if self.use_moe and self.num_experts_per_tok > self.num_experts:
            raise ValueError("num_experts_per_tok cannot exceed num_experts")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @classmethod
    def from_file(cls, path: str) -> "DeepSeekConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)

    def save(self, path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)


# ---------------------------------------------------------------- primitives
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()                                  # normalise in fp32 for stability
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def build_rope_cache(head_dim: int, max_seq_len: int, theta: float,
                     device=None, dtype=torch.float32):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)                   # (T, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)            # (T, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, head_dim); cos/sin: (T, head_dim)."""
    cos = cos.unsqueeze(0).unsqueeze(0).to(x.dtype)
    sin = sin.unsqueeze(0).unsqueeze(0).to(x.dtype)
    return x * cos + _rotate_half(x) * sin


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.n_rep = config.num_heads // config.num_kv_heads
        self.head_dim = config.head_dim
        self.dropout_p = config.dropout

        d = config.hidden_size
        self.q_proj = nn.Linear(d, config.num_heads * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(d, config.num_kv_heads * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(d, config.num_kv_heads * self.head_dim, bias=config.bias)
        self.o_proj = nn.Linear(config.num_heads * self.head_dim, d, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self._sdpa = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x, cos, sin, past_kv=None, use_cache=False):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        present = (k, v) if use_cache else None

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # A single query token attending to the whole cache needs no mask;
        # a multi-token chunk (prefill) needs the causal mask.
        is_causal = t > 1
        if self._sdpa:
            y = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=is_causal,
            )
        else:  # pragma: no cover - fallback for torch < 2.0
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if is_causal:
                mask = torch.ones(t, k.size(2), dtype=torch.bool, device=x.device).tril(
                    diagonal=k.size(2) - t
                )
                scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            y = F.softmax(scores.float(), dim=-1).to(q.dtype) @ v

        y = y.transpose(1, 2).contiguous().view(b, t, -1)
        return self.resid_dropout(self.o_proj(y)), present


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEFeedForward(nn.Module):
    """Top-k routed experts plus always-on shared experts (DeepSeekMoE style)."""

    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.experts = nn.ModuleList(
            [SwiGLU(config.hidden_size, config.moe_intermediate_size, config.bias)
             for _ in range(config.num_experts)]
        )
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.shared_experts = (
            SwiGLU(config.hidden_size,
                   config.moe_intermediate_size * config.num_shared_experts,
                   config.bias)
            if config.num_shared_experts > 0 else None
        )

    def forward(self, x):
        b, t, d = x.shape
        flat = x.view(-1, d)
        probs = F.softmax(self.gate(flat).float(), dim=-1)          # (N, E)
        weights, indices = torch.topk(probs, self.top_k, dim=-1)
        weights = (weights / weights.sum(dim=-1, keepdim=True)).to(x.dtype)

        out = torch.zeros_like(flat)
        flat_idx = indices.reshape(-1)                               # (N * top_k,)
        flat_w = weights.reshape(-1)
        for e in range(self.num_experts):
            slot = (flat_idx == e).nonzero(as_tuple=True)[0]
            if slot.numel() == 0:
                continue
            tokens = torch.div(slot, self.top_k, rounding_mode="floor")
            out.index_add_(0, tokens, self.experts[e](flat[tokens]) * flat_w[slot].unsqueeze(-1))

        # Switch-transformer load-balancing loss: fraction routed x mean gate prob.
        counts = torch.zeros(self.num_experts, device=x.device, dtype=probs.dtype)
        counts.index_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=probs.dtype))
        frac = counts / flat_idx.numel()
        aux_loss = self.num_experts * torch.sum(frac.detach() * probs.mean(dim=0))

        if self.shared_experts is not None:
            out = out + self.shared_experts(flat)
        return out.view(b, t, d), aux_loss


class TransformerBlock(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.post_attn_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.mlp = MoEFeedForward(config) if config.use_moe else SwiGLU(
            config.hidden_size, config.intermediate_size, config.bias
        )
        self.is_moe = config.use_moe
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, cos, sin, past_kv=None, use_cache=False):
        attn_out, present = self.attn(self.input_norm(x), cos, sin, past_kv, use_cache)
        x = x + attn_out
        if self.is_moe:
            ff_out, aux = self.mlp(self.post_attn_norm(x))
        else:
            ff_out, aux = self.mlp(self.post_attn_norm(x)), None
        x = x + self.dropout(ff_out)
        return x, present, aux


# -------------------------------------------------------------------- model
class DeepSeekForCausalLM(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.gradient_checkpointing = False

        self.apply(self._init_weights)
        # GPT-2 style scaled init for the residual output projections.
        std = config.initializer_range / math.sqrt(2 * config.num_layers)
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(p, mean=0.0, std=std)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        cos, sin = build_rope_cache(config.head_dim, config.context_length, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    # ------------------------------------------------------------- utilities
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def num_parameters(self, non_embedding: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.token_embedding.weight.numel()
            if not self.config.tie_word_embeddings:
                total -= self.lm_head.weight.numel()
        return total

    def _rope_slice(self, start: int, length: int, device, dtype):
        end = start + length
        if end > self.rope_cos.size(0):   # grow the cache for longer contexts
            cos, sin = build_rope_cache(self.config.head_dim, end * 2,
                                        self.config.rope_theta, device=device)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        cos = self.rope_cos[start:end].to(device=device, dtype=dtype)
        sin = self.rope_sin[start:end].to(device=device, dtype=dtype)
        return cos, sin

    # --------------------------------------------------------------- forward
    def forward(self, input_ids, labels=None, past_key_values=None, use_cache=False):
        b, t = input_ids.shape
        if t == 0:
            raise ValueError("input_ids must contain at least one token")
        past_len = past_key_values[0][0].size(2) if past_key_values else 0
        if past_len + t > self.config.context_length:
            raise ValueError(
                f"sequence length {past_len + t} exceeds context_length "
                f"{self.config.context_length}"
            )

        x = self.dropout(self.token_embedding(input_ids))
        cos, sin = self._rope_slice(past_len, t, x.device, x.dtype)

        presents: KVCache = []
        aux_total = None
        for i, block in enumerate(self.blocks):
            past = past_key_values[i] if past_key_values else None
            if self.gradient_checkpointing and self.training and not use_cache:
                x, present, aux = torch.utils.checkpoint.checkpoint(
                    block, x, cos, sin, None, False, use_reentrant=False
                )
            else:
                x, present, aux = block(x, cos, sin, past, use_cache)
            if use_cache:
                presents.append(present)
            if aux is not None:
                aux_total = aux if aux_total is None else aux_total + aux

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # predict token i+1 from position i
            shift_logits = logits[:, :-1].reshape(-1, logits.size(-1))
            shift_labels = labels[:, 1:].reshape(-1)
            loss = F.cross_entropy(shift_logits.float(), shift_labels, ignore_index=-100)
            if aux_total is not None:
                loss = loss + self.config.moe_aux_loss_coef * aux_total

        return {
            "logits": logits,
            "loss": loss,
            "aux_loss": aux_total,
            "past_key_values": presents if use_cache else None,
        }

    # -------------------------------------------------------------- sampling
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Autoregressive sampling with a KV cache. Returns prompt + continuation."""
        idx = input_ids.to(self.device)
        pieces = [idx] + list(self.stream_generate(input_ids, **kwargs))
        return torch.cat(pieces, dim=1)

    @torch.no_grad()
    def stream_generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
        restrict_vocab: Optional[int] = None,
    ):
        """Yield one ``(batch, 1)`` token tensor at a time."""
        self.eval()
        device = self.device
        idx = input_ids.to(device)
        if idx.size(1) >= self.config.context_length:
            idx = idx[:, -(self.config.context_length - 1):]

        past, cur = None, idx
        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            if past is not None and past[0][0].size(2) + 1 > self.config.context_length:
                # Slide the window: drop the cache and re-prefill on the tail.
                idx = idx[:, -(self.config.context_length - 1):]
                past, cur = None, idx

            out = self(cur, past_key_values=past, use_cache=True)
            past = out["past_key_values"]
            logits = out["logits"][:, -1, :].detach().clone().float()

            if restrict_vocab is not None and restrict_vocab < logits.size(-1):
                logits[:, restrict_vocab:] = float("-inf")
            if repetition_penalty != 1.0:
                for b in range(idx.size(0)):
                    seen = torch.unique(idx[b])
                    score = logits[b, seen]
                    logits[b, seen] = torch.where(score < 0, score * repetition_penalty,
                                                  score / repetition_penalty)

            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k and top_k > 0:
                    k = min(top_k, logits.size(-1))
                    kth = logits.topk(k, dim=-1).values[:, -1, None]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                if top_p is not None and 0 < top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                    sorted_probs = F.softmax(sorted_logits, dim=-1)
                    # exclusive cumulative mass: keep the smallest prefix reaching top_p
                    remove = sorted_probs.cumsum(dim=-1) - sorted_probs > top_p
                    remove[:, 0] = False                      # always keep the top token
                    logits = logits.masked_fill(
                        remove.scatter(1, sorted_idx, remove), float("-inf")
                    )
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            if eos_token_id is not None:
                next_token = torch.where(finished.unsqueeze(1),
                                         torch.full_like(next_token, eos_token_id), next_token)
                finished = finished | (next_token.squeeze(1) == eos_token_id)

            idx = torch.cat([idx, next_token], dim=1)
            cur = next_token
            yield next_token
            if eos_token_id is not None and bool(finished.all()):
                break

    @torch.no_grad()
    def generate_text(self, tokenizer: ByteBPETokenizer, prompt: str,
                      max_new_tokens: int = 64, return_full_text: bool = True, **kwargs) -> str:
        ids = tokenizer.encode(prompt)
        if not ids:                                   # empty prompt -> start from <bos>
            ids = [tokenizer.bos_token_id]
        kwargs.setdefault("eos_token_id", tokenizer.eos_token_id)
        # Never sample ids the tokenizer cannot decode (matters for untrained models).
        kwargs.setdefault("restrict_vocab", tokenizer.vocab_size)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.generate(input_ids, max_new_tokens=max_new_tokens, **kwargs)
        tokens = out[0].tolist()
        if not return_full_text:
            tokens = tokens[len(ids):]
        return tokenizer.decode(tokens)

    # ------------------------------------------------------------ persistence
    def configure_optimizers(self, lr: float, weight_decay: float = 0.1,
                             betas: Tuple[float, float] = (0.9, 0.95)):
        """Decay matrices, never decay norms/biases/embeddings."""
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused = torch.cuda.is_available()
        try:
            return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)
        except (TypeError, RuntimeError):
            return torch.optim.AdamW(groups, lr=lr, betas=betas)

    def save_pretrained(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        self.config.save(os.path.join(directory, "config.json"))
        torch.save(self.state_dict(), os.path.join(directory, "model.pt"))

    @classmethod
    def from_pretrained(cls, directory: str, map_location="cpu") -> "DeepSeekForCausalLM":
        config = DeepSeekConfig.from_file(os.path.join(directory, "config.json"))
        model = cls(config)
        state = _safe_torch_load(os.path.join(directory, "model.pt"), map_location)
        missing, unexpected = model.load_state_dict(state, strict=False)
        ignorable = {"rope_cos", "rope_sin", "lm_head.weight"}
        hard_missing = [k for k in missing if k not in ignorable]
        if hard_missing or unexpected:
            raise RuntimeError(f"checkpoint mismatch: missing={hard_missing} unexpected={unexpected}")
        return model


def _safe_torch_load(path: str, map_location="cpu"):
    """Load weights with ``weights_only=True`` when the torch version supports it."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # torch < 2.0
        return torch.load(path, map_location=map_location)


def generate_text(model: DeepSeekForCausalLM, tokenizer: ByteBPETokenizer, prompt: str,
                  max_new_tokens: int = 64, temperature: float = 0.8, top_k: int = 50,
                  **kwargs) -> str:
    """Backwards-compatible functional wrapper."""
    return model.generate_text(tokenizer, prompt, max_new_tokens=max_new_tokens,
                               temperature=temperature, top_k=top_k, **kwargs)


def build_model(config: Optional[DeepSeekConfig] = None) -> DeepSeekForCausalLM:
    return DeepSeekForCausalLM(config or DeepSeekConfig())
