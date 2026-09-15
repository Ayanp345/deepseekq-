from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SPLIT_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d{1,3}| ?(?:(?![^\W\d_])[^\s\d])+|\s+(?!\S)|\s+|."""
)

DEFAULT_SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]

Pair = Tuple[int, int]


class ByteBPETokenizer:
    """Trainable byte-level BPE tokenizer."""

    def __init__(
        self,
        merges: Optional[Dict[Pair, int]] = None,
        special_tokens: Optional[Sequence[str]] = None,
    ):
        self.special_tokens: List[str] = list(special_tokens or DEFAULT_SPECIALS)
        if len(set(self.special_tokens)) != len(self.special_tokens):
            raise ValueError("special_tokens must be unique")
        self.special_id_map: Dict[str, int] = {t: i for i, t in enumerate(self.special_tokens)}
        self.n_special = len(self.special_tokens)
        self.merges: Dict[Pair, int] = dict(merges or {})
        self._rebuild()

    def _rebuild(self) -> None:
        """Rebuild id -> bytes table and the merge-rank table."""
        self.vocab: Dict[int, bytes] = {}
        for token, idx in self.special_id_map.items():
            self.vocab[idx] = token.encode("utf-8")
        for b in range(256):
            self.vocab[self.n_special + b] = bytes([b])
        for (a, b), new_id in self.merges.items():
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
        self.ranks: Dict[Pair, int] = {pair: i for i, pair in enumerate(self.merges)}
        self._cache: Dict[str, List[int]] = {}

    @property
    def vocab_size(self) -> int:
        return self.n_special + 256 + len(self.merges)

    @property
    def pad_token_id(self) -> int:
        return self.special_id_map.get("<pad>", 0)

    @property
    def bos_token_id(self) -> int:
        return self.special_id_map.get("<bos>", 2)

    @property
    def eos_token_id(self) -> int:
        return self.special_id_map.get("<eos>", 3)

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> "ByteBPETokenizer":
        """Learn merges from ``text`` until the vocabulary reaches ``vocab_size``."""
        floor = self.n_special + 256
        if vocab_size < floor:
            raise ValueError(f"vocab_size must be >= {floor} (specials + 256 bytes)")
        num_merges = vocab_size - floor

        words: Counter = Counter()
        for chunk in SPLIT_PATTERN.findall(text):
            words[chunk] += 1
        # Each word is a list of byte-ids plus its corpus frequency.
        corpus: List[Tuple[List[int], int]] = [
            ([self.n_special + b for b in word.encode("utf-8")], count)
            for word, count in words.items()
        ]

        self.merges = {}
        for i in range(num_merges):
            stats: Counter = Counter()
            for ids, count in corpus:
                for pair in zip(ids, ids[1:]):
                    stats[pair] += count
            if not stats:
                break
            # Deterministic tie-break so training is reproducible.
            pair = max(stats.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))[0]
            new_id = floor + i
            self.merges[pair] = new_id
            corpus = [(_merge(ids, pair, new_id), count) for ids, count in corpus]
            if verbose and (i + 1) % 100 == 0:
                print(f"  merge {i + 1}/{num_merges} -> id {new_id} (count {stats[pair]})")

        self._rebuild()
        return self

    def _encode_chunk(self, chunk: str) -> List[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        # `surrogatepass` keeps malformed strings (lone surrogates from broken
        # upstream decoding) from raising in the middle of a request.
        ids = [self.n_special + b for b in chunk.encode("utf-8", errors="surrogatepass")]
        while len(ids) >= 2:
            pair = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, float("inf")))
            if pair not in self.ranks:
                break
            ids = _merge(ids, pair, self.merges[pair])
        if len(self._cache) < 100_000:
            self._cache[chunk] = ids
        return ids

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a str")
        ids: List[int] = [self.bos_token_id] if add_bos else []
        for chunk in SPLIT_PATTERN.findall(text):
            ids.extend(self._encode_chunk(chunk))
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, token_ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        """Decode ids to text.

        Ids outside the vocabulary are dropped rather than raising, so an
        untrained / mismatched model can never crash the caller.
        """
        specials = set(self.special_id_map.values())
        parts: List[bytes] = []
        for tid in token_ids:
            tid = int(tid)
            if skip_special_tokens and tid in specials:
                continue
            piece = self.vocab.get(tid)
            if piece is not None:
                parts.append(piece)
        return b"".join(parts).decode("utf-8", errors="replace")

    def token_bytes(self, token_id: int, skip_special_tokens: bool = True) -> bytes:
        """Raw bytes for a single id - used by incremental/streaming decoders."""
        token_id = int(token_id)
        if skip_special_tokens and token_id in set(self.special_id_map.values()):
            return b""
        return self.vocab.get(token_id, b"")

    def save(self, path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {
            "version": 2,
            "special_tokens": self.special_tokens,
            "merges": [[a, b, new] for (a, b), new in self.merges.items()],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "ByteBPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        merges = {(a, b): new for a, b, new in payload.get("merges", [])}
        return cls(merges=merges, special_tokens=payload.get("special_tokens", DEFAULT_SPECIALS))

    def __repr__(self) -> str:
        return f"ByteBPETokenizer(vocab_size={self.vocab_size}, merges={len(self.merges)})"


def _merge(ids: List[int], pair: Pair, new_id: int) -> List[int]:
    out: List[int] = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


# Backwards-compatible alias for the original class name.
SimpleTokenizer = ByteBPETokenizer
