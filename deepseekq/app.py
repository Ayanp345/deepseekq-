"""Flask inference service.

Environment
-----------
    MODEL_DIR   directory holding config.json / model.pt / tokenizer.json
                (default: "checkpoints")
    DEVICE      cuda | cpu | mps (default: auto)
    API_KEY     if set, requests must send  Authorization: Bearer <key>
    MAX_NEW_TOKENS  hard server-side ceiling (default 512)
    PORT        default 5000

Run in production with a WSGI server rather than the dev server:
    gunicorn -w 1 -t 120 -b 0.0.0.0:5000 app:app
(one worker: each worker loads its own copy of the weights)
"""

from __future__ import annotations

import codecs
import json
import logging
import os
import threading
import time
from typing import Any, Dict, Tuple

import torch
from flask import Flask, Response, jsonify, request, stream_with_context
from werkzeug.exceptions import HTTPException

from model import DeepSeekConfig, DeepSeekForCausalLM, _safe_torch_load
from tokenizer import ByteBPETokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("deepseekq")

MODEL_DIR = os.getenv("MODEL_DIR", "checkpoints")
API_KEY = os.getenv("API_KEY")
MAX_NEW_TOKENS_CAP = int(os.getenv("MAX_NEW_TOKENS", "512"))

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# The model is a single shared object and is not thread-safe; serialise access.
_lock = threading.Lock()


def _pick_device() -> torch.device:
    requested = os.getenv("DEVICE")
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_artifacts() -> Tuple[DeepSeekForCausalLM, ByteBPETokenizer, torch.device, bool]:
    device = _pick_device()
    tokenizer_path = os.path.join(MODEL_DIR, "tokenizer.json")
    if os.path.exists(tokenizer_path):
        tokenizer = ByteBPETokenizer.load(tokenizer_path)
    else:
        log.warning("no tokenizer.json in %s - falling back to raw byte-level", MODEL_DIR)
        tokenizer = ByteBPETokenizer()

    config_path = os.path.join(MODEL_DIR, "config.json")
    if os.path.exists(config_path):
        config = DeepSeekConfig.from_file(config_path)
    else:
        config = DeepSeekConfig(vocab_size=tokenizer.vocab_size)
        log.warning("no config.json in %s - using defaults", MODEL_DIR)

    if config.vocab_size < tokenizer.vocab_size:
        raise RuntimeError(
            f"config.vocab_size ({config.vocab_size}) is smaller than the tokenizer "
            f"vocabulary ({tokenizer.vocab_size}); these artifacts do not match"
        )

    model = DeepSeekForCausalLM(config)
    weights_path = os.path.join(MODEL_DIR, "model.pt")
    trained = os.path.exists(weights_path)
    if trained:
        state = _safe_torch_load(weights_path, map_location="cpu")
        state = state.get("model", state) if isinstance(state, dict) else state
        missing, unexpected = model.load_state_dict(state, strict=False)
        ignorable = {"rope_cos", "rope_sin", "lm_head.weight"}
        hard = [k for k in missing if k not in ignorable]
        if hard or unexpected:
            raise RuntimeError(f"checkpoint mismatch: missing={hard} unexpected={unexpected}")
        log.info("loaded weights from %s", weights_path)
    else:
        log.warning("no weights at %s - serving a RANDOMLY INITIALISED model", weights_path)

    model.to(device).eval()
    return model, tokenizer, device, trained


model, tokenizer, DEVICE, TRAINED = load_artifacts()
log.info("ready: %s params on %s (trained=%s)", f"{model.num_parameters():,}", DEVICE, TRAINED)


# ------------------------------------------------------------------ helpers
class BadRequest(Exception):
    pass


def _require_auth() -> None:
    if not API_KEY:
        return
    header = request.headers.get("Authorization", "")
    if header != f"Bearer {API_KEY}":
        raise PermissionError("invalid or missing API key")


def _number(payload: Dict[str, Any], key: str, default: float,
            low: float, high: float, cast=float):
    value = payload.get(key, default)
    try:
        value = cast(value)
    except (TypeError, ValueError):
        raise BadRequest(f"'{key}' must be a number")
    if not low <= value <= high:
        raise BadRequest(f"'{key}' must be between {low} and {high}")
    return value


def parse_payload() -> Dict[str, Any]:
    payload = request.get_json(silent=True)
    if payload is None:
        raise BadRequest("body must be valid JSON with Content-Type: application/json")
    if not isinstance(payload, dict):
        raise BadRequest("body must be a JSON object")

    prompt = payload.get("prompt", "")
    if not isinstance(prompt, str):
        raise BadRequest("'prompt' must be a string")
    if len(prompt) > 100_000:
        raise BadRequest("'prompt' is too long")

    return {
        "prompt": prompt,
        "max_new_tokens": _number(payload, "max_new_tokens", 32, 1, MAX_NEW_TOKENS_CAP, int),
        "temperature": _number(payload, "temperature", 0.8, 0.0, 5.0),
        "top_k": _number(payload, "top_k", 50, 0, model.config.vocab_size, int),
        "top_p": _number(payload, "top_p", 0.95, 0.0, 1.0),
        "repetition_penalty": _number(payload, "repetition_penalty", 1.1, 1.0, 2.0),
    }


# ------------------------------------------------------------------- routes
@app.get("/")
def index():
    return jsonify({
        "service": "deepseekq-from-scratch",
        "model": {
            "parameters": model.num_parameters(),
            "vocab_size": model.config.vocab_size,
            "context_length": model.config.context_length,
            "trained": TRAINED,
        },
        "routes": ["GET /health", "POST /generate", "POST /generate/stream"],
    })


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "device": str(DEVICE),
        "parameters": model.num_parameters(),
        "non_embedding_parameters": model.num_parameters(non_embedding=True),
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "trained_weights": TRAINED,
    })


@app.post("/generate")
def generate():
    _require_auth()
    params = parse_payload()
    started = time.time()
    with _lock, torch.inference_mode():
        text = model.generate_text(
            tokenizer, params["prompt"],
            max_new_tokens=params["max_new_tokens"],
            temperature=params["temperature"],
            top_k=params["top_k"],
            top_p=params["top_p"],
            repetition_penalty=params["repetition_penalty"],
            return_full_text=False,
        )
    elapsed = time.time() - started
    return jsonify({
        "prompt": params["prompt"],
        "completion": text,
        "max_new_tokens": params["max_new_tokens"],
        "elapsed_seconds": round(elapsed, 3),
        "warning": None if TRAINED else "model weights are randomly initialised",
    })


@app.post("/generate/stream")
def generate_stream():
    """Server-sent events; one JSON `data:` frame per decoded chunk."""
    _require_auth()
    params = parse_payload()
    ids = tokenizer.encode(params["prompt"]) or [tokenizer.bos_token_id]
    input_ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    def stream():
        # Incremental UTF-8 decoding: a token can end mid-codepoint.
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        try:
            with _lock, torch.inference_mode():
                for token in model.stream_generate(
                    input_ids,
                    max_new_tokens=params["max_new_tokens"],
                    temperature=params["temperature"],
                    top_k=params["top_k"],
                    top_p=params["top_p"],
                    repetition_penalty=params["repetition_penalty"],
                    eos_token_id=tokenizer.eos_token_id,
                    restrict_vocab=tokenizer.vocab_size,
                ):
                    piece = decoder.decode(tokenizer.token_bytes(int(token[0, 0])))
                    if piece:
                        yield f"data: {json.dumps({'token': piece})}\n\n"
            tail = decoder.decode(b"", final=True)
            if tail:
                yield f"data: {json.dumps({'token': tail})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:  # surface errors inside the stream
            log.exception("stream failed")
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return Response(stream_with_context(stream()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# --------------------------------------------------------------- error paths
@app.errorhandler(BadRequest)
def _bad_request(exc):
    return jsonify({"error": str(exc)}), 400


@app.errorhandler(PermissionError)
def _unauthorised(exc):
    return jsonify({"error": str(exc)}), 401


@app.errorhandler(404)
def _not_found(_):
    return jsonify({"error": "not found"}), 404


@app.errorhandler(Exception)
def _server_error(exc):
    if isinstance(exc, HTTPException):      # 404/405/413 keep their own status
        return jsonify({"error": exc.description}), exc.code
    log.exception("unhandled error")
    return jsonify({"error": "internal server error", "detail": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False, threaded=True)
