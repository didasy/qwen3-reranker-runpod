import os
import time
from typing import List, Dict, Any

import torch
import runpod
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# ----------------------------
# Env
# ----------------------------
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-Reranker-8B")
HF_TOKEN = os.getenv("HF_TOKEN")  # put as a Secret in RunPod if needed

# dtype heuristic
DTYPE = (
    torch.bfloat16
    if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    else (torch.float16 if torch.cuda.is_available() else torch.float32)
)

if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

# ----------------------------
# Load once per pod
# ----------------------------
print(f"[boot] loading reranker: {MODEL_ID} dtype={DTYPE} device_map=auto")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, use_fast=True, token=HF_TOKEN, trust_remote_code=True
)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, torch_dtype=DTYPE, device_map="auto", token=HF_TOKEN, trust_remote_code=True
)
model.eval()

# pad fallback
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


# ----------------------------
# Core scoring
# ----------------------------
@torch.inference_mode()
def score_pairs(query: str, docs: List[str], max_length: int = 1024) -> List[float]:
    """Return a score per document (higher = more relevant)."""
    device = model.device
    # Batch tokenize as sentence pairs
    enc = tokenizer(
        [query] * len(docs),
        docs,
        padding=True,
        truncation=True,               # truncate longest first
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    out = model(**enc)
    # cross-encoder head: [batch, 1] or [batch]
    logits = out.logits.squeeze(-1).detach()
    # convert to Python floats
    return logits.float().cpu().tolist()


def rerank(inp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input (from /runsync):
    {
      "query": "<string>",
      "documents": ["doc1", "doc2", ...],
      "batch_size": 8,         # optional, internal micro-batch
      "max_length": 1024       # optional, tokenizer truncation length
    }

    Output (what your Node code expects):
    { "reranked": [ { "index": <int>, "score": <float>, "text": "<doc>" }, ... ] }
    """
    query = inp.get("query")
    docs = inp.get("documents")

    if not isinstance(query, str) or not query.strip():
        return {"error": "Missing 'query' (non-empty string)."}
    if not isinstance(docs, list) or len(docs) == 0 or not all(isinstance(d, str) for d in docs):
        return {"error": "Missing 'documents' (non-empty list of strings)."}

    # Controls
    max_length = int(inp.get("max_length", 1024))
    max_length = max(64, min(max_length, 8192))  # guardrails

    batch_size = int(inp.get("batch_size", 8))
    batch_size = max(1, min(batch_size, 32))

    t0 = time.time()
    scores: List[float] = [0.0] * len(docs)

    # Chunk over documents to control memory
    for start in range(0, len(docs), batch_size):
        end = min(start + batch_size, len(docs))
        chunk = docs[start:end]
        try:
            s = score_pairs(query, chunk, max_length=max_length)
        except torch.cuda.OutOfMemoryError:
            # fall back to smaller micro-batch
            torch.cuda.empty_cache()
            half = max(1, batch_size // 2)
            if half < batch_size:
                # retry per-doc
                for i, d in enumerate(chunk):
                    s1 = score_pairs(query, [d], max_length=max_length)
                    scores[start + i] = float(s1[0])
                continue
            else:
                raise
        # write scores back
        for i, sc in enumerate(s):
            scores[start + i] = float(sc)

    # Build results with original indices
    results = [{"index": i, "score": float(scores[i]), "text": docs[i]} for i in range(len(docs))]
    # Sort by score desc
    results.sort(key=lambda x: x["score"], reverse=True)

    elapsed = round(time.time() - t0, 3)
    return {
        "model": MODEL_ID,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "count": len(docs),
        "latency_sec": elapsed,
        "reranked": results,      # <-- what your Node parser reads
    }


# ----------------------------
# RunPod serverless handler
# ----------------------------
def handler(job):
    """
    RunPod invokes with:
    { "input": { "query": "...", "documents": ["..."], ... } }
    """
    try:
        params = job.get("input", {}) or {}
        return rerank(params)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)}"}


runpod.serverless.start({"handler": handler})
