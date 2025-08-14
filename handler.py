# handler.py
# Qwen3-Reranker-8B scoring via CausalLM next-token "yes/no" probabilities.
# - Loads once per pod and exposes a Runpod Serverless handler.
# - Input (to /runsync or /run -> /status):
#   {
#     "query": "<string>",
#     "documents": ["doc1", "doc2", ...],
#     "batch_size": 8,            # optional (1..32), default 8
#     "max_length": 1024,         # optional (64..8192), default 1024
#     "task": "<custom instruct>" # optional; default provided below
#   }
# - Output (sorted by score desc):
#   {
#     "model": "...",
#     "device": "cuda" | "cpu",
#     "count": <int>,
#     "latency_sec": <float>,
#     "reranked": [ { "index": <int>, "score": <float>, "text": "<doc>" }, ... ]
#   }

import os
import time
from typing import List, Dict, Any

import torch
import runpod
from transformers import AutoTokenizer, AutoModelForCausalLM


# ----------------------------
# Env & dtype
# ----------------------------
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-Reranker-8B")
HF_TOKEN = os.getenv("HF_TOKEN")  # optional; put as a Secret in Runpod if needed
USE_FLASH_ATTENTION_2 = os.getenv("USE_FLASH_ATTENTION_2", "0").lower() not in ("0", "false", "")

DTYPE = (
    torch.bfloat16
    if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    else (torch.float16 if torch.cuda.is_available() else torch.float32)
)

if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


# ----------------------------
# Utilities
# ----------------------------
def _candidate_last_token_ids(tokenizer: AutoTokenizer, candidates: List[str]) -> List[int]:
    """
    Return a unique list of the *last* token IDs for several surface forms of a short word.
    This makes us robust to space/case tokenization variants (e.g., " yes", "Yes", "yes").
    """
    ids = []
    for t in candidates:
        toks = tokenizer(t, add_special_tokens=False).input_ids
        if toks:
            ids.append(toks[-1])
    uniq = []
    seen = set()
    for i in ids:
        if i not in seen:
            uniq.append(i)
            seen.add(i)
    # very last-resort fallback
    if not uniq:
        try:
            uniq = [tokenizer.convert_tokens_to_ids("yes")]
        except Exception:
            uniq = [1]  # usually <unk>
    return uniq


def _build_prompts(query: str, docs: List[str], task: str) -> List[str]:
    """
    Per model card: force a strict yes/no judgment.
    We form a chat-style prompt and read the next-token distribution at the last position.
    """
    sys = (
        'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
        'Note that the answer can only be "yes" or "no".'
    )
    prefix = f"<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    pairs = []
    for d in docs:
        pairs.append(f"{prefix}<Instruct>: {task}\n<Query>: {query}\n<Document>: {d}{suffix}")
    return pairs


# ----------------------------
# Load once per pod
# ----------------------------
print(f"[boot] loading reranker (CausalLM): {MODEL_ID} dtype={DTYPE} device_map=auto")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_fast=True,
    token=HF_TOKEN,
    trust_remote_code=True,
    padding_side="left",  # left-pad so logits[:, -1, :] aligns to last non-pad token
)

model_kwargs = dict(
    torch_dtype=DTYPE,
    device_map="auto",
    token=HF_TOKEN,
    trust_remote_code=True,
)

model = None
if USE_FLASH_ATTENTION_2:
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            attn_implementation="flash_attention_2",
            **model_kwargs,
        ).eval()
        print("[boot] using flash_attention_2")
    except Exception as e:
        print(f"[boot] flash_attention_2 unavailable, falling back. Reason: {type(e).__name__}: {e}")

if model is None:
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs).eval()

# Ensure PAD exists (set PAD = EOS if missing)
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
    model.config.pad_token_id = tokenizer.pad_token_id

print("[boot] pad_token_id =", tokenizer.pad_token_id, "eos_token_id =", tokenizer.eos_token_id)
print("[boot] padding_side =", tokenizer.padding_side)

# Precompute candidate token id sets for "yes"/"no"
YES_IDS = _candidate_last_token_ids(tokenizer, [" yes", "Yes", " yes.", "YES", "yes"])
NO_IDS = _candidate_last_token_ids(tokenizer, [" no", "No", " no.", "NO", "no"])
print(f"[boot] YES_IDS={YES_IDS} NO_IDS={NO_IDS}")


# ----------------------------
# Core scoring via next-token logits
# ----------------------------
@torch.inference_mode()
def score_pairs(query: str, docs: List[str], *, max_length: int = 1024, task: str = None) -> List[float]:
    """
    Return P('yes') per document (higher = more relevant).
    We read the final-position next-token distribution and compare mass for yes vs no.
    """
    task = task or "Given a web search query, retrieve relevant passages that answer the query"
    prompts = _build_prompts(query, docs, task)

    enc = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}

    out = model(**enc)
    last_logits = out.logits[:, -1, :]  # [batch, vocab]

    yes_idx = torch.tensor(YES_IDS, device=last_logits.device, dtype=torch.long)
    no_idx = torch.tensor(NO_IDS, device=last_logits.device, dtype=torch.long)

    yes_logit = torch.logsumexp(last_logits.index_select(1, yes_idx), dim=1)
    no_logit = torch.logsumexp(last_logits.index_select(1, no_idx), dim=1)
    logits_2 = torch.stack([no_logit, yes_logit], dim=1)  # [batch, 2]
    probs = torch.softmax(logits_2, dim=1)[:, 1]          # P("yes")
    return probs.float().cpu().tolist()


def rerank(inp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input:
    {
      "query": "<string>",
      "documents": ["doc1", "doc2", ...],
      "batch_size": 8,         # optional, internal micro-batch
      "max_length": 1024,      # optional, tokenizer truncation length
      "task": "<custom instruct>"  # optional
    }

    Output:
    {
      "model": MODEL_ID,
      "device": "cuda"|"cpu",
      "count": <int>,
      "latency_sec": <float>,
      "reranked": [ { "index": <int>, "score": <float>, "text": "<doc>" }, ... ]
    }
    """
    query = inp.get("query")
    docs = inp.get("documents")
    task = inp.get("task")

    if not isinstance(query, str) or not query.strip():
        return {"error": "Missing 'query' (non-empty string)."}

    if not isinstance(docs, list) or len(docs) == 0 or not all(isinstance(d, str) for d in docs):
        return {"error": "Missing 'documents' (non-empty list of strings)."}

    # Controls
    max_length = int(inp.get("max_length", 1024))
    max_length = max(64, min(max_length, 8192))  # guardrails

    base_bs = int(inp.get("batch_size", 8))
    base_bs = max(1, min(base_bs, 32))

    t0 = time.time()
    scores: List[float] = [0.0] * len(docs)

    # Chunk over documents with OOM fallback
    start = 0
    cur_bs = min(base_bs, len(docs))
    while start < len(docs):
        end = min(start + cur_bs, len(docs))
        chunk = docs[start:end]
        try:
            s = score_pairs(query, chunk, max_length=max_length, task=task)
            for i, sc in enumerate(s):
                scores[start + i] = float(sc)
            start = end
            cur_bs = min(base_bs, len(docs) - start) or 0
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if cur_bs > 1:
                cur_bs = max(1, cur_bs // 2)
                continue
            # already at 1 -> re-raise
            raise

    # Build results with original indices
    results = [{"index": i, "score": float(scores[i]), "text": docs[i]} for i in range(len(docs))]
    results.sort(key=lambda x: x["score"], reverse=True)

    elapsed = round(time.time() - t0, 3)
    return {
        "model": MODEL_ID,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "count": len(docs),
        "latency_sec": elapsed,
        "reranked": results,
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


# Entrypoint for Runpod
runpod.serverless.start({"handler": handler})
