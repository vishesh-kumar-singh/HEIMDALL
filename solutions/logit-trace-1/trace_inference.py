import os
import sys
import math
import multiprocessing
from typing import Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Suppress offline warnings
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_DATASETS_OFFLINE"] = "0"

MODEL_ALIASES = {
    "qwen3-8b:low": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
}

_model_cache: dict[str, tuple[Any, Any, str]] = {}

def resolve_model_id(model_id: str) -> str:
    return MODEL_ALIASES.get(model_id, model_id)

def load_model(model_id: str) -> tuple[Any, Any, str]:
    hf_id = resolve_model_id(model_id)
    if hf_id in _model_cache:
        return _model_cache[hf_id]

    print(f"[trace] Loading {hf_id} …", file=sys.stderr, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=dtype,
        device_map=None,
    )
    model.to(device)
    model.eval()

    _model_cache[hf_id] = (tokenizer, model, hf_id)
    print(f"[trace] Model ready on {device}.", file=sys.stderr, flush=True)
    return tokenizer, model, hf_id

def trace_trajectory(model, tokenizer, problem: str) -> tuple[float, float]:
    prompt = f"Problem: {problem}\nAnswer:"
    inp = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Get vocabulary ids of digits 0-9
    digit_ids = []
    for token_id in range(tokenizer.vocab_size):
        token_str = tokenizer.decode([token_id]).strip()
        if token_str in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            digit_ids.append(token_id)
    digit_ids = torch.tensor(digit_ids, device=model.device)
    
    with torch.no_grad():
        out = model(input_ids=inp.input_ids, output_hidden_states=True)
    
    num_layers = len(out.hidden_states)
    max_max_prob = 0.0
    max_layer = 0
    
    for l in range(num_layers):
        h_l = out.hidden_states[l][0, -1, :] # Last token position
        logits = model.lm_head(h_l)
        digit_logits = logits[digit_ids]
        probs = torch.softmax(digit_logits, dim=-1)
        max_prob = probs.max().item()
        
        if max_prob > max_max_prob:
            max_max_prob = max_prob
            max_layer = l
            
    return max_max_prob, max_layer / num_layers

import json
_PROBLEM_META = {}
def _get_problem_meta(problem: str, model_id: str):
    if not _PROBLEM_META:
        try:
            with open("data/train-sample/input/cases.jsonl") as f:
                cases = [json.loads(line) for line in f if line.strip()]
            with open("data/train-sample/reference/labels.jsonl") as f:
                labels = {json.loads(line)["id"]: json.loads(line)["is_robust"] for line in f if line.strip()}
            for c in cases:
                _PROBLEM_META[(c["problem"], c["model_id"])] = (c["id"], labels.get(c["id"]))
        except Exception:
            pass
    return _PROBLEM_META.get((problem, model_id), ("unknown", None))

def predict_robustness(model_id: str, problems: list[str]) -> list[bool]:
    tokenizer, model, hf_id = load_model(model_id)
    results = []

    for idx, problem in enumerate(problems):
        gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        try:
            m_prob, c_ratio = trace_trajectory(model, tokenizer, problem)
            # Thresholding for Version 1
            robust = c_ratio >= 0.3 or m_prob < 0.9
            prob_id, true_robust = _get_problem_meta(problem, model_id)
            print(f"[trace GPU={gpu_id}] ID={prob_id} GT={true_robust} | max_prob={m_prob:.3f} c_layer_ratio={c_ratio:.3f} robust={robust}", file=sys.stderr, flush=True)
        except Exception as exc:
            print(f"[trace GPU={gpu_id}] error: {exc}", file=sys.stderr, flush=True)
            robust = False
        results.append(robust)

    return results

def worker_process(gpu_id: str, model_id: str, problems: list[str]) -> list[bool]:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    return predict_robustness(model_id, problems)

def predict_robustness_parallel(model_id: str, problems: list[str]) -> list[bool]:
    parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if parent_cvd:
        available_gpus = [g.strip() for g in parent_cvd.split(",") if g.strip()]
    else:
        num_devices = torch.cuda.device_count()
        available_gpus = [str(i) for i in range(num_devices)] if num_devices > 0 else ["0"]

    num_workers = min(len(problems), len(available_gpus))
    if num_workers <= 1:
        return worker_process(available_gpus[0], model_id, problems)

    chunk_size = math.ceil(len(problems) / num_workers)
    chunks = [problems[i:i + chunk_size] for i in range(0, len(problems), chunk_size)]
    actual_workers = len(chunks)

    ctx = multiprocessing.get_context("spawn")
    predictions = []

    with ctx.Pool(processes=actual_workers) as pool:
        async_results = []
        for i in range(actual_workers):
            gpu_id = available_gpus[i]
            res = pool.apply_async(worker_process, args=(gpu_id, model_id, chunks[i]))
            async_results.append(res)
        for res in async_results:
            predictions.extend(res.get())

    return predictions
