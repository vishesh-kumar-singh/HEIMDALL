"""Logit-lens / text-feature robustness detector (DeepSeek, Olmo, Skywork, Qwen-family rules).

Same features and thresholds as the original. Changes: imports work from any
working directory, the 4096-token generation only runs where a rule needs the text
(Skywork), no label lookup, per-chunk failure isolation, soft deadline,
non-fatal logging.
"""
from __future__ import annotations

import json
import math
import multiprocessing
import os
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from logit_features import extract_features_fast, get_digit_ids

MODEL_ALIASES = {"qwen3-8b:low": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"}
THINKING_MODELS = {"deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"}
MAX_THINKING_TOKENS = 4096
MATH_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
FALLBACK_ON_ERROR = False
FALLBACK_ON_DEADLINE = False


def resolve_model_id(model_id: str) -> str:
    return MODEL_ALIASES.get(model_id, model_id)


def load_model(model_id: str):
    hf_id = resolve_model_id(model_id)
    tokenizer = AutoTokenizer.from_pretrained(hf_id, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, dtype=dtype, device_map=None, local_files_only=True
        )
    except TypeError:  # older transformers
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=dtype, device_map=None, local_files_only=True
        )
    model.to(device)
    model.eval()
    return tokenizer, model, hf_id


def format_prompt(tokenizer, problem: str) -> str:
    messages = [{"role": "user", "content": problem + MATH_SUFFIX}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return problem + MATH_SUFFIX


def generate_text(model, tokenizer, problem: str) -> str:
    prompt = format_prompt(tokenizer, problem)
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inp = {k: v.to(model.device) for k, v in inp.items()}
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=MAX_THINKING_TOKENS, do_sample=False)
    gen_ids = out[0, inp["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=False)


def thinking_portion(text: str, is_thinking: bool) -> str:
    import re
    if is_thinking:
        m = re.search(r"<think>(.*?)(?:</think>|$)", text, re.DOTALL)
        return m.group(1) if m else text
    idx = text.find("\\boxed{")
    return text[:idx] if idx > 0 else text


def decide(model_id: str, features: dict, wait_count: int, recheck_count: int) -> bool:
    """Hard-coded per-model rules (unchanged)."""
    if "Olmo-3-7B" in model_id:
        return features.get("kl_mean", 1.0) <= 0.02
    if "DeepSeek" in model_id:
        if features.get("de_early", 0.0) <= 2.23:
            return False
        return features.get("fp_early", 1.0) <= 0.00
    if "Skywork" in model_id:
        return wait_count > 0 or recheck_count > 0
    # Qwen3.5-4B / Qwen-family / general fallback
    if features.get("dp_max", 1.0) <= 0.79:
        return features.get("hn_slope", 3.0) <= 2.08
    return False


def needs_text(model_id: str) -> bool:
    """Only the Skywork rule reads the generated trace."""
    return "Skywork" in model_id


def predict_robustness(model_id, problems, flip_threshold: int = 1, deadline=None):
    try:
        tokenizer, model, hf_id = load_model(model_id)
    except Exception as exc:
        print(f"[logit] model load failed: {exc}", file=sys.stderr, flush=True)
        return [FALLBACK_ON_ERROR] * len(problems)

    digit_ids = None
    results = []
    for idx, problem in enumerate(problems):
        if deadline is not None and time.time() > deadline:
            results.append(FALLBACK_ON_DEADLINE)
            continue
        try:
            if digit_ids is None:
                digit_ids = get_digit_ids(tokenizer, model.device)
            features = extract_features_fast(model, tokenizer, problem, digit_ids)

            wait_count = recheck_count = 0
            if needs_text(model_id):
                full_text = generate_text(model, tokenizer, problem)
                low = thinking_portion(full_text, hf_id in THINKING_MODELS).lower()
                wait_count = low.count("wait")
                recheck_count = low.count("recheck") + low.count("let me check") + low.count("recalculat")

            robust = bool(decide(model_id, features, wait_count, recheck_count))

            if os.environ.get("AIMO_DEBUG"):
                try:
                    dbg = {k: float(v) for k, v in features.items()}
                    dbg.update(idx=idx, wait=wait_count, recheck=recheck_count)
                    print(f"[logit] {json.dumps(dbg)} | PRED={robust}", file=sys.stderr, flush=True)
                except Exception:
                    pass
        except Exception as exc:
            print(f"[logit] #{idx} error: {exc}", file=sys.stderr, flush=True)
            robust = FALLBACK_ON_ERROR
        results.append(bool(robust))
    return results


def worker_process(gpu_id, model_id, problems, flip_threshold, deadline=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    return predict_robustness(model_id, problems, flip_threshold=flip_threshold, deadline=deadline)


def predict_robustness_parallel(model_id, problems, flip_threshold: int = 1, deadline=None):
    """Data-parallel over the visible GPUs; a failed chunk only loses that chunk."""
    parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if parent_cvd:
        gpus = [g.strip() for g in parent_cvd.split(",") if g.strip()]
    else:
        n_dev = torch.cuda.device_count()
        gpus = [str(i) for i in range(n_dev)] if n_dev > 0 else ["0"]

    num_workers = min(len(problems), len(gpus))
    if num_workers <= 1:
        return worker_process(gpus[0], model_id, problems, flip_threshold, deadline)

    chunk_size = math.ceil(len(problems) / num_workers)
    chunks = [problems[i:i + chunk_size] for i in range(0, len(problems), chunk_size)]

    ctx = multiprocessing.get_context("spawn")
    predictions = []
    with ctx.Pool(processes=len(chunks)) as pool:
        pending = [pool.apply_async(worker_process, args=(gpus[i], model_id, chunks[i], flip_threshold, deadline))
                   for i in range(len(chunks))]
        for i, res in enumerate(pending):
            try:
                predictions.extend(res.get())
            except Exception as exc:
                print(f"[logit] chunk {i} failed: {exc!r}", file=sys.stderr, flush=True)
                predictions.extend([FALLBACK_ON_ERROR] * len(chunks[i]))
    return predictions
