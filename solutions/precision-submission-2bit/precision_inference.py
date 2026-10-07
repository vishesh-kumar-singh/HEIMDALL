"""Precision-drop answer flip robustness detection.

Runs clean generation and then quantized (masked mantissa) generation.
If the answer flips, it marks it as False (non-robust).
"""

from __future__ import annotations

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
import re
import sys
import math
import multiprocessing
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ALIASES: dict[str, str] = {
    "qwen3-8b:low": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
}

MATH_SUFFIX: str = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
MAX_THINKING_TOKENS = 4096
MAX_ANSWER_TOKENS = 64
BATCH_SIZE = 2

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_model_id(model_id: str) -> str:
    # During testing, route all the fake track aliases to DeepSeek
    if ":low" in model_id or "gpt" in model_id or "gemini" in model_id or "GLM" in model_id:
        return "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    return MODEL_ALIASES.get(model_id, model_id)

def format_prompt(tokenizer, problem: str) -> str:
    messages = [{"role": "user", "content": problem + MATH_SUFFIX}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        return problem + MATH_SUFFIX

_BOXED_RE = re.compile(r"\\boxed\{")

def extract_boxed(text: str) -> str | None:
    matches = list(_BOXED_RE.finditer(text))
    if not matches:
        return None
    start = matches[-1].end()
    depth, pos = 1, start
    while pos < len(text) and depth > 0:
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
        pos += 1
    return text[start : pos - 1].strip() if depth == 0 else text[start:pos].strip()

def norm(answer: str | None) -> str:
    if answer is None:
        return "__NONE__"
    a = answer.strip()
    for l, r in (("$", "$"), ("\\(", "\\)")):
        if a.startswith(l) and a.endswith(r):
            a = a[len(l) : -len(r)].strip()
    return a or "__EMPTY__"

def mask_precision(model, mask=~0x3):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dtype == torch.bfloat16:
                new_data = param.view(torch.int16).bitwise_and_(mask).view(torch.bfloat16)
                param.copy_(new_data)

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

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def process_batch(model, tokenizer, problems, prefix="CLEAN"):
    prompts = [format_prompt(tokenizer, p) for p in problems]
    inp = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    inp = {k: v.to(model.device) for k, v in inp.items()}
    
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=MAX_THINKING_TOKENS, do_sample=False)
        
    answers = []
    for i, p in enumerate(problems):
        gen_ids = out[i, inp["input_ids"].shape[1] :]
        full_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
        
        ans_str = extract_boxed(full_text)
        if not ans_str:
            if "</think>" not in full_text:
                force_prompt = full_text + "\n</think>\n\nTime is up. Therefore, my best guess for the final mathematical answer is \\boxed{"
            else:
                force_prompt = full_text + "\n\nTime is up. Therefore, my best guess for the final mathematical answer is \\boxed{"
            
            force_ids = tokenizer(force_prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
            with torch.no_grad():
                fo = model.generate(force_ids, max_new_tokens=MAX_ANSWER_TOKENS, do_sample=False)
            fo_text = tokenizer.decode(fo[0, force_ids.shape[1]:], skip_special_tokens=False)
            ans_str = extract_boxed("\\boxed{" + fo_text)

        answers.append(norm(ans_str))
        
    return answers


def predict_robustness(
    model_id: str,
    problems: list[str],
) -> list[bool]:
    
    hf_id = resolve_model_id(model_id)
    print(f"[precision] Loading {hf_id}...", file=sys.stderr, flush=True)
    
    tokenizer = AutoTokenizer.from_pretrained(hf_id, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=dtype,
        device_map=None,
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    
    clean_answers = []
    print(f"[precision] Running clean batched generation on {len(problems)} problems...", file=sys.stderr, flush=True)
    for i in range(0, len(problems), BATCH_SIZE):
        batch = problems[i:i+BATCH_SIZE]
        clean_answers.extend(process_batch(model, tokenizer, batch, "CLEAN"))
        
    print(f"[precision] Dropping precision (~0xF mask)...", file=sys.stderr, flush=True)
    mask_precision(model, mask=~0xF)
    
    noisy_answers = []
    print(f"[precision] Running noisy batched generation on {len(problems)} problems...", file=sys.stderr, flush=True)
    for i in range(0, len(problems), BATCH_SIZE):
        batch = problems[i:i+BATCH_SIZE]
        noisy_answers.extend(process_batch(model, tokenizer, batch, "NOISY"))
        
    # Free memory
    del model
    torch.cuda.empty_cache()
        
    results = []
    for p, c, n in zip(problems, clean_answers, noisy_answers):
        # True if robust (didn't flip), False if flipped
        is_robust = (c == n)
        results.append(is_robust)
        
        prob_id, true_robust = _get_problem_meta(p, model_id)
        print(f"[precision_extract] ID={prob_id} | GT={true_robust} | CLEAN={c} | NOISY={n} | PRED={is_robust}", file=sys.stderr, flush=True)
        
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
        
    # Spawn 2 workers per GPU to maximize utilization
    available_gpus = [g for g in available_gpus for _ in range(2)]

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
