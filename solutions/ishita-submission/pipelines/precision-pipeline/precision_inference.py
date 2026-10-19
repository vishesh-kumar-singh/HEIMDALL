"""Precision-drop answer-flip robustness detector.

Solve each problem with the clean bf16 model, then again after zeroing the 4 lowest
mantissa bits of every bf16 weight. If the final answer flips, predict not robust.

Changes vs. the original: no label-file lookup, dtype= for Transformers 5, deterministic
(greedy) decoding only, soft deadline with partial results, per-chunk failure isolation.
Logic, prompts, token limits and the mask are unchanged.
"""
from __future__ import annotations

import math
import multiprocessing
import os
import re
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEEPSEEK = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
MODEL_ALIASES = {"qwen3-8b:low": DEEPSEEK}
MATH_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
MAX_THINKING_TOKENS = 4096
MAX_ANSWER_TOKENS = 64
BATCH_SIZE = 2
WORKERS_PER_GPU = 2
FALLBACK_ON_ERROR = False
FALLBACK_ON_DEADLINE = False


def resolve_model_id(model_id: str) -> str:
    # Track aliases / API-model ids use the DeepSeek checkpoint as their white-box proxy.
    if ":low" in model_id or "gpt" in model_id or "gemini" in model_id or "GLM" in model_id:
        return DEEPSEEK
    return MODEL_ALIASES.get(model_id, model_id)


def format_prompt(tokenizer, problem: str) -> str:
    messages = [{"role": "user", "content": problem + MATH_SUFFIX}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return problem + MATH_SUFFIX


_BOXED_RE = re.compile(r"\\boxed\{")


def extract_boxed(text: str):
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
    return text[start: pos - 1].strip() if depth == 0 else text[start:pos].strip()


def norm(answer):
    if answer is None:
        return "__NONE__"
    a = answer.strip()
    for l, r in (("$", "$"), ("\\(", "\\)")):
        if a.startswith(l) and a.endswith(r):
            a = a[len(l): -len(r)].strip()
    return a or "__EMPTY__"


def mask_precision(model, mask=~0xF):
    with torch.no_grad():
        for _, param in model.named_parameters():
            if param.dtype == torch.bfloat16:
                new_data = param.view(torch.int16).bitwise_and_(mask).view(torch.bfloat16)
                param.copy_(new_data)


def process_batch(model, tokenizer, problems):
    prompts = [format_prompt(tokenizer, p) for p in problems]
    inp = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    inp = {k: v.to(model.device) for k, v in inp.items()}
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=MAX_THINKING_TOKENS, do_sample=False)

    answers = []
    for i, _ in enumerate(problems):
        gen_ids = out[i, inp["input_ids"].shape[1]:]
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


def predict_robustness(model_id, problems, deadline=None):
    n = len(problems)
    hf_id = resolve_model_id(model_id)
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_id, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        try:
            model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=dtype, device_map=None, local_files_only=True)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=dtype, device_map=None, local_files_only=True)
        model.to(device)
        model.eval()
    except Exception as exc:
        print(f"[precision] model load failed: {exc}", file=sys.stderr, flush=True)
        return [FALLBACK_ON_ERROR] * n

    t0 = time.time()
    mid_deadline = None if deadline is None else t0 + 0.5 * max(0.0, deadline - t0)

    clean = []
    try:
        for i in range(0, n, BATCH_SIZE):
            if mid_deadline is not None and time.time() > mid_deadline:
                break  # only problems finished here get a noisy pass too
            clean.extend(process_batch(model, tokenizer, problems[i:i + BATCH_SIZE]))
    except Exception as exc:
        print(f"[precision] clean phase error: {exc!r}", file=sys.stderr, flush=True)
        clean = clean[: (len(clean) // BATCH_SIZE) * BATCH_SIZE]

    noisy = []
    if clean:
        try:
            mask_precision(model, mask=~0xF)
            for i in range(0, len(clean), BATCH_SIZE):
                if deadline is not None and time.time() > deadline:
                    break
                noisy.extend(process_batch(model, tokenizer, problems[i:i + BATCH_SIZE]))
        except Exception as exc:
            print(f"[precision] noisy phase error: {exc!r}", file=sys.stderr, flush=True)

    del model
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    k = min(len(clean), len(noisy))
    results = [bool(clean[i] == noisy[i]) for i in range(k)]
    results += [FALLBACK_ON_DEADLINE] * (n - k)
    return results


def worker_process(gpu_id, model_id, problems, deadline=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    return predict_robustness(model_id, problems, deadline=deadline)


def predict_robustness_parallel(model_id, problems, deadline=None):
    parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if parent_cvd:
        gpus = [g.strip() for g in parent_cvd.split(",") if g.strip()]
    else:
        n_dev = torch.cuda.device_count()
        gpus = [str(i) for i in range(n_dev)] if n_dev > 0 else ["0"]
    gpus = [g for g in gpus for _ in range(WORKERS_PER_GPU)]

    num_workers = min(len(problems), len(gpus))
    if num_workers <= 1:
        return worker_process(gpus[0], model_id, problems, deadline)

    chunk_size = math.ceil(len(problems) / num_workers)
    chunks = [problems[i:i + chunk_size] for i in range(0, len(problems), chunk_size)]
    ctx = multiprocessing.get_context("spawn")
    predictions = []
    with ctx.Pool(processes=len(chunks)) as pool:
        pending = [pool.apply_async(worker_process, (gpus[i], model_id, chunks[i], deadline))
                   for i in range(len(chunks))]
        for i, res in enumerate(pending):
            try:
                predictions.extend(res.get())
            except Exception as exc:
                print(f"[precision] chunk {i} failed: {exc!r}", file=sys.stderr, flush=True)
                predictions.extend([FALLBACK_ON_ERROR] * len(chunks[i]))
    return predictions
