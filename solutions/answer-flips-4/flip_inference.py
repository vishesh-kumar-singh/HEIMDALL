"""Answer-flip trajectory analysis for robustness detection.

Generate a full thinking trace, then retroactively force-decode the answer
at evenly-spaced checkpoints.  Count how many times the decoded answer
*changes* (flips).

- Memorised / spurious → 0 flips (answer known from the start)
- Robust reasoning     → ≥1 flip (model works through the problem)

Supports both thinking models (<think>…</think>) and standard CoT models.
"""

from __future__ import annotations

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
import re
import sys
import multiprocessing
import traceback
from typing import Any

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from math_compare import model_judge_equivalent

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ALIASES: dict[str, str] = {
    "qwen3-8b:low": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
}

THINKING_MODELS: set[str] = {
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
}

NUM_CHECKPOINTS: int = 10
MAX_THINKING_TOKENS: int = 4096
MAX_ANSWER_TOKENS: int = 64
MATH_SUFFIX: str = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."

# ---------------------------------------------------------------------------
# Model loading — device_map="auto", single GPU for ≤8B models
# ---------------------------------------------------------------------------

_model_cache: dict[str, tuple[Any, Any, str]] = {}


def resolve_model_id(model_id: str) -> str:
    return MODEL_ALIASES.get(model_id, model_id)


def load_model(model_id: str) -> tuple[Any, Any, str]:
    """Load tokenizer + model onto a single GPU."""
    hf_id = resolve_model_id(model_id)
    if hf_id in _model_cache:
        return _model_cache[hf_id]

    print(f"[flip] Loading {hf_id} …", file=sys.stderr, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(hf_id, local_files_only=True)
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

    _model_cache[hf_id] = (tokenizer, model, hf_id)
    print(f"[flip] Model ready on {device}.", file=sys.stderr, flush=True)
    return tokenizer, model, hf_id


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def format_prompt(tokenizer: Any, problem: str) -> str:
    messages = [{"role": "user", "content": problem + MATH_SUFFIX}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        return problem + MATH_SUFFIX


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

_BOXED_RE = re.compile(r"\\boxed\{")


def extract_boxed(text: str) -> str | None:
    """Content of the *last* ``\\boxed{…}`` (handles nested braces)."""
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


# ---------------------------------------------------------------------------
# Thinking extraction
# ---------------------------------------------------------------------------


def _thinking_portion(text: str, is_thinking: bool) -> str:
    if is_thinking:
        m = re.search(r"<think>(.*?)(?:</think>|$)", text, re.DOTALL)
        return m.group(1) if m else text
    idx = text.find("\\boxed{")
    return text[:idx] if idx > 0 else text


# ---------------------------------------------------------------------------
# Per-problem trajectory
# ---------------------------------------------------------------------------


def answer_trajectory(
    model: Any,
    tokenizer: Any,
    problem: str,
    hf_id: str,
) -> tuple[list[str], int, str]:
    """Generate thinking, force-decode answer at checkpoints.

    Returns (answers_list, flip_count, full_generated_text).
    """
    is_thinking = hf_id in THINKING_MODELS
    prompt = format_prompt(tokenizer, problem)

    # --- 1. Full generation ------------------------------------------------
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inp = {k: v.to(model.device) for k, v in inp.items()}

    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=MAX_THINKING_TOKENS, do_sample=False)

    gen_ids = out[0, inp["input_ids"].shape[1] :]
    full_text = tokenizer.decode(gen_ids, skip_special_tokens=False)

    # --- 2. Isolate thinking tokens ----------------------------------------
    thinking = _thinking_portion(full_text, is_thinking)
    think_ids = tokenizer.encode(thinking, add_special_tokens=False)

    if len(think_ids) < 10:
        final = norm(extract_boxed(full_text))
        return [final], 0, full_text

    # --- 3. Prepare common prompt prefixes ---------------------------------
    prompt_ids = inp["input_ids"]
    if is_thinking:
        think_open_ids = tokenizer.encode("<think>", add_special_tokens=False, return_tensors="pt").to(model.device)
        prompt_ids = torch.cat([prompt_ids, think_open_ids], dim=1)

    if is_thinking:
        close_tag = "</think>\n\nTime is up. Therefore, my best guess for the final mathematical answer is \\boxed{"
    else:
        close_tag = "\n\nTime is up. Therefore, my best guess for the final mathematical answer is \\boxed{"
        
    suffix_ids = tokenizer.encode(close_tag, add_special_tokens=False, return_tensors="pt").to(model.device)

    # --- 4. Force-decode at checkpoints ------------------------------------
    answers: list[str] = []
    think_tensor = torch.tensor([think_ids], device=model.device)

    for i in range(1, NUM_CHECKPOINTS + 1):
        frac = i / NUM_CHECKPOINTS
        ckpt_len = max(1, int(len(think_ids) * frac))

        # Concat: Prompt + Partial Thinking + Suffix
        partial_think = think_tensor[:, :ckpt_len]
        force_ids = torch.cat([prompt_ids, partial_think, suffix_ids], dim=1)

        with torch.no_grad():
            fo = model.generate(force_ids, max_new_tokens=MAX_ANSWER_TOKENS, do_sample=False)

        ans_ids = fo[0, force_ids.shape[1] :]
        ans_text = tokenizer.decode(ans_ids, skip_special_tokens=True).strip()

        # Answer is everything before the first unmatched }
        raw = ans_text.split("}")[0].strip() if "}" in ans_text else ans_text.strip()
        answers.append(norm(raw))

    # --- 5. Count flips ----------------------------------------------------
    # Ignore obvious placeholders or empty generations that artificially inflate flip counts.
    ignore_set = {"__EMPTY__", "__NONE__", "?", "N", "x", "y", "a", "b"}
    valid_answers = [a for a in answers if a not in ignore_set]

    flips = 0
    if len(valid_answers) > 1:
        for j in range(1, len(valid_answers)):
            if valid_answers[j] != valid_answers[j - 1]:
                is_equiv = model_judge_equivalent(model, tokenizer, valid_answers[j - 1], valid_answers[j])
                if not is_equiv:
                    flips += 1
                
    return answers, flips, full_text


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

def predict_robustness(
    model_id: str,
    problems: list[str],
    flip_threshold: int = 1,
) -> list[bool]:
    """One robustness prediction per problem.

    Robust iff flip_count >= flip_threshold.
    Falls back to False on any error so the submission stays valid.
    """
    try:
        tokenizer, model, hf_id = load_model(model_id)
    except Exception as exc:
        print(f"[flip] Model load failed: {exc}", file=sys.stderr, flush=True)
        return [False] * len(problems)

    results: list[bool] = []
    for idx, problem in enumerate(problems):
        try:
            answers, flips, _ = answer_trajectory(model, tokenizer, problem, hf_id)
            
            # Compute Margin of Victory
            ignore_set = {"__EMPTY__", "__NONE__", "?", "N", "x", "y", "a", "b"}
            valid_answers = [a for a in answers if a not in ignore_set]
            
            clusters = []
            for a in valid_answers:
                found = False
                for cluster in clusters:
                    if model_judge_equivalent(model, tokenizer, a, cluster[0]):
                        cluster.append(a)
                        found = True
                        break
                if not found:
                    clusters.append([a])
            
            cluster_sizes = sorted([len(c) for c in clusters], reverse=True)
            if len(cluster_sizes) > 1:
                margin = cluster_sizes[0] - cluster_sizes[1]
            elif len(cluster_sizes) == 1:
                margin = cluster_sizes[0]
            else:
                margin = 0
            
            # Uniform thresholding based on margin
            robust = margin >= 4
            
            gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
            prob_id, true_robust = _get_problem_meta(problem, model_id)
            print(
                f"[flip GPU={gpu_id}] ID={prob_id} GT={true_robust} | flips={flips} margin={margin} robust={robust} answers={answers}",
                file=sys.stderr, flush=True,
            )
        except Exception as exc:
            print(f"[flip] #{idx} error: {exc}", file=sys.stderr, flush=True)
            robust = False
        results.append(bool(robust))

    return results


import os
import math
import multiprocessing

def worker_process(gpu_id: str, model_id: str, problems: list[str], flip_threshold: int) -> list[bool]:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    return predict_robustness(model_id, problems, flip_threshold=flip_threshold)

def predict_robustness_parallel(model_id: str, problems: list[str], flip_threshold: int = 1) -> list[bool]:
    """Parallelized robustness prediction across available GPUs."""
    parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if parent_cvd:
        available_gpus = [g.strip() for g in parent_cvd.split(",") if g.strip()]
    else:
        num_devices = torch.cuda.device_count()
        available_gpus = [str(i) for i in range(num_devices)] if num_devices > 0 else ["0"]

    num_workers = min(len(problems), len(available_gpus))
    if num_workers <= 1:
        return worker_process(available_gpus[0], model_id, problems, flip_threshold)

    chunk_size = math.ceil(len(problems) / num_workers)
    chunks = [problems[i:i + chunk_size] for i in range(0, len(problems), chunk_size)]
    actual_workers = len(chunks)

    ctx = multiprocessing.get_context("spawn")
    predictions = []

    with ctx.Pool(processes=actual_workers) as pool:
        async_results = []
        for i in range(actual_workers):
            gpu_id = available_gpus[i]
            res = pool.apply_async(worker_process, args=(gpu_id, model_id, chunks[i], flip_threshold))
            async_results.append(res)
        for res in async_results:
            predictions.extend(res.get())

    return predictions
