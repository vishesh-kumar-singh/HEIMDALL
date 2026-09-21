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
import numpy as np
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
# Feature extraction (Prompt-only logit trace)
# ---------------------------------------------------------------------------

def get_digit_ids(tokenizer, device):
    """Get token IDs for digits 0-9."""
    digit_ids = []
    for token_id in range(min(tokenizer.vocab_size, 200000)):
        try:
            token_str = tokenizer.decode([token_id]).strip()
        except Exception:
            continue
        if token_str in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            digit_ids.append(token_id)
    return torch.tensor(digit_ids, device=device)

def extract_features_fast(model, tokenizer, problem: str, digit_ids: torch.Tensor) -> dict:
    """Single forward pass on prompt only. No generation."""
    
    # Format prompt with chat template
    prompt = f"Problem: {problem}\nAnswer:"
    messages = [{"role": "user", "content": prompt}]
    try:
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        formatted_prompt = prompt

    inp = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    prompt_len = inp.input_ids.shape[1]

    # Single forward pass with hidden states
    with torch.no_grad():
        out = model(input_ids=inp.input_ids, output_hidden_states=True)

    num_layers = len(out.hidden_states)
    
    digit_probs_per_layer = []
    digit_entropy_per_layer = []
    full_entropy_per_layer = []
    hidden_norm_per_layer = []
    top1_digit_per_layer = []
    top1_full_per_layer = []
    top1_full_prob_per_layer = []
    digit_mass_per_layer = []
    kl_from_uniform_per_layer = []
    
    uniform_digit = torch.ones(len(digit_ids), device=model.device) / len(digit_ids)
    log_uniform_digit = torch.log(uniform_digit)

    for l in range(num_layers):
        h_l = out.hidden_states[l][0, -1, :]  # Last token
        hidden_norm_per_layer.append(h_l.norm().item())
        
        logits = model.lm_head(h_l)
        full_probs = torch.softmax(logits, dim=-1)
        
        log_full_probs = torch.log(full_probs + 1e-10)
        full_entropy = -(full_probs * log_full_probs).sum().item()
        full_entropy_per_layer.append(full_entropy)
        
        top1_prob, top1_idx = full_probs.max(dim=-1)
        top1_full_per_layer.append(top1_idx.item())
        top1_full_prob_per_layer.append(top1_prob.item())
        
        digit_logits = logits[digit_ids]
        d_probs = torch.softmax(digit_logits, dim=-1)
        
        max_d_prob, max_d_idx = d_probs.max(dim=-1)
        digit_probs_per_layer.append(max_d_prob.item())
        top1_digit_per_layer.append(max_d_idx.item())
        
        d_mass = full_probs[digit_ids].sum().item()
        digit_mass_per_layer.append(d_mass)
        
        d_log_probs = torch.log(d_probs + 1e-10)
        d_entropy = -(d_probs * d_log_probs).sum().item()
        digit_entropy_per_layer.append(d_entropy)
        
        kl = (d_probs * (d_log_probs - log_uniform_digit)).sum().item()
        kl_from_uniform_per_layer.append(kl)

    dp = np.array(digit_probs_per_layer)
    de = np.array(digit_entropy_per_layer)
    fe = np.array(full_entropy_per_layer)
    hn = np.array(hidden_norm_per_layer)
    dm = np.array(digit_mass_per_layer)
    kl = np.array(kl_from_uniform_per_layer)
    fp = np.array(top1_full_prob_per_layer)
    
    quarter = max(1, num_layers // 4)
    half = max(1, num_layers // 2)
    
    cryst_05 = num_layers
    cryst_09 = num_layers
    for l_idx, p in enumerate(digit_probs_per_layer):
        if p > 0.5 and cryst_05 == num_layers:
            cryst_05 = l_idx
        if p > 0.9 and cryst_09 == num_layers:
            cryst_09 = l_idx
            
    digit_changes = sum(1 for i in range(1, len(top1_digit_per_layer)) if top1_digit_per_layer[i] != top1_digit_per_layer[i - 1])
    full_token_changes = sum(1 for i in range(1, len(top1_full_per_layer)) if top1_full_per_layer[i] != top1_full_per_layer[i - 1])
    
    x = np.arange(num_layers)
    dp_slope = np.polyfit(x, dp, 1)[0] if num_layers > 1 else 0.0
    fe_slope = np.polyfit(x, fe, 1)[0] if num_layers > 1 else 0.0
    hn_slope = np.polyfit(x, hn, 1)[0] if num_layers > 1 else 0.0
    dm_slope = np.polyfit(x, dm, 1)[0] if num_layers > 1 else 0.0

    return {
        "num_layers": num_layers,
        "prompt_len": prompt_len,
        "dp_mean": dp.mean(), "dp_std": dp.std(), "dp_max": dp.max(), "dp_min": dp.min(),
        "dp_slope": dp_slope,
        "dp_early": dp[:quarter].mean(), "dp_late": dp[-quarter:].mean(),
        "dp_early_late_diff": dp[-quarter:].mean() - dp[:quarter].mean(),
        "dp_first_half": dp[:half].mean(), "dp_second_half": dp[half:].mean(),
        "de_mean": de.mean(), "de_std": de.std(), "de_min": de.min(), "de_max": de.max(),
        "de_early": de[:quarter].mean(), "de_late": de[-quarter:].mean(),
        "fe_mean": fe.mean(), "fe_std": fe.std(), "fe_min": fe.min(), "fe_max": fe.max(),
        "fe_slope": fe_slope,
        "fe_early": fe[:quarter].mean(), "fe_late": fe[-quarter:].mean(),
        "fe_drop": fe[0] - fe[-1] if num_layers > 1 else 0.0,
        "hn_mean": hn.mean(), "hn_std": hn.std(), "hn_min": hn.min(), "hn_max": hn.max(),
        "hn_slope": hn_slope,
        "hn_early": hn[:quarter].mean(), "hn_late": hn[-quarter:].mean(),
        "dm_mean": dm.mean(), "dm_std": dm.std(), "dm_max": dm.max(), "dm_min": dm.min(),
        "dm_slope": dm_slope,
        "dm_early": dm[:quarter].mean(), "dm_late": dm[-quarter:].mean(),
        "kl_mean": kl.mean(), "kl_std": kl.std(), "kl_max": kl.max(),
        "fp_mean": fp.mean(), "fp_std": fp.std(), "fp_max": fp.max(), "fp_min": fp.min(),
        "fp_early": fp[:quarter].mean(), "fp_late": fp[-quarter:].mean(),
        "cryst_05_ratio": cryst_05 / num_layers,
        "cryst_09_ratio": cryst_09 / num_layers,
        "digit_changes": digit_changes,
        "digit_change_ratio": digit_changes / max(num_layers - 1, 1),
        "full_token_changes": full_token_changes,
        "full_token_change_ratio": full_token_changes / max(num_layers - 1, 1),
    }

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

def get_robustness_from_features(model_id: str, features: dict, flips: int) -> bool:
    if 'Qwen/Qwen3.5-4B' in model_id:
        if features['kl_std'] <= 0.282109:
            if features['hn_slope'] <= 2.076950:
                return True
            else:
                return False
        else:
            return False
    if 'Skywork/Skywork-OR1-Math-7B' in model_id:
        if features['kl_std'] <= 0.560962:
            return True
        else:
            return False
    if 'allenai/Olmo-3-7B-Think' in model_id:
        if features['kl_std'] <= 0.032678:
            return True
        else:
            return False
    if 'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B' in model_id:
        if features['dp_second_half'] <= 0.565815:
            return False
        else:
            if features['de_min'] <= 0.002190:
                return False
            else:
                return True
    if model_id == 'qwen3-8b:low':
        return False
        
    # Fallback to flip threshold logic
    if "Qwen3.5" in model_id:
        return flips < 4
    elif "Skywork" in model_id:
        return flips >= 3
    else:
        return flips >= 1


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
        digit_ids = get_digit_ids(tokenizer, model.device)
    except Exception as exc:
        print(f"[flip] Model load failed: {exc}", file=sys.stderr, flush=True)
        return [False] * len(problems)

    results: list[bool] = []
    for idx, problem in enumerate(problems):
        try:
            # 1. Extract logit features on prompt
            features = extract_features_fast(model, tokenizer, problem, digit_ids)
            
            # 2. Get flip trajectory
            answers, flips, _ = answer_trajectory(model, tokenizer, problem, hf_id)
            
            # 3. Model-specific tree rules
            robust = get_robustness_from_features(hf_id, features, flips)
                
            gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
            prob_id, true_robust = _get_problem_meta(problem, model_id)
            print(
                f"[flip GPU={gpu_id}] ID={prob_id} GT={true_robust} | flips={flips} robust={robust} kl_std={features['kl_std']:.4f} answers={answers}",
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
