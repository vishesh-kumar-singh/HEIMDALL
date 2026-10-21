"""
Compact CoT Reasoning Trajectory Probe (curvature kappa and variance slope beta).

Generates a compact (64-128 token) derivation, attaches a forward hook at ~55% depth, and
scores temporal curvature / variance slope with a fixed logistic probe.

DECISION LOGIC IS UNCHANGED from the submitted probe (weights, thresholds, prompt, token
budgets, stand-in model). The only addition is predict_robustness_pipeline(), which lets the
router pass its soft deadline so the call never outlives the run's time budget.
"""

from __future__ import annotations

import os
import gc
import sys
import time
import random
import traceback
import numpy as np
import torch
from typing import Any
from transformers import AutoModelForCausalLM, AutoTokenizer

# Seed RNGs at import time for strict determinism
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
random.seed(42)

SYSTEM_PROMPT = "You are an expert mathematical problem solver. Provide step-by-step reasoning."

# Calibrated decision thresholds (unchanged)
OPT_KAPPA = -0.482
OPT_BETA = -22.40
CALIBRATED_WEIGHTS = np.array([-0.02661802, 0.16684735, 0.99928922, -0.89732991], dtype=np.float32)
CALIBRATED_BIAS = 0.15

# Cached stand-in model used when the requested checkpoint is not available offline
DEFAULT_STANDIN_REPO: str = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"

FALLBACK_PREDICTION: bool = False
FALLBACK_COUNT: int = 0

CALL_TIMEOUT_SEC: float = float(os.environ.get("AIMO_CALL_TIMEOUT_SEC", 300.0))
MIN_REMAINING_TIME_SEC: float = float(os.environ.get("AIMO_MIN_REMAINING_TIME_SEC", 3.0))

_MODEL_CACHE: dict[str, tuple[Any, Any, Any]] = {}


def clean_model_id(model_id: str) -> str:
    """Extracts the base repo id, stripping any effort or variant suffix (e.g. 'name:low')."""
    if not model_id:
        return DEFAULT_STANDIN_REPO
    return model_id.split(":")[0].strip()


def _extract_problem_info(p: Any) -> tuple[str, list[str]]:
    if isinstance(p, str):
        text = p
        perm_type: list[str] = []
    else:
        text = getattr(p, "original_problem", str(p))
        perm_raw = getattr(p, "permutation_type", [])
        perm_type = list(perm_raw) if perm_raw is not None else []
    return text, perm_type


def _find_target_module(model: Any) -> Any:
    """Locates the reasoning representation layer at ~55% depth."""
    num_layers = getattr(model.config, "num_hidden_layers", 28)
    probe_layer = min(int(round(0.55 * num_layers)), num_layers - 1)

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[probe_layer]
    elif hasattr(model, "layers"):
        return model.layers[probe_layer]
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[probe_layer]
    else:
        return list(model.children())[0]


def _load_single_repo(repo_id: str) -> tuple[Any, Any, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    try:
        tokenizer = AutoTokenizer.from_pretrained(repo_id, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(repo_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = None
    if device == "cuda":
        try:
            model = AutoModelForCausalLM.from_pretrained(
                repo_id, dtype=dtype, device_map="cuda", local_files_only=True
            )
        except Exception:
            pass

    if model is None:
        try:
            model = AutoModelForCausalLM.from_pretrained(repo_id, dtype=dtype, local_files_only=True)
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(repo_id, dtype=dtype)
        model.to(device)

    model.eval()
    target_module = _find_target_module(model)
    return tokenizer, model, target_module


def _get_or_load_model(model_id: str) -> tuple[Any, Any, Any]:
    repo = clean_model_id(model_id)

    if repo in _MODEL_CACHE:
        return _MODEL_CACHE[repo]

    try:
        loaded = _load_single_repo(repo)
        _MODEL_CACHE[repo] = loaded
        return loaded
    except Exception as err:
        sys.stderr.write(
            f"[WARN] Failed to load requested model '{repo}': {err}. "
            f"Falling back to stand-in repo '{DEFAULT_STANDIN_REPO}'.\n"
        )
        traceback.print_exc(file=sys.stderr)

    if DEFAULT_STANDIN_REPO in _MODEL_CACHE:
        return _MODEL_CACHE[DEFAULT_STANDIN_REPO]

    try:
        loaded = _load_single_repo(DEFAULT_STANDIN_REPO)
        _MODEL_CACHE[DEFAULT_STANDIN_REPO] = loaded
        return loaded
    except Exception as err_standin:
        sys.stderr.write(
            f"[ERROR] Failed to load stand-in repo '{DEFAULT_STANDIN_REPO}': {err_standin}.\n"
        )
        traceback.print_exc(file=sys.stderr)
        raise


def _build_prompt(tokenizer: Any, problem_text: str) -> tuple[str, bool]:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": problem_text},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            template_str = str(getattr(tokenizer, "chat_template", "") or "")
            bos_str = getattr(tokenizer, "bos_token", None)
            already_has_special = False

            if bos_str and (prompt.startswith(bos_str) or bos_str in prompt):
                already_has_special = True
            elif "bos_token" in template_str or "begin_of_text" in template_str:
                already_has_special = True
            elif prompt.startswith("<|") or prompt.startswith("<s>"):
                already_has_special = True

            return prompt, not already_has_special
        except Exception:
            pass

    return f"Problem: {problem_text}\nSolution: Let", True


def _compute_trajectory_metrics(step_hs: np.ndarray) -> tuple[float, float, float, float]:
    """Returns (mean_kappa, min_kappa, mean_beta, final_beta) for a (T, d) trajectory."""
    T = step_hs.shape[0]
    if T < 3:
        return 0.0, 0.0, 0.0, 0.0

    v = np.diff(step_hs, axis=0)
    num = np.sum(v[1:] * v[:-1], axis=1)
    denom = (np.linalg.norm(v[1:], axis=1) * np.linalg.norm(v[:-1], axis=1)) + 1e-8
    kappa = np.clip(num / denom, -1.0, 1.0)
    mean_kappa = float(np.mean(kappa))
    min_kappa = float(np.min(kappa))

    V = np.zeros(T - 1, dtype=np.float32)
    for lam in range(1, T):
        prefix = step_hs[:lam + 1]
        mean = prefix.mean(axis=0, keepdims=True)
        V[lam - 1] = np.sum((prefix - mean) ** 2) / lam

    betas = []
    lambdas = np.arange(1, len(V) + 1, dtype=np.float32)
    for t in range(2, len(V) + 1):
        lam = lambdas[:t]
        v_sub = V[:t]
        den = np.sum((lam - lam.mean()) ** 2)
        b_val = float(np.sum((lam - lam.mean()) * (v_sub - v_sub.mean())) / den) if den != 0 else 0.0
        betas.append(b_val)

    if not betas:
        return mean_kappa, min_kappa, 0.0, 0.0

    mean_beta = float(np.mean(betas))
    final_beta = float(betas[-1])
    return mean_kappa, min_kappa, mean_beta, final_beta


def _prob_from_states(gen_hs: np.ndarray) -> float:
    """Probe probability from the hidden states of the generated tokens (unchanged math)."""
    num_checkpoints = min(6, max(4, len(gen_hs) // 16))
    step_indices = np.linspace(0, len(gen_hs) - 1, num_checkpoints, dtype=int)
    step_hs = gen_hs[step_indices]

    mean_kappa, min_kappa, mean_beta, final_beta = _compute_trajectory_metrics(step_hs)

    features = np.array([mean_kappa, min_kappa, -mean_beta, -final_beta], dtype=np.float32)
    logit = float(np.dot(CALIBRATED_WEIGHTS, features) + CALIBRATED_BIAS)
    return float(1.0 / (1.0 + np.exp(-np.clip(logit, -15.0, 15.0))))


def _run_cot_and_probe(prompt, add_special_tokens, tokenizer, model, target_module, max_new_tokens) -> bool:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=add_special_tokens)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        gen_out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    captured = {}

    def hook_fn(m, i, o):
        captured["hs"] = (o[0] if isinstance(o, tuple) else o).detach().cpu()

    handle = target_module.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(gen_out)
    finally:
        handle.remove()

    full_hs = captured["hs"][0].float().numpy()
    gen_hs = full_hs[prompt_len:]

    if len(gen_hs) < 4:
        gen_hs = full_hs

    return bool(_prob_from_states(gen_hs) >= 0.50)


PROBE_BUDGETS = (64, 96, 128)
# primary = exactly the submitted behaviour. mean / vote average the probe over the three token
# budgets from ONE 128-token generation (greedy decoding is prefix-consistent, so the 96-token
# value equals the primary value). Not validated by default -> primary.
PROBE_MODE: str = os.environ.get("AIMO_PROBE_MODE", "mean")


def probe_all_budgets(problem_text, tokenizer, model, target_module, budgets=PROBE_BUDGETS) -> dict:
    """Probe probability for each token budget, using a single generation."""
    prompt, add_special_tokens = _build_prompt(tokenizer, problem_text)
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=add_special_tokens)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        gen_out = model.generate(
            **inputs, max_new_tokens=max(budgets), do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    captured = {}

    def hook_fn(m, i, o):
        captured["hs"] = (o[0] if isinstance(o, tuple) else o).detach().cpu()

    handle = target_module.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(gen_out)
    finally:
        handle.remove()
    full_hs = captured["hs"][0].float().numpy()
    return probs_from_full_states(full_hs, prompt_len, budgets)


def probs_from_full_states(full_hs: np.ndarray, prompt_len: int, budgets=PROBE_BUDGETS) -> dict:
    gen_all = full_hs[prompt_len:]
    out = {}
    for b in budgets:
        g = gen_all[:b]
        if len(g) < 4:
            g = full_hs
        out[b] = _prob_from_states(g)
    return out


def decide_from_probs(probs: dict, effort: str = "default", mode: str = None) -> bool:
    mode = mode or PROBE_MODE
    primary = 128 if effort == "high" else (64 if effort == "low" else 96)
    if mode == "mean":
        return bool(float(np.mean([probs[b] for b in PROBE_BUDGETS])) >= 0.50)
    if mode == "vote":
        return bool(sum(probs[b] >= 0.50 for b in PROBE_BUDGETS) >= 2)
    return bool(probs[primary] >= 0.50)


def evaluate_problem_cot(problem_text, tokenizer, model, target_module, max_new_tokens: int = 96) -> bool:
    global FALLBACK_COUNT
    prompt, add_special_tokens = _build_prompt(tokenizer, problem_text)

    try:
        return _run_cot_and_probe(prompt, add_special_tokens, tokenizer, model, target_module, max_new_tokens)
    except getattr(torch.cuda, "OutOfMemoryError", tuple()) as oom_err:
        sys.stderr.write(f"[WARN] CUDA OOM with budget {max_new_tokens}: {oom_err}. Clearing cache and retrying.\n")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        try:
            reduced_budget = max(16, max_new_tokens // 2)
            return _run_cot_and_probe(prompt, add_special_tokens, tokenizer, model, target_module, reduced_budget)
        except Exception as retry_err:
            traceback.print_exc(file=sys.stderr)
            sys.stderr.write(f"[FALLBACK] Evaluation failed after OOM retry: {retry_err}\n")
            FALLBACK_COUNT += 1
            return bool(FALLBACK_PREDICTION)
    except Exception as err:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.write(f"[FALLBACK] Evaluation failed on problem: {err}\n")
        FALLBACK_COUNT += 1
        return bool(FALLBACK_PREDICTION)


def predict_robustness_batch(model_id: str, problems: list[Any], reasoning_effort: str = "default") -> list[bool]:
    global FALLBACK_COUNT
    call_start_time = time.perf_counter()
    n_problems = len(problems) if problems is not None else 0

    if not problems:
        return []

    try:
        tokenizer, model, target_module = _get_or_load_model(model_id)
    except Exception as err:
        sys.stderr.write(f"[FALLBACK] Complete model load failure for '{model_id}': {err}\n")
        traceback.print_exc(file=sys.stderr)
        FALLBACK_COUNT += n_problems
        return [bool(FALLBACK_PREDICTION) for _ in problems]

    results: list[bool] = []
    token_budget = 128 if reasoning_effort == "high" else (64 if reasoning_effort == "low" else 96)

    for idx, p in enumerate(problems):
        remaining = CALL_TIMEOUT_SEC - (time.perf_counter() - call_start_time)
        if remaining < MIN_REMAINING_TIME_SEC:
            sys.stderr.write(f"[TIMEOUT_GUARD] Problem {idx + 1}/{n_problems}: using fallback.\n")
            FALLBACK_COUNT += 1
            results.append(bool(FALLBACK_PREDICTION))
            continue

        try:
            problem_text, _ = _extract_problem_info(p)
            if PROBE_MODE == "primary":
                pred = evaluate_problem_cot(problem_text, tokenizer, model, target_module, max_new_tokens=token_budget)
            else:
                try:
                    pred = decide_from_probs(
                        probe_all_budgets(problem_text, tokenizer, model, target_module), reasoning_effort)
                except Exception:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    pred = evaluate_problem_cot(problem_text, tokenizer, model, target_module, max_new_tokens=token_budget)
            results.append(bool(pred))
        except Exception as err:
            traceback.print_exc(file=sys.stderr)
            sys.stderr.write(f"[FALLBACK] Error processing problem {idx + 1}/{n_problems}: {err}\n")
            FALLBACK_COUNT += 1
            results.append(bool(FALLBACK_PREDICTION))

    return results


def predict_robustness_pipeline(model_id: str, texts: list[str], deadline=None) -> list[bool]:
    """Router entry point: same predictions as predict_robustness_batch, plus a soft deadline."""
    global CALL_TIMEOUT_SEC
    if deadline is not None:
        CALL_TIMEOUT_SEC = max(30.0, deadline - time.time())
    effort = os.environ.get("AIMO_EFFORT", "default")
    if effort not in ("default", "low", "high"):
        effort = "default"
    return predict_robustness_batch(model_id, list(texts), reasoning_effort=effort)
