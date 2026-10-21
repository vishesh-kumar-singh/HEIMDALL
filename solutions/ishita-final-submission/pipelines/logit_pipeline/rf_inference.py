"""Random-forest robustness detector for DeepSeek-R1-0528-Qwen3-8B (alias qwen3-8b:low).

Feature extraction is the same as the pipeline the forest was trained on:
  * 8 trace features: answer flips across 10 forced-answer checkpoints, margin of victory
    between answer clusters, and keyword counts / length of the thinking trace;
  * 55 logit-lens features from one forward pass over the prompt.
Hardening only: soft deadline with a per-problem time estimate (so the router's hard cap is
never hit mid-problem), per-problem and per-chunk failure isolation, two workers per GPU,
no label lookups.
"""
from __future__ import annotations

import math
import multiprocessing
import os
import pickle
import re
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
from rf_math_compare import model_judge_equivalent

MODEL_ALIASES = {"qwen3-8b:low": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"}
THINKING_MODELS = {"deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"}
NUM_CHECKPOINTS = 10
MAX_THINKING_TOKENS = 4096
MAX_ANSWER_TOKENS = 64
WORKERS_PER_GPU = 2
MATH_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
IGNORE_SET = {"__EMPTY__", "__NONE__", "?", "N", "x", "y", "a", "b"}
ARTIFACT = os.path.join(_HERE, "rf_artifacts", "rf_artifact.pkl")

_rf = None
_model_cache: dict = {}


def get_rf():
    global _rf
    if _rf is None:
        with open(ARTIFACT, "rb") as f:
            _rf = pickle.load(f)
    return _rf


def prior_fallback() -> bool:
    """Majority class of the data the forest was trained on (used when a problem can't be run)."""
    try:
        import numpy as np
        rf = get_rf()
        pri = np.mean([e.tree_.value[0][0] / e.tree_.value[0][0].sum() for e in rf.estimators_], axis=0)
        classes = list(rf.classes_)
        return bool(classes[int(np.argmax(pri))])
    except Exception:  # noqa: BLE001
        return False


def resolve_model_id(model_id: str) -> str:
    return MODEL_ALIASES.get(model_id, model_id)


def load_model(model_id: str):
    hf_id = resolve_model_id(model_id)
    if hf_id in _model_cache:
        return _model_cache[hf_id]
    tokenizer = AutoTokenizer.from_pretrained(hf_id, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=dtype, device_map=None, local_files_only=True)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=dtype, device_map=None, local_files_only=True)
    model.to(device)
    model.eval()
    _model_cache[hf_id] = (tokenizer, model, hf_id)
    return tokenizer, model, hf_id


# --------------------------------------------------------------------------- trace helpers
def format_prompt(tokenizer, problem: str) -> str:
    messages = [{"role": "user", "content": problem + MATH_SUFFIX}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:  # noqa: BLE001
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


def thinking_portion(text: str, is_thinking: bool) -> str:
    if is_thinking:
        m = re.search(r"<think>(.*?)(?:</think>|$)", text, re.DOTALL)
        return m.group(1) if m else text
    idx = text.find("\\boxed{")
    return text[:idx] if idx > 0 else text


def count_flips(answers, same) -> int:
    valid = [a for a in answers if a not in IGNORE_SET]
    flips = 0
    for j in range(1, len(valid)):
        if valid[j] != valid[j - 1] and not same(valid[j - 1], valid[j]):
            flips += 1
    return flips


def margin_of_victory(answers, same) -> int:
    valid = [a for a in answers if a not in IGNORE_SET]
    clusters = []
    for a in valid:
        for c in clusters:
            if same(a, c[0]):
                c.append(a)
                break
        else:
            clusters.append([a])
    sizes = sorted((len(c) for c in clusters), reverse=True)
    if len(sizes) > 1:
        return sizes[0] - sizes[1]
    return sizes[0] if sizes else 0


def answer_trajectory(model, tokenizer, problem: str, hf_id: str):
    is_thinking = hf_id in THINKING_MODELS
    prompt = format_prompt(tokenizer, problem)
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inp = {k: v.to(model.device) for k, v in inp.items()}

    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=MAX_THINKING_TOKENS, do_sample=False)
    gen_ids = out[0, inp["input_ids"].shape[1]:]
    full_text = tokenizer.decode(gen_ids, skip_special_tokens=False)

    thinking = thinking_portion(full_text, is_thinking)
    think_ids = tokenizer.encode(thinking, add_special_tokens=False)
    if len(think_ids) < 10:
        return [norm(extract_boxed(full_text))], 0, full_text

    prompt_ids = inp["input_ids"]
    if is_thinking:
        think_open = tokenizer.encode("<think>", add_special_tokens=False, return_tensors="pt").to(model.device)
        prompt_ids = torch.cat([prompt_ids, think_open], dim=1)
        close_tag = "</think>\n\nTime is up. Therefore, my best guess for the final mathematical answer is \\boxed{"
    else:
        close_tag = "\n\nTime is up. Therefore, my best guess for the final mathematical answer is \\boxed{"
    suffix_ids = tokenizer.encode(close_tag, add_special_tokens=False, return_tensors="pt").to(model.device)

    answers = []
    think_tensor = torch.tensor([think_ids], device=model.device)
    for i in range(1, NUM_CHECKPOINTS + 1):
        ckpt_len = max(1, int(len(think_ids) * i / NUM_CHECKPOINTS))
        force_ids = torch.cat([prompt_ids, think_tensor[:, :ckpt_len], suffix_ids], dim=1)
        with torch.no_grad():
            fo = model.generate(force_ids, max_new_tokens=MAX_ANSWER_TOKENS, do_sample=False)
        ans_text = tokenizer.decode(fo[0, force_ids.shape[1]:], skip_special_tokens=True).strip()
        raw = ans_text.split("}")[0].strip() if "}" in ans_text else ans_text.strip()
        answers.append(norm(raw))

    same = lambda x, y: model_judge_equivalent(model, tokenizer, x, y)  # noqa: E731
    return answers, count_flips(answers, same), full_text


def build_features(model, tokenizer, hf_id, problem, digit_ids) -> dict:
    logit_feats = extract_features_fast(model, tokenizer, problem, digit_ids)
    answers, flips, full_text = answer_trajectory(model, tokenizer, problem, hf_id)
    same = lambda x, y: model_judge_equivalent(model, tokenizer, x, y)  # noqa: E731
    margin = margin_of_victory(answers, same)

    low = thinking_portion(full_text, hf_id in THINKING_MODELS).lower()
    features = {
        "flips": flips,
        "margin": margin,
        "wait": low.count("wait"),
        "oops": low.count("oops"),
        "recheck": low.count("recheck") + low.count("let me check") + low.count("recalculat"),
        "alt": low.count("alternatively"),
        "act": low.count("actually"),
        "len": len(tokenizer.encode(thinking_portion(full_text, hf_id in THINKING_MODELS), add_special_tokens=False)),
    }
    features.update(logit_feats)
    return features


def rf_predict(features: dict) -> bool:
    import pandas as pd
    rf = get_rf()
    names = list(rf.feature_names_in_)
    row = {n: float(features.get(n, 0.0)) for n in names}
    X = pd.DataFrame([row], columns=names)
    return bool(rf.predict(X)[0])


# --------------------------------------------------------------------------- driver
def predict_robustness(model_id, problems, flip_threshold: int = 1, deadline=None):
    fallback = prior_fallback()
    try:
        tokenizer, model, hf_id = load_model(model_id)
    except Exception as exc:  # noqa: BLE001
        print(f"[rf] model load failed: {exc}", file=sys.stderr, flush=True)
        return [fallback] * len(problems)

    digit_ids = None
    results = []
    max_dur = 0.0
    for idx, problem in enumerate(problems):
        # Don't start a problem we expect to overrun the deadline (router hard cap kills everything).
        if deadline is not None and time.time() + 1.15 * max_dur > deadline:
            results.append(fallback)
            continue
        t0 = time.time()
        try:
            if digit_ids is None:
                digit_ids = get_digit_ids(tokenizer, model.device)
            feats = build_features(model, tokenizer, hf_id, problem, digit_ids)
            robust = rf_predict(feats)
        except Exception as exc:  # noqa: BLE001
            print(f"[rf] #{idx} error: {exc}", file=sys.stderr, flush=True)
            robust = fallback
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
        max_dur = max(max_dur, time.time() - t0)
        results.append(bool(robust))
    return results


def worker_process(gpu_id, model_id, problems, flip_threshold, deadline=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    return predict_robustness(model_id, problems, flip_threshold=flip_threshold, deadline=deadline)


def predict_robustness_parallel(model_id, problems, flip_threshold: int = 1, deadline=None):
    parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if parent_cvd:
        gpus = [g.strip() for g in parent_cvd.split(",") if g.strip()]
    else:
        n_dev = torch.cuda.device_count()
        gpus = [str(i) for i in range(n_dev)] if n_dev > 0 else ["0"]
    gpus = [g for g in gpus for _ in range(WORKERS_PER_GPU)]

    num_workers = min(len(problems), len(gpus))
    if num_workers <= 1:
        return worker_process(gpus[0], model_id, problems, flip_threshold, deadline)

    chunk_size = math.ceil(len(problems) / num_workers)
    chunks = [problems[i:i + chunk_size] for i in range(0, len(problems), chunk_size)]

    ctx = multiprocessing.get_context("spawn")
    predictions = []
    fb = prior_fallback()
    with ctx.Pool(processes=len(chunks)) as pool:
        pending = [pool.apply_async(worker_process, args=(gpus[i], model_id, chunks[i], flip_threshold, deadline))
                   for i in range(len(chunks))]
        for i, res in enumerate(pending):
            try:
                predictions.extend(res.get())
            except Exception as exc:  # noqa: BLE001
                print(f"[rf] chunk {i} failed: {exc!r}", file=sys.stderr, flush=True)
                predictions.extend([fb] * len(chunks[i]))
    return predictions
