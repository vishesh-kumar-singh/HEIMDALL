"""Perturbation-stability detector: rewrite the problem 4 ways, solve original + rewrites,
and call the problem robust if enough answers match the greedy base answer.

Changes vs. the original: no hard-coded local paths or label lookup, deterministic sampling
(seeded per problem), per-problem failure isolation, soft deadline, truncated tail restored.
Prompts, token limits and the match threshold (MATCH_THRESHOLD) follow the newer dev version.
"""
from __future__ import annotations

import hashlib
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

from math_compare import model_judge_equivalent

DEEPSEEK = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
ALIASES = {"qwen3-8b:low": DEEPSEEK}
MATCH_THRESHOLD = 3          # out of 5 answers must match the greedy base (older version used 4)
SAMPLE_TEMPERATURE = 0.7
FALLBACK_ON_ERROR = False
FALLBACK_ON_DEADLINE = False


def resolve_checkpoint_model_id(model_id: str) -> str:
    return ALIASES.get(model_id, model_id)


def load_model_and_tokenizer(model_id: str):
    try:
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id, local_files_only=True, trust_remote_code=False)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError(f"tokenizer for {model_id!r} has no pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map=None,
                                                     local_files_only=True, trust_remote_code=False)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=None,
                                                     local_files_only=True, trust_remote_code=False)
    model.to(device)
    model.eval()
    return model, tokenizer


def format_problem(tokenizer, text: str, role: str = "user") -> str:
    try:
        formatted = tokenizer.apply_chat_template(
            [{"role": role, "content": text}], tokenize=False, add_generation_prompt=True
        )
    except (AttributeError, TypeError, ValueError):
        return text
    return formatted if isinstance(formatted, str) else text


def extract_answer(text: str):
    think_idx = text.rfind("</think>")
    if think_idx != -1:
        text = text[think_idx + 8:]
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return None
    brace_count = 0
    start_idx = idx + 7
    for i in range(start_idx, len(text)):
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            if brace_count == 0:
                return text[start_idx:i].strip()
            brace_count -= 1
    return text[start_idx:].strip()


def force_extract_answer(model, tokenizer, full_generated_text: str):
    tail = "\n\nTime is up. Therefore, my best guess for the final mathematical answer is \\boxed{"
    force_prompt = full_generated_text + (tail if "</think>" in full_generated_text else "\n</think>" + tail)
    force_ids = tokenizer(force_prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(force_ids, max_new_tokens=48, do_sample=False, use_cache=True)
    final_text = tokenizer.decode(out[0][force_ids.shape[1]:], skip_special_tokens=True)
    # brace-matched extraction (the old split("}")[0] cut \frac{1}{2} down to "\frac{1")
    depth = 0
    for i, ch in enumerate(final_text):
        if ch == "{":
            depth += 1
        elif ch == "}":
            if depth == 0:
                ans = final_text[:i].strip()
                return ans if ans else None
            depth -= 1
    ans = final_text.strip()
    return ans if ans else None


def get_answer(model, tokenizer, input_ids, output_ids, idx: int = 0):
    prompt_len = input_ids.shape[1]
    generated_text = tokenizer.decode(output_ids[idx][prompt_len:], skip_special_tokens=True)
    ans = extract_answer(generated_text)
    if ans is None or ans == "":
        full_text = tokenizer.decode(output_ids[idx], skip_special_tokens=True)
        return force_extract_answer(model, tokenizer, full_text)
    return ans


PERTURBATION_MODES = [
    "Rewrite the following math problem by changing the names of people, places, or objects, but DO NOT change any numbers or the underlying math. Only output the rewritten problem and nothing else.\n\nProblem: {problem}",
    "Add some useless, irrelevant background information or distracting details to the following math problem without changing the core mathematical question or the numbers. Only output the rewritten problem and nothing else.\n\nProblem: {problem}",
    "Rewrite the following math problem in a highly formal, academic tone, but keep all numbers and the mathematical structure identical. Only output the rewritten problem and nothing else.\n\nProblem: {problem}",
    "Rewrite the following math problem using the simplest language possible, like a children's story, but keep the exact same numbers and math. Only output the rewritten problem and nothing else.\n\nProblem: {problem}",
]
BOXED_HINT = "\n\nPlease put your final mathematical answer inside \\boxed{}."


def _seed_for(problem: str) -> int:
    return int(hashlib.md5(problem.encode("utf-8")).hexdigest()[:8], 16)


def solve_one(model, tokenizer, problem: str) -> bool:
    tokenizer.padding_side = "left"
    seed = _seed_for(problem)

    # 1. four rewrites (sampled, seeded)
    rewrite_prompts = [format_problem(tokenizer, m.format(problem=problem)) for m in PERTURBATION_MODES]
    rw_in = tokenizer(rewrite_prompts, return_tensors="pt", padding=True).to(model.device)
    torch.manual_seed(seed)
    with torch.no_grad():
        rw_out = model.generate(**rw_in, max_new_tokens=1024, do_sample=True,
                                temperature=SAMPLE_TEMPERATURE, use_cache=True)
    perturbed = []
    for j in range(4):
        text = tokenizer.decode(rw_out[j][rw_in.input_ids.shape[1]:], skip_special_tokens=True).strip()
        # Reasoning models "think" before rewriting: keep only the rewrite itself. If the
        # thinking never finished there is no rewrite, so fall back to the original problem.
        if "</think>" in text:
            text = text.rsplit("</think>", 1)[1].strip()
        elif "<think>" in text:
            text = problem
        if len(text) < 10 or "Rewritten" in text or "Problem:" in text[:20]:
            text = problem
        perturbed.append(text)
    del rw_in, rw_out
    torch.cuda.empty_cache()

    # 2. greedy base answer
    orig_formatted = format_problem(tokenizer, problem + BOXED_HINT)
    orig_in = tokenizer(orig_formatted, return_tensors="pt").to(model.device)
    with torch.no_grad():
        orig_out = model.generate(**orig_in, max_new_tokens=4096, do_sample=False, use_cache=True)
    base_answer = get_answer(model, tokenizer, orig_in.input_ids, orig_out)
    del orig_in, orig_out
    torch.cuda.empty_cache()
    if base_answer is None:
        return False

    # 3. base + 4 rewrites in one sampled batch (seeded)
    prompts = [orig_formatted] + [format_problem(tokenizer, p + BOXED_HINT) for p in perturbed]
    b_in = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    torch.manual_seed(seed + 1)
    with torch.no_grad():
        b_out = model.generate(**b_in, max_new_tokens=4096, do_sample=True,
                               temperature=SAMPLE_TEMPERATURE, use_cache=True)
    match_count = 0
    for j in range(5):
        ans = get_answer(model, tokenizer, b_in.input_ids, b_out, j)
        if model_judge_equivalent(model, tokenizer, base_answer, ans):
            match_count += 1
    del b_in, b_out
    torch.cuda.empty_cache()
    return match_count >= MATCH_THRESHOLD


def predict_robustness(model_id, problem_texts, deadline=None):
    texts = list(problem_texts)
    if not texts:
        return []
    ckpt = resolve_checkpoint_model_id(model_id)
    try:
        model, tokenizer = load_model_and_tokenizer(ckpt)
    except Exception as exc:
        print(f"[stability] model load failed: {exc}", file=sys.stderr, flush=True)
        return [FALLBACK_ON_ERROR] * len(texts)

    preds = []
    for idx, problem in enumerate(texts):
        if deadline is not None and time.time() > deadline:
            preds.append(FALLBACK_ON_DEADLINE)
            continue
        try:
            preds.append(bool(solve_one(model, tokenizer, problem)))
        except Exception as exc:
            print(f"[stability] #{idx} error: {exc!r}", file=sys.stderr, flush=True)
            preds.append(FALLBACK_ON_ERROR)
    return preds


def worker_process(gpu_id, model_id, chunk, deadline=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    return predict_robustness(model_id, chunk, deadline=deadline)


def predict_robustness_parallel(model_id, problem_texts, deadline=None):
    texts = list(problem_texts)
    if not texts:
        return []
    parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if parent_cvd:
        gpus = [g.strip() for g in parent_cvd.split(",") if g.strip()]
    else:
        n_dev = torch.cuda.device_count()
        gpus = [str(i) for i in range(n_dev)] if n_dev > 0 else ["0"]

    num_workers = min(len(texts), len(gpus))
    if num_workers <= 1:
        return worker_process(gpus[0], model_id, texts, deadline)
    chunk_size = math.ceil(len(texts) / num_workers)
    chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
    ctx = multiprocessing.get_context("spawn")
    out = []
    with ctx.Pool(processes=len(chunks)) as pool:
        pending = [pool.apply_async(worker_process, (gpus[i], model_id, chunks[i], deadline))
                   for i in range(len(chunks))]
        for i, res in enumerate(pending):
            try:
                out.extend(res.get())
            except Exception as exc:
                print(f"[stability] chunk {i} failed: {exc!r}", file=sys.stderr, flush=True)
                out.extend([FALLBACK_ON_ERROR] * len(chunks[i]))
    return out
