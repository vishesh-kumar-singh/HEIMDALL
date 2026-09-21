"""Submission-side robustness prediction using perturbation + model-as-judge."""

from __future__ import annotations

from collections.abc import Sequence
import os

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from uncertainty_profile.config import resolve_checkpoint_model_id
from uncertainty_profile.extraction import (
    load_model_and_tokenizer,
    release_model,
    format_problem,
)
from uncertainty_profile.math_compare import model_judge_equivalent


# ── Answer extraction ────────────────────────────────────────────────

def extract_answer(text: str) -> str | None:
    """Extract the last \\boxed{...} answer, ignoring <think> blocks."""
    think_idx = text.rfind("</think>")
    if think_idx != -1:
        text = text[think_idx + 8:]

    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return None

    brace_count = 0
    start_idx = idx + 7
    for i in range(start_idx, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            if brace_count == 0:
                return text[start_idx:i].strip()
            brace_count -= 1
    return text[start_idx:].strip()


def force_extract_answer(model, tokenizer, full_generated_text: str) -> str | None:
    """Force the model to produce a boxed answer if it didn't naturally."""
    if "</think>" not in full_generated_text:
        force_prompt = full_generated_text + "\n</think>\n\nTime is up. Therefore, my best guess for the final mathematical answer is \\boxed{"
    else:
        force_prompt = full_generated_text + "\n\nTime is up. Therefore, my best guess for the final mathematical answer is \\boxed{"

    force_ids = tokenizer(force_prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)

    with torch.no_grad():
        out = model.generate(force_ids, max_new_tokens=20, do_sample=False, use_cache=True)

    final_text = tokenizer.decode(out[0][force_ids.shape[1]:], skip_special_tokens=True)
    ans = final_text.split('}')[0].strip()
    return ans if ans else None


def get_answer(model, tokenizer, input_ids, output_ids, idx: int = 0) -> str | None:
    """Extract answer from generation output, with force-extraction fallback."""
    prompt_len = input_ids.shape[1]
    generated_text = tokenizer.decode(output_ids[idx][prompt_len:], skip_special_tokens=True)
    ans = extract_answer(generated_text)

    if ans is None or ans == "":
        full_text = tokenizer.decode(output_ids[idx], skip_special_tokens=True)
        return force_extract_answer(model, tokenizer, full_text)
    return ans


# ── Perturbation modes ───────────────────────────────────────────────

PERTURBATION_MODES = [
    "Rewrite the following math problem by changing the names of people, places, or objects, but DO NOT change any numbers or the underlying math. Only output the rewritten problem and nothing else.\n\nProblem: {problem}",
    "Add some useless, irrelevant background information or distracting details to the following math problem without changing the core mathematical question or the numbers. Only output the rewritten problem and nothing else.\n\nProblem: {problem}",
    "Rewrite the following math problem in a highly formal, academic tone, but keep all numbers and the mathematical structure identical. Only output the rewritten problem and nothing else.\n\nProblem: {problem}",
    "Rewrite the following math problem using the simplest language possible, like a children's story, but keep the exact same numbers and math. Only output the rewritten problem and nothing else.\n\nProblem: {problem}",
]


# ── Worker (runs on a single GPU) ────────────────────────────────────

def worker_process(
    gpu_id: str,
    checkpoint_model_id: str,
    is_offline: bool,
    problem_chunk: list[dict],
) -> list[bool]:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    model: PreTrainedModel | None = None
    tokenizer: PreTrainedTokenizerBase | None = None

    try:
        model, tokenizer = load_model_and_tokenizer(
            checkpoint_model_id,
            local_files_only=is_offline,
        )

        # Allow user to specify temperature through env var for testing (defaults to 0.7 as per code)
        pert_temp = float(os.environ.get("PERTURB_TEMP", "0.7"))
        
        predictions = []
        for idx, item in enumerate(problem_chunk):
            problem = item["problem"]
            prob_id = item["id"]
            gt = item["gt"]
            
            print(f"[GPU {gpu_id}] Problem {idx + 1}/{len(problem_chunk)} | ID: {prob_id} | GT: {gt}", flush=True)

            # ── Step 1: Generate 4 perturbations ──────────────────────
            # We must set padding side to left for batched generation!
            tokenizer.padding_side = "left"
            
            rewrite_prompts = [
                format_problem(tokenizer, mode.format(problem=problem), role="user")
                for mode in PERTURBATION_MODES
            ]
            rewrite_inputs = tokenizer(
                rewrite_prompts, return_tensors="pt", padding=True
            ).to(model.device)

            with torch.no_grad():
                rewrite_outputs = model.generate(
                    **rewrite_inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.7,
                    use_cache=True,
                )

            perturbed_problems = []
            for j in range(4):
                text = tokenizer.decode(
                    rewrite_outputs[j][rewrite_inputs.input_ids.shape[1]:],
                    skip_special_tokens=True,
                ).strip()
                
                # Failsafe: if the model failed to rewrite and just output garbage, fallback to the original problem
                if len(text) < 10 or "Rewritten" in text or "Problem:" in text[:20]:
                    text = problem
                    
                perturbed_problems.append(text)

            del rewrite_inputs, rewrite_outputs
            torch.cuda.empty_cache()

            # ── Step 2: Solve original problem (greedy, temp=0.0) ────
            orig_formatted = format_problem(
                tokenizer,
                problem + "\n\nPlease put your final mathematical answer inside \\boxed{}.",
                role="user",
            )
            orig_inputs = tokenizer(orig_formatted, return_tensors="pt").to(model.device)

            with torch.no_grad():
                orig_outputs = model.generate(
                    **orig_inputs, max_new_tokens=4096, do_sample=False, use_cache=True,
                )
            base_answer = get_answer(model, tokenizer, orig_inputs.input_ids, orig_outputs)
            del orig_inputs, orig_outputs
            torch.cuda.empty_cache()

            worker_prefix = f"[{os.environ.get('CUDA_VISIBLE_DEVICES', 'GPU')}]"
            print(f"{worker_prefix}  Greedy Base answer: {base_answer}", flush=True)

            if base_answer is None:
                # Can't even solve the original → not robust
                print(f"{worker_prefix}  → Could not extract base answer! Defaulting to False.", flush=True)
                predictions.append(False)
                continue

            # ── Step 3: Solve base + 4 perturbed in a single batched pass ──
            batched_prompts = [orig_formatted]
            for pert_problem in perturbed_problems:
                batched_prompts.append(format_problem(
                    tokenizer,
                    pert_problem + "\n\nPlease put your final mathematical answer inside \\boxed{}.",
                    role="user",
                ))
            
            tokenizer.padding_side = "left"
            batched_inputs = tokenizer(batched_prompts, return_tensors="pt", padding=True).to(model.device)

            with torch.no_grad():
                batched_outputs = model.generate(
                    **batched_inputs, 
                    max_new_tokens=4096, 
                    do_sample=(pert_temp > 0.0), 
                    temperature=pert_temp if pert_temp > 0.0 else None, 
                    use_cache=True,
                )
            
            match_count = 0
            for j in range(5):
                ans = get_answer(model, tokenizer, batched_inputs.input_ids, batched_outputs, j)
                if j == 0:
                    name = "Base2"
                else:
                    name = f"Pert {j}"
                
                # Model-as-judge comparison against the GREEDY base answer
                is_match = model_judge_equivalent(model, tokenizer, base_answer, ans)
                print(f"{worker_prefix}  {name} answer: {ans} | Match: {is_match}", flush=True)

                if is_match:
                    match_count += 1
            
            del batched_inputs, batched_outputs
            torch.cuda.empty_cache()

            # ── Step 4: Decision — ≥4 of 5 must match base ───────────
            predicted_robust = match_count >= 3
            print(f"{worker_prefix}  → {match_count}/5 match base -> {'ROBUST (True)' if predicted_robust else 'NOT ROBUST (False)'} | GT: {gt}", flush=True)
            predictions.append(bool(predicted_robust))

        return predictions
    finally:
        if model is not None or tokenizer is not None:
            model = None
            tokenizer = None
            release_model()


# ── Entry point ──────────────────────────────────────────────────────

def predict_robustness(model_id: str, problem_texts: Sequence[str]) -> list[bool]:
    texts = list(problem_texts)
    if not texts:
        return []
    checkpoint_model_id = resolve_checkpoint_model_id(model_id)

    import os
    import math
    import multiprocessing
    import json
    
    # Attempt to load ground truth and IDs from the reference directory for local testing
    problem_dicts = []
    cases_path = "/home/vishesh/Robustness_test/data/train-sample/input/cases.jsonl"
    ref_path = "/home/vishesh/Robustness_test/data/train-sample/reference/labels.jsonl"
    
    id_to_gt = {}
    if os.path.exists(ref_path):
        with open(ref_path, "r") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    id_to_gt[obj["id"]] = obj.get("is_robust", "unknown")
                    
    prob_to_id = {}
    if os.path.exists(cases_path):
        with open(cases_path, "r") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    prob_to_id[obj["problem"]] = obj["id"]
                    
    for text in texts:
        pid = prob_to_id.get(text, "unknown")
        gt = id_to_gt.get(pid, "unknown")
        problem_dicts.append({
            "problem": text,
            "id": pid,
            "gt": gt
        })

    is_offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    if parent_cvd:
        available_gpus = [g.strip() for g in parent_cvd.split(",") if g.strip()]
    else:
        # Keep only GPU 2 and 3 as explicitly requested
        available_gpus = ["2", "3"]

    num_workers = min(len(problem_dicts), len(available_gpus))
    if num_workers <= 1:
        return worker_process(available_gpus[0], checkpoint_model_id, is_offline, problem_dicts)

    chunk_size = math.ceil(len(problem_dicts) / num_workers)
    chunks = [problem_dicts[i:i + chunk_size] for i in range(0, len(problem_dicts), chunk_size)]
    actual_workers = len(chunks)

    ctx = multiprocessing.get_context("spawn")
    predictions = []

    with ctx.Pool(processes=actual_workers) as pool:
        async_results = []
        for i in range(actual_workers):
            gpu_id = available_gpus[i]
            res = pool.apply_async(
                worker_process, (gpu_id, checkpoint_model_id, is_offline, chunks[i])
            )
            async_results.append(res)

        for res in async_results:
            predictions.extend(res.get())

    return predictions
