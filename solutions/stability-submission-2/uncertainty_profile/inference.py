"""Submission-side uncertainty feature extraction and heuristic inference."""

from __future__ import annotations

from collections.abc import Sequence

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from uncertainty_profile.config import GenerationConfidenceConfig, resolve_checkpoint_model_id
from uncertainty_profile.extraction import (
    extract_generation_features,
    load_model_and_tokenizer,
    release_model,
)
import torch

# Threshold for robustness (lower entropy = more certain = robust)
ENTROPY_THRESHOLD = 0.5

# Helper functions from evaluate_stability.py
def extract_answer(text):
    # Chop off at </think> to avoid extracting intermediate thoughts
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

def force_extract_answer(model, tokenizer, full_generated_text):
    if "</think>" not in full_generated_text:
        force_prompt = full_generated_text + "\n</think>\n\nTime is up. Therefore, my best guess for the final mathematical answer is \\boxed{"
    else:
        force_prompt = full_generated_text + "\n\nTime is up. Therefore, my best guess for the final mathematical answer is \\boxed{"
        
    force_ids = tokenizer(force_prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    
    with torch.no_grad():
        out = model.generate(force_ids, max_new_tokens=20, do_sample=False)
        
    final_text = tokenizer.decode(out[0][force_ids.shape[1]:], skip_special_tokens=True)
    ans = final_text.split('}')[0].strip()
    return ans if ans else None

def get_answer_with_fallback(model, tokenizer, inputs, outputs, i=0):
    if hasattr(inputs, 'input_ids'):
        prompt_len = inputs.input_ids.shape[1]
    else:
        prompt_len = inputs.shape[1] if isinstance(inputs, torch.Tensor) else len(inputs[0])
        
    generated_text = tokenizer.decode(outputs[i][prompt_len:], skip_special_tokens=True)
    ans = extract_answer(generated_text)
    
    if ans is None or ans == "":
        full_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
        return force_extract_answer(model, tokenizer, full_text)
    return ans


def worker_process(gpu_id: str, checkpoint_model_id: str, is_offline: bool, problem_chunk: list[str]) -> list[bool]:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizerBase
    from collections import Counter
    from uncertainty_profile.extraction import load_model_and_tokenizer, release_model, format_problem

    model: PreTrainedModel | None = None
    tokenizer: PreTrainedTokenizerBase | None = None

    try:
        model, tokenizer = load_model_and_tokenizer(
            checkpoint_model_id,
            local_files_only=is_offline,
        )
        
        predictions = []
        for idx, problem in enumerate(problem_chunk):
            print(f"[GPU {gpu_id}] Processing problem {idx + 1}/{len(problem_chunk)}...", flush=True)
            # 1. Batched Semantic Perturbation Generation (4 samples with distinct modes)
            perturbation_prompts = [
                f"Rewrite the following math problem by changing the names of people, places, or objects, but DO NOT change any numbers or the underlying math. Only output the rewritten problem and nothing else.\n\nProblem: {problem}",
                f"Add some useless, irrelevant background information or distracting details to the following math problem without changing the core mathematical question or the numbers. Only output the rewritten problem and nothing else.\n\nProblem: {problem}",
                f"Rewrite the following math problem in a highly formal, academic tone, but keep all numbers and the mathematical structure identical. Only output the rewritten problem and nothing else.\n\nProblem: {problem}",
                f"Rewrite the following math problem using the simplest language possible, like a children's story, but keep the exact same numbers and math. Only output the rewritten problem and nothing else.\n\nProblem: {problem}"
            ]
            
            rewrite_texts = [format_problem(tokenizer, p, role="user") for p in perturbation_prompts]
            
            rewrite_inputs = tokenizer(rewrite_texts, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                outputs = model.generate(**rewrite_inputs, max_new_tokens=512, do_sample=True, temperature=0.7)
                
            perturbed_problems = []
            for j in range(4):
                pert_text = tokenizer.decode(outputs[j][rewrite_inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
                perturbed_problems.append(pert_text)
                
            # 2. Batched Greedy Solving (1 Original + 4 Perturbed)
            all_problems = [problem] + perturbed_problems
            formatted_problems = [format_problem(tokenizer, p + "\n\nPlease put your final mathematical answer inside \\boxed{}.", role="user") for p in all_problems]
            
            solve_inputs = tokenizer(formatted_problems, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                solve_outputs = model.generate(**solve_inputs, max_new_tokens=4096, do_sample=False)
                
            # 3. Extract Answers
            extracted_answers = []
            for j in range(5):
                ans = get_answer_with_fallback(model, tokenizer, solve_inputs, solve_outputs, j)
                extracted_answers.append(ans)
                
            # 4. Strict Consistency Check
            valid = [a for a in extracted_answers if a is not None]
            if len(valid) == 5 and len(set(valid)) == 1:
                predicted_robust = True
            else:
                predicted_robust = False
                
            predictions.append(bool(predicted_robust))
            
        return predictions
    finally:
        if model is not None or tokenizer is not None:
            model = None
            tokenizer = None
            release_model()


def predict_robustness(model_id: str, problem_texts: Sequence[str]) -> list[bool]:
    texts = list(problem_texts)
    if not texts:
        return []
    checkpoint_model_id = resolve_checkpoint_model_id(model_id)
    
    import os
    import math
    import multiprocessing
    import torch

    is_offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    
    if parent_cvd:
        available_gpus = [g.strip() for g in parent_cvd.split(",") if g.strip()]
    else:
        num_devices = torch.cuda.device_count()
        available_gpus = [str(i) for i in range(num_devices)] if num_devices > 0 else ["0"]
        
    num_workers = min(len(texts), len(available_gpus))
    if num_workers <= 1:
        # Fallback to sequential execution
        return worker_process(available_gpus[0], checkpoint_model_id, is_offline, texts)
        
    chunk_size = math.ceil(len(texts) / num_workers)
    chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
    actual_workers = len(chunks)
    
    # Spawn workers using the 'spawn' context to ensure a clean CUDA initialization
    ctx = multiprocessing.get_context("spawn")
    predictions = []
    
    with ctx.Pool(processes=actual_workers) as pool:
        async_results = []
        for i in range(actual_workers):
            gpu_id = available_gpus[i]
            res = pool.apply_async(worker_process, (gpu_id, checkpoint_model_id, is_offline, chunks[i]))
            async_results.append(res)
            
        for res in async_results:
            predictions.extend(res.get())
            
    return predictions
