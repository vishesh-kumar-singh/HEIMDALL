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
        for problem in problem_chunk:
            # 1. Semantic Perturbation Prompt
            rewrite_prompt = f"Rewrite the following math problem by changing the names of people, places, or objects, but DO NOT change any numbers or the underlying math. Only output the rewritten problem and nothing else.\n\nProblem: {problem}"
            text = format_problem(tokenizer, rewrite_prompt, role="user")
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7)
            perturbed_problem = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            
            # 2. Original Solve (T=0)
            text = format_problem(tokenizer, problem + "\n\nPlease put your final mathematical answer inside \\boxed{}.", role="user")
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
            orig_ans = get_answer_with_fallback(model, tokenizer, inputs, outputs)
            
            # 3. Perturbed Solve (T=0)
            text = format_problem(tokenizer, perturbed_problem + "\n\nPlease put your final mathematical answer inside \\boxed{}.", role="user")
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
            pert_ans = get_answer_with_fallback(model, tokenizer, inputs, outputs)
            
            m1_match = (orig_ans == pert_ans) and (orig_ans is not None)
            
            # 4. Self Consistency (T=0.7, 5 passes in one batch)
            text = format_problem(tokenizer, problem + "\n\nPlease put your final mathematical answer inside \\boxed{}.", role="user")
            
            m2_answers = []
            batched_inputs = tokenizer([text]*5, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                outputs = model.generate(**batched_inputs, max_new_tokens=4096, do_sample=True, temperature=0.7)
            for j in range(5):
                ans = get_answer_with_fallback(model, tokenizer, batched_inputs, outputs, j)
                m2_answers.append(ans)
                
            valid_answers = [a for a in m2_answers if a is not None]
            if orig_ans is not None:
                valid_answers.append(orig_ans)
                
            mode_count = max(Counter(valid_answers).values()) if valid_answers else 0
            predicted_robust = m1_match and (mode_count >= 5)
            
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
