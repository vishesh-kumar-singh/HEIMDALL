import os
import sys
import multiprocessing
import torch
import math
from typing import Any
import gc

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer

MATH_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."

def generate_responses(model_id: str, problems: list[str]) -> list[str]:
    print(f"[JUDGE] Loading TARGET model {model_id}...", file=sys.stderr, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=None,
        local_files_only=True,
    )
    model.to(device)
    model.eval()

    is_thinking = "DeepSeek" in model_id or "Olmo" in model_id
    
    responses = []
    with torch.no_grad():
        for problem in problems:
            messages = [{"role": "user", "content": problem + MATH_SUFFIX}]
            try:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                prompt = problem + MATH_SUFFIX
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            out = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            gen_ids = out[0, inputs.input_ids.shape[1]:]
            response = tokenizer.decode(gen_ids, skip_special_tokens=True)
            
            # Force decode if answer wasn't reached
            if "boxed{" not in response:
                if is_thinking:
                    close_tag = "</think>\n\nTime is up. Therefore, my best guess for the final mathematical answer is \\boxed{"
                else:
                    close_tag = "\n\nTime is up. Therefore, my best guess for the final mathematical answer is \\boxed{"
                    
                suffix_ids = tokenizer.encode(close_tag, add_special_tokens=False, return_tensors="pt").to(device)
                force_ids = torch.cat([out, suffix_ids], dim=1)
                
                fo = model.generate(
                    force_ids,
                    max_new_tokens=64,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                ans_ids = fo[0, inputs.input_ids.shape[1]:]
                response = tokenizer.decode(ans_ids, skip_special_tokens=True)
                
            responses.append(response)

    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[JUDGE] Unloaded TARGET model {model_id}.", file=sys.stderr, flush=True)
    
    return responses

def judge_responses(judge_model_id: str, problems: list[str], responses: list[str]) -> list[bool]:
    print(f"[JUDGE] Loading JUDGE model {judge_model_id}...", file=sys.stderr, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(judge_model_id, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        judge_model_id,
        torch_dtype=dtype,
        device_map=None,
        local_files_only=True,
    )
    model.to(device)
    model.eval()

    predictions = []
    with torch.no_grad():
        for problem, response in zip(problems, responses):
            judge_prompt = f"""You are an expert mathematical judge. Your task is to evaluate the following reasoning and answer provided by an AI model for a math problem.
You must rate the response on a scale of 1 to 10 based on the following rubric:
1-3: Completely wrong, hallucinates properties, or gets trapped in repetitive gibberish loops.
4-6: Uncertain, incomplete, explicitly states it does not know, or fails to reach a final conclusion.
7-8: Mostly correct logic but contains flawed calculations or leaps in reasoning.
9-10: Perfect, robust logic. Confident, mathematically sound, and reaches the definitively correct answer without hesitation.

Here is the problem:
{problem}

Here is the model's response:
{response}

Analyze the response carefully. Then, provide your final score out of 10. Your final line MUST contain ONLY the integer score (e.g., "Score: 9")."""

            messages = [{"role": "user", "content": judge_prompt}]
            try:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                prompt = judge_prompt
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            judge_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # Parse the score from the end of the text
            score = 0
            for line in reversed(judge_text.strip().split('\n')):
                line = line.strip().lower()
                if "score:" in line:
                    try:
                        # Extract the first integer after 'score:'
                        import re
                        match = re.search(r'score:\s*(\d+)', line)
                        if match:
                            score = int(match.group(1))
                        break
                    except:
                        pass
                else:
                    # Fallback: just try to find the last integer in the response
                    try:
                        import re
                        matches = re.findall(r'\b(10|[1-9])\b', judge_text)
                        if matches:
                            score = int(matches[-1])
                    except:
                        pass
            
            is_robust = score >= 9
            predictions.append(is_robust)
            print(f"[JUDGE RESULT] Score: {score} | Robust: {is_robust}", file=sys.stderr, flush=True)

    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[JUDGE] Unloaded JUDGE model {judge_model_id}.", file=sys.stderr, flush=True)
    
    return predictions


def worker_process(gpu_id: int, target_model_id: str, problems: list[str]) -> list[bool]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 1. Generate answers
    responses = generate_responses(target_model_id, problems)
    
    # 2. Pick judge model
    if "Qwen3.5" in target_model_id:
        judge_model_id = "Skywork/Skywork-OR1-Math-7B"
    else:
        judge_model_id = "Qwen/Qwen3.5-4B"
        
    # 3. Judge answers
    predictions = judge_responses(judge_model_id, problems, responses)
    return predictions


def evaluate_models(model_id: str, problems: list[str]) -> list[bool]:
    if not problems:
        return []

    available_gpus = []
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        available_gpus = [int(x.strip()) for x in cuda_visible.split(",") if x.strip()]
    else:
        num_gpus = torch.cuda.device_count()
        available_gpus = list(range(num_gpus))

    if not available_gpus:
        available_gpus = [0]
        
    num_workers = len(available_gpus)

    if num_workers == 1:
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
            res = pool.apply_async(
                worker_process, (gpu_id, model_id, chunks[i])
            )
            async_results.append(res)

        for res in async_results:
            predictions.extend(res.get())

    return predictions
