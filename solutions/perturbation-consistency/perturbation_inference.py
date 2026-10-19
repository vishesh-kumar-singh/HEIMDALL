"""Offline, answer-only perturbation-consistency inference.

This candidate measures whether a model's final boxed answer changes after two
fixed irrelevant statements are prepended. It is not an independent correctness
test for the original answer.
"""

from __future__ import annotations

import gc
import os
from collections.abc import Sequence
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from math_compare import model_judge_equivalent


MAX_PROMPT_TOKENS = 2048
MAX_NEW_TOKENS = 2048
MAX_ANSWER_EXTRACTION_TOKENS = 20
DISTRACTORS = (
    ("Separately, blue is a color.",True),
    # ("The library has 17 dryards.",True),
    ("Separately, the library has 17 windows.",True),
)
SOLUTION_SUFFIX = "\n\nPlease put your final mathematical answer inside \\boxed{}."


def _input_device(model: Any) -> torch.device:
    embeddings = model.get_input_embeddings()
    return embeddings.weight.device if embeddings is not None else next(model.parameters()).device


def _format_problem(tokenizer: Any, text: str) -> str:
    try:
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except (AttributeError, TypeError, ValueError):
        return text
    return formatted if isinstance(formatted, str) else text


def extract_answer(text: str) -> str | None:
    """Extract the last complete ``\\boxed{...}`` answer from model output."""
    think_end = text.rfind("</think>")
    if think_end != -1:
        text = text[think_end + len("</think>") :]

    start = text.rfind(r"\boxed{")
    if start == -1:
        return None
    answer_start = start + len(r"\boxed{")
    nested_braces = 0
    for index in range(answer_start, len(text)):
        character = text[index]
        if character == "{":
            nested_braces += 1
        elif character == "}":
            if nested_braces == 0:
                answer = text[answer_start:index].strip()
                return answer or None
            nested_braces -= 1
    return None


def _force_extract_answer(model: Any, tokenizer: Any, full_text: str) -> str | None:
    """Ask for a final boxed answer only when ordinary extraction failed."""
    if "</think>" not in full_text:
        continuation = "\n</think>\n\nTime is up. Final mathematical answer: \\boxed{"
    else:
        continuation = "\n\nTime is up. Final mathematical answer: \\boxed{"
    forced_prompt = full_text + continuation
    forced_inputs = tokenizer(forced_prompt, return_tensors="pt", add_special_tokens=False)
    forced_inputs = {
        name: value.to(_input_device(model)) for name, value in forced_inputs.items()
    }
    with torch.inference_mode():
        output_ids = model.generate(
            **forced_inputs,
            max_new_tokens=MAX_ANSWER_EXTRACTION_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    completion = tokenizer.decode(
        output_ids[0, forced_inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )
    answer = completion.split("}", 1)[0].strip()
    return answer or None


def generate_answer(model: Any, tokenizer: Any, problem: str) -> str | None:
    """Greedily generate one answer, using a short forced-answer fallback."""
    prompt = _format_problem(tokenizer, problem + SOLUTION_SUFFIX)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_PROMPT_TOKENS,
    )
    inputs = {name: value.to(_input_device(model)) for name, value in inputs.items()}
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_text = tokenizer.decode(
        output_ids[0, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )
    answer = extract_answer(generated_text)
    if answer is not None:
        return answer
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return _force_extract_answer(model, tokenizer, full_text)



def load_model_and_tokenizer(model_id: str) -> tuple[Any, Any]:
    """Load only the evaluator-supplied, locally cached checkpoint."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError(f"tokenizer for {model_id!r} has no pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            device_map="auto",
            local_files_only=True,
            trust_remote_code=False,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float32,
            local_files_only=True,
            trust_remote_code=False,
        )
        model.to("cpu")
    model.eval()
    return model, tokenizer

_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}


def get_model_and_tokenizer(model_id: str) -> tuple[Any, Any]:
    """Reuse a model already loaded in this process."""
    if model_id not in _MODEL_CACHE:
        _MODEL_CACHE[model_id] = load_model_and_tokenizer(model_id)
    return _MODEL_CACHE[model_id]

def release_model() -> None:
    """Release dereferenced resources after a completed invocation."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def predict_robustness(model_id: str, problems: Sequence[str]) -> list[bool]:
    """Return one conservative consistency prediction per input problem."""
    texts = list(problems)
    if not texts:
        return []

    model: Any | None = None
    tokenizer: Any | None = None
    try:
        try:
            model, tokenizer = get_model_and_tokenizer(model_id)
        except Exception:
            return [False] * len(texts)

        predictions: list[bool] = []
        for problem in texts:
            try:
                original_answer = generate_answer(model, tokenizer, problem)
                if original_answer is None:
                    predictions.append(False)
                    continue

                perturbation_matches: list[bool] = []
                for distractor, enabled in DISTRACTORS:
                    if not enabled:
                        continue
                    perturbed_answer = generate_answer(
                        model,
                        tokenizer,
                        f"{distractor}\n\n{problem}",
                    )
                    if perturbed_answer is None:
                        perturbation_matches.append(False)
                        continue
                    perturbation_matches.append(
                        bool(model_judge_equivalent(
                            model,
                            tokenizer,
                            original_answer,
                            perturbed_answer,
                        ))
                    )
                predictions.append(bool(all(perturbation_matches)))
            except Exception:
                predictions.append(False)
        return predictions
    finally:
        pass
