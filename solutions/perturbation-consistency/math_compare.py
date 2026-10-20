"""Conservative mathematical answer-equivalence helpers."""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Any

import torch


def normalize_answer(answer: str | None) -> str:
    """Normalize simple formatting differences without attempting symbolic algebra."""
    if answer is None:
        return ""
    normalized = answer.strip()
    for command in (
        r"\text{",
        r"\mathrm{",
        r"\mathbf{",
        r"\textbf{",
        r"\displaystyle",
    ):
        normalized = normalized.replace(command, "")
    while normalized.endswith("}") and normalized.count("}") > normalized.count("{"):
        normalized = normalized[:-1]
    normalized = normalized.replace(r"\,", "").replace(r"\;", "").replace(r"\ ", "")
    normalized = normalized.replace(" ", "").replace("−", "-")
    if normalized.endswith(".0"):
        normalized = normalized[:-2]
    return normalized


def try_numeric_equal(answer_a: str, answer_b: str) -> bool | None:
    """Compare ordinary numeric and simple fractional answers when possible."""
    try:
        return abs(float(answer_a) - float(answer_b)) < 1e-9
    except (ValueError, OverflowError):
        pass

    fraction_pattern = re.compile(r"\\frac\{([^}]+)\}\{([^}]+)\}")

    def as_fraction(value: str) -> Fraction | None:
        match = fraction_pattern.fullmatch(value)
        try:
            if match:
                return Fraction(int(match.group(1)), int(match.group(2)))
            return Fraction(value)
        except (ValueError, ZeroDivisionError):
            return None

    fraction_a = as_fraction(answer_a)
    fraction_b = as_fraction(answer_b)
    if fraction_a is not None and fraction_b is not None:
        return fraction_a == fraction_b
    return None


def answers_match(answer_a: str | None, answer_b: str | None) -> bool | None:
    """Return a definite quick comparison, or ``None`` when judging is needed."""
    if answer_a is None or answer_b is None:
        return None
    normalized_a = normalize_answer(answer_a)
    normalized_b = normalize_answer(answer_b)
    if not normalized_a or not normalized_b:
        return None
    if normalized_a == normalized_b:
        return True
    return try_numeric_equal(normalized_a, normalized_b)


def _input_device(model: Any) -> torch.device:
    embeddings = model.get_input_embeddings()
    return embeddings.weight.device if embeddings is not None else next(model.parameters()).device


def model_judge_equivalent(model: Any, tokenizer: Any, answer_a: str, answer_b: str) -> bool:
    """Use the already loaded model only when a quick comparison is inconclusive.

    An ambiguous judge output is deliberately treated as non-equivalence.
    """
    quick_result = answers_match(answer_a, answer_b)
    if quick_result is not None:
        return bool(quick_result)

    judge_prompt = (
        "Are these two mathematical answers exactly equivalent?\n"
        f"Answer A: {answer_a}\n"
        f"Answer B: {answer_b}\n"
        "Reply with ONLY Yes or No."
    )
    try:
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": judge_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except (AttributeError, TypeError, ValueError):
        formatted = judge_prompt

    inputs = tokenizer(formatted, return_tensors="pt")
    inputs = {name: value.to(_input_device(model)) for name, value in inputs.items()}
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(
        output_ids[0, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    ).replace("</think>", " ")
    labels = re.findall(r"\b(yes|no)\b", response.casefold())
    return bool(labels and labels[-1] == "yes")
