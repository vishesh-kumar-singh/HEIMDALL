"""Model-as-judge math answer comparison (same logic; parsing errors can no longer crash)."""
from __future__ import annotations

import re
from fractions import Fraction
from typing import Optional

import torch


def normalize_answer(answer: str) -> str:
    if answer is None:
        return ""
    s = answer.strip()
    for cmd in [r"\text{", r"\mathrm{", r"\mathbf{", r"\textbf{", r"\displaystyle"]:
        s = s.replace(cmd, "")
    while s.endswith("}") and s.count("}") > s.count("{"):
        s = s[:-1]
    s = s.replace(r"\,", "").replace(r"\;", "").replace(r"\ ", "")
    s = s.replace(" ", "")
    s = s.replace("\u2212", "-")
    if s.endswith(".0"):
        s = s[:-2]
    return s


def try_numeric_equal(a: str, b: str) -> Optional[bool]:
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (ValueError, OverflowError):
        pass

    frac_pat = re.compile(r"\\frac\{([^}]+)\}\{([^}]+)\}")

    def to_fraction(s):
        m = frac_pat.fullmatch(s)
        try:
            if m:
                return Fraction(int(m.group(1)), int(m.group(2)))
            return Fraction(s)
        except (ValueError, ZeroDivisionError):
            return None

    fa, fb = to_fraction(a), to_fraction(b)
    if fa is not None and fb is not None:
        return fa == fb
    return None


def answers_match(a, b) -> Optional[bool]:
    if a is None or b is None:
        return None
    na, nb = normalize_answer(a), normalize_answer(b)
    if na == nb:
        return True
    return try_numeric_equal(na, nb)


def model_judge_equivalent(model, tokenizer, answer_a, answer_b) -> bool:
    quick = answers_match(answer_a, answer_b)
    if quick is not None:
        return quick
    if answer_a is None or answer_b is None:
        return False

    prompt = (
        f"Are these two mathematical answers exactly equivalent?\n"
        f"Answer A: {answer_a}\n"
        f"Answer B: {answer_b}\n"
        f"Reply with ONLY 'Yes' or 'No'."
    )
    if hasattr(tokenizer, "apply_chat_template"):
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
    else:
        formatted = prompt

    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False, use_cache=True)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().lower()
    response_clean = response.replace("</think>", "").strip()
    if "yes" in response_clean[-20:]:
        return True
    if "no" in response_clean[-20:]:
        return False
    return "yes" in response_clean
