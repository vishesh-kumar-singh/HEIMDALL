"""Model-as-judge math answer comparison (verbatim from the RF training pipeline).

Kept byte-for-byte equivalent in behaviour so the flip/margin features match the
features the random forest was trained on.
"""

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
        fa = float(a)
        fb = float(b)
        return abs(fa - fb) < 1e-9
    except (ValueError, OverflowError):
        pass

    frac_pat = re.compile(r"\\frac\{([^}]+)\}\{([^}]+)\}")

    def to_fraction(s):
        m = frac_pat.fullmatch(s)
        if m:
            try:
                return Fraction(int(m.group(1)), int(m.group(2)))
            except (ValueError, ZeroDivisionError):
                return None
        try:
            return Fraction(s)
        except (ValueError, ZeroDivisionError):
            return None

    fa = to_fraction(a)
    fb = to_fraction(b)
    if fa is not None and fb is not None:
        return fa == fb
    return None


def answers_match(a: str, b: str) -> Optional[bool]:
    if a is None or b is None:
        return None
    na = normalize_answer(a)
    nb = normalize_answer(b)
    if na == nb:
        return True
    numeric = try_numeric_equal(na, nb)
    if numeric is not None:
        return numeric
    return None


def model_judge_equivalent(model, tokenizer, answer_a: str, answer_b: str) -> bool:
    quick = answers_match(answer_a, answer_b)
    if quick is not None:
        return quick

    prompt = (
        f"Are these two mathematical answers exactly equivalent?\n"
        f"Answer A: {answer_a}\n"
        f"Answer B: {answer_b}\n"
        f"Reply with ONLY 'Yes' or 'No'."
    )
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        formatted = prompt

    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False, use_cache=True)

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip().lower()
    response_clean = response.replace("</think>", "").strip()
    if "yes" in response_clean[-20:]:
        return True
    elif "no" in response_clean[-20:]:
        return False
    else:
        return "yes" in response_clean
