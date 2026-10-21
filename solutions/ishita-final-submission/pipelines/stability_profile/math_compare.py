"""Model-as-judge math answer comparison.

Cheap canonicalisation + numeric/fraction comparison is tried first, so formatting-only
differences (\\dfrac vs \\frac, 1,000 vs 1000, 90^\\circ vs 90, x=5 vs 5, ...) are never
mistaken for different answers. The model judge is only used for genuinely ambiguous pairs.
"""
from __future__ import annotations

import re
from fractions import Fraction
from typing import Optional

import torch

_WRAP_CMDS = re.compile(r"\\(?:text|mathrm|mathbf|textbf|mathit|operatorname)\{([^{}]*)\}")
_THOUSANDS = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")
_LEADING_VAR = re.compile(r"^[A-Za-z][A-Za-z0-9_]*=(?!.*=)")
_FRAC = re.compile(r"^(-?)\\frac\{(-?\d+)\}\{(-?\d+)\}$")


def normalize_answer(answer: Optional[str]) -> str:
    if answer is None:
        return ""
    s = answer.strip()
    for l, r in (("$$", "$$"), ("$", "$"), ("\\(", "\\)"), ("\\[", "\\]")):
        if len(s) >= len(l) + len(r) and s.startswith(l) and s.endswith(r):
            s = s[len(l): -len(r)].strip()
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    for cmd in ("\\left", "\\right", "\\displaystyle", "\\!"):
        s = s.replace(cmd, "")
    s = _WRAP_CMDS.sub(r"\1", s)
    s = s.replace("^\\circ", "").replace("^{\\circ}", "").replace("\u00b0", "")
    s = s.replace("\\%", "").replace("%", "").replace("\\$", "")
    s = s.replace("\\,", "").replace("\\;", "").replace("\\:", "").replace("\\ ", "")
    s = s.replace("\u2212", "-").replace(" ", "")
    s = s.rstrip(".")
    if _THOUSANDS.match(s):
        s = s.replace(",", "")
    s = _LEADING_VAR.sub("", s)
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _to_fraction(s: str) -> Optional[Fraction]:
    m = _FRAC.match(s)
    try:
        if m:
            f = Fraction(int(m.group(2)), int(m.group(3)))
            return -f if m.group(1) else f
        if "/" in s and s.count("/") == 1:
            a, b = s.split("/")
            return Fraction(int(a), int(b))
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


def try_numeric_equal(a: str, b: str) -> Optional[bool]:
    fa, fb = _to_fraction(a), _to_fraction(b)
    if fa is not None and fb is not None:
        return fa == fb
    try:
        x, y = float(a), float(b)
        return abs(x - y) <= 1e-9 * max(1.0, abs(x), abs(y))
    except (ValueError, OverflowError):
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
