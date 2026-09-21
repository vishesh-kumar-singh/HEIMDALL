"""Model-as-judge math answer comparison.

Uses the loaded LLM itself to determine whether two mathematical answers
are semantically equivalent, avoiding brittle string matching.
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def normalize_answer(answer: str) -> str:
    """Basic normalization: strip whitespace, LaTeX commands, trailing .0, etc."""
    if answer is None:
        return ""
    s = answer.strip()
    # Remove common LaTeX wrappers
    for cmd in [r"\text{", r"\mathrm{", r"\mathbf{", r"\textbf{", r"\displaystyle"]:
        s = s.replace(cmd, "")
    # Remove trailing unmatched braces from above
    while s.endswith("}") and s.count("}") > s.count("{"):
        s = s[:-1]
    # Remove spaces, \, and \;
    s = s.replace(r"\,", "").replace(r"\;", "").replace(r"\ ", "")
    s = s.replace(" ", "")
    # Normalize minus signs
    s = s.replace("−", "-")  # unicode minus to ascii
    # Strip trailing .0
    if s.endswith(".0"):
        s = s[:-2]
    return s


def try_numeric_equal(a: str, b: str) -> Optional[bool]:
    """Try to compare as numbers. Returns None if either isn't numeric."""
    try:
        fa = float(a)
        fb = float(b)
        return abs(fa - fb) < 1e-9
    except (ValueError, OverflowError):
        pass

    # Try fraction parsing: \frac{num}{den}
    frac_pat = re.compile(r"\\frac\{([^}]+)\}\{([^}]+)\}")
    def to_fraction(s):
        m = frac_pat.fullmatch(s)
        if m:
            return Fraction(int(m.group(1)), int(m.group(2)))
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
    """Quick check: do these two answers obviously match or not?
    
    Returns True/False if we can determine, None if uncertain (needs model judge).
    """
    if a is None or b is None:
        return None
    
    na = normalize_answer(a)
    nb = normalize_answer(b)
    
    # Exact match after normalization
    if na == nb:
        return True
    
    # Try numeric comparison
    numeric = try_numeric_equal(na, nb)
    if numeric is not None:
        return numeric
    
    # Can't determine — need the model
    return None


def model_judge_equivalent(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    answer_a: str,
    answer_b: str,
) -> bool:
    """Ask the model whether two math answers are equivalent.
    
    This generates only 1-2 tokens (Yes/No), so it's very fast (~50ms).
    """
    # First try the fast normalizer
    quick = answers_match(answer_a, answer_b)
    if quick is not None:
        return quick
    
    prompt = (
        f"Are these two mathematical answers exactly equivalent?\n"
        f"Answer A: {answer_a}\n"
        f"Answer B: {answer_b}\n"
        f"Reply with ONLY 'Yes' or 'No'."
    )
    
    # Use chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        formatted = prompt
    
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,  # Just need Yes/No but allow some thinking
            do_sample=False,
            use_cache=True,
        )
    
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip().lower()
    
    # Extract Yes/No from response (model might think first)
    # Check the last occurrence since thinking models put answer at end
    response_clean = response.replace("</think>", "").strip()
    
    if "yes" in response_clean[-20:]:
        return True
    elif "no" in response_clean[-20:]:
        return False
    else:
        # If model is ambiguous, default to checking if "yes" appears anywhere
        return "yes" in response_clean
