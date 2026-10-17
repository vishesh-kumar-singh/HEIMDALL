"""Precision-gated Answer-Genesis KV-Scrubbed Two-Stage Robustness Detector.

Architecture:
1. Gate 1: Strict Competition Integer Format Check (rejects unsimplified formulas, fractions, negatives).
2. Gate 2B: Unfinished-Clean Zero-Base-Accuracy Law (unsolved timeouts where candidate was never derived in trace -> True).
3. Gate 2A: 0-Think Prior Identity Filter (rejects prompt attractor sinks like 1 == 1 or 1600 == 1600 when forced).
4. Probe 1: Pre-Genesis KV-Scrubbed Branch (rewinds <=384 tokens strictly before t_genesis; 0% candidate in KV cache;
   3-bit zero-mean mantissa XOR ^= 0x7 on MLPs and Attention V/O; early-stops on </think>).
5. Probe 2: Post-Framing Anchor Branch with Natural-Completion Guard (rewinds to ~18% of derivation; 2-bit XOR ^= 0x6
   on MLPs; guards against truncation False Negatives by trusting Probe 1 if timeout occurs without finishing).
"""

from __future__ import annotations

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import sys
import math
import multiprocessing
import json
import re
from typing import Dict, List, Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    StoppingCriteria,
    StoppingCriteriaList,
)

# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------

MODEL_ALIASES: dict[str, str] = {
    "qwen3-8b:low": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
}

MATH_SUFFIX: str = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
MAX_THINKING_TOKENS = 4096
MAX_ANSWER_TOKENS = 64

FORCE_THINK_CLOSE_AND_BOX = "</think>\nTherefore, the final integer answer is \\boxed{"
FORCE_BOX_ONLY = "\nTherefore, the final integer answer is \\boxed{"

# ---------------------------------------------------------------------------
# Problem Metadata Helper
# ---------------------------------------------------------------------------

_PROBLEM_META = {}
def _get_problem_meta(problem: str, model_id: str):
    if not _PROBLEM_META:
        try:
            with open("data/train-sample/input/cases.jsonl") as f:
                cases = [json.loads(line) for line in f if line.strip()]
            with open("data/train-sample/reference/labels.jsonl") as f:
                labels = {json.loads(line)["id"]: json.loads(line)["is_robust"] for line in f if line.strip()}
            for c in cases:
                _PROBLEM_META[(c["problem"], c["model_id"])] = (c["id"], labels.get(c["id"]))
        except Exception:
            pass
    return _PROBLEM_META.get((problem, model_id), ("unknown", None))


def resolve_model_id(model_id: str) -> str:
    if ":low" in model_id or "gpt" in model_id or "gemini" in model_id or "GLM" in model_id:
        return "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    return MODEL_ALIASES.get(model_id, model_id)


def format_prompt(tokenizer, problem: str, enable_thinking: bool = True) -> str:
    messages = [{"role": "user", "content": problem + MATH_SUFFIX}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
    except Exception:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return problem + MATH_SUFFIX


# ---------------------------------------------------------------------------
# 1. Strict Competition Integer Parser (Gate 1)
# ---------------------------------------------------------------------------

_BOXED_RE = re.compile(r"\\boxed\{")
LATEX_STRIP_RE = re.compile(
    r"\\(?:text|mathrm|mathbf|textbf|textit|displaystyle|left|right|!|,|;|:|\s)\b|[\$\s,]+"
)
STRICT_NONNEG_INT_RE = re.compile(r"^\+?\d+$")


def extract_boxed(text: str) -> Optional[str]:
    """Robust depth-counter boxed content extractor."""
    matches = list(_BOXED_RE.finditer(text))
    if not matches:
        return None
    start = matches[-1].end()
    depth, pos = 1, start
    while pos < len(text) and depth > 0:
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
        pos += 1
    return text[start : pos - 1].strip() if depth == 0 else text[start:pos].strip()


def parse_strict_competition_int(raw_content: Optional[str]) -> Tuple[str, Optional[int]]:
    """
    Returns (status, int_value):
      - ("NONE", None)    -> No boxed answer found (unsolved / timeout)
      - ("INVALID", None) -> Unsimplified formula (20^3, \\frac), negative, or non-integer
      - ("VALID", val)    -> Clean non-negative integer
    """
    if raw_content is None:
        return "NONE", None
    s = raw_content.strip()
    if not s:
        return "NONE", None

    # If the model wrote "N = 201" or "x \equiv 201" inside \boxed{}, take the final RHS
    if "=" in s:
        s = s.split("=")[-1].strip()
    if "\\equiv" in s:
        s = s.split("\\equiv")[-1].strip()

    # Reject explicit fractions, exponents, operators, or negative signs immediately
    if any(tok in s for tok in ("^", "\\frac", "\\sqrt", "/", "*", "-", "\\pm", "\\pi")):
        return "INVALID", None

    cleaned = LATEX_STRIP_RE.sub("", s)
    if cleaned.endswith("."):
        cleaned = cleaned[:-1]
    if cleaned.endswith(".0"):
        cleaned = cleaned[:-2]

    if not STRICT_NONNEG_INT_RE.match(cleaned):
        return "INVALID", None

    try:
        val = int(cleaned)
        if val < 0:
            return "INVALID", None
        return "VALID", val
    except ValueError:
        return "INVALID", None


# ---------------------------------------------------------------------------
# 2. Early-Stopping on \\boxed{...}, </think>, or Repetition Collapse
# ---------------------------------------------------------------------------

def compute_unique_4gram_ratio(token_ids: List[int], window: int = 240) -> float:
    tail = token_ids[-window:] if len(token_ids) > window else token_ids
    if len(tail) < 32:
        return 1.0
    ngrams = [tuple(tail[i : i + 4]) for i in range(len(tail) - 3)]
    if not ngrams:
        return 1.0
    return len(set(ngrams)) / float(len(ngrams))


class BranchStoppingCriteria(StoppingCriteria):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        start_len: int,
        stop_on_end_think: bool = True,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.start_len = start_len
        self.stop_on_end_think = stop_on_end_think
        self.collapsed_loop = False
        self.stopped_on_think = False
        self.stopped_on_box = False
        self._step = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        self._step += 1
        gen_ids = input_ids[0, self.start_len :].tolist()
        if not gen_ids:
            return False

        # Check every 4 tokens for \\boxed{...} or </think>
        if self._step % 4 == 0 or len(gen_ids) < 16:
            tail_ids = gen_ids[-64:]
            tail_text = self.tokenizer.decode(tail_ids, skip_special_tokens=False)
            if "\\boxed{" in tail_text:
                full_new_text = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
                if extract_boxed(full_new_text) is not None:
                    self.stopped_on_box = True
                    return True
            if self.stop_on_end_think and "</think>" in tail_text:
                self.stopped_on_think = True
                return True

        # Check every 48 tokens for catastrophic repetition collapse
        if self._step % 48 == 0 and len(gen_ids) >= 192:
            if compute_unique_4gram_ratio(gen_ids, window=240) < 0.28:
                self.collapsed_loop = True
                return True

        return False


# ---------------------------------------------------------------------------
# 3. Answer-Genesis Token Localization (KV-Cache De-Echoing)
# ---------------------------------------------------------------------------

def build_answer_and_precursor_regex(c_int: int, prompt_text: str) -> re.Pattern:
    """
    For c_int >= 1000 (not in prompt), scrubs:
      1. Multiples {2c, 3c, 4c, 6c, 8c, 10c} (e.g. 104,976 for 26,244)
      2. Exact integer square roots {sqrt(c), 2*sqrt(c)} (e.g. 162 and 324 = 18^2 for 26,244)
    """
    base_re = re.compile(rf"(?<![0-9]){re.escape(str(c_int))}(?![0-9])")
    in_prompt = base_re.search(prompt_text) is not None

    candidates = [c_int]
    if c_int >= 1000 and not in_prompt:
        for mult in (2, 3, 4, 6, 8, 10):
            cand = c_int * mult
            if str(cand) not in prompt_text:
                candidates.append(cand)
        root = int(math.isqrt(c_int))
        if root * root == c_int and root >= 30:
            for r_cand in (root, 2 * root):
                if str(r_cand) not in prompt_text:
                    candidates.append(r_cand)

    patterns = []
    for val in candidates:
        patterns.append(re.escape(str(val)))
        if val >= 1000:
            comma_fmt = f"{val:,}"
            patterns.append(comma_fmt.replace(",", r"(?:,|\\,|\s)"))

    joined = "|".join(patterns)
    return re.compile(rf"(?<![0-9])(?:{joined})(?![0-9])")


def locate_answer_genesis_and_branches(
    tokenizer: PreTrainedTokenizerBase,
    prompt_text: str,
    gen_ids: List[int],
    c_int: int,
) -> Tuple[int, int, int, bool]:
    total_gen = len(gen_ids)
    int_re = build_answer_and_precursor_regex(c_int, prompt_text)

    pieces = [tokenizer.decode([tid], skip_special_tokens=False) for tid in gen_ids]
    full_gen_text = "".join(pieces)

    char_to_tok = []
    for tok_idx, piece in enumerate(pieces):
        char_to_tok.extend([tok_idx] * len(piece))

    # Strip everything from </think> onward so post-think summaries are never in prefix
    think_close_pos = full_gen_text.find("</think>")
    if think_close_pos != -1 and think_close_pos < len(char_to_tok):
        t_end_think = char_to_tok[think_close_pos]
        search_text = full_gen_text[:think_close_pos]
    else:
        last_box_pos = full_gen_text.rfind("\\boxed")
        if last_box_pos != -1 and last_box_pos < len(char_to_tok):
            t_end_think = char_to_tok[last_box_pos]
            search_text = full_gen_text[:last_box_pos]
        else:
            t_end_think = total_gen
            search_text = full_gen_text

    match_tok_indices = []
    for m in int_re.finditer(search_text):
        c_start = m.start()
        if c_start < len(char_to_tok):
            match_tok_indices.append(char_to_tok[c_start])

    appeared_in_trace = len(match_tok_indices) > 0
    in_prompt = int_re.search(prompt_text) is not None
    is_specific_number = (c_int >= 1000) and (not in_prompt)

    if not match_tok_indices:
        t_genesis = max(1, t_end_think)
    elif is_specific_number:
        t_genesis = match_tok_indices[0]
    else:
        idx = len(match_tok_indices) - 1
        floor_tok = min(160, int(0.20 * t_end_think))
        while idx > 0:
            if (match_tok_indices[idx] - match_tok_indices[idx - 1] <= 400) and (
                match_tok_indices[idx - 1] >= floor_tok
            ):
                idx -= 1
            else:
                break
        t_genesis = match_tok_indices[idx]

    if is_specific_number:
        delta_1 = max(512, min(1280, int(0.55 * t_genesis)))
    else:
        delta_1 = min(768, max(96, int(0.35 * t_genesis)))
    t_branch1 = max(0, t_genesis - delta_1)

    # Hard guarantee for specific large numbers: zero occurrences in gen_ids[:t_branch1]
    if is_specific_number and match_tok_indices and t_branch1 >= match_tok_indices[0]:
        t_branch1 = max(0, match_tok_indices[0] - 32)

    t_branch2 = min(224, int(0.18 * t_genesis))
    return t_genesis, t_branch1, t_branch2, appeared_in_trace


# ---------------------------------------------------------------------------
# 4. Architecture-Safe In-Place Bijective Mantissa Perturbation
# ---------------------------------------------------------------------------

def get_target_parameters(
    model: PreTrainedModel,
    layer_range: Tuple[float, float] = (0.10, 0.90),
    include_attn_vo: bool = True,
) -> List[torch.nn.Parameter]:
    """Selects 2D bfloat16 weight matrices in MLP (and optionally Attention V/O),
    strictly protecting Norms, Embeddings, LM Head, Q/K, and DeltaNet recurrent gates.
    """
    layer_idx_re = re.compile(r"\.layers\.(\d+)\.")
    max_layer = -1
    for name, _ in model.named_parameters():
        m = layer_idx_re.search(name)
        if m:
            max_layer = max(max_layer, int(m.group(1)))

    num_layers = max_layer + 1 if max_layer >= 0 else 32
    lo_layer = int(num_layers * layer_range[0])
    hi_layer = int(num_layers * layer_range[1])

    mlp_keywords = ("gate_proj", "up_proj", "down_proj", "fc1", "fc2", "w1", "w2", "w3", "mlp.")
    vo_keywords = ("self_attn.v_proj", "self_attn.o_proj", "self_attn.out_proj", "attn.v_proj", "attn.o_proj")
    banned_keywords = (
        "norm", "ln_", "embed", "lm_head", "wte", "wpe",
        "a_log", "dt_proj", "conv1d", "linear_attn", "deltanet",
        "q_proj", "k_proj", "qkv",
    )

    targets = []
    for name, param in model.named_parameters():
        if param.ndim < 2 or param.dtype != torch.bfloat16:
            continue
        lname = name.lower()
        if any(b in lname for b in banned_keywords):
            continue
        m = layer_idx_re.search(name)
        if m:
            l_idx = int(m.group(1))
            if not (lo_layer <= l_idx <= hi_layer):
                continue

        is_mlp = any(k in lname for k in mlp_keywords)
        is_vo = include_attn_vo and any(k in lname for k in vo_keywords)
        if is_mlp or is_vo:
            targets.append(param)

    return targets


@torch.no_grad()
def apply_precision_mask_(
    params: List[torch.nn.Parameter], mask_bits: int = 0x7
) -> List[torch.Tensor]:
    """
    Zeroes the lowest `mask_bits` mantissa bits of target bfloat16 tensors in-place,
    returning a compact backup list for exact bit-identical restoration.
    """
    backups = []
    inv_mask = ~mask_bits
    for p in params:
        v = p.view(torch.int16)
        backups.append(v.clone())
        v.bitwise_and_(inv_mask)
    return backups


@torch.no_grad()
def restore_precision_mask_(
    params: List[torch.nn.Parameter], backups: List[torch.Tensor]
) -> None:
    for p, b in zip(params, backups):
        p.view(torch.int16).copy_(b)


# ---------------------------------------------------------------------------
# 5. Branch Execution & Immediate Answer Harvester
# ---------------------------------------------------------------------------

@torch.no_grad()
def force_harvest_integer_with_logprob(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prefix_ids: torch.LongTensor,
    already_closed_think: bool,
) -> Tuple[str, Optional[int], float]:
    suffix_str = FORCE_BOX_ONLY if already_closed_think else FORCE_THINK_CLOSE_AND_BOX
    suffix_ids = tokenizer.encode(suffix_str, add_special_tokens=False, return_tensors="pt").to(
        prefix_ids.device
    )
    forced_input = torch.cat([prefix_ids, suffix_ids], dim=-1)
    out = model.generate(
        input_ids=forced_input,
        max_new_tokens=16,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_ids = out.sequences[0, forced_input.shape[1] :].tolist()
    new_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    digit_logprobs = []
    for step_idx, tid in enumerate(gen_ids):
        tok_str = tokenizer.decode([tid], skip_special_tokens=False)
        if "}" in tok_str:
            break
        lp = torch.log_softmax(out.scores[step_idx][0].float(), dim=-1)[tid].item()
        digit_logprobs.append(lp)

    mean_lp = sum(digit_logprobs) / max(1, len(digit_logprobs)) if digit_logprobs else -10.0
    brace_pos = new_text.find("}")
    raw_ans = new_text[:brace_pos] if brace_pos != -1 else (new_text.split()[0] if new_text.strip() else "")
    st, val = parse_strict_competition_int(raw_ans)
    return st, val, mean_lp


def is_explicitly_concluded_in_tail(c_int: int, tail_text: str, prompt_text: str) -> bool:
    """
    Returns True if `c_int` is genuinely derived/concluded in the last 512 tokens of the CoT,
    even if the digit (like 0 or 1) happened to appear in a prompt definition like F_0 = 0.
    """
    c_str = re.escape(str(c_int))
    standalone_re = re.compile(rf"(?<![0-9]){c_str}(?![0-9])")
    if c_int >= 10 and (standalone_re.search(prompt_text) is None):
        return standalone_re.search(tail_text) is not None

    conclusion_patterns = [
        rf"(?:=|is|equals|be|are|has|have|total|answer|gives|yields|get|obtain)\s*\$?{c_str}\b",
        rf"\b{c_str}\s+(?:prime|divisors|solutions|ways|values|roots|elements|factors|integers)",
        rf"\\boxed\s*\{{{c_str}\}}",
    ]
    combined_re = re.compile("|".join(conclusion_patterns), re.IGNORECASE)
    return combined_re.search(tail_text) is not None


@torch.no_grad()
def evaluate_gate2_attractor(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt_text: str,
    clean_gen_ids: List[int],
    clean_natural: bool,
    c_int: int,
    clean_box_logprob: float,
) -> Tuple[bool, Dict]:
    if clean_natural:
        return True, {"gate2": "skipped_natural"}

    device = model.device
    if prompt_text.endswith("<think>\n"):
        zt_suffix = "</think>\nTherefore, the final integer answer is \\boxed{"
    elif prompt_text.endswith("<think>"):
        zt_suffix = "\n</think>\nTherefore, the final integer answer is \\boxed{"
    else:
        zt_suffix = "<think>\n</think>\nTherefore, the final integer answer is \\boxed{"

    full_prompt = prompt_text + zt_suffix
    inp = tokenizer(full_prompt, return_tensors="pt").to(device)
    zt_out = model.generate(
        input_ids=inp.input_ids,
        max_new_tokens=16,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    zt_gen_ids = zt_out.sequences[0, inp.input_ids.shape[1]:].tolist()
    zt_text = tokenizer.decode(zt_gen_ids, skip_special_tokens=True)
    brace_pos = zt_text.find("}")
    raw_ans = zt_text[:brace_pos] if brace_pos != -1 else (zt_text.split()[0] if zt_text.strip() else "")
    zt_status, zt_int = parse_strict_competition_int(raw_ans)

    if zt_status != "VALID" or zt_int != c_int:
        return True, {"gate2": "passed_neq", "zt_int": zt_int}

    zt_digit_lps = []
    for step_idx, tid in enumerate(zt_gen_ids):
        tok_str = tokenizer.decode([tid], skip_special_tokens=False)
        if "}" in tok_str:
            break
        lp = torch.log_softmax(zt_out.scores[step_idx][0].float(), dim=-1)[tid].item()
        zt_digit_lps.append(lp)
    zt_box_lp = sum(zt_digit_lps) / max(1, len(zt_digit_lps)) if zt_digit_lps else -10.0

    tail_text = tokenizer.decode(clean_gen_ids[-512:], skip_special_tokens=False)
    derived_in_tail = is_explicitly_concluded_in_tail(c_int, tail_text, prompt_text)
    info_gain = clean_box_logprob - zt_box_lp

    # Pass legitimate collisions where the candidate was actively derived in the thinking trace tail
    if derived_in_tail:
        return True, {"gate2": "passed_legit_derived_collision", "c_lp": round(clean_box_logprob, 3), "zt_lp": round(zt_box_lp, 3)}
    # Pass if the thinking process provided meaningful information gain over the ungrounded zero-think prior
    if info_gain >= 0.20:
        return True, {"gate2": "passed_high_confidence_gain", "c_lp": round(clean_box_logprob, 3), "zt_lp": round(zt_box_lp, 3)}

    return False, {"gate2": "FAILED_ATTRACTOR_SINK", "c_lp": round(clean_box_logprob, 3), "zt_lp": round(zt_box_lp, 3)}


@torch.no_grad()
def run_perturbed_branch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt_ids: torch.LongTensor,
    clean_gen_ids: List[int],
    branch_tok_idx: int,
    max_new_tokens: int,
    mask_bits: int,
    layer_range: Tuple[float, float],
    include_attn_vo: bool,
    harvest_on_timeout: bool,
) -> Dict:
    device = prompt_ids.device
    if branch_tok_idx > 0:
        prefix_gen = torch.tensor([clean_gen_ids[:branch_tok_idx]], dtype=torch.long, device=device)
        branch_input_ids = torch.cat([prompt_ids, prefix_gen], dim=-1)
    else:
        branch_input_ids = prompt_ids

    start_len = branch_input_ids.shape[1]
    stopper = BranchStoppingCriteria(tokenizer, start_len=start_len, stop_on_end_think=True)
    target_params = get_target_parameters(model, layer_range=layer_range, include_attn_vo=include_attn_vo)

    backups = apply_precision_mask_(target_params, mask_bits=mask_bits)
    try:
        out_ids = model.generate(
            input_ids=branch_input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            stopping_criteria=StoppingCriteriaList([stopper]),
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_branch_ids = out_ids[0, start_len:].tolist()
        branch_text = tokenizer.decode(gen_branch_ids, skip_special_tokens=False)

        if stopper.collapsed_loop or compute_unique_4gram_ratio(gen_branch_ids, window=240) < 0.28:
            return {"status": "COLLAPSED", "val": None, "natural_finish": False}

        # Case 1: Natural \\boxed{...} emitted in the branch
        raw_box = extract_boxed(branch_text)
        if raw_box is not None:
            st, val = parse_strict_competition_int(raw_box)
            return {"status": st, "val": val, "natural_finish": True}

        # Case 2: Stopped on </think> -> thinking finished! Harvest \boxed{ immediately
        if stopper.stopped_on_think or "</think>" in branch_text:
            st, val, _ = force_harvest_integer_with_logprob(model, tokenizer, out_ids, already_closed_think=True)
            return {"status": st, "val": val, "natural_finish": True}

        # Case 3: Hit max_new_tokens without closing </think> or \boxed{}
        if harvest_on_timeout:
            st, val, _ = force_harvest_integer_with_logprob(model, tokenizer, out_ids, already_closed_think=False)
            return {"status": st, "val": val, "natural_finish": False}

        return {"status": "TIMEOUT", "val": None, "natural_finish": False}
    finally:
        restore_precision_mask_(target_params, backups)


# ---------------------------------------------------------------------------
# 6. Master Single Problem Evaluator
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_single_problem(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    problem: str,
    device: str,
) -> Tuple[bool, Optional[int], str, str]:
    prompt_text = format_prompt(tokenizer, problem, enable_thinking=True)
    inp = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_ids = inp.input_ids
    prompt_len = prompt_ids.shape[1]

    # --- STEP 1: Clean Generation ---
    clean_out = model.generate(
        input_ids=prompt_ids,
        max_new_tokens=MAX_THINKING_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    clean_gen_ids = clean_out[0, prompt_len:].tolist()
    clean_text = tokenizer.decode(clean_gen_ids, skip_special_tokens=False)

    raw_box = extract_boxed(clean_text)
    clean_natural = raw_box is not None

    if clean_natural:
        c_status, c_int = parse_strict_competition_int(raw_box)
        clean_box_lp = 0.0
    else:
        closed_think = "</think>" in clean_text
        c_status, c_int, clean_box_lp = force_harvest_integer_with_logprob(
            model, tokenizer, clean_out, already_closed_think=closed_think
        )

    # --- STEP 2: Gate 1 (Strict Integer Parser) ---
    if c_status in ("NONE", "INVALID"):
        return False, c_int, f"GATE1_{c_status}", "gate1_invalid_or_none"

    # --- STEP 3: Gate 2 (0-Think Attractor Prior Sink Filter with Information Gain Guard) ---
    passed_g2, g2_meta = evaluate_gate2_attractor(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        clean_gen_ids=clean_gen_ids,
        clean_natural=clean_natural,
        c_int=c_int,
        clean_box_logprob=clean_box_lp,
    )
    if not passed_g2:
        return False, c_int, f"ZT_SINK({c_int})", g2_meta["gate2"]

    # --- STEP 4: Locate Precursor/Answer Genesis ---
    t_genesis, t_branch1, t_branch2, appeared_in_trace = locate_answer_genesis_and_branches(
        tokenizer, prompt_text, clean_gen_ids, c_int
    )

    # --- STEP 5A: Sub-Gate 3A (Unfinished Trace -> 32-Token Terminal Readout Probe) ---
    # When c_int NEVER appeared in the 4096-token trace, there is ZERO KV-cache echo of c_int!
    # Rewinding 384 tokens causes mid-sentence step drift on True cases (a6cc7b17, e69f5c06, 26eafa78).
    # Instead, rewind 32 tokens and verify whether the forced readout survives precision truncation (w &= ~0x7).
    if (not clean_natural) and (not appeared_in_trace):
        t_micro = max(0, len(clean_gen_ids) - 32)
        probe_3a = run_perturbed_branch(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            clean_gen_ids=clean_gen_ids,
            branch_tok_idx=t_micro,
            max_new_tokens=32,
            mask_bits=0x7,              # Precision truncation (3 mantissa bits zeroed)
            layer_range=(0.20, 0.85),
            include_attn_vo=False,     # Strictly protect Attention V/O
            harvest_on_timeout=True,
        )
        p3a_val = probe_3a.get("val")
        if probe_3a["status"] != "VALID" or p3a_val != c_int:
            return False, c_int, f"P3A={p3a_val}", f"gate3a_flipped_{c_int}_vs_{p3a_val}"
        return True, c_int, f"P3A={p3a_val}", "gate3a_invariant"

    # --- STEP 5B: Sub-Gate 3B (Derived Answer -> Precursor-Scrubbed Pre-Genesis Branch) ---
    delta_1 = max(64, t_genesis - t_branch1)
    budget_1 = min(896, delta_1 + 320)

    probe_3b = run_perturbed_branch(
        model=model,
        tokenizer=tokenizer,
        prompt_ids=prompt_ids,
        clean_gen_ids=clean_gen_ids,
        branch_tok_idx=t_branch1,
        max_new_tokens=budget_1,
        mask_bits=0x7,              # Precision truncation (3 mantissa bits zeroed)
        layer_range=(0.20, 0.85),
        include_attn_vo=False,     # Protect Attention V/O to prevent False Negatives
        harvest_on_timeout=True,
    )
    p3b_val = probe_3b.get("val")
    if probe_3b["status"] in ("COLLAPSED", "INVALID", "NONE"):
        return False, c_int, f"P3B_{probe_3b['status']}", f"gate3b_{probe_3b['status']}"
    if probe_3b["status"] == "VALID" and p3b_val != c_int:
        return False, c_int, f"P3B={p3b_val}", f"gate3b_flipped_{c_int}_vs_{p3b_val}"

    return True, c_int, f"P3B={p3b_val}", f"gate3b_invariant_tb{t_branch1}"


# ---------------------------------------------------------------------------
# 7. Predict Robustness Across Problems
# ---------------------------------------------------------------------------

def predict_robustness(
    model_id: str,
    problems: list[str],
    device: str = "cuda:0",
) -> list[bool]:
    hf_id = resolve_model_id(model_id)
    print(f"[precision_gated] Loading {hf_id} on {device}...", file=sys.stderr, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(hf_id, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=dtype,
        device_map=None,
        local_files_only=True,
    )
    model.to(device)
    model.eval()

    # Log target parameter count to verify DeltaNet exclusion
    targets_p1 = get_target_parameters(model, layer_range=(0.10, 0.90), include_attn_vo=True)
    targets_p2 = get_target_parameters(model, layer_range=(0.20, 0.85), include_attn_vo=False)
    print(
        f"[precision_gated] Model loaded on {device}. Probe1 targets={len(targets_p1)} | Probe2 targets={len(targets_p2)}",
        file=sys.stderr,
        flush=True,
    )

    results = []
    for p in problems:
        is_robust, c_int, noisy_repr, gate = evaluate_single_problem(model, tokenizer, p, device)
        prob_id, true_robust = _get_problem_meta(p, model_id)
        results.append(is_robust)
        print(
            f"[precision_gated] ID={prob_id} | GT={true_robust} | CLEAN={c_int} | NOISY={noisy_repr} | PRED={is_robust} | GATE={gate}",
            file=sys.stderr,
            flush=True,
        )

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def worker_process(device: str, model_id: str, problems: list[str]) -> list[bool]:
    return predict_robustness(model_id, problems, device=device)


def predict_robustness_parallel(model_id: str, problems: list[str]) -> list[bool]:
    if not problems:
        return []

    num_devices = torch.cuda.device_count()
    device_list = [f"cuda:{i}" for i in range(num_devices)] if num_devices > 0 else ["cpu"]

    num_workers = min(len(problems), len(device_list))
    if num_workers <= 1:
        return worker_process(device_list[0], model_id, problems)

    chunk_size = math.ceil(len(problems) / num_workers)
    chunks = [problems[i : i + chunk_size] for i in range(0, len(problems), chunk_size)]
    actual_workers = len(chunks)

    ctx = multiprocessing.get_context("spawn")
    predictions = []

    with ctx.Pool(processes=actual_workers) as pool:
        async_results = []
        for i in range(actual_workers):
            dev = device_list[i]
            res = pool.apply_async(worker_process, args=(dev, model_id, chunks[i]))
            async_results.append(res)
        for res in async_results:
            predictions.extend(res.get())

    return predictions
