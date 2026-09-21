"""
FAST Logit Trace Feature Extraction (Prompt-Only, No Generation).

Does a single forward pass on just the problem prompt.
Extracts layer-wise features at the LAST prompt token position.
No generation needed - runs in ~1-2 seconds per sample.

Features:
  - Digit probability trajectory across ALL layers
  - Full vocabulary entropy across layers
  - Hidden state norm trajectory
  - Top-k token analysis
  - Layer-wise confidence metrics
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import sys
import json
import pickle
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_ALIASES = {
    "qwen3-8b:low": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
}


def load_model(model_id: str):
    hf_id = MODEL_ALIASES.get(model_id, model_id)
    print(f"[extract] Loading {hf_id} ...", file=sys.stderr, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_id, local_files_only=True)
    dtype = torch.bfloat16 if DEVICE == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=dtype, device_map=None, local_files_only=True
    )
    model.to(DEVICE)
    model.eval()
    print(f"[extract] Model ready on {DEVICE}.", file=sys.stderr, flush=True)
    return tokenizer, model, hf_id


def get_digit_ids(tokenizer, device):
    """Get token IDs for digits 0-9."""
    digit_ids = []
    for token_id in range(min(tokenizer.vocab_size, 200000)):
        try:
            token_str = tokenizer.decode([token_id]).strip()
        except Exception:
            continue
        if token_str in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            digit_ids.append(token_id)
    return torch.tensor(digit_ids, device=device)


def extract_features_fast(model, tokenizer, problem: str, digit_ids: torch.Tensor) -> dict:
    """Single forward pass on prompt only. No generation."""
    
    # Format prompt with chat template
    prompt = f"Problem: {problem}\nAnswer:"
    messages = [{"role": "user", "content": prompt}]
    try:
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        formatted_prompt = prompt

    inp = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    prompt_len = inp.input_ids.shape[1]

    # Single forward pass with hidden states
    with torch.no_grad():
        out = model(input_ids=inp.input_ids, output_hidden_states=True)

    num_layers = len(out.hidden_states)
    
    # ===== Features at LAST token position =====
    digit_probs_per_layer = []   # max digit prob at each layer
    digit_entropy_per_layer = []  # entropy over digit distribution
    full_entropy_per_layer = []   # entropy over full vocabulary
    hidden_norm_per_layer = []    # norm of hidden state
    top1_digit_per_layer = []     # which digit is top
    top1_full_per_layer = []      # which token is top across full vocab
    top1_full_prob_per_layer = [] # prob of top token in full vocab
    digit_mass_per_layer = []     # total probability mass on digits
    kl_from_uniform_per_layer = []
    
    uniform_digit = torch.ones(len(digit_ids), device=model.device) / len(digit_ids)
    log_uniform_digit = torch.log(uniform_digit)

    for l in range(num_layers):
        h_l = out.hidden_states[l][0, -1, :]  # Last token
        
        # Hidden state norm
        hidden_norm_per_layer.append(h_l.norm().item())
        
        # Full vocabulary logits
        logits = model.lm_head(h_l)
        full_probs = torch.softmax(logits, dim=-1)
        
        # Full vocab entropy
        log_full_probs = torch.log(full_probs + 1e-10)
        full_entropy = -(full_probs * log_full_probs).sum().item()
        full_entropy_per_layer.append(full_entropy)
        
        # Top-1 across full vocab
        top1_prob, top1_idx = full_probs.max(dim=-1)
        top1_full_per_layer.append(top1_idx.item())
        top1_full_prob_per_layer.append(top1_prob.item())
        
        # Digit-specific analysis
        digit_logits = logits[digit_ids]
        d_probs = torch.softmax(digit_logits, dim=-1)
        
        max_d_prob, max_d_idx = d_probs.max(dim=-1)
        digit_probs_per_layer.append(max_d_prob.item())
        top1_digit_per_layer.append(max_d_idx.item())
        
        # Digit mass (how much of total prob is on digits)
        d_mass = full_probs[digit_ids].sum().item()
        digit_mass_per_layer.append(d_mass)
        
        # Digit entropy
        d_log_probs = torch.log(d_probs + 1e-10)
        d_entropy = -(d_probs * d_log_probs).sum().item()
        digit_entropy_per_layer.append(d_entropy)
        
        # KL from uniform (digit dist)
        kl = (d_probs * (d_log_probs - log_uniform_digit)).sum().item()
        kl_from_uniform_per_layer.append(kl)

    # Convert to numpy for stats
    dp = np.array(digit_probs_per_layer)
    de = np.array(digit_entropy_per_layer)
    fe = np.array(full_entropy_per_layer)
    hn = np.array(hidden_norm_per_layer)
    dm = np.array(digit_mass_per_layer)
    kl = np.array(kl_from_uniform_per_layer)
    fp = np.array(top1_full_prob_per_layer)
    
    # Derived features
    quarter = max(1, num_layers // 4)
    half = max(1, num_layers // 2)
    
    # Crystallization layers
    cryst_05 = num_layers
    cryst_09 = num_layers
    for l_idx, p in enumerate(digit_probs_per_layer):
        if p > 0.5 and cryst_05 == num_layers:
            cryst_05 = l_idx
        if p > 0.9 and cryst_09 == num_layers:
            cryst_09 = l_idx
    
    # Digit stability
    digit_changes = sum(
        1 for i in range(1, len(top1_digit_per_layer))
        if top1_digit_per_layer[i] != top1_digit_per_layer[i - 1]
    )
    
    # Top token stability (full vocab)
    full_token_changes = sum(
        1 for i in range(1, len(top1_full_per_layer))
        if top1_full_per_layer[i] != top1_full_per_layer[i - 1]
    )
    
    # Slopes
    x = np.arange(num_layers)
    dp_slope = np.polyfit(x, dp, 1)[0] if num_layers > 1 else 0.0
    fe_slope = np.polyfit(x, fe, 1)[0] if num_layers > 1 else 0.0
    hn_slope = np.polyfit(x, hn, 1)[0] if num_layers > 1 else 0.0
    dm_slope = np.polyfit(x, dm, 1)[0] if num_layers > 1 else 0.0

    features = {
        "num_layers": num_layers,
        "prompt_len": prompt_len,
        
        # Digit prob trajectory stats
        "dp_mean": dp.mean(), "dp_std": dp.std(), "dp_max": dp.max(), "dp_min": dp.min(),
        "dp_slope": dp_slope,
        "dp_early": dp[:quarter].mean(), "dp_late": dp[-quarter:].mean(),
        "dp_early_late_diff": dp[-quarter:].mean() - dp[:quarter].mean(),
        "dp_first_half": dp[:half].mean(), "dp_second_half": dp[half:].mean(),
        
        # Digit entropy stats
        "de_mean": de.mean(), "de_std": de.std(), "de_min": de.min(), "de_max": de.max(),
        "de_early": de[:quarter].mean(), "de_late": de[-quarter:].mean(),
        
        # Full vocab entropy stats
        "fe_mean": fe.mean(), "fe_std": fe.std(), "fe_min": fe.min(), "fe_max": fe.max(),
        "fe_slope": fe_slope,
        "fe_early": fe[:quarter].mean(), "fe_late": fe[-quarter:].mean(),
        "fe_drop": fe[0] - fe[-1] if num_layers > 1 else 0.0,
        
        # Hidden norm stats
        "hn_mean": hn.mean(), "hn_std": hn.std(), "hn_min": hn.min(), "hn_max": hn.max(),
        "hn_slope": hn_slope,
        "hn_early": hn[:quarter].mean(), "hn_late": hn[-quarter:].mean(),
        
        # Digit mass stats
        "dm_mean": dm.mean(), "dm_std": dm.std(), "dm_max": dm.max(), "dm_min": dm.min(),
        "dm_slope": dm_slope,
        "dm_early": dm[:quarter].mean(), "dm_late": dm[-quarter:].mean(),
        
        # KL from uniform
        "kl_mean": kl.mean(), "kl_std": kl.std(), "kl_max": kl.max(),
        
        # Full vocab top-1 prob
        "fp_mean": fp.mean(), "fp_std": fp.std(), "fp_max": fp.max(), "fp_min": fp.min(),
        "fp_early": fp[:quarter].mean(), "fp_late": fp[-quarter:].mean(),
        
        # Crystallization
        "cryst_05_ratio": cryst_05 / num_layers,
        "cryst_09_ratio": cryst_09 / num_layers,
        
        # Stability
        "digit_changes": digit_changes,
        "digit_change_ratio": digit_changes / max(num_layers - 1, 1),
        "full_token_changes": full_token_changes,
        "full_token_change_ratio": full_token_changes / max(num_layers - 1, 1),
        
        # Raw trajectories for later analysis
        "_dp_trajectory": digit_probs_per_layer,
        "_de_trajectory": digit_entropy_per_layer,
        "_fe_trajectory": full_entropy_per_layer,
        "_hn_trajectory": hidden_norm_per_layer,
        "_dm_trajectory": digit_mass_per_layer,
        "_kl_trajectory": kl_from_uniform_per_layer,
        "_fp_trajectory": top1_full_prob_per_layer,
    }
    
    return features


def main():
    # Load cases and labels
    cases = []
    with open("data/train-sample/input/cases.jsonl") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))

    labels = {}
    with open("data/train-sample/reference/labels.jsonl") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                labels[d["id"]] = d["is_robust"]

    # Group by model
    model_groups = {}
    for c in cases:
        mid = c["model_id"]
        model_groups.setdefault(mid, []).append(c)

    all_results = []
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    for model_id, group_cases in model_groups.items():
        tokenizer, model, hf_id = load_model(model_id)
        digit_ids = get_digit_ids(tokenizer, model.device)

        for c in group_cases:
            case_id = c["id"]
            problem = c["problem"]
            gt = labels.get(case_id)

            print(f"[extract GPU={gpu_id}] {case_id} (model={model_id}, GT={gt})...", file=sys.stderr, flush=True)

            try:
                features = extract_features_fast(model, tokenizer, problem, digit_ids)
                features["case_id"] = case_id
                features["model_id"] = model_id
                features["hf_id"] = hf_id
                features["gt_robust"] = gt
                all_results.append(features)

                print(
                    f"[extract GPU={gpu_id}] {case_id} GT={gt} | "
                    f"dp_max={features['dp_max']:.3f} dp_late={features['dp_late']:.3f} "
                    f"fe_late={features['fe_late']:.3f} dm_late={features['dm_late']:.6f} "
                    f"digit_changes={features['digit_changes']} "
                    f"cryst_05={features['cryst_05_ratio']:.3f}",
                    file=sys.stderr, flush=True,
                )
            except Exception as exc:
                print(f"[extract GPU={gpu_id}] ERROR {case_id}: {exc}", file=sys.stderr, flush=True)
                import traceback
                traceback.print_exc(file=sys.stderr)

        del model, tokenizer
        torch.cuda.empty_cache()

    # Save
    with open("logit_trace_features.pkl", "wb") as f:
        pickle.dump(all_results, f)
    print(f"\n[extract] Saved {len(all_results)} feature sets to logit_trace_features.pkl", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
