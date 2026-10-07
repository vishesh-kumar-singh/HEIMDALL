import torch
import numpy as np
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
    prompt = f"Problem: {problem}\\nAnswer:"
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
    
    digit_probs_per_layer = []
    digit_entropy_per_layer = []
    full_entropy_per_layer = []
    hidden_norm_per_layer = []
    top1_digit_per_layer = []
    top1_full_per_layer = []
    top1_full_prob_per_layer = []
    digit_mass_per_layer = []
    kl_from_uniform_per_layer = []
    
    uniform_digit = torch.ones(len(digit_ids), device=model.device) / len(digit_ids)
    log_uniform_digit = torch.log(uniform_digit)

    for l in range(num_layers):
        h_l = out.hidden_states[l][0, -1, :]  # Last token
        hidden_norm_per_layer.append(h_l.norm().item())
        
        logits = model.lm_head(h_l)
        full_probs = torch.softmax(logits, dim=-1)
        
        log_full_probs = torch.log(full_probs + 1e-10)
        full_entropy = -(full_probs * log_full_probs).sum().item()
        full_entropy_per_layer.append(full_entropy)
        
        top1_prob, top1_idx = full_probs.max(dim=-1)
        top1_full_per_layer.append(top1_idx.item())
        top1_full_prob_per_layer.append(top1_prob.item())
        
        digit_logits = logits[digit_ids]
        d_probs = torch.softmax(digit_logits, dim=-1)
        
        max_d_prob, max_d_idx = d_probs.max(dim=-1)
        digit_probs_per_layer.append(max_d_prob.item())
        top1_digit_per_layer.append(max_d_idx.item())
        
        d_mass = full_probs[digit_ids].sum().item()
        digit_mass_per_layer.append(d_mass)
        
        d_log_probs = torch.log(d_probs + 1e-10)
        d_entropy = -(d_probs * d_log_probs).sum().item()
        digit_entropy_per_layer.append(d_entropy)
        
        kl = (d_probs * (d_log_probs - log_uniform_digit)).sum().item()
        kl_from_uniform_per_layer.append(kl)

    dp = np.array(digit_probs_per_layer)
    de = np.array(digit_entropy_per_layer)
    fe = np.array(full_entropy_per_layer)
    hn = np.array(hidden_norm_per_layer)
    dm = np.array(digit_mass_per_layer)
    kl = np.array(kl_from_uniform_per_layer)
    fp = np.array(top1_full_prob_per_layer)
    
    quarter = max(1, num_layers // 4)
    half = max(1, num_layers // 2)
    
    cryst_05 = num_layers
    cryst_09 = num_layers
    for l_idx, p in enumerate(digit_probs_per_layer):
        if p > 0.5 and cryst_05 == num_layers:
            cryst_05 = l_idx
        if p > 0.9 and cryst_09 == num_layers:
            cryst_09 = l_idx
            
    digit_changes = sum(1 for i in range(1, len(top1_digit_per_layer)) if top1_digit_per_layer[i] != top1_digit_per_layer[i - 1])
    full_token_changes = sum(1 for i in range(1, len(top1_full_per_layer)) if top1_full_per_layer[i] != top1_full_per_layer[i - 1])
    
    x = np.arange(num_layers)
    dp_slope = np.polyfit(x, dp, 1)[0] if num_layers > 1 else 0.0
    fe_slope = np.polyfit(x, fe, 1)[0] if num_layers > 1 else 0.0
    hn_slope = np.polyfit(x, hn, 1)[0] if num_layers > 1 else 0.0
    dm_slope = np.polyfit(x, dm, 1)[0] if num_layers > 1 else 0.0

    return {
        "num_layers": num_layers,
        "prompt_len": prompt_len,
        "dp_mean": dp.mean(), "dp_std": dp.std(), "dp_max": dp.max(), "dp_min": dp.min(),
        "dp_slope": dp_slope,
        "dp_early": dp[:quarter].mean(), "dp_late": dp[-quarter:].mean(),
        "dp_early_late_diff": dp[-quarter:].mean() - dp[:quarter].mean(),
        "dp_first_half": dp[:half].mean(), "dp_second_half": dp[half:].mean(),
        "de_mean": de.mean(), "de_std": de.std(), "de_min": de.min(), "de_max": de.max(),
        "de_early": de[:quarter].mean(), "de_late": de[-quarter:].mean(),
        "fe_mean": fe.mean(), "fe_std": fe.std(), "fe_min": fe.min(), "fe_max": fe.max(),
        "fe_slope": fe_slope,
        "fe_early": fe[:quarter].mean(), "fe_late": fe[-quarter:].mean(),
        "fe_drop": fe[0] - fe[-1] if num_layers > 1 else 0.0,
        "hn_mean": hn.mean(), "hn_std": hn.std(), "hn_min": hn.min(), "hn_max": hn.max(),
        "hn_slope": hn_slope,
        "hn_early": hn[:quarter].mean(), "hn_late": hn[-quarter:].mean(),
        "dm_mean": dm.mean(), "dm_std": dm.std(), "dm_max": dm.max(), "dm_min": dm.min(),
        "dm_slope": dm_slope,
        "dm_early": dm[:quarter].mean(), "dm_late": dm[-quarter:].mean(),
        "kl_mean": kl.mean(), "kl_std": kl.std(), "kl_max": kl.max(),
        "fp_mean": fp.mean(), "fp_std": fp.std(), "fp_max": fp.max(), "fp_min": fp.min(),
        "fp_early": fp[:quarter].mean(), "fp_late": fp[-quarter:].mean(),
        "cryst_05_ratio": cryst_05 / num_layers,
        "cryst_09_ratio": cryst_09 / num_layers,
        "digit_changes": digit_changes,
        "digit_change_ratio": digit_changes / max(num_layers - 1, 1),
        "full_token_changes": full_token_changes,
        "full_token_change_ratio": full_token_changes / max(num_layers - 1, 1),
    }
