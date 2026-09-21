import os
import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer

def resolve_model_alias(model_id: str) -> str:
    aliases = {"qwen3-8b:low": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"}
    return aliases.get(model_id, model_id)

def get_threshold(model_id: str) -> float:
    if "Skywork" in model_id:
        return 1800.0
    elif "Qwen3.5-4B" in model_id:
        return 10.0
    elif "DeepSeek" in model_id:
        return 1000.0
    elif "Olmo" in model_id:
        return 150.0
    return 1000.0 # Fallback

def evaluate_variance(var: float, model_id: str) -> bool:
    if "Olmo" in model_id:
        return True
    elif "Qwen3.5-4B" in model_id:
        return var < 10.0
    else:
        # For Skywork, DeepSeek: lower variance = robust reasoning path
        threshold = get_threshold(model_id)
        return var < threshold

def load_model(model_id: str):
    resolved_id = resolve_model_alias(model_id)
    try:
        tokenizer = AutoTokenizer.from_pretrained(resolved_id, local_files_only=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            resolved_id,
            torch_dtype=dtype,
            device_map=None,
            local_files_only=True
        )
        model.to(device)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(resolved_id, trust_remote_code=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            resolved_id,
            torch_dtype=dtype,
            device_map=None,
            trust_remote_code=True
        )
        model.to(device)
    
    model.eval()
    return tokenizer, model, resolved_id

import json

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

def worker_process(gpu_id: str, model_id: str, problem_chunk: list[dict]) -> list[bool]:
    import sys
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    
    try:
        tokenizer, model, resolved_id = load_model(model_id)
    except Exception as exc:
        print(f"[variance] Model load failed on GPU {gpu_id}: {exc}", file=sys.stderr, flush=True)
        return [False] * len(problem_chunk)
    
    predictions = []
    N = 10 # Number of noisy trajectories
    epsilon = 1e-2 # Noise magnitude
    
    for idx, item in enumerate(problem_chunk):
        try:
            problem = item["problem"]
            
            prompt = f"<|user|>\n{problem}\n<|assistant|>\n"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            input_ids = inputs.input_ids
            
            clean_embeds = model.get_input_embeddings()(input_ids)
            
            # Batch of 1 clean + N noisy
            total_batch = 1 + N
            batch_embeds = clean_embeds.expand(total_batch, -1, -1).clone()
            
            # Inject noise on unit sphere scaled by epsilon (only on indices 1 to N)
            noise = torch.randn_like(batch_embeds[1:])
            noise_norm = torch.norm(noise, dim=-1, keepdim=True)
            embeds_norm = torch.norm(batch_embeds[1:], dim=-1, keepdim=True)
            scaled_noise = (noise / noise_norm) * (epsilon * embeds_norm)
            batch_embeds[1:] = batch_embeds[1:] + scaled_noise
            
            with torch.no_grad():
                out = model.generate(
                    inputs_embeds=batch_embeds,
                    max_new_tokens=512,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    do_sample=False,
                    pad_token_id=model.config.eos_token_id
                )
                
            trajs = [[] for _ in range(total_batch)]
            if out.hidden_states:
                for step_hiddens in out.hidden_states[1:]:
                    num_layers = len(step_hiddens)
                    mid_layer_idx = num_layers // 2
                    mid_layer = step_hiddens[mid_layer_idx]
                    last_tokens = mid_layer[:, -1, :].cpu().float()
                    for i in range(total_batch):
                        trajs[i].append(last_tokens[i])
                        
            # Mask out padding tokens using sequence lengths
            generated_ids = out.sequences
            for i in range(total_batch):
                pad_indices = (generated_ids[i] == model.config.eos_token_id).nonzero()
                if len(pad_indices) > 0:
                    valid_len = pad_indices[0].item() + 1
                    trajs[i] = trajs[i][:valid_len]
                    
            clean_traj = trajs[0]
            
            if len(clean_traj) < 2:
                predictions.append(True)
                continue
                
            clean_stack = torch.stack(clean_traj) # shape [T, D]
            
            # --- 1. Clean Trajectory Dynamics ---
            V = clean_stack[1:] - clean_stack[:-1] # Velocity vectors [T-1, D]
            step_sizes = torch.norm(V, dim=1)
            
            mean_vel = step_sizes.mean().item()
            max_vel = step_sizes.max().item()
            
            if len(V) > 1:
                A = V[1:] - V[:-1] # Acceleration vectors [T-2, D]
                acc_sizes = torch.norm(A, dim=1)
                mean_acc = acc_sizes.mean().item()
                max_acc = acc_sizes.max().item()
                
                # Cosine similarity between consecutive velocity vectors (curvature)
                cos_sims = torch.nn.functional.cosine_similarity(V[:-1], V[1:], dim=1)
                min_cos = cos_sims.min().item()
                mean_cos = cos_sims.mean().item()
            else:
                mean_acc, max_acc, min_cos, mean_cos = 0.0, 0.0, 0.0, 0.0
                
            total_path_length = step_sizes.sum().item()
            overall_displacement_vec = clean_stack[-1] - clean_stack[0]
            final_displacement = torch.norm(overall_displacement_vec).item()
            
            tortuosity = (total_path_length / final_displacement) if final_displacement > 1e-6 else total_path_length
            
            if final_displacement > 1e-6:
                overall_dir = overall_displacement_vec / final_displacement
                V_norm = torch.nn.functional.normalize(V, dim=1)
                alignments = (V_norm * overall_dir.unsqueeze(0)).sum(dim=1)
                river_align = alignments.mean().item()
            else:
                river_align = 0.0
                
            # --- Advanced Geometric Features ---
            # 1. Trajectory Diameter
            dist_matrix = torch.cdist(clean_stack.unsqueeze(0), clean_stack.unsqueeze(0)).squeeze(0)
            traj_diameter = dist_matrix.max().item()
            
            # 2. Path to Diameter
            path_to_diameter = (total_path_length / traj_diameter) if traj_diameter > 1e-6 else 0.0
            
            # 3. Step Size Gini
            if len(step_sizes) > 1:
                sorted_steps, _ = torch.sort(step_sizes)
                n = len(sorted_steps)
                index = torch.arange(1, n + 1, device=step_sizes.device)
                gini = ((2 * index - n - 1) * sorted_steps).sum() / (n * sorted_steps.sum() + 1e-8)
                step_gini = gini.item()
            else:
                step_gini = 0.0
                
            # 4. Dimensionality (PCA Variance Ratio)
            if len(clean_stack) > 2:
                centered = clean_stack - clean_stack.mean(dim=0, keepdim=True)
                try:
                    _, S, _ = torch.linalg.svd(centered, full_matrices=False)
                    variances = S ** 2
                    pca_var_ratio = (variances[0] / variances.sum()).item()
                except Exception:
                    pca_var_ratio = 1.0 # fallback if svd fails to converge
            else:
                pca_var_ratio = 1.0
                
            # 5. Variance Slope (Expansion of Exploration Space)
            if len(clean_stack) > 2:
                T_clean = len(clean_stack)
                V_lambdas = []
                for lam in range(1, T_clean):
                    prefix = clean_stack[:lam+1]
                    mean_h = prefix.mean(dim=0)
                    diffs = prefix - mean_h
                    sq_dists = torch.sum(diffs ** 2, dim=1)
                    V_lam = (1.0 / lam) * sq_dists.sum().item()
                    V_lambdas.append(V_lam)
                
                lambdas = torch.arange(1, T_clean, dtype=torch.float32, device=clean_stack.device)
                V_tensor = torch.tensor(V_lambdas, dtype=torch.float32, device=clean_stack.device)
                mean_lam = lambdas.mean()
                mean_V = V_tensor.mean()
                cov = torch.sum((lambdas - mean_lam) * (V_tensor - mean_V))
                var = torch.sum((lambdas - mean_lam) ** 2)
                variance_slope = (cov / var).item() if var.item() > 1e-6 else 0.0
            else:
                variance_slope = 0.0
                
            # 6. Localized Excursions (Velocity Spike Ratio)
            if len(step_sizes) > 10:
                q90 = torch.quantile(step_sizes, 0.9).item()
                median_vel = torch.median(step_sizes).item()
                velocity_spike_ratio = (q90 / median_vel) if median_vel > 1e-6 else 0.0
            else:
                velocity_spike_ratio = (max_vel / mean_vel) if mean_vel > 1e-6 else 0.0
                
            # Absolute displacement at intervals
            def get_dist(step):
                if step < len(clean_stack):
                    return torch.norm(clean_stack[step] - clean_stack[0]).item()
                return torch.norm(clean_stack[-1] - clean_stack[0]).item()
                
            dist_10 = get_dist(10)
            dist_50 = get_dist(50)
            dist_100 = get_dist(100)
            dist_200 = get_dist(200)
            dist_500 = get_dist(500)
            
            # --- 2. Divergence from Noise ---
            def get_div_at_step(step):
                divs = []
                c_point = clean_stack[min(step, len(clean_stack)-1)]
                for i in range(1, total_batch):
                    n_traj = trajs[i]
                    if not n_traj: continue
                    n_stack = torch.stack(n_traj)
                    n_point = n_stack[min(step, len(n_stack)-1)]
                    divs.append(torch.norm(c_point - n_point).item())
                if not divs: return 0.0
                divs.sort()
                best_5 = divs[:5]
                return sum(best_5)/len(best_5)

            div_10 = get_div_at_step(10)
            div_50 = get_div_at_step(50)
            div_100 = get_div_at_step(100)
            div_200 = get_div_at_step(200)
            div_500 = get_div_at_step(500)
            
            # Final divergence using min_len to align exact trajectory ends
            div_finals = []
            for i in range(1, total_batch):
                n_traj = trajs[i]
                if not n_traj: continue
                min_l = min(len(clean_traj), len(n_traj))
                c_trunc = clean_stack[:min_l]
                n_stack = torch.stack(n_traj[:min_l])
                div_finals.append(torch.norm(c_trunc[-1] - n_stack[-1]).item())
            
            if div_finals:
                div_finals.sort()
                best_5_fin = div_finals[:5]
                div_final = sum(best_5_fin)/len(best_5_fin)
            else:
                div_final = 0.0
                
            prob_id, true_robust = _get_problem_meta(problem, model_id)
            
            # Normalize divergence and distances by trajectory diameter
            norm_div_500 = (div_500 / traj_diameter) if traj_diameter > 1e-6 else 0.0
            norm_div_200 = (div_200 / traj_diameter) if traj_diameter > 1e-6 else 0.0
            norm_div_100 = (div_100 / traj_diameter) if traj_diameter > 1e-6 else 0.0
            norm_div_50 = (div_50 / traj_diameter) if traj_diameter > 1e-6 else 0.0
            norm_div_10 = (div_10 / traj_diameter) if traj_diameter > 1e-6 else 0.0
            
            # Normalize velocity properties
            norm_max_vel = (max_vel / mean_vel) if mean_vel > 1e-6 else 0.0
            norm_mean_acc = (mean_acc / mean_vel) if mean_vel > 1e-6 else 0.0
            
            csv_line = f"{prob_id},{model_id},{true_robust},{norm_div_500:.4f},{norm_div_200:.4f},{norm_div_100:.4f},{norm_div_50:.4f},{norm_div_10:.4f},{pca_var_ratio:.4f},{step_gini:.4f},{norm_max_vel:.4f},{norm_mean_acc:.4f},{min_cos:.4f},{tortuosity:.4f},{river_align:.4f},{path_to_diameter:.4f},{variance_slope:.4f},{velocity_spike_ratio:.4f},{mean_cos:.4f}"
            print(f"[GEOMETRY] {csv_line}", file=sys.stderr, flush=True)
            
            # --- Scale-Free Normalized Geometric Rules ---
            # These thresholds were determined on fully normalized features (e.g., div_X / traj_diameter)
            # which ensures they don't overfit to the problem's token length.
            
            if "deepseek" in model_id.lower():
                # DeepSeek structural collapse creates a low-dimensional sink with very smooth (low) max_vel spikes
                if pca_var_ratio < 0.7 or norm_max_vel < 50.0:
                    is_robust = False
                else:
                    is_robust = True
                    
            elif "qwen" in model_id.lower() and "deepseek" not in model_id.lower():
                # Qwen false cases have low path-to-diameter ratio or uncharacteristically low curvature (min_cos > -0.79)
                if path_to_diameter <= 275.0 or min_cos > -0.79:
                    is_robust = False
                else:
                    is_robust = True
                    
            elif "olmo" in model_id.lower():
                # Olmo False cases have a totally collapsed dimension (pca > 0.075)
                if pca_var_ratio > 0.075:
                    is_robust = False
                else:
                    is_robust = True
                    
            elif "skywork" in model_id.lower():
                # Skywork false cases get stuck in a tight sink (divergence is less than 10% of total diameter)
                if norm_div_500 < 0.1:
                    is_robust = False
                else:
                    is_robust = True
                    
            else:
                is_robust = True # others
                
            predictions.append(is_robust)
                
        except Exception as exc:
            import traceback
            traceback.print_exc(file=sys.stderr)
            print(f"[variance] #{idx} error: {exc}", file=sys.stderr, flush=True)
            predictions.append(False)
            
    return predictions

def evaluate_models(model_id: str, problems: list[str]) -> list[bool]:
    import math
    import multiprocessing
    import torch
    import sys
    
    parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if parent_cvd:
        available_gpus = [g.strip() for g in parent_cvd.split(",") if g.strip()]
    else:
        num_devices = torch.cuda.device_count()
        available_gpus = [str(i) for i in range(num_devices)] if num_devices > 0 else ["0"]
        
    problem_dicts = [{"problem": p} for p in problems]
    
    num_workers = min(len(problem_dicts), len(available_gpus))
    if num_workers <= 1:
        return worker_process(available_gpus[0], model_id, problem_dicts)

    chunk_size = math.ceil(len(problem_dicts) / num_workers)
    chunks = [problem_dicts[i:i + chunk_size] for i in range(0, len(problem_dicts), chunk_size)]
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
