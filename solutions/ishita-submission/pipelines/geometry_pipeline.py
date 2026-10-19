"""Geometry pipeline (Olmo route). Features and thresholds are unchanged from the
original; changes are: deterministic noise, no label lookup, optional soft deadline,
dtype/eos handling, dead code removed. Set AIMO_DEBUG=1 to print features to stderr."""
import math
import multiprocessing
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

N_NOISY = 10
EPSILON = 1e-2
MAX_NEW_TOKENS = int(os.environ.get("AIMO_GEO_TOKENS", 512))  # 512 in final-submission; geometric-submission used 4096
FALLBACK_ON_ERROR = False       # prediction for a problem whose processing crashed
FALLBACK_ON_DEADLINE = False    # prediction for problems skipped because time ran out


def resolve_model_alias(model_id: str) -> str:
    aliases = {"qwen3-8b:low": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"}
    return aliases.get(model_id, model_id)


def load_model(model_id: str):
    resolved_id = resolve_model_alias(model_id)
    tokenizer = AutoTokenizer.from_pretrained(resolved_id, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(
            resolved_id, dtype=dtype, device_map=None, local_files_only=True
        )
    except TypeError:  # older transformers
        model = AutoModelForCausalLM.from_pretrained(
            resolved_id, torch_dtype=dtype, device_map=None, local_files_only=True
        )
    model.to(device)
    model.eval()
    return tokenizer, model, resolved_id


def _eos_id(model):
    eos = model.config.eos_token_id
    if isinstance(eos, (list, tuple)):
        eos = eos[0]
    return eos


def classify(model_id, feats):
    """Per-model rules (unchanged). feats is a dict of scale-free features."""
    m = model_id.lower()
    if "deepseek" in m:
        return not (feats["pca_var_ratio"] < 0.7 or feats["norm_max_vel"] < 50.0)
    if "qwen" in m:
        return not (feats["path_to_diameter"] <= 275.0 or feats["min_cos"] > -0.79)
    if "olmo" in m:
        return not (feats["norm_div_100"] < 0.05)
    if "skywork" in m:
        return not (feats["norm_div_500"] < 0.1)
    return True


def worker_process(gpu_id, model_id, problem_chunk, deadline=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    try:
        tokenizer, model, _ = load_model(model_id)
    except Exception as exc:
        print(f"[geometry] model load failed on GPU {gpu_id}: {exc}", file=sys.stderr, flush=True)
        return [FALLBACK_ON_ERROR] * len(problem_chunk)

    eos_id = _eos_id(model)
    predictions = []
    total_batch = 1 + N_NOISY

    for idx, item in enumerate(problem_chunk):
        if deadline is not None and time.time() > deadline:
            predictions.append(FALLBACK_ON_DEADLINE)
            continue
        try:
            problem = item["problem"]
            prompt = f"<|user|>\n{problem}\n<|assistant|>\n"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            input_ids = inputs.input_ids

            clean_embeds = model.get_input_embeddings()(input_ids)
            batch_embeds = clean_embeds.expand(total_batch, -1, -1).clone()

            # Deterministic noise: seeded per problem index, independent of batch order.
            gen = torch.Generator(device=batch_embeds.device)
            gen.manual_seed(1234 + idx)
            noise = torch.randn(
                batch_embeds[1:].shape, generator=gen,
                device=batch_embeds.device, dtype=batch_embeds.dtype,
            )
            noise_norm = torch.norm(noise, dim=-1, keepdim=True)
            embeds_norm = torch.norm(batch_embeds[1:], dim=-1, keepdim=True)
            batch_embeds[1:] = batch_embeds[1:] + (noise / noise_norm) * (EPSILON * embeds_norm)

            with torch.no_grad():
                out = model.generate(
                    inputs_embeds=batch_embeds,
                    max_new_tokens=MAX_NEW_TOKENS,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    do_sample=False,
                    pad_token_id=eos_id,
                )

            trajs = [[] for _ in range(total_batch)]
            if out.hidden_states:
                for step_hiddens in out.hidden_states[1:]:
                    mid_layer = step_hiddens[len(step_hiddens) // 2]
                    last_tokens = mid_layer[:, -1, :].cpu().float()
                    for i in range(total_batch):
                        trajs[i].append(last_tokens[i])

            generated_ids = out.sequences
            for i in range(total_batch):
                pad_indices = (generated_ids[i] == eos_id).nonzero()
                if len(pad_indices) > 0:
                    trajs[i] = trajs[i][: pad_indices[0].item() + 1]

            clean_traj = trajs[0]
            if len(clean_traj) < 2:
                predictions.append(True)
                continue

            clean_stack = torch.stack(clean_traj)
            V = clean_stack[1:] - clean_stack[:-1]
            step_sizes = torch.norm(V, dim=1)
            mean_vel = step_sizes.mean().item()
            max_vel = step_sizes.max().item()

            if len(V) > 1:
                A = V[1:] - V[:-1]
                mean_acc = torch.norm(A, dim=1).mean().item()
                cos_sims = torch.nn.functional.cosine_similarity(V[:-1], V[1:], dim=1)
                min_cos = cos_sims.min().item()
            else:
                mean_acc, min_cos = 0.0, 0.0

            total_path_length = step_sizes.sum().item()
            dist_matrix = torch.cdist(clean_stack.unsqueeze(0), clean_stack.unsqueeze(0)).squeeze(0)
            traj_diameter = dist_matrix.max().item()
            path_to_diameter = (total_path_length / traj_diameter) if traj_diameter > 1e-6 else 0.0

            if len(step_sizes) > 1:
                sorted_steps, _ = torch.sort(step_sizes)
                n = len(sorted_steps)
                index = torch.arange(1, n + 1, device=step_sizes.device)
                step_gini = (((2 * index - n - 1) * sorted_steps).sum() / (n * sorted_steps.sum() + 1e-8)).item()
            else:
                step_gini = 0.0

            if len(clean_stack) > 2:
                centered = clean_stack - clean_stack.mean(dim=0, keepdim=True)
                try:
                    _, S, _ = torch.linalg.svd(centered, full_matrices=False)
                    variances = S ** 2
                    pca_var_ratio = (variances[0] / variances.sum()).item()
                except Exception:
                    pca_var_ratio = 1.0
            else:
                pca_var_ratio = 1.0

            noisy_stacks = [torch.stack(t) for t in trajs[1:] if t]  # stacked once, reused below

            def div_at_step(step):
                divs = []
                c_point = clean_stack[min(step, len(clean_stack) - 1)]
                for n_stack in noisy_stacks:
                    n_point = n_stack[min(step, len(n_stack) - 1)]
                    divs.append(torch.norm(c_point - n_point).item())
                if not divs:
                    return 0.0
                divs.sort()
                return sum(divs[:5]) / len(divs[:5])

            def nd(step):
                return (div_at_step(step) / traj_diameter) if traj_diameter > 1e-6 else 0.0

            feats = {
                "norm_div_500": nd(500), "norm_div_200": nd(200), "norm_div_100": nd(100),
                "norm_div_50": nd(50), "norm_div_10": nd(10),
                "pca_var_ratio": pca_var_ratio, "step_gini": step_gini,
                "norm_max_vel": (max_vel / mean_vel) if mean_vel > 1e-6 else 0.0,
                "norm_mean_acc": (mean_acc / mean_vel) if mean_vel > 1e-6 else 0.0,
                "min_cos": min_cos, "path_to_diameter": path_to_diameter,
            }
            if os.environ.get("AIMO_DEBUG"):
                print(f"[GEOMETRY] {idx},{model_id}," + ",".join(f"{k}={v:.4f}" for k, v in feats.items()),
                      file=sys.stderr, flush=True)

            predictions.append(bool(classify(model_id, feats)))

            del out, batch_embeds, clean_embeds, inputs, trajs, clean_stack, noisy_stacks
            torch.cuda.empty_cache()

        except Exception as exc:
            import traceback
            traceback.print_exc(file=sys.stderr)
            print(f"[geometry] #{idx} error: {exc}", file=sys.stderr, flush=True)
            predictions.append(FALLBACK_ON_ERROR)

    return predictions


def evaluate_models(model_id: str, problems: list, deadline=None) -> list:
    parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if parent_cvd:
        gpus = [g.strip() for g in parent_cvd.split(",") if g.strip()]
    else:
        n_dev = torch.cuda.device_count()
        gpus = [str(i) for i in range(n_dev)] if n_dev > 0 else ["0"]

    problem_dicts = [{"problem": p} for p in problems]
    num_workers = min(len(problem_dicts), len(gpus))
    if num_workers <= 1:
        return worker_process(gpus[0], model_id, problem_dicts, deadline)

    chunk_size = math.ceil(len(problem_dicts) / num_workers)
    chunks = [problem_dicts[i:i + chunk_size] for i in range(0, len(problem_dicts), chunk_size)]
    ctx = multiprocessing.get_context("spawn")
    predictions = []
    with ctx.Pool(processes=len(chunks)) as pool:
        results = [pool.apply_async(worker_process, (gpus[i], model_id, chunks[i], deadline))
                   for i in range(len(chunks))]
        for i, r in enumerate(results):
            try:
                predictions.extend(r.get())
            except Exception as exc:
                print(f"[geometry] chunk {i} failed: {exc!r}", file=sys.stderr, flush=True)
                predictions.extend([FALLBACK_ON_ERROR] * len(chunks[i]))
    return predictions
