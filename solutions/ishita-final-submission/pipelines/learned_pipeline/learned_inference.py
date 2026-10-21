"""Applies the one-feature rules written by tools/fit_rules.py (learned_rules.json).

One forward pass per problem (logit-lens features), then a single threshold test.
Models without a saved rule get the fallback answer.
"""
from __future__ import annotations

import json
import math
import multiprocessing
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))

import torch

FALLBACK_ON_ERROR = False
FALLBACK_ON_DEADLINE = False


def load_rule(model_id: str):
    try:
        with open(os.path.join(_HERE, "learned_rules.json"), encoding="utf-8") as f:
            return json.load(f).get(model_id)
    except Exception as exc:  # noqa: BLE001
        print(f"[learned] cannot read rules: {exc!r}", file=sys.stderr, flush=True)
        return None


def apply_rule(rule, features: dict) -> bool:
    v = float(features[rule["feature"]])
    return bool(v > rule["threshold"]) if rule["robust_if"] == "gt" else bool(v <= rule["threshold"])


def predict_robustness(model_id, problems, deadline=None):
    n = len(problems)
    rule = load_rule(model_id)
    if rule is None:
        print(f"[learned] no rule for {model_id}", file=sys.stderr, flush=True)
        return [FALLBACK_ON_ERROR] * n
    try:
        from pipelines.logit_pipeline import flip_inference as fi
        from pipelines.logit_pipeline.logit_features import extract_features_fast, get_digit_ids
        tokenizer, model, _ = fi.load_model(model_id)
        digit_ids = get_digit_ids(tokenizer, model.device)
    except Exception as exc:  # noqa: BLE001
        print(f"[learned] load failed: {exc!r}", file=sys.stderr, flush=True)
        return [FALLBACK_ON_ERROR] * n

    out = []
    for idx, problem in enumerate(problems):
        if deadline is not None and time.time() > deadline:
            out.append(FALLBACK_ON_DEADLINE)
            continue
        try:
            out.append(apply_rule(rule, extract_features_fast(model, tokenizer, problem, digit_ids)))
        except Exception as exc:  # noqa: BLE001
            print(f"[learned] #{idx} error: {exc!r}", file=sys.stderr, flush=True)
            out.append(FALLBACK_ON_ERROR)
    return out


def worker_process(gpu_id, model_id, chunk, deadline=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    return predict_robustness(model_id, chunk, deadline=deadline)


def predict_robustness_parallel(model_id, problems, deadline=None):
    parent = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if parent:
        gpus = [g.strip() for g in parent.split(",") if g.strip()]
    else:
        nd = torch.cuda.device_count()
        gpus = [str(i) for i in range(nd)] if nd > 0 else ["0"]
    workers = min(len(problems), len(gpus))
    if workers <= 1:
        return worker_process(gpus[0], model_id, problems, deadline)
    size = math.ceil(len(problems) / workers)
    chunks = [problems[i:i + size] for i in range(0, len(problems), size)]
    ctx = multiprocessing.get_context("spawn")
    preds = []
    with ctx.Pool(processes=len(chunks)) as pool:
        pending = [pool.apply_async(worker_process, (gpus[i], model_id, chunks[i], deadline)) for i in range(len(chunks))]
        for i, r in enumerate(pending):
            try:
                preds.extend(r.get())
            except Exception as exc:  # noqa: BLE001
                print(f"[learned] chunk {i} failed: {exc!r}", file=sys.stderr, flush=True)
                preds.extend([FALLBACK_ON_ERROR] * len(chunks[i]))
    return preds
