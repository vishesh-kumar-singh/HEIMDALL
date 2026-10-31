"""Router for the AIMO Interpretability submission.

Routes models to their designated diagnostic pipeline:
- DeepSeek          -> precision (Mantissa precision masking pipeline)
- Skywork           -> precision (Mantissa precision masking pipeline)
- Olmo              -> precision (Mantissa precision masking pipeline)
- Qwen              -> cot_geometry (Kushagra's CoT geometry probe)
- openai/gpt-oss-120b -> error (Intentional error route for invalid predictions)

Deterministic execution, full problem evaluation without artificial timeouts.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

DEFAULT_ROUTES = {
    "by_model": {
        "openai/gpt-oss-120b": "error",
    },
    "by_family": {
        "deepseek": "precision",
        "skywork": "precision",
        "olmo": "precision",
        "qwen": "cot_geometry",
        "gpt-oss": "error",
        "120b": "error",
    },
    "default_pipeline": "error",
    "fallback": False,
}


def log(msg: str) -> None:
    print(f"[Router] {msg}", file=sys.stderr, flush=True)


def load_routes() -> dict:
    routes = json.loads(json.dumps(DEFAULT_ROUTES))
    try:
        with open(os.path.join(HERE, "routes.json"), encoding="utf-8") as f:
            data = json.load(f)
        for key in routes:
            if key in data:
                routes[key] = data[key]
    except FileNotFoundError:
        pass
    except Exception as exc:  # noqa: BLE001
        log(f"could not read routes.json ({exc!r}); using defaults")
    return routes


ROUTES = load_routes()


def route(model_id: str) -> str:
    """Return the pipeline name for a model id."""
    if "by_model" in ROUTES and model_id in ROUTES["by_model"]:
        return ROUTES["by_model"][model_id]
    m = model_id.lower()
    # Check deepseek first because DeepSeek model names contain 'qwen3'
    if "deepseek" in m:
        return ROUTES.get("by_family", {}).get("deepseek", "precision")
    for key in ("skywork", "olmo", "qwen", "gpt-oss", "120b"):
        if key in m and key in ROUTES.get("by_family", {}):
            return ROUTES["by_family"][key]
    return ROUTES.get("default_pipeline", "error")


def fallback_value(model_id: str) -> bool:
    if "fallback_by_model" in ROUTES and model_id in ROUTES["fallback_by_model"]:
        return bool(ROUTES["fallback_by_model"][model_id])
    return bool(ROUTES.get("fallback", False))


def to_texts(problems: Any) -> list[str]:
    """Accept str, Problem(original_problem=...), or dict rows."""
    out = []
    for p in problems:
        if isinstance(p, str):
            out.append(p)
        elif isinstance(p, dict):
            out.append(p.get("original_problem") or p.get("problem"))
        else:
            out.append(getattr(p, "original_problem", str(p)))
    return out


def call_pipeline(name: str, model_id: str, texts: list[str], soft_deadline=None) -> list[bool]:
    """Lazy imports: a broken/missing pipeline only breaks its own models."""
    if name in ("error", "invalid"):
        raise ValueError(f"Intentional error: model '{model_id}' routed to '{name}' for invalid predictions.")
    if name == "const_true":
        return [True] * len(texts)
    if name == "const_false":
        return [False] * len(texts)
    if name == "cot_geometry":
        from pipelines.cot_geometry import predict_robustness_pipeline as f
        return f(model_id, texts, deadline=soft_deadline)
    if name == "geometry":
        from pipelines.geometry_pipeline import evaluate_models as f
        return f(model_id, texts, deadline=soft_deadline)
    if name == "precision":
        from pipelines.precision_pipeline import predict_robustness_parallel as f
        return f(model_id, texts, deadline=soft_deadline)
    raise ValueError(f"unknown pipeline {name!r}")


def _validate(preds: Any, n: int) -> list[bool] | None:
    if preds is None or len(preds) != n:
        return None
    return [bool(x) for x in preds]


def are_robust(model_id: str, problems: Any, reasoning_effort: Any = None) -> list[bool]:
    os.environ["AIMO_EFFORT"] = reasoning_effort if isinstance(reasoning_effort, str) else "default"
    texts = to_texts(problems)
    n = len(texts)
    name = route(model_id)

    if name in ("error", "invalid"):
        log(f"{model_id} -> {name}: raising error for invalid predictions.")
        raise ValueError(f"UNRECOGNIZED / INVALID MODEL: {model_id}. Routed to {name}.")

    if name.startswith("const_") or n == 0:
        return [name == "const_true"] * n if name.startswith("const_") else []

    log(f"{model_id} -> {name} ({n} problems)")
    t0 = time.time()
    try:
        raw_preds = call_pipeline(name, model_id, texts, soft_deadline=None)
        preds = _validate(raw_preds, n)
    except Exception as exc:
        log(f"{name} pipeline execution failed: {exc!r}")
        traceback.print_exc(file=sys.stderr)
        preds = None

    elapsed = time.time() - t0
    log(f"{model_id} finished in {elapsed:.1f}s ({'ok' if preds is not None else 'FALLBACK'})")
    return preds if preds is not None else [fallback_value(model_id)] * n
