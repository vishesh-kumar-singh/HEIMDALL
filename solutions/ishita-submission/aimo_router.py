"""Router + safety guard for the AIMO Interpretability final submission.

Lives next to solution.py (not inside it) so that spawned child processes can
import it by a stable module name regardless of how the evaluator loads solution.py.

Routing is data-driven: routes.json (next to this file) can map exact model ids or model
families to a pipeline. Generate it from your validation data with evaluate_routes.py.
Pipelines: logit | geometry | precision | stability | const_true | const_false
"""
import json
import multiprocessing
import os
import queue
import signal
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# ----------------------------------------------------------------------------
# Budget settings. The evaluator gives 3600 s for ALL models and problems.
# ----------------------------------------------------------------------------
_T0 = time.time()
TOTAL_BUDGET_S = 3600
SAFETY_RESERVE_S = 240
MODEL_LOAD_ALLOWANCE_S = 150      # soft deadline = hard cap minus this
USE_SUBPROCESS_GUARD = True       # hard timeout via child process (set False to debug)

# Max seconds one are_robust() call may use, per pipeline (also bounded by what is left overall).
PIPELINE_CAP_S = {
    "logit": 600, "geometry": 900, "precision": 1400, "stability": 1400,
    "const_true": 5, "const_false": 5,
}

DEFAULT_ROUTES = {
    "by_model": {},
    "by_family": {"deepseek": "logit", "skywork": "stability", "olmo": "geometry", "qwen": "precision"},
    "default_pipeline": "const_false",   # models with no entry (e.g. gpt-oss, gemma)
    "fallback": False,                   # answer used if a pipeline crashes / times out
    "fallback_by_model": {},
}


def log(msg):
    print(f"[Router] {msg}", file=sys.stderr, flush=True)


def load_routes():
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


def route(model_id: str):
    """Return the pipeline name for a model id."""
    if model_id in ROUTES["by_model"]:
        return ROUTES["by_model"][model_id]
    m = model_id.lower()
    for key in ("deepseek", "skywork", "olmo", "qwen"):   # deepseek first: its name contains "Qwen3"
        if key in m and key in ROUTES["by_family"]:
            return ROUTES["by_family"][key]
    return ROUTES.get("default_pipeline", "const_false")


def fallback_value(model_id: str) -> bool:
    return bool(ROUTES["fallback_by_model"].get(model_id, ROUTES.get("fallback", False)))


def to_texts(problems):
    """Accept str, Problem(original_problem=...), or dict rows."""
    out = []
    for p in problems:
        if isinstance(p, str):
            out.append(p)
        elif isinstance(p, dict):
            out.append(p.get("original_problem") or p.get("problem"))
        else:
            out.append(getattr(p, "original_problem"))
    return out


def call_pipeline(name, model_id, texts, soft_deadline):
    """Lazy imports: a broken/missing pipeline only breaks its own models."""
    if name == "const_true":
        return [True] * len(texts)
    if name == "const_false":
        return [False] * len(texts)
    if name == "logit":
        from pipelines.logit_pipeline.flip_inference import predict_robustness_parallel as f
        return f(model_id, texts, flip_threshold=1, deadline=soft_deadline)
    if name == "stability":
        from pipelines.stability_profile.inference import predict_robustness_parallel as f
        return f(model_id, texts, deadline=soft_deadline)
    if name == "geometry":
        from pipelines.geometry_pipeline import evaluate_models as f
        return f(model_id, texts, deadline=soft_deadline)
    if name == "precision":
        from pipelines.precision_pipeline.precision_inference import predict_robustness_parallel as f
        return f(model_id, texts, deadline=soft_deadline)
    raise ValueError(f"unknown pipeline {name!r}")


def child_main(name, model_id, texts, soft_deadline, q):
    try:
        os.setsid()  # own process group, so a timeout can kill grandchildren (per-GPU pool workers) too
    except Exception:  # noqa: BLE001
        pass
    try:
        preds = call_pipeline(name, model_id, texts, soft_deadline)
        q.put(("ok", [bool(x) for x in preds]))
    except BaseException as exc:  # noqa: BLE001
        q.put(("err", repr(exc)))


def _validate(preds, n):
    if preds is None or len(preds) != n:
        return None
    return [bool(x) for x in preds]


def run_guarded(name, model_id, texts, budget_s, soft_deadline):
    n = len(texts)
    if name.startswith("const_"):
        return _validate(call_pipeline(name, model_id, texts, None), n)
    if not USE_SUBPROCESS_GUARD:
        try:
            return _validate(call_pipeline(name, model_id, texts, soft_deadline), n)
        except Exception as exc:  # noqa: BLE001
            log(f"{name} failed in-process: {exc!r}")
            return None

    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=child_main, args=(name, model_id, texts, soft_deadline, q))
    p.start()
    result = None
    try:
        status, payload = q.get(timeout=budget_s)
        if status == "ok":
            result = _validate(payload, n)
        else:
            log(f"{name} raised in child: {payload}")
    except queue.Empty:
        log(f"{name} exceeded {budget_s:.0f}s hard cap; terminating")
    finally:
        if p.is_alive():
            try:
                os.killpg(p.pid, signal.SIGKILL)   # whole tree, frees GPU memory
            except Exception:  # noqa: BLE001
                p.kill()
            p.join(10)
    return result


def are_robust(model_id: str, problems, reasoning_effort=None):
    texts = to_texts(problems)
    n = len(texts)
    name = route(model_id)
    fallback = [fallback_value(model_id)] * n

    if name.startswith("const_") or n == 0:
        return [name == "const_true"] * n if name.startswith("const_") else []

    remaining = TOTAL_BUDGET_S - SAFETY_RESERVE_S - (time.time() - _T0)
    if remaining < 60:
        log(f"only {remaining:.0f}s left; using fallback for {model_id}")
        return fallback
    budget = min(PIPELINE_CAP_S.get(name, 600), remaining)
    soft_deadline = time.time() + max(30.0, budget - MODEL_LOAD_ALLOWANCE_S)

    log(f"{model_id} -> {name} ({n} problems, hard cap {budget:.0f}s)")
    t = time.time()
    preds = run_guarded(name, model_id, texts, budget, soft_deadline)
    log(f"{model_id} finished in {time.time() - t:.0f}s ({'ok' if preds else 'FALLBACK'})")
    return preds if preds is not None else fallback
