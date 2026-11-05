#!/usr/bin/env python3
"""Extract and calibrate the compact CoT trajectory-geometry probe.

The extractor deliberately shares the prompt construction, layer selection, feature
calculation, and prefix behaviour of ``heimdall.bifrost.pipelines.cot_geometry``.
It generates one greedy 128-token rollout per case and saves the four features for
each of the 64, 96, and 128-token prefixes.

The ``fit`` command implements the objective and two-stage optimisation protocol in
Appendix ``app:appendix_probe_hyperparams`` of report.tex.  It writes every
regularisation candidate as well as the selected one, so calibration choices remain
auditable rather than being hidden in a Python module.

Examples:
    # Extract labelled features from a competition-style dataset.
    uv run python scripts/cot_geometry_calibration.py extract \
        --cases data/train-sample/input/cases.jsonl \
        --labels data/train-sample/reference/labels.jsonl \
        --output artifacts/cot_geometry_features.jsonl

    # Fit the model-wise probes described in the report.
    uv run python scripts/cot_geometry_calibration.py fit \
        --features artifacts/cot_geometry_features.jsonl \
        --output artifacts/cot_geometry_weights.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# These helpers are the source of truth used by the submitted Bifrost solution.
from heimdall.bifrost.pipelines import cot_geometry as submitted_probe


PROBE_BUDGETS = (64, 96, 128)
FEATURE_NAMES = ("mean_kappa", "min_kappa", "neg_mean_beta", "neg_final_beta")
LAMBDA_GRID = (0.0, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 0.5)
EPSILON = 1e-7


def _set_seed(seed: int) -> None:
    """Match the deterministic seed policy of the submitted probe."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    except FileNotFoundError as exc:
        raise SystemExit(f"file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSONL in {path}: {exc}") from exc


def _labels_for_cases(
    cases: list[dict[str, Any]], labels_path: Path | None
) -> list[bool | None]:
    """Load labels from cases or a companion labels JSONL file.

    Matching by row preserves datasets that intentionally have repeated case IDs.
    If the label rows are not aligned, a unique ID lookup is used as a fallback.
    """
    labels_from_cases = [case.get("is_robust") for case in cases]
    if all(value is not None for value in labels_from_cases):
        return [bool(value) for value in labels_from_cases]
    if labels_path is None:
        return [bool(value) if value is not None else None for value in labels_from_cases]

    label_rows = _read_jsonl(labels_path)
    aligned = len(label_rows) == len(cases) and all(
        not case.get("id") or not row.get("id") or case["id"] == row["id"]
        for case, row in zip(cases, label_rows)
    )
    if aligned:
        return [
            bool(case["is_robust"]) if case.get("is_robust") is not None
            else (bool(row["is_robust"]) if row.get("is_robust") is not None else None)
            for case, row in zip(cases, label_rows)
        ]

    by_id = {
        str(row["id"]): bool(row["is_robust"])
        for row in label_rows
        if row.get("id") is not None and row.get("is_robust") is not None
    }
    return [
        bool(case["is_robust"]) if case.get("is_robust") is not None
        else by_id.get(str(case.get("id")))
        for case in cases
    ]


def _load_model(
    model_id: str, *, device: str, allow_download: bool
) -> tuple[Any, Any, Any]:
    """Load a model with the same dtype and target-layer selection as Bifrost."""
    repo_id = submitted_probe.clean_model_id(model_id)
    resolved_device = torch.device(device)
    dtype = (
        torch.bfloat16
        if resolved_device.type == "cuda" and torch.cuda.is_bf16_supported()
        else (torch.float16 if resolved_device.type == "cuda" else torch.float32)
    )
    load_options: dict[str, Any] = {}
    if not allow_download:
        load_options["local_files_only"] = True

    tokenizer = AutoTokenizer.from_pretrained(repo_id, **load_options)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(repo_id, dtype=dtype, **load_options)
    except TypeError:  # transformers versions before the dtype alias
        model = AutoModelForCausalLM.from_pretrained(repo_id, torch_dtype=dtype, **load_options)
    model.to(resolved_device)
    model.eval()
    return tokenizer, model, submitted_probe._find_target_module(model)


def _features_from_states(full_hs: np.ndarray, prompt_len: int) -> dict[str, dict[str, Any]]:
    """Use the submitted probe's exact prefix fallback and feature calculation."""
    generated_states = full_hs[prompt_len:]
    by_budget: dict[str, dict[str, Any]] = {}
    for budget in PROBE_BUDGETS:
        states = generated_states[:budget]
        # This is intentionally identical to probs_from_full_states().
        if len(states) < 4:
            states = full_hs
        checkpoints = min(6, max(4, len(states) // 16))
        indices = np.linspace(0, len(states) - 1, checkpoints, dtype=int)
        mean_kappa, min_kappa, mean_beta, final_beta = submitted_probe._compute_trajectory_metrics(
            states[indices]
        )
        values = [mean_kappa, min_kappa, -mean_beta, -final_beta]
        by_budget[str(budget)] = {
            "checkpoint_count": checkpoints,
            "checkpoint_indices": indices.tolist(),
            "mean_kappa": float(mean_kappa),
            "min_kappa": float(min_kappa),
            "neg_mean_beta": float(-mean_beta),
            "neg_final_beta": float(-final_beta),
            "values": [float(value) for value in values],
        }
    return by_budget


def _extract_case(
    case: dict[str, Any], tokenizer: Any, model: Any, target_module: Any
) -> tuple[dict[str, dict[str, Any]], int, int]:
    """Run the same single-rollout hook sequence used in probe_all_budgets()."""
    prompt, add_special_tokens = submitted_probe._build_prompt(tokenizer, str(case["problem"]))
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=add_special_tokens)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    prompt_len = int(inputs["input_ids"].shape[1])

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max(PROBE_BUDGETS),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    captured: dict[str, torch.Tensor] = {}

    def hook_fn(_module: Any, _input: Any, output: Any) -> None:
        captured["hidden_states"] = (
            output[0] if isinstance(output, tuple) else output
        ).detach().cpu()

    handle = target_module.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(generated)
    finally:
        handle.remove()

    full_hs = captured["hidden_states"][0].float().numpy()
    generated_count = max(0, int(full_hs.shape[0]) - prompt_len)
    return _features_from_states(full_hs, prompt_len), prompt_len, generated_count


def extract(args: argparse.Namespace) -> None:
    if args.dataset is not None:
        cases_path = args.dataset / "input" / "cases.jsonl"
        labels_path = args.labels or args.dataset / "reference" / "labels.jsonl"
        output_path = args.output or args.dataset / "cot_geometry_features.jsonl"
    else:
        cases_path = args.cases
        labels_path = args.labels
        output_path = args.output
    if output_path is None:  # guarded by argparse for --cases; keeps type checkers honest.
        raise SystemExit("--output is required when using --cases")

    cases = _read_jsonl(cases_path)
    if not cases:
        raise SystemExit("cases file has no rows")
    for index, case in enumerate(cases):
        missing = {"model_id", "problem"}.difference(case)
        if missing:
            raise SystemExit(f"case {index} is missing {', '.join(sorted(missing))}")

    labels = _labels_for_cases(cases, labels_path if labels_path and labels_path.exists() else None)
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"output already exists: {output_path}; pass --overwrite to replace it")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _set_seed(args.seed)

    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for row_index, case in enumerate(cases):
        grouped[str(case["model_id"])].append((row_index, case))

    total = len(cases)
    with output_path.open("w", encoding="utf-8") as stream:
        for model_id, model_cases in grouped.items():
            print(f"Loading {model_id} for {len(model_cases)} case(s)", file=sys.stderr, flush=True)
            tokenizer, model, target_module = _load_model(
                model_id, device=args.device, allow_download=args.allow_download
            )
            try:
                for position, (row_index, case) in enumerate(model_cases, start=1):
                    features, prompt_tokens, generated_tokens = _extract_case(
                        case, tokenizer, model, target_module
                    )
                    record = {
                        "schema_version": 1,
                        "case_id": str(case.get("id", f"case-{row_index:05d}")),
                        "case_index": row_index,
                        "model_id": model_id,
                        "reasoning_effort": case.get("reasoning_effort", "default"),
                        "is_robust": labels[row_index],
                        "prompt_token_count": prompt_tokens,
                        "generated_token_count": generated_tokens,
                        "max_new_tokens": max(PROBE_BUDGETS),
                        "budgets": list(PROBE_BUDGETS),
                        "feature_names": list(FEATURE_NAMES),
                        "features": features,
                    }
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                    stream.flush()
                    print(
                        f"[{model_id}] {position}/{len(model_cases)}: {record['case_id']}",
                        file=sys.stderr,
                        flush=True,
                    )
            finally:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print(f"Wrote {total} feature rows to {output_path}", file=sys.stderr)


def _model_family(model_id: str) -> str:
    """Match get_model_probe_weights() dispatch precedence in the submission."""
    normalized = model_id.lower()
    if "deepseek" in normalized:
        return "deepseek"
    if "qwen" in normalized:
        return "qwen"
    if "skywork" in normalized:
        return "skywork"
    if "olmo" in normalized:
        return "olmo"
    return "default"


def _stable_sigmoid(logits: np.ndarray) -> np.ndarray:
    bounded = np.clip(logits, -15.0, 15.0)
    return 1.0 / (1.0 + np.exp(-bounded))


def _objective(theta: np.ndarray, features: np.ndarray, labels: np.ndarray, l2: float) -> float:
    weights, bias = theta[:4], theta[4]
    probabilities = _stable_sigmoid(np.einsum("nbf,f->nb", features, weights) + bias).mean(axis=1)
    probabilities = np.clip(probabilities, EPSILON, 1.0 - EPSILON)
    cross_entropy = -np.mean(
        labels * np.log(probabilities) + (1.0 - labels) * np.log(1.0 - probabilities)
    )
    return float(cross_entropy + l2 * np.dot(weights, weights))


def _metrics(theta: np.ndarray, features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    probabilities = _stable_sigmoid(np.einsum("nbf,f->nb", features, theta[:4]) + theta[4]).mean(axis=1)
    predictions = probabilities >= 0.5
    accuracy = float(np.mean(predictions == labels))
    positive = labels == 1
    negative = ~positive
    true_recall = float(np.mean(predictions[positive])) if np.any(positive) else float("nan")
    false_recall = float(np.mean(~predictions[negative])) if np.any(negative) else float("nan")
    balanced_accuracy = float(np.nanmean([true_recall, false_recall]))
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "true_recall": true_recall,
        "false_recall": false_recall,
    }


def _initial_points(
    baseline: np.ndarray, global_theta: np.ndarray | None, rng: np.random.Generator
) -> list[np.ndarray]:
    """Ten starts: baseline, global fit where applicable, then Gaussian starts."""
    points = [baseline.copy()]
    if global_theta is not None:
        points.append(global_theta.copy())
    while len(points) < 10:
        points.append(rng.normal(loc=0.0, scale=0.5, size=5))
    return points


def _optimise(
    features: np.ndarray,
    labels: np.ndarray,
    l2: float,
    starts: Iterable[np.ndarray],
    maxiter: int,
    bound: float,
) -> tuple[np.ndarray, float]:
    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise SystemExit(
            "SciPy is required for fitting. Run `uv sync` after pulling this revision."
        ) from exc

    objective = lambda theta: _objective(theta, features, labels, l2)
    simplex_results = []
    for start in starts:
        result = minimize(
            objective,
            np.asarray(start, dtype=np.float64),
            method="Nelder-Mead",
            options={"maxiter": maxiter, "fatol": 1e-10, "xatol": 1e-8},
        )
        simplex_results.append(result)

    # The appendix refers to the top simplex candidates. Polishing every one avoids
    # an undocumented ranking cutoff while retaining the specified two-stage search.
    polished = []
    bounds = [(-bound, bound)] * 5
    for result in sorted(simplex_results, key=lambda item: float(item.fun)):
        polished.append(
            minimize(
                objective,
                result.x,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 20_000, "ftol": 1e-15, "gtol": 1e-10},
            )
        )
    best = min(polished, key=lambda item: float(item.fun))
    return np.asarray(best.x, dtype=np.float64), float(best.fun)


def _records_to_arrays(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    feature_rows = []
    labels = []
    for record in records:
        if record.get("is_robust") is None:
            raise SystemExit(
                "fitting requires is_robust labels in every feature record; "
                "rerun extract with --labels or use labelled cases"
            )
        try:
            feature_rows.append(
                [record["features"][str(budget)]["values"] for budget in PROBE_BUDGETS]
            )
        except (KeyError, TypeError) as exc:
            raise SystemExit(
                f"feature record {record.get('case_id', '<unknown>')} does not match extractor schema"
            ) from exc
        labels.append(float(bool(record["is_robust"])))
    features = np.asarray(feature_rows, dtype=np.float64)
    if features.ndim != 3 or features.shape[1:] != (3, 4) or not np.isfinite(features).all():
        raise SystemExit("features must be finite arrays of shape (cases, 3 budgets, 4 features)")
    return features, np.asarray(labels, dtype=np.float64)


def fit(args: argparse.Namespace) -> None:
    records = _read_jsonl(args.features)
    if not records:
        raise SystemExit("feature file has no rows")
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"output already exists: {args.output}; pass --overwrite to replace it")
    _set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    baseline = np.asarray(args.baseline, dtype=np.float64)
    if baseline.shape != (5,):
        raise SystemExit("--baseline must contain exactly five values: w1 w2 w3 w4 bias")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if "model_id" not in record:
            raise SystemExit("each feature record needs model_id")
        grouped[_model_family(str(record["model_id"]))].append(record)

    global_features, global_labels = _records_to_arrays(records)
    global_by_lambda: dict[float, np.ndarray] = {}
    for l2 in LAMBDA_GRID:
        theta, _loss = _optimise(
            global_features,
            global_labels,
            l2,
            _initial_points(baseline, None, rng),
            args.maxiter,
            args.bound,
        )
        global_by_lambda[l2] = theta

    report: dict[str, Any] = {
        "schema_version": 1,
        "feature_names": list(FEATURE_NAMES),
        "budgets": list(PROBE_BUDGETS),
        "protocol": {
            "objective": "mean-of-sigmoids BCE + lambda * ||w||^2; bias unregularized",
            "seed": args.seed,
            "nelder_mead_maxiter": args.maxiter,
            "nelder_mead_fatol": 1e-10,
            "nelder_mead_xatol": 1e-8,
            "random_starts": 10,
            "gaussian_start_stddev": 0.5,
            "l_bfgs_b_maxiter": 20_000,
            "l_bfgs_b_bound": args.bound,
            "lambda_grid": list(LAMBDA_GRID),
            "selection": args.select,
            "baseline": baseline.tolist(),
        },
        "global_initialisations": {
            str(l2): global_by_lambda[l2].tolist() for l2 in LAMBDA_GRID
        },
        "models": {},
    }

    selection_key = "loss" if args.select == "loss" else args.select.replace("-", "_")
    for family, family_records in sorted(grouped.items()):
        features, labels = _records_to_arrays(family_records)
        candidates = []
        for l2 in LAMBDA_GRID:
            theta, loss = _optimise(
                features,
                labels,
                l2,
                _initial_points(baseline, global_by_lambda[l2], rng),
                args.maxiter,
                args.bound,
            )
            metrics = _metrics(theta, features, labels)
            candidates.append(
                {
                    "l2": l2,
                    "weights": theta[:4].tolist(),
                    "bias": float(theta[4]),
                    "loss": loss,
                    **metrics,
                }
            )

        if args.select == "loss":
            selected = min(candidates, key=lambda candidate: candidate["loss"])
        else:
            # Prefer the specified score, then the lower regularised loss for ties.
            selected = max(
                candidates,
                key=lambda candidate: (candidate[selection_key], -candidate["loss"]),
            )
        report["models"][family] = {
            "case_count": len(family_records),
            "robust_count": int(labels.sum()),
            "spurious_count": int(len(labels) - labels.sum()),
            "selected": selected,
            "candidates": candidates,
        }
        print(
            f"{family}: l2={selected['l2']}, accuracy={selected['accuracy']:.3f}, "
            f"weights={selected['weights']}, bias={selected['bias']:.8f}",
            file=sys.stderr,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote calibration report to {args.output}", file=sys.stderr)


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _dataset_path(value: str) -> Path:
    """Accept either ``data/train-sample`` or the shorter ``train-sample``."""
    provided = Path(value).expanduser()
    if provided.exists():
        return provided.resolve()
    return (ROOT / "data" / provided).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    extract_parser = commands.add_parser("extract", help="generate and save CoT geometry features")
    source = extract_parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--dataset", type=_dataset_path,
        help="dataset folder, for example train-sample or data/train-sample; resolves input/cases.jsonl",
    )
    source.add_argument("--cases", type=_path, help="JSONL cases with model_id and problem")
    extract_parser.add_argument("--labels", type=_path, help="optional JSONL labels with id and is_robust")
    extract_parser.add_argument(
        "--output", type=_path,
        help="output JSONL feature file; defaults to DATASET/cot_geometry_features.jsonl",
    )
    extract_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    extract_parser.add_argument("--seed", type=int, default=42)
    extract_parser.add_argument("--allow-download", action="store_true", help="allow Hugging Face downloads")
    extract_parser.add_argument("--overwrite", action="store_true")
    extract_parser.set_defaults(handler=extract)

    fit_parser = commands.add_parser("fit", help="fit model-wise probe weights from extracted features")
    fit_parser.add_argument("--features", type=_path, required=True, help="JSONL produced by extract")
    fit_parser.add_argument("--output", type=_path, required=True, help="calibration JSON output")
    fit_parser.add_argument("--seed", type=int, default=42)
    fit_parser.add_argument("--maxiter", type=int, default=30_000)
    fit_parser.add_argument(
        "--bound", type=float, default=1_000.0,
        help="symmetric L-BFGS-B parameter bound; must contain the desired solution",
    )
    fit_parser.add_argument(
        "--baseline", type=float, nargs=5, default=(0.0, 0.0, 0.0, 0.0, 0.0),
        metavar=("W1", "W2", "W3", "W4", "BIAS"),
        help="baseline initialisation used by the first Nelder-Mead trial",
    )
    fit_parser.add_argument(
        "--select", choices=("loss", "accuracy", "balanced-accuracy"), default="loss",
        help="criterion for selecting one regularisation candidate per model family",
    )
    fit_parser.add_argument("--overwrite", action="store_true")
    fit_parser.set_defaults(handler=fit)

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    arguments.handler(arguments)
