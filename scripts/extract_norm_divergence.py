#!/usr/bin/env python3
"""Collect normalized embedding-perturbation divergence features.

This is the data-collection counterpart of Bifrost's
``pipelines/geometry_pipeline.py``.  For every case it uses the production prompt,
one clean plus ten 1%-spherically perturbed embedding trajectories, greedy 512-token
generation, the middle transformer layer, and the mean of the five closest noisy
trajectories.  It writes one CSV row per case, including all five normalized
divergence milestones used in report.tex.

Examples:
    # The usual form; resolves data/train-sample/input/cases.jsonl and labels.
    uv run python scripts/extract_norm_divergence.py --dataset train-sample

    # A custom dataset and output location.
    uv run python scripts/extract_norm_divergence.py \
        --dataset data/full-small-labeled \
        --output artifacts/full_small_norm_divergence.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Use the submitted implementation for model loading and its exact model aliases.
from heimdall.bifrost.pipelines import geometry_pipeline as submitted_pipeline


MILESTONES = (10, 50, 100, 200, 500)

CSV_FIELDS = (
    "case_id",
    "case_index",
    "model_id",
    "reasoning_effort",
    "is_robust",
    "noise_seed",
    "epsilon",
    "n_noisy",
    "max_new_tokens",
    "trajectory_length",
    "trajectory_diameter",
    "div_10",
    "div_50",
    "div_100",
    "div_200",
    "div_500",
    "norm_div_10",
    "norm_div_50",
    "norm_div_100",
    "norm_div_200",
    "norm_div_500",
    "pca_var_ratio",
    "step_gini",
    "norm_max_vel",
    "norm_mean_acc",
    "min_cos",
    "path_to_diameter",
    "status",
    "error",
)


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _dataset_path(value: str) -> Path:
    provided = Path(value).expanduser()
    if provided.exists():
        return provided.resolve()
    return (ROOT / "data" / provided).resolve()


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
    in_cases = [case.get("is_robust") for case in cases]
    if all(value is not None for value in in_cases):
        return [bool(value) for value in in_cases]
    if labels_path is None:
        return [bool(value) if value is not None else None for value in in_cases]
    rows = _read_jsonl(labels_path)
    aligned = len(rows) == len(cases) and all(
        not case.get("id") or not row.get("id") or case["id"] == row["id"]
        for case, row in zip(cases, rows)
    )
    if aligned:
        return [
            bool(case["is_robust"]) if case.get("is_robust") is not None
            else (bool(row["is_robust"]) if row.get("is_robust") is not None else None)
            for case, row in zip(cases, rows)
        ]
    by_id = {
        str(row["id"]): bool(row["is_robust"])
        for row in rows
        if row.get("id") is not None and row.get("is_robust") is not None
    }
    return [
        bool(case["is_robust"]) if case.get("is_robust") is not None
        else by_id.get(str(case.get("id")))
        for case in cases
    ]


def _eos_id(model: Any) -> int:
    eos = model.config.eos_token_id
    return int(eos[0] if isinstance(eos, (tuple, list)) else eos)


def _empty_row(case: dict[str, Any], case_index: int, label: bool | None, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "case_id": str(case.get("id", f"case-{case_index:05d}")),
        "case_index": case_index,
        "model_id": str(case["model_id"]),
        "reasoning_effort": case.get("reasoning_effort", "default"),
        "is_robust": label,
        "noise_seed": args.seed + case_index,
        "epsilon": args.epsilon,
        "n_noisy": args.n_noisy,
        "max_new_tokens": args.max_new_tokens,
        "status": "error",
        "error": "",
    }


def _extract_case(
    case: dict[str, Any],
    case_index: int,
    label: bool | None,
    tokenizer: Any,
    model: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Feature extraction copied from the Bifrost geometry pipeline.

    The case index is the per-model position.  That is the seed used by the
    submitted single-worker pipeline (1234 + idx), which makes this collector
    deterministic and directly comparable to a single-GPU submission run.
    """
    row = _empty_row(case, case_index, label, args)
    row["noise_seed"] = args.seed + case_index
    try:
        prompt = f"<|user|>\n{case['problem']}\n<|assistant|>\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        clean_embeds = model.get_input_embeddings()(inputs.input_ids)
        total_batch = 1 + args.n_noisy
        batch_embeds = clean_embeds.expand(total_batch, -1, -1).clone()

        # The submitted pipeline uses independently sampled, norm-scaled noise for
        # every token and each of its ten perturbed trajectories.
        generator = torch.Generator(device=batch_embeds.device)
        generator.manual_seed(row["noise_seed"])
        noise = torch.randn(
            batch_embeds[1:].shape,
            generator=generator,
            device=batch_embeds.device,
            dtype=batch_embeds.dtype,
        )
        noise_norm = torch.norm(noise, dim=-1, keepdim=True)
        embed_norm = torch.norm(batch_embeds[1:], dim=-1, keepdim=True)
        batch_embeds[1:] = batch_embeds[1:] + (noise / noise_norm) * (args.epsilon * embed_norm)

        eos_id = _eos_id(model)
        with torch.no_grad():
            output = model.generate(
                inputs_embeds=batch_embeds,
                max_new_tokens=args.max_new_tokens,
                return_dict_in_generate=True,
                output_hidden_states=True,
                do_sample=False,
                pad_token_id=eos_id,
            )

        trajectories: list[list[torch.Tensor]] = [[] for _ in range(total_batch)]
        if output.hidden_states:
            for step_hidden_states in output.hidden_states[1:]:
                middle = step_hidden_states[len(step_hidden_states) // 2]
                final_tokens = middle[:, -1, :].detach().cpu().float()
                for index in range(total_batch):
                    trajectories[index].append(final_tokens[index])

        # Preserve the solution's early-EOS trimming semantics.
        for index in range(total_batch):
            eos_positions = (output.sequences[index] == eos_id).nonzero()
            if len(eos_positions) > 0:
                trajectories[index] = trajectories[index][: eos_positions[0].item() + 1]

        clean_trajectory = trajectories[0]
        if len(clean_trajectory) < 2:
            row.update(status="insufficient_trajectory", error="fewer than two generated hidden states")
            return row

        clean_states = torch.stack(clean_trajectory)
        velocity = clean_states[1:] - clean_states[:-1]
        step_sizes = torch.norm(velocity, dim=1)
        mean_velocity = step_sizes.mean().item()
        max_velocity = step_sizes.max().item()
        if len(velocity) > 1:
            acceleration = velocity[1:] - velocity[:-1]
            mean_acceleration = torch.norm(acceleration, dim=1).mean().item()
            min_cos = torch.nn.functional.cosine_similarity(velocity[:-1], velocity[1:], dim=1).min().item()
        else:
            mean_acceleration, min_cos = 0.0, 0.0

        path_length = step_sizes.sum().item()
        distance_matrix = torch.cdist(clean_states.unsqueeze(0), clean_states.unsqueeze(0)).squeeze(0)
        diameter = distance_matrix.max().item()
        path_to_diameter = path_length / diameter if diameter > 1e-6 else 0.0

        if len(step_sizes) > 1:
            sorted_steps, _ = torch.sort(step_sizes)
            count = len(sorted_steps)
            ranks = torch.arange(1, count + 1, device=step_sizes.device)
            step_gini = (((2 * ranks - count - 1) * sorted_steps).sum() / (count * sorted_steps.sum() + 1e-8)).item()
        else:
            step_gini = 0.0

        if len(clean_states) > 2:
            centered = clean_states - clean_states.mean(dim=0, keepdim=True)
            try:
                _, singular_values, _ = torch.linalg.svd(centered, full_matrices=False)
                variances = singular_values**2
                pca_var_ratio = (variances[0] / variances.sum()).item()
            except Exception:
                pca_var_ratio = 1.0
        else:
            pca_var_ratio = 1.0

        noisy_states = [torch.stack(trajectory) for trajectory in trajectories[1:] if trajectory]

        def divergence_at(step: int) -> float:
            clean_point = clean_states[min(step, len(clean_states) - 1)]
            distances = [
                torch.norm(clean_point - trajectory[min(step, len(trajectory) - 1)]).item()
                for trajectory in noisy_states
            ]
            return sum(sorted(distances)[:5]) / min(5, len(distances)) if distances else 0.0

        raw_divergence = {step: divergence_at(step) for step in MILESTONES}
        row.update(
            trajectory_length=len(clean_trajectory),
            trajectory_diameter=diameter,
            pca_var_ratio=pca_var_ratio,
            step_gini=step_gini,
            norm_max_vel=max_velocity / mean_velocity if mean_velocity > 1e-6 else 0.0,
            norm_mean_acc=mean_acceleration / mean_velocity if mean_velocity > 1e-6 else 0.0,
            min_cos=min_cos,
            path_to_diameter=path_to_diameter,
            status="ok",
        )
        for step, raw_value in raw_divergence.items():
            row[f"div_{step}"] = raw_value
            row[f"norm_div_{step}"] = raw_value / diameter if diameter > 1e-6 else 0.0
        return row
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main(args: argparse.Namespace) -> None:
    cases_path = args.dataset / "input" / "cases.jsonl"
    labels_path = args.labels or args.dataset / "reference" / "labels.jsonl"
    output_path = args.output or args.dataset / "norm_divergence_features.csv"
    cases = _read_jsonl(cases_path)
    if not cases:
        raise SystemExit("cases file has no rows")
    for index, case in enumerate(cases):
        missing = {"model_id", "problem"}.difference(case)
        if missing:
            raise SystemExit(f"case {index} is missing {', '.join(sorted(missing))}")
    labels = _labels_for_cases(cases, labels_path if labels_path.exists() else None)
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"output already exists: {output_path}; pass --overwrite to replace it")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for case_index, case in enumerate(cases):
        grouped[str(case["model_id"])].append((case_index, case))

    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for model_id, model_cases in grouped.items():
            print(f"Loading {model_id} for {len(model_cases)} case(s)", file=sys.stderr, flush=True)
            tokenizer, model, _resolved_id = submitted_pipeline.load_model(model_id)
            try:
                for model_position, (case_index, case) in enumerate(model_cases):
                    row = _extract_case(
                        case, model_position, labels[case_index], tokenizer, model, args
                    )
                    # Keep the original dataset index independently of the seed index.
                    row["case_index"] = case_index
                    writer.writerow(row)
                    stream.flush()
                    print(
                        f"[{model_id}] {model_position + 1}/{len(model_cases)}: "
                        f"{row['case_id']} ({row['status']})",
                        file=sys.stderr,
                        flush=True,
                    )
            finally:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print(f"Wrote {len(cases)} rows to {output_path}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=_dataset_path, required=True,
        help="dataset folder, for example train-sample or data/train-sample",
    )
    parser.add_argument("--labels", type=_path, help="optional labels JSONL override")
    parser.add_argument(
        "--output", type=_path,
        help="output CSV; defaults to DATASET/norm_divergence_features.csv",
    )
    parser.add_argument("--seed", type=int, default=1234, help="base seed for per-case noise")
    parser.add_argument("--epsilon", type=float, default=1e-2, help="relative embedding perturbation size")
    parser.add_argument("--n-noisy", type=int, default=10, help="number of perturbed trajectories")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
