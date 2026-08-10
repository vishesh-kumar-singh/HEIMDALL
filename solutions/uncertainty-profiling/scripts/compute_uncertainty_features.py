#!/usr/bin/env python3
"""Generate and cache uncertainty features for the pinned training dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

SOLUTION_ROOT = Path(__file__).resolve().parents[1]
ROOT = SOLUTION_ROOT.parents[1]
for import_path in (ROOT / "scripts", SOLUTION_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from import_hf_dataset import fetch_all_rows  # noqa: E402
from uncertainty_profile.artifact import safe_model_id  # noqa: E402
from uncertainty_profile.config import (  # noqa: E402
    FEATURE_NAMES,
    GenerationConfidenceConfig,
)
from uncertainty_profile.extraction import (  # noqa: E402
    iter_generation_feature_batches,
    load_model_and_tokenizer,
    release_model,
)


DEFAULT_DATASET = "aimo-interp/augmented-sample-math-agg"
DEFAULT_REVISION = "f972ced0705096f8d7ca7fac30825900b8b7fb6a"
DEFAULT_CONFIG = "default"
DEFAULT_SPLIT = "validation"
DEFAULT_FEATURE_MODEL = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
REQUIRED_COLUMNS = {
    "model_id",
    "dataset_id",
    "problem_id",
    "original_problem",
    "absolute_accuracy_decay",
    "model_is_robust",
}


def prompt_key(problem_text: str) -> str:
    """Return a stable content key for a problem statement.

    Args:
        problem_text: Original mathematical problem statement.

    Returns:
        SHA-256 digest of the UTF-8 prompt text.
    """

    return hashlib.sha256(problem_text.encode("utf-8")).hexdigest()


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Atomically write a DataFrame as Parquet.

    Args:
        frame: Feature rows to serialize.
        path: Final Parquet destination.

    Raises:
        OSError: If the temporary or final file cannot be written.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".parquet",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        frame.to_parquet(temporary_path, index=False)
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def validate_source_rows(
    rows: Sequence[Mapping[str, object]],
) -> pd.DataFrame:
    """Validate source rows and attach stable row and prompt identifiers.

    Args:
        rows: Records fetched from the pinned Hugging Face dataset.

    Returns:
        Normalized source DataFrame with ``source_row_index`` and ``prompt_key``.

    Raises:
        ValueError: If required data is missing, malformed, or label-inconsistent.
    """

    if not rows:
        raise ValueError("source dataset is empty")
    frame = pd.DataFrame.from_records(rows)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"source dataset is missing columns: {missing}")
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("source dataset contains missing required values")
    if (frame["original_problem"].astype(str).str.len() == 0).any():
        raise ValueError("source dataset contains an empty original_problem")
    if (frame["problem_id"].astype(str).str.len() == 0).any():
        raise ValueError("source dataset contains an empty problem_id")
    if not all(type(value) is bool for value in frame["model_is_robust"].tolist()):
        raise ValueError("model_is_robust must contain native boolean values")

    target = pd.to_numeric(frame["absolute_accuracy_decay"], errors="coerce")
    if target.isna().any() or not np.isfinite(target.to_numpy()).all():
        raise ValueError("absolute_accuracy_decay must contain finite numbers")
    if not target.between(0.0, 1.0).all():
        raise ValueError("absolute_accuracy_decay must be in [0, 1]")
    frame["absolute_accuracy_decay"] = target.astype(float)
    frame["original_problem"] = frame["original_problem"].astype(str)
    frame["problem_id"] = frame["problem_id"].astype(str)
    frame["model_id"] = frame["model_id"].astype(str)

    conflicting = (
        frame.groupby("original_problem", sort=False)["model_is_robust"].nunique() > 1
    )
    if conflicting.any():
        raise ValueError(
            "repeated original_problem values have conflicting robustness labels"
        )
    frame.insert(0, "source_row_index", np.arange(len(frame), dtype=int))
    frame["prompt_key"] = frame["original_problem"].map(prompt_key)
    return frame


def validate_partial_frame(
    partial: pd.DataFrame,
    *,
    feature_model_id: str,
    generation_config_json: str,
) -> pd.DataFrame:
    """Validate a resumable prompt-level feature cache.

    Args:
        partial: Cached unique-prompt feature rows.
        feature_model_id: Model identifier expected in every cached row.
        generation_config_json: Canonical generation configuration string.

    Returns:
        ``partial`` after successful validation.

    Raises:
        ValueError: If schema, provenance, or prompt-key uniqueness is invalid.
    """

    required = {
        "prompt_key",
        "original_problem",
        "feature_model_id",
        "generation_config_json",
        "generated_text",
        "generation_num_tokens",
        *FEATURE_NAMES,
    }
    missing = sorted(required - set(partial.columns))
    if missing:
        raise ValueError(f"partial feature cache is missing columns: {missing}")
    if set(partial["feature_model_id"].astype(str)) != {feature_model_id}:
        raise ValueError("partial feature cache was generated with another model")
    if set(partial["generation_config_json"].astype(str)) != {
        generation_config_json
    }:
        raise ValueError("partial feature cache uses another generation configuration")
    duplicate_keys = partial["prompt_key"].duplicated(keep=False)
    if duplicate_keys.any():
        raise ValueError("partial feature cache contains duplicate prompt keys")
    return partial


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed feature-generation options.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Generate model-response uncertainty features for regressor training."
        )
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Hugging Face dataset repository containing training rows.",
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="Exact dataset revision or commit hash to fetch.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Dataset configuration name.",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help="Dataset split to load.",
    )
    parser.add_argument(
        "--feature-model-id",
        default=DEFAULT_FEATURE_MODEL,
        help="Model used to generate responses and uncertainty features.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional Hugging Face model cache directory.",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=2048,
        help="Maximum prompt length in tokens.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Maximum number of tokens generated per response.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of prompts generated in each model batch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducible generation setup.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Limit source rows before deduplication; useful for smoke tests.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Number of deterministic prompt shards.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based prompt shard generated by this process.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Parquet destination; defaults under data/uncertainty-profiling/.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require the training model to exist in the local Hugging Face cache.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing final and partial feature cache.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate one cached feature row per prompt and restore source rows.

    Raises:
        ValueError: If arguments or source data are invalid.
        RuntimeError: If prompt generation or cache validation is incomplete.
    """

    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("num_shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")

    generation_config = GenerationConfidenceConfig(
        max_prompt_length=args.max_prompt_length,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )
    generation_config.validate()
    generation_config_json = json.dumps(
        generation_config.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    )

    output_name = f"{safe_model_id(args.feature_model_id)}_confidence_features"
    if args.num_shards > 1:
        output_name += f".shard-{args.shard_index:02d}-of-{args.num_shards:02d}"
    output_path = args.output or (
        ROOT / "data" / "uncertainty-profiling" / f"{output_name}.parquet"
    )
    partial_path = output_path.with_name(f"{output_path.name}.partial.parquet")

    if output_path.is_file() and not args.overwrite:
        print(f"Feature cache already exists: {output_path}")
        return
    if args.overwrite:
        output_path.unlink(missing_ok=True)
        partial_path.unlink(missing_ok=True)

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Get data
    print(
        f"Fetching {args.dataset}@{args.revision} "
        f"config={args.config} split={args.split}",
        flush=True,
    )
    rows = fetch_all_rows(args.dataset, args.revision, args.config, args.split)
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("max_samples must be positive")
        rows = rows[: args.max_samples]
    source = validate_source_rows(rows)
    unique_prompts = source[["prompt_key", "original_problem"]].drop_duplicates(
        "prompt_key", keep="first"
    )
    if unique_prompts["original_problem"].nunique() != len(unique_prompts):
        raise RuntimeError("prompt hash collision detected")

    # Shard after deduplication so each prompt is generated by one process.
    selected_prompts = unique_prompts.iloc[
        args.shard_index :: args.num_shards
    ].copy()
    selected_keys = set(selected_prompts["prompt_key"])
    source = source[source["prompt_key"].isin(selected_keys)].copy()
    unique_prompts = selected_prompts.reset_index(drop=True)

    # Resume only from a cache matching the exact model and generation config.
    if partial_path.is_file():
        completed = validate_partial_frame(
            pd.read_parquet(partial_path),
            feature_model_id=args.feature_model_id,
            generation_config_json=generation_config_json,
        )
    else:
        completed = pd.DataFrame()
    completed_keys = set(completed.get("prompt_key", pd.Series(dtype=str)).astype(str))
    missing_prompts = unique_prompts[
        ~unique_prompts["prompt_key"].isin(completed_keys)
    ].reset_index(drop=True)
    print(
        f"Source rows={len(source)}, unique prompts={len(unique_prompts)}, "
        f"remaining prompts={len(missing_prompts)}",
        flush=True,
    )

    # Run model on data
    model: PreTrainedModel | None = None
    tokenizer: PreTrainedTokenizerBase | None = None
    try:
        if len(missing_prompts):
            model, tokenizer = load_model_and_tokenizer(
                args.feature_model_id,
                cache_dir=str(args.cache_dir) if args.cache_dir else None,
                local_files_only=args.local_files_only,
            )
            problem_texts = missing_prompts["original_problem"].tolist()
            with tqdm(
                total=len(unique_prompts),
                initial=len(completed),
                desc="Generating prompt features",
                unit="prompt",
            ) as progress:
                for start, feature_rows in iter_generation_feature_batches(
                    model=model,
                    tokenizer=tokenizer,
                    problem_texts=problem_texts,
                    config=generation_config,
                    include_generated_text=True,
                ):
                    prompt_batch = missing_prompts.iloc[
                        start : start + len(feature_rows)
                    ].reset_index(drop=True)
                    batch = pd.concat(
                        [prompt_batch, pd.DataFrame.from_records(feature_rows)],
                        axis=1,
                    )
                    batch["feature_model_id"] = args.feature_model_id
                    batch["generation_config_json"] = generation_config_json
                    completed = pd.concat([completed, batch], ignore_index=True)
                    completed = completed.drop_duplicates("prompt_key", keep="last")
                    atomic_write_parquet(completed, partial_path)
                    progress.update(len(feature_rows))
    finally:
        if model is not None or tokenizer is not None:
            model = None
            tokenizer = None
            release_model()

    # Save feature frame
    completed = validate_partial_frame(
        completed,
        feature_model_id=args.feature_model_id,
        generation_config_json=generation_config_json,
    )
    expected_keys = set(unique_prompts["prompt_key"])
    if set(completed["prompt_key"]) != expected_keys:
        missing = sorted(expected_keys - set(completed["prompt_key"]))
        raise RuntimeError(
            f"feature extraction did not complete; missing {len(missing)} prompts"
        )

    feature_columns = [
        "prompt_key",
        "generated_text",
        "generation_num_tokens",
        *FEATURE_NAMES,
    ]

    # Restore duplicate source rows after generating every unique prompt once.
    final = source.merge(
        completed[feature_columns],
        on="prompt_key",
        how="left",
        validate="many_to_one",
    ).sort_values("source_row_index", kind="stable")
    final["source_dataset"] = args.dataset
    final["source_revision"] = args.revision
    final["source_config"] = args.config
    final["source_split"] = args.split
    final["feature_model_id"] = args.feature_model_id
    final["generation_config_json"] = generation_config_json
    atomic_write_parquet(final.reset_index(drop=True), output_path)
    partial_path.unlink(missing_ok=True)
    print(f"Saved {len(final)} feature rows to {output_path}", flush=True)


if __name__ == "__main__":
    main()
