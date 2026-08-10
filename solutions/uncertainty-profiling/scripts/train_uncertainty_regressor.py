#!/usr/bin/env python3
"""Select, fit, validate, and export the uncertainty regression artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired, TypedDict

import accelerate
import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

SOLUTION_ROOT = Path(__file__).resolve().parents[1]
ROOT = SOLUTION_ROOT.parents[1]
if str(SOLUTION_ROOT) not in sys.path:
    sys.path.insert(0, str(SOLUTION_ROOT))

from uncertainty_profile.artifact import (  # noqa: E402
    default_artifact_path,
    dump_artifact,
    make_artifact_payload,
    validate_artifact_payload,
)
from uncertainty_profile.config import (  # noqa: E402
    FEATURE_NAMES,
    GenerationConfidenceConfig,
)


TARGET_COLUMN = "absolute_accuracy_decay"
LABEL_COLUMN = "model_is_robust"
GROUP_COLUMN = "original_problem"


ParameterValue = int | float | str | None


class EvaluationResult(TypedDict):
    """Cross-validation metrics for one regressor candidate."""

    identifier: str
    decision_threshold: float
    balanced_accuracy: float
    accuracy: float
    mae: float
    mse: float
    r2: float
    family: NotRequired[str]
    params: NotRequired[dict[str, ParameterValue]]


@dataclass(frozen=True)
class CandidateSpec:
    """Describe one deterministic hyperparameter candidate.

    Attributes:
        family: Supported regressor family name.
        params: Estimator keyword arguments selected from the search grid.
    """

    family: str
    params: dict[str, ParameterValue]

    @property
    def identifier(self) -> str:
        """Return a stable identifier used for ordering and reporting."""

        return f"{self.family}:{json.dumps(self.params, sort_keys=True)}"


def iter_parquet_files(path: Path) -> Iterable[Path]:
    """Yield completed feature-cache files in deterministic order.

    Args:
        path: A completed Parquet file or directory containing cache shards.

    Yields:
        Completed Parquet paths in lexical order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a directly supplied file is a partial cache.
    """

    if path.is_file():
        if path.name.endswith(".partial.parquet"):
            raise ValueError(f"refusing incomplete feature cache: {path}")
        yield path
        return
    if not path.is_dir():
        raise FileNotFoundError(f"feature data path does not exist: {path}")
    yield from sorted(
        file for file in path.glob("*.parquet")
        if not file.name.endswith(".partial.parquet")
    )


def load_feature_data(path: Path) -> pd.DataFrame:
    """Load and concatenate completed feature-cache shards.

    Args:
        path: Completed Parquet file or directory of completed shards.

    Returns:
        Concatenated feature rows with source filenames attached.

    Raises:
        ValueError: If no completed Parquet files are found.
    """

    files = list(iter_parquet_files(path))
    if not files:
        raise ValueError(f"no parquet files found under {path}")
    frames = []
    for file in files:
        frame = pd.read_parquet(file)
        frame["source_feature_file"] = file.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def validate_feature_data(frame: pd.DataFrame, n_splits: int) -> None:
    """Validate feature schema, targets, groups, and shared provenance.

    Args:
        frame: Training feature rows.
        n_splits: Number of grouped cross-validation folds.

    Raises:
        ValueError: If the feature data cannot support valid grouped CV.
    """

    required = {
        *FEATURE_NAMES,
        TARGET_COLUMN,
        LABEL_COLUMN,
        GROUP_COLUMN,
        "feature_model_id",
        "generation_config_json",
        "source_dataset",
        "source_revision",
        "model_id",
    }
    missing = sorted(required - set(frame.columns))

    if missing:
        raise ValueError(f"feature data is missing columns: {missing}")
    if len(frame) < n_splits:
        raise ValueError(f"feature data has {len(frame)} rows for {n_splits} folds")
    if not all(type(value) is bool for value in frame[LABEL_COLUMN].tolist()):
        raise ValueError(f"{LABEL_COLUMN} must contain native boolean values")

    target = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")

    if target.isna().any() or not np.isfinite(target.to_numpy()).all():
        raise ValueError(f"{TARGET_COLUMN} must contain finite values")
    if not target.between(0.0, 1.0).all():
        raise ValueError(f"{TARGET_COLUMN} must be in [0, 1]")
    if frame[GROUP_COLUMN].isna().any():
        raise ValueError(f"{GROUP_COLUMN} contains missing values")
    
    conflicts = frame.groupby(GROUP_COLUMN, sort=False)[LABEL_COLUMN].nunique() > 1

    if conflicts.any():
        raise ValueError("a CV group contains conflicting robustness labels")
    if frame[LABEL_COLUMN].nunique() != 2:
        raise ValueError("feature data must contain both robustness classes")

    class_group_counts = (
        frame[[GROUP_COLUMN, LABEL_COLUMN]]
        .drop_duplicates(GROUP_COLUMN)[LABEL_COLUMN]
        .value_counts()
    )

    if int(class_group_counts.min()) < n_splits:
        raise ValueError("each class must contain at least one group per CV fold")
    for column in (
        "feature_model_id",
        "generation_config_json",
        "source_dataset",
        "source_revision",
    ):
        if frame[column].nunique(dropna=False) != 1:
            raise ValueError(f"feature data contains multiple values for {column}")


def build_candidate_specs() -> list[CandidateSpec]:
    """Build the complete, deterministically ordered model search grid.

    Returns:
        Candidate specifications for Extra Trees, histogram gradient boosting,
        and random forests.
    """

    tree_grid = {
        "n_estimators": [200, 500],
        "max_depth": [None, 8],
        "min_samples_leaf": [1, 4],
        "max_features": [1.0, "sqrt"],
    }
    histogram_grid = {
        "max_iter": [200, 500],
        "learning_rate": [0.05, 0.1],
        "max_leaf_nodes": [7, 15],
        "l2_regularization": [0.0, 1.0],
    }
    specs = [
        CandidateSpec(family=family, params=dict(params))
        for family, grid in (
            ("extra_trees", tree_grid),
            ("hist_gradient_boosting", histogram_grid),
            ("random_forest", tree_grid),
        )
        for params in ParameterGrid(grid)
    ]
    return sorted(specs, key=lambda spec: spec.identifier)


def build_pipeline(spec: CandidateSpec, seed: int) -> Pipeline:
    """Build an imputation and regression pipeline for one candidate.

    Args:
        spec: Estimator family and hyperparameters.
        seed: Random seed applied to stochastic estimators.

    Returns:
        Unfitted scikit-learn pipeline.

    Raises:
        ValueError: If ``spec.family`` is unsupported.
    """

    if spec.family == "random_forest":
        estimator = RandomForestRegressor(
            random_state=seed,
            n_jobs=1,
            **spec.params,
        )
    elif spec.family == "extra_trees":
        estimator = ExtraTreesRegressor(
            random_state=seed,
            n_jobs=1,
            **spec.params,
        )
    elif spec.family == "hist_gradient_boosting":
        estimator = HistGradientBoostingRegressor(
            random_state=seed,
            **spec.params,
        )
    else:
        raise ValueError(f"unsupported estimator family: {spec.family}")
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", estimator),
        ]
    )


def make_grouped_splits(
    frame: pd.DataFrame,
    *,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Construct stratified grouped folds and verify group isolation.

    Args:
        frame: Validated training feature rows.
        n_splits: Number of cross-validation folds.
        seed: Fold-shuffling random seed.

    Returns:
        Training and validation row-index arrays for each fold.

    Raises:
        RuntimeError: If any group appears in both sides of a fold.
    """

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    labels = frame[LABEL_COLUMN].astype(bool).to_numpy()
    groups = frame[GROUP_COLUMN].astype(str).to_numpy()
    splits = list(splitter.split(frame, labels, groups))

    for train_indices, validation_indices in splits:
        train_groups = set(groups[train_indices])
        validation_groups = set(groups[validation_indices])
        if train_groups & validation_groups:
            raise RuntimeError("cross-validation leaked a group across splits")
    return splits


def threshold_candidates(predictions: Sequence[float]) -> list[float]:
    """Enumerate all classification-distinct score thresholds.

    Args:
        predictions: Finite continuous out-of-fold predictions.

    Returns:
        Thresholds bracketing and bisecting all unique scores.

    Raises:
        ValueError: If predictions are empty or non-finite.
    """

    unique = np.unique(np.asarray(predictions, dtype=float))
    if unique.size == 0 or not np.isfinite(unique).all():
        raise ValueError("regressor predictions must be finite")

    candidates = [float(np.nextafter(unique[0], -math.inf))]
    candidates.extend(
        float(left + (right - left) / 2.0)
        for left, right in zip(unique[:-1], unique[1:])
    )
    candidates.append(float(np.nextafter(unique[-1], math.inf)))
    return candidates


def select_threshold(
    labels: Sequence[bool],
    predictions: Sequence[float],
) -> tuple[float, float, float]:
    """Select a robustness threshold from out-of-fold predictions.

    Scores below the threshold are labeled robust. Balanced accuracy is
    maximized first, followed by plain accuracy and then the lower threshold.

    Args:
        labels: Binary robustness labels.
        predictions: Continuous regression predictions.

    Returns:
        Selected threshold, balanced accuracy, and plain accuracy.

    Raises:
        RuntimeError: If threshold enumeration produces no candidates.
    """

    y_true = np.asarray(labels, dtype=bool)
    y_pred = np.asarray(predictions, dtype=float)
    best: tuple[tuple[float, float, float], float, float, float] | None = None

    for threshold in threshold_candidates(y_pred):
        binary = y_pred < threshold
        balanced = float(balanced_accuracy_score(y_true, binary))
        accuracy = float(accuracy_score(y_true, binary))
        key = (balanced, accuracy, -threshold)
        if best is None or key > best[0]:
            best = (key, threshold, balanced, accuracy)

    if best is None:
        raise RuntimeError("threshold selection produced no candidates")
    return best[1], best[2], best[3]


def compute_oof_predictions(
    pipeline: Pipeline,
    frame: pd.DataFrame,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Fit each fold and return one out-of-fold prediction per row.

    Args:
        pipeline: Unfitted regression pipeline.
        frame: Validated training feature rows.
        splits: Training and validation row indices for each fold.

    Returns:
        Finite predictions aligned with ``frame``.

    Raises:
        RuntimeError: If any row lacks a finite out-of-fold prediction.
    """

    features = frame[list(FEATURE_NAMES)]
    target = frame[TARGET_COLUMN].astype(float)
    predictions = np.full(len(frame), np.nan, dtype=float)

    for train_indices, validation_indices in splits:
        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(features.iloc[train_indices], target.iloc[train_indices])
        predictions[validation_indices] = fold_pipeline.predict(
            features.iloc[validation_indices]
        )

    if np.isnan(predictions).any() or not np.isfinite(predictions).all():
        raise RuntimeError(
            "cross-validation did not produce finite predictions for all rows"
        )
    return predictions


def evaluate_predictions(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    *,
    identifier: str,
) -> EvaluationResult:
    """Calculate regression metrics and thresholded classification metrics.

    Args:
        frame: Training rows containing regression targets and binary labels.
        predictions: Continuous predictions aligned with ``frame``.
        identifier: Stable candidate identifier for reporting.

    Returns:
        JSON-serializable candidate metrics.
    """

    threshold, balanced, accuracy = select_threshold(
        frame[LABEL_COLUMN].astype(bool).to_numpy(), predictions
    )
    target = frame[TARGET_COLUMN].astype(float).to_numpy()

    return {
        "identifier": identifier,
        "decision_threshold": threshold,
        "balanced_accuracy": balanced,
        "accuracy": accuracy,
        "mae": float(mean_absolute_error(target, predictions)),
        "mse": float(mean_squared_error(target, predictions)),
        "r2": float(r2_score(target, predictions)),
    }


def run_model_selection(
    frame: pd.DataFrame,
    *,
    specs: Sequence[CandidateSpec],
    n_splits: int,
    seed: int,
) -> tuple[
    CandidateSpec,
    EvaluationResult,
    list[EvaluationResult],
    EvaluationResult,
]:
    """Evaluate candidates and select the deterministic CV winner.

    Candidates are ranked by balanced accuracy, plain accuracy, regression MAE,
    and their pre-sorted deterministic parameter order.

    Args:
        frame: Validated training feature rows.
        specs: Candidate specifications in deterministic order.
        n_splits: Number of grouped cross-validation folds.
        seed: Shared fold and estimator random seed.

    Returns:
        Winning specification, its metrics, every candidate result, and dummy
        mean-regressor metrics.

    Raises:
        RuntimeError: If no candidates are supplied.
    """

    splits = make_grouped_splits(frame, n_splits=n_splits, seed=seed)
    dummy = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", DummyRegressor(strategy="mean")),
        ]
    )
    dummy_predictions = compute_oof_predictions(dummy, frame, splits)
    dummy_result = evaluate_predictions(
        frame, dummy_predictions, identifier="mean_baseline"
    )

    results: list[EvaluationResult] = []
    best_spec = None
    best_result = None
    best_key = None

    for index, spec in enumerate(specs, start=1):
        predictions = compute_oof_predictions(build_pipeline(spec, seed), frame, splits)
        result = evaluate_predictions(frame, predictions, identifier=spec.identifier)
        result["family"] = spec.family
        result["params"] = spec.params
        results.append(result)
        key = (
            result["balanced_accuracy"],
            result["accuracy"],
            -result["mae"],
        )
        if best_key is None or key > best_key:
            best_key = key
            best_spec = spec
            best_result = result
        print(
            f"[{index}/{len(specs)}] {spec.identifier}: "
            f"balanced_accuracy={result['balanced_accuracy']:.4f}, "
            f"accuracy={result['accuracy']:.4f}, mae={result['mae']:.4f}",
            flush=True,
        )

    if best_spec is None or best_result is None:
        raise RuntimeError("model selection had no candidates")

    return best_spec, best_result, results, dummy_result


def sha256_file(path: Path) -> str:
    """Compute the SHA-256 checksum of a file.

    Args:
        path: File to hash.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically write a JSON report.

    Args:
        path: Final report destination.
        payload: JSON-serializable report data.

    Raises:
        ValueError: If ``payload`` contains non-JSON values or NaNs.
        OSError: If the report cannot be committed atomically.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".json",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)

    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed training and export options.
    """

    parser = argparse.ArgumentParser(
        description="Select and export the uncertainty robustness regressor."
    )
    parser.add_argument(
        "--feature-data-path",
        type=Path,
        required=True,
        help="Completed feature Parquet file or directory of cache shards.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of stratified grouped cross-validation folds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for CV splitting and estimator fitting.",
    )
    parser.add_argument(
        "--artifact-path",
        type=Path,
        help="Joblib destination; defaults to the model-specific solution path.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=ROOT / "data" / "uncertainty-profiling" / "cv-results.json",
        help="JSON destination for CV metrics and artifact provenance.",
    )
    return parser.parse_args()


def main() -> None:
    """Run grouped CV, refit the winner, and export artifact and report.

    Raises:
        ValueError: If arguments or feature data are invalid.
        RuntimeError: If selection, serialization, or round-trip checks fail.
    """

    args = parse_args()
    if args.n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    frame = load_feature_data(args.feature_data_path)
    validate_feature_data(frame, args.n_splits)
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype(float)
    feature_model_id = str(frame["feature_model_id"].iloc[0])

    generation_config = GenerationConfidenceConfig.from_dict(
        json.loads(str(frame["generation_config_json"].iloc[0]))
    )
    specs = build_candidate_specs()
    print(
        f"Selecting among {len(specs)} candidates on {len(frame)} rows and "
        f"{frame[GROUP_COLUMN].nunique()} groups",
        flush=True,
    )

    best_spec, best_result, all_results, dummy_result = run_model_selection(
        frame,
        specs=specs,
        n_splits=args.n_splits,
        seed=args.seed,
    )

    # Refit the selected candidate on every available training row.
    final_pipeline = build_pipeline(best_spec, args.seed)
    final_pipeline.fit(
        frame[list(FEATURE_NAMES)],
        frame[TARGET_COLUMN].astype(float),
    )

    source_model_ids = sorted(set(frame["model_id"].astype(str)))
    training_provenance = {
        "source_dataset": str(frame["source_dataset"].iloc[0]),
        "source_revision": str(frame["source_revision"].iloc[0]),
        "source_model_ids": source_model_ids,
        "feature_model_id": feature_model_id,
        "target_column": TARGET_COLUMN,
        "label_column": LABEL_COLUMN,
        "group_column": GROUP_COLUMN,
        "row_count": int(len(frame)),
        "group_count": int(frame[GROUP_COLUMN].nunique()),
        "n_splits": int(args.n_splits),
        "seed": int(args.seed),
    }
    selected_params = {
        "family": best_spec.family,
        **best_spec.params,
    }
    cv_results = {
        "selection_metric": "balanced_accuracy",
        "tie_breakers": ["accuracy", "mae", "deterministic_parameter_order"],
        "best": best_result,
        "mean_baseline": dummy_result,
        "candidates": all_results,
    }
    library_versions = {
        "python": platform.python_version(),
        "joblib": joblib.__version__,
        "accelerate": accelerate.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    payload = make_artifact_payload(
        model_id=feature_model_id,
        generation_config=generation_config,
        estimator=final_pipeline,
        decision_threshold=float(best_result["decision_threshold"]),
        selected_params=selected_params,
        cv_results=cv_results,
        training_provenance=training_provenance,
        library_versions=library_versions,
    )
    artifact_path = args.artifact_path or default_artifact_path(feature_model_id)
    dump_artifact(payload, artifact_path)

    # Confirm serialization preserves both scores and thresholded labels.
    loaded = validate_artifact_payload(joblib.load(artifact_path), path=artifact_path)
    before = final_pipeline.predict(frame[list(FEATURE_NAMES)])
    after = loaded.estimator.predict(frame[list(FEATURE_NAMES)])
    if not np.allclose(before, after, rtol=0.0, atol=1e-12):
        raise RuntimeError("artifact round trip changed regressor scores")
    threshold = float(best_result["decision_threshold"])
    if not np.array_equal(before < threshold, after < threshold):
        raise RuntimeError("artifact round trip changed robustness predictions")

    results_payload = {
        "artifact_path": str(artifact_path),
        "artifact_sha256": sha256_file(artifact_path),
        "artifact_size_bytes": artifact_path.stat().st_size,
        "selected_params": selected_params,
        "decision_threshold": float(best_result["decision_threshold"]),
        "best_cv_metrics": best_result,
        "mean_baseline_cv_metrics": dummy_result,
        "training_provenance": training_provenance,
        "library_versions": library_versions,
        "all_candidate_results": all_results,
    }
    write_json(args.results_path, results_payload)
    print(json.dumps(results_payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
