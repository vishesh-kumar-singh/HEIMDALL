#!/usr/bin/env python3
"""Run ``are_robust(model_id: str, problems: list[str])`` on input data."""

import argparse
import importlib.util
import inspect
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType
from typing import Any, get_type_hints

PROBLEM_FIELDS = {"original_problem", "permutation_type"}


class IngestionError(RuntimeError):
    pass


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise IngestionError(f"{path.name}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise IngestionError(f"{path.name}:{line_number}: expected a JSON object")
            yield value


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen_ids = set()
    for line_number, case in enumerate(read_jsonl(path), start=1):
        case_id = case.get("id")
        model_id = case.get("model_id")
        problem = case.get("problem")
        if not isinstance(case_id, str) or not case_id:
            raise IngestionError(f"{path.name}:{line_number}: invalid id")
        if case_id in seen_ids:
            raise IngestionError(f"{path.name}:{line_number}: duplicate id {case_id!r}")
        if not isinstance(model_id, str) or not model_id:
            raise IngestionError(f"{path.name}:{line_number}: invalid model_id")
        if not isinstance(problem, dict) or set(problem) != PROBLEM_FIELDS:
            raise IngestionError(
                f"{path.name}:{line_number}: problem must contain exactly "
                "original_problem and permutation_type"
            )
        if not isinstance(problem["original_problem"], str) or not problem["original_problem"]:
            raise IngestionError(f"{path.name}:{line_number}: invalid problem.original_problem")
        permutation_type = problem["permutation_type"]
        if (
            not isinstance(permutation_type, list)
            or not permutation_type
            or any(not isinstance(value, str) or not value for value in permutation_type)
        ):
            raise IngestionError(f"{path.name}:{line_number}: invalid problem.permutation_type")
        seen_ids.add(case_id)
        cases.append(case)
    if not cases:
        raise IngestionError("input dataset is empty")
    return cases


def validate_are_robust(function: Any) -> None:
    """Validate the participant entry-point signature and annotations.

    Args:
        function: Object exported as ``are_robust`` by the submission.

    Raises:
        IngestionError: If the callable does not match the required contract.
    """

    expected_message = (
        "solution.py must define "
        "are_robust(model_id: str, problems: list[str]) -> list[bool]"
    )
    if not callable(function):
        raise IngestionError(expected_message)
    try:
        signature = inspect.signature(function)
        annotations = get_type_hints(function)
    except (NameError, TypeError, ValueError) as exc:
        raise IngestionError(expected_message) from exc

    parameters = list(signature.parameters.values())
    if (
        len(parameters) != 2
        or [parameter.name for parameter in parameters]
        != ["model_id", "problems"]
        or any(
            parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD
            or parameter.default is not inspect.Parameter.empty
            for parameter in parameters
        )
        or annotations.get("model_id") is not str
        or annotations.get("problems") != list[str]
        or annotations.get("return") != list[bool]
    ):
        raise IngestionError(expected_message)


def load_solution(submission_dir: Path) -> ModuleType:
    solution_path = submission_dir / "solution.py"
    if not solution_path.is_file():
        raise IngestionError("submission must contain solution.py at its root")
    spec = importlib.util.spec_from_file_location("participant_solution", solution_path)
    if spec is None or spec.loader is None:
        raise IngestionError("could not load solution.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        validate_are_robust(getattr(module, "are_robust", None))
    except Exception:
        if sys.modules.get(spec.name) is module:
            del sys.modules[spec.name]
        raise
    return module


def run(input_dir: Path, output_dir: Path, submission_dir: Path) -> None:
    submission_import_path = str(submission_dir.resolve())
    sys.path.insert(0, submission_import_path)
    try:
        _run(input_dir, output_dir, submission_dir)
    finally:
        # Remove the exact entry inserted above, even if participant code changed
        # sys.path or added the same path independently.
        for index, entry in enumerate(sys.path):
            if entry is submission_import_path:
                del sys.path[index]
                break


def _run(input_dir: Path, output_dir: Path, submission_dir: Path) -> None:
    cases = load_cases(input_dir / "cases.jsonl")
    solution = load_solution(submission_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batches: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, case in enumerate(cases):
        batches.setdefault(case["model_id"], []).append((index, case))

    predictions = [
        {"id": case["id"], "is_robust": False, "valid": False} for case in cases
    ]
    failures = 0
    for model_id, batch in batches.items():
        batch_predictions = [False] * len(batch)
        batch_valid = [False] * len(batch)
        try:
            problems = [
                case["problem"]["original_problem"] for _, case in batch
            ]
            results = solution.are_robust(model_id, problems)
            if (
                type(results) is list
                and len(results) == len(problems)
                and all(type(result) is bool for result in results)
            ):
                batch_predictions = results
                batch_valid = [True] * len(batch)
        except Exception:
            # Participant exception details are intentionally not copied to results.
            pass

        failures += batch_valid.count(False)
        for (index, case), prediction, valid in zip(batch, batch_predictions, batch_valid):
            predictions[index] = {
                "id": case["id"],
                "is_robust": prediction,
                "valid": valid,
            }

    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8", newline="\n") as stream:
        for prediction in predictions:
            stream.write(json.dumps(prediction, sort_keys=True, separators=(",", ":")))
            stream.write("\n")

    summary = {"cases": len(cases), "invalid_predictions": failures}
    (output_dir / "ingestion_summary.json").write_text(
        json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("submission_dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.input_dir, arguments.output_dir, arguments.submission_dir)
