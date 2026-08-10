"""Tests for the Codabench participant entry-point contract."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
INGESTION_PATH = ROOT / "components" / "ingestion_program" / "ingestion.py"


def load_ingestion_module() -> ModuleType:
    """Load the ingestion program as a test module."""

    spec = importlib.util.spec_from_file_location(
        "ingestion_contract_test",
        INGESTION_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load ingestion program")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


INGESTION = load_ingestion_module()


class IngestionContractTests(unittest.TestCase):
    """Exercise entry-point validation and problem-string forwarding."""

    @staticmethod
    def write_solution(directory: Path, source: str) -> None:
        """Write a temporary participant entry point."""

        directory.mkdir(parents=True)
        (directory / "solution.py").write_text(source, encoding="utf-8")

    def test_accepts_exact_string_list_signature(self) -> None:
        """Accept the documented annotated entry-point signature."""

        with tempfile.TemporaryDirectory() as temporary:
            submission = Path(temporary) / "submission"
            self.write_solution(
                submission,
                "def are_robust(model_id: str, problems: list[str]) "
                "-> list[bool]:\n"
                "    return [False for _ in problems]\n",
            )
            module = INGESTION.load_solution(submission)
        self.assertTrue(callable(module.are_robust))

    def test_rejects_legacy_problem_dataclass_signature(self) -> None:
        """Reject the superseded list-of-Problem adapter contract."""

        with tempfile.TemporaryDirectory() as temporary:
            submission = Path(temporary) / "submission"
            self.write_solution(
                submission,
                "from dataclasses import dataclass\n\n"
                "@dataclass\n"
                "class Problem:\n"
                "    original_problem: str\n"
                "    permutation_type: list[str]\n\n"
                "def are_robust(model_id: str, problems: list[Problem]) "
                "-> list[bool]:\n"
                "    return [False for _ in problems]\n",
            )
            with self.assertRaisesRegex(
                INGESTION.IngestionError,
                r"problems: list\[str\]",
            ):
                INGESTION.load_solution(submission)

    def test_run_passes_only_original_problem_strings(self) -> None:
        """Pass ordered original-problem strings to participant code."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "output"
            submission = root / "submission"
            input_dir.mkdir()
            self.write_solution(
                submission,
                "def are_robust(model_id: str, problems: list[str]) "
                "-> list[bool]:\n"
                "    assert model_id == 'example/model'\n"
                "    assert all(type(problem) is str for problem in problems)\n"
                "    return [problem.startswith('robust') for problem in problems]\n",
            )
            cases = [
                {
                    "id": "first",
                    "model_id": "example/model",
                    "problem": {
                        "original_problem": "robust example",
                        "permutation_type": ["rephrase"],
                    },
                },
                {
                    "id": "second",
                    "model_id": "example/model",
                    "problem": {
                        "original_problem": "non-robust example",
                        "permutation_type": ["domain"],
                    },
                },
            ]
            (input_dir / "cases.jsonl").write_text(
                "".join(json.dumps(case) + "\n" for case in cases),
                encoding="utf-8",
            )

            INGESTION.run(input_dir, output_dir, submission)
            predictions = [
                json.loads(line)
                for line in (output_dir / "predictions.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual(
            [prediction["is_robust"] for prediction in predictions],
            [True, False],
        )
        self.assertTrue(all(prediction["valid"] for prediction in predictions))


if __name__ == "__main__":
    unittest.main()
