"""Codabench entry point. All logic lives in aimo_router.py."""
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aimo_router import are_robust as _are_robust


@dataclass(frozen=True)
class Problem:
    original_problem: str
    permutation_type: list[str]


def are_robust(model_id: str, problems, reasoning_effort=None) -> list[bool]:
    return _are_robust(model_id, problems, reasoning_effort)
