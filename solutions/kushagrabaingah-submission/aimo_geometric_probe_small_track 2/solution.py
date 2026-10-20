"""
Codabench entry point for the Geometric Trajectory Robustness Probe.
AIMO Interpretability Challenge at NeurIPS 2026.
"""

from __future__ import annotations
from typing import Any
from geometry_inference import predict_robustness_batch


def are_robust(
    model_id: str,
    problems: Any = None,
    reasoning_effort: str = "default",
    **kwargs
) -> list[bool]:
    """
    Predicts whether model_id's mathematical reasoning on each problem is robust.

    Official interface:
        are_robust(model_id: str, problems: list[Problem]) -> list[bool]
    where Problem is a frozen dataclass with 'original_problem: str' and 'permutation_type: list[str]'.

    Also supports:
        - Legacy 2-argument positional: are_robust(model_id, problems)
        - Keyword argument: are_robust(model_id, problems=problems)
        - 3-argument call: are_robust(model_id, reasoning_effort="default", problems=problems)
        - 3-argument positional: are_robust(model_id, "default", problems)

    Args:
        model_id: Identifier of the candidate model (e.g. 'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B' or 'model:low')
        problems: List of Problem objects or mathematical problem strings
        reasoning_effort: 'default', 'low', or 'high'
        **kwargs: Optional keyword arguments

    Returns:
        List of native Python bools (True for Robust, False for Spurious)
    """
    actual_problems = None
    effort = "default"

    # 1. Resolve 'problems' passed via kwargs
    if "problems" in kwargs and kwargs["problems"] is not None:
        actual_problems = kwargs["problems"]

    # 2. Resolve positional arguments
    if actual_problems is None:
        if isinstance(problems, list):
            actual_problems = problems
        elif isinstance(reasoning_effort, list):
            # 2-arg legacy positional: are_robust(model_id, [problems])
            actual_problems = reasoning_effort
        elif problems is not None and not isinstance(problems, str):
            actual_problems = list(problems)
        else:
            actual_problems = []

    # 3. Resolve 'reasoning_effort'
    if isinstance(reasoning_effort, str):
        effort = reasoning_effort
    elif isinstance(problems, str):
        effort = problems
    elif "reasoning_effort" in kwargs and isinstance(kwargs["reasoning_effort"], str):
        effort = kwargs["reasoning_effort"]

    return predict_robustness_batch(
        model_id=model_id,
        problems=actual_problems,
        reasoning_effort=effort
    )
