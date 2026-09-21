"""Codabench entry point for the uncertainty-profiling baseline."""

from uncertainty_profile.inference import predict_robustness


def are_robust(model_id: str, problems: list[str], reasoning_effort: list[str] = None) -> list[bool]:
    """Predict one native boolean robustness label per problem.

    Args:
        model_id: Codabench model alias or exact checkpoint identifier.
        problems: Mathematical problem statements in evaluation order.
        reasoning_effort: Reasoning effort levels for each problem.

    Returns:
        Robustness predictions in the same order as ``problems``.
    """

    return predict_robustness(model_id=model_id, problem_texts=problems)
