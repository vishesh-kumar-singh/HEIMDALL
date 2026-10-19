"""Codabench entry point for perturbation-consistency prediction."""

from perturbation_inference import predict_robustness


def are_robust(model_id: str, problems: list[str]) -> list[bool]:
    """Predict whether answers stay consistent under two fixed distractors."""
    return predict_robustness(model_id, problems)
