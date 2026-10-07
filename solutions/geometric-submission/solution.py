from variance_inference import evaluate_models

def are_robust(model_id: str, problems: list[str]) -> list[bool]:
    """
    Main entrypoint for Codabench evaluator.
    Returns a list of booleans indicating whether each problem is robust.
    """
    return evaluate_models(model_id, problems)
