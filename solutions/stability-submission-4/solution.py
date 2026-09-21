from flip_inference import predict_robustness

def are_robust(model_id: str, problems: list[str]) -> list[bool]:
    return predict_robustness(model_id, problems)
