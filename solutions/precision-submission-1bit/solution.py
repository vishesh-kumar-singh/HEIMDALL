import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from precision_inference import predict_robustness_parallel

def are_robust(model_id: str, problems: list[str], reasoning_effort: list[str] = None) -> list[bool]:
    """
    Precision-drop based robustness detection.
    """
    print(f"[Router] Dispatching {model_id} to precision_inference", file=sys.stderr, flush=True)
    return predict_robustness_parallel(model_id, problems)
