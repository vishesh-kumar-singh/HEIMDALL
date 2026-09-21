"""LogitTrace (Layerwise Logit Trajectories) Version 1.

Implements the zero-shot threshold heuristic based on "MemLens / LogitTrace".
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from trace_inference import predict_robustness_parallel

def are_robust(model_id: str, problems: list[str], reasoning_effort: list[str] = None) -> list[bool]:
    return predict_robustness_parallel(model_id, problems)
