"""Codabench entry — answer-flip robustness detector (threshold = 1).

Predicts robust when the force-decoded answer flips less than flip_threshold times
across checkpoints during the model's thinking trace.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flip_inference import predict_robustness_parallel

FLIP_THRESHOLD = 1


def are_robust(model_id: str, problems: list[str], reasoning_effort: list[str] = None) -> list[bool]:
    return predict_robustness_parallel(model_id, problems, flip_threshold=FLIP_THRESHOLD)

