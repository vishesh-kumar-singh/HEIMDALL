import sys
import os

# Ensure the pipelines directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipelines.geometry_pipeline import evaluate_models as geometry_pipeline
from pipelines.stability_profile.inference import predict_robustness as stability_pipeline
from pipelines.logit_pipeline.flip_inference import predict_robustness_parallel as logit_pipeline

def are_robust(model_id: str, problems: list[str], reasoning_effort: list[str] = None) -> list[bool]:
    """
    Master router for the final submission.
    Dispatches different models to different evaluation pipelines based on what works best.
    """
    model_id_lower = model_id.lower()
    
    if "deepseek" in model_id_lower:
        print(f"[Router] Dispatching DeepSeek to logit_pipeline", file=sys.stderr, flush=True)
        # We pass flip_threshold=1 because the decision tree in flip_inference handles it
        return logit_pipeline(model_id, problems, flip_threshold=1)
        
    elif "skywork" in model_id_lower:
        print(f"[Router] Dispatching Skywork to stability_pipeline", file=sys.stderr, flush=True)
        return stability_pipeline(model_id, problems)
        
    else:
        # Default placeholder (we will build flips/text pipelines for Qwen/Olmo)
        print(f"[Router] {model_id} routed to default (True)", file=sys.stderr, flush=True)
        return [True] * len(problems)
