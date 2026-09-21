import sys
import json
import os

# Add root directory to python path for features import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.extract_features import evaluate_models

if __name__ == "__main__":
    cases_path = "data/train-sample/input/cases.jsonl"
    if not os.path.exists(cases_path):
        print(f"Cases not found at {cases_path}")
        sys.exit(1)
        
    with open(cases_path) as f:
        cases = [json.loads(line) for line in f if line.strip()]

    # Run extraction per model
    models = set(c["model_id"] for c in cases)
    for model_id in models:
        # filter problems for this model
        problems = [c["problem"] for c in cases if c["model_id"] == model_id]
        print(f"\n==============================================", flush=True)
        print(f"Running extraction for {model_id} with {len(problems)} problems", flush=True)
        print(f"==============================================\n", flush=True)
        evaluate_models(model_id, problems)
