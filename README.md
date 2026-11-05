# HEIMDALL: Seeing Through Spurious Reasoning
### AIMO Interpretability Challenge (NeurIPS 2026)

<p align="center">
  <img src="image.png" alt="Heimdall - The All-Seeing Guardian" width="700" style="border-radius: 14px; box-shadow: 0 8px 24px rgba(0,0,0,0.5);"/>
</p>

<p align="center">
  <em>"Like the all-seeing Norse watchman who guards the Bifröst bridge, <strong>Heimdall</strong> inspects the internal geometry and weight-space landscape of the model itself rather than relying on surface token generation—unveiling whether mathematical derivations stem from grounded attractor basins or fragile, memorized shortcuts."</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/AIMO_Interpretability_Challenge-NeurIPS_2026-blueviolet?style=for-the-badge" alt="AIMO Challenge"/>
  <img src="https://img.shields.io/badge/Live_Leaderboard-12%2F14_(86%25)-success?style=for-the-badge" alt="Leaderboard"/>
  <img src="https://img.shields.io/badge/Track-Small_Models-orange?style=for-the-badge" alt="Tracks"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="MIT License"/>
</p>

---

## Overview

Large language models (LLMs) frequently arrive at correct final answers through brittle heuristics, surface pattern matching, or memorized shortcuts rather than sound deductive derivations. Consequently, standard accuracy benchmarks fail to certify genuine reasoning fidelity. While behavioral stress-testing requires generating millions of tokens across extensive prompt variations, **Heimdall** provides model-internal, white-box interpretability diagnostics that discern robust from spurious derivations without prompt rewriting or external judges.

Developed for the **AIMO Interpretability Challenge (NeurIPS 2026)**, Heimdall operationalizes three complementary internal modalities:
1. **Weight Space**: Testing loss landscape flatness via **Mantissa Precision Masking**.
2. **Hidden-State Space**: Detecting premature derivation collapse via **CoT Trajectory Geometry** ($\kappa, \beta$).
3. **Embedding Space**: Measuring continuous derivation stability via **Perturbation Trajectory Divergence**.

Because internal robustness signatures prove highly architecture-dependent, Heimdall deploys two routed multi-pipeline systems:
- **Heimdall-Bifröst**: The full tripartite bridge routing models across all three diagnostics (**12/14, 85.7%** on live leaderboard).
- **Heimdall-Gjallarhorn**: The conservative divergence-free variant sounding the alert on fragile representations (**11/14, 78.6%** on live leaderboard).

---

### 1. Mantissa Precision Masking (`pipelines/precision_pipeline.py`)
Spurious reasoning often balances on knife-edge arithmetic dependencies in parameter space, whereas genuine mathematical derivations reside in broad, flat attractor basins. 
- **Methodology**: In-place bitwise masking zeroes the lowest mantissa bits in `torch.bfloat16` weight tensors (`param.view(torch.int16).bitwise_and_(mask).view(torch.bfloat16)`).
- **Invariance Criterion**: We compare the model's clean derivation with its generation under precision-truncated weights. If the extracted answer flips, the derivation is diagnosed as non-robust.
- **Empirical Threshold**: As established in our ablation study, genuine derivations remain stable under aggressive 4-bit mantissa zeroing (`~0xF`), whereas spurious reasoning derails under the mildest perturbation.

### 2. Chain-of-Thought Trajectory Geometry (`pipelines/cot_geometry.py`)
Rather than waiting for extended rollouts (thousands of tokens), the geometric progression of hidden states over compact prefixes ($L \le 128$ tokens) carries distinct structural signals.
- **Menger Curvature ($\kappa$)**: Quantifies the extrinsic geometric turning angle between successive hidden states $h_{t-1}, h_t, h_{t+1}$. Spurious traces exhibit premature trajectory collapse and sharp erratic turns.
- **Representation-Dispersion Slope ($\beta$)**: Measures the rate of feature-space spread over tokens. Structurally grounded deductions steadily expand across latent dimensions, while memorized shortcuts remain confined to narrow subspaces.
- **Multi-Budget Logistic Ensembling**: Features extracted across token budgets $\mathcal{B} = \{64, 96, 128\}$ are fused through model-specific calibrated logistic classifiers.

### 3. Spherical Embedding Trajectory Divergence (`pipelines/geometry_pipeline.py`)
Tests how the model's hidden representation manifold responds to continuous infinitesimal noise in the input space.
- **Methodology**: Applies norm-bounded spherical Gaussian perturbations directly to input token embeddings and traces layer-wise divergence against the unperturbed trajectory.
- **Separation**: Structurally resilient derivations maintain directional coherence, whereas spurious proofs diverge rapidly away from the correct solution manifold.

---

## The Heimdall Architectures

Heimdall routes each model family to its optimal diagnostic based on empirical representation signatures:

### 1. Heimdall-Bifröst (`heimdall/bifrost`)
The complete tripartite routing bridge:
- **DeepSeek** (`deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`) $\longrightarrow$ **Mantissa Precision Masking**
- **Skywork** (`Skywork/Skywork-OR1-Math-7B`) $\longrightarrow$ **Mantissa Precision Masking**
- **Olmo** (`allenai/Olmo-3-7B-Think`) $\longrightarrow$ **Perturbation Trajectory Divergence**
- **Qwen** (`Qwen/Qwen3.5-4B`) $\longrightarrow$ **CoT Trajectory Geometry**
- **GPT-OSS-120B** (`openai/gpt-oss-120b`) $\longrightarrow$ **Intentional Error Route** (Main Track)

*Performance*: **12/14 (85.7% accuracy)** on the live competition benchmark.

### 2. Heimdall-Gjallarhorn (`heimdall/gjallarhorn`)
The conservative divergence-free variant, unifying Olmo, Skywork, and DeepSeek under weight-space precision testing:
- **DeepSeek** $\longrightarrow$ **Mantissa Precision Masking**
- **Skywork** $\longrightarrow$ **Mantissa Precision Masking**
- **Olmo** $\longrightarrow$ **Mantissa Precision Masking**
- **Qwen** $\longrightarrow$ **CoT Trajectory Geometry**
- **GPT-OSS-120B** $\longrightarrow$ **Intentional Error Route** (Main Track)

*Performance*: **11/14 (78.6% accuracy)** on the live competition benchmark.

---

## Repository Structure

```text
Robustness_test/
├── heimdall/                             # Official Heimdall Submissions
│   ├── bifrost/                          # Heimdall-Bifröst Submission
│   │   ├── solution.py                   # Codabench callable entry point: are_robust()
│   │   ├── aimo_router.py                # Dynamic model family router
│   │   ├── routes.json                   # Pipeline dispatch configuration
│   │   └── pipelines/
│   │       ├── precision_pipeline.py     # Weight mantissa bitmasking
│   │       ├── cot_geometry.py           # Hidden-state curvature & slope probe
│   │       └── geometry_pipeline.py      # Embedding trajectory divergence
│   └── gjallarhorn/                      # Heimdall-Gjallarhorn Submission
│       ├── solution.py
│       ├── aimo_router.py
│       ├── routes.json
│       └── pipelines/
│           ├── precision_pipeline.py
│           └── cot_geometry.py
├── components/                           # Official competition ingestion & scoring harness
│   ├── ingestion_program/
│   └── scoring_program/
├── scripts/                              # Build, test, and dataset utilities
│   ├── build.py                          # Submissions packager for Codabench
│   ├── run_local.py                      # Local offline evaluation harness
│   └── import_hf_dataset.py              # HF dataset sync script
│   ├── cot_geometry_calibration.py        # CoT feature extraction and probe fitting
│   └── extract_norm_divergence.py         # Embedding-perturbation feature collection
├── Dockerfile.competition                # Competition evaluation runtime environment
├── image.png                             # Heimdall artwork
└── pyproject.toml                        # Environment configuration
```

---

## Quick Start & Usage

### 1. Installation & Environment
Ensure you have Python 3.11+ and [`uv`](https://docs.astral.sh/uv/) installed:
```bash
git clone https://github.com/vishesh-kumar-singh/HEIMDALL.git
cd HEIMDALL
uv sync
```

### Reproducing CoT and divergence calibration data

The calibration utilities take a dataset directory, so `train-sample` resolves to
`data/train-sample`. They read `input/cases.jsonl` and, when present,
`reference/labels.jsonl`. The defaults save generated artifacts next to that dataset.

```bash
# One greedy 128-token rollout per case; writes data/train-sample/cot_geometry_features.jsonl
uv run python scripts/cot_geometry_calibration.py extract --dataset train-sample

# Fits the model-wise mean-of-sigmoids probe in the report appendix.
uv run python scripts/cot_geometry_calibration.py fit \
  --features data/train-sample/cot_geometry_features.jsonl \
  --output data/train-sample/cot_geometry_weights.json

# Runs the 1 clean + 10 perturbed 512-token trajectories and writes all five
# normalized divergence milestones to data/train-sample/norm_divergence_features.csv.
uv run python scripts/extract_norm_divergence.py --dataset train-sample
```

`cot_geometry_calibration.py fit` records every regularization candidate and the
selected one in its JSON output. Its defaults match the appendix: 64/96/128-token
prefixes, 10 Nelder--Mead starts, 30,000 simplex iterations, and L-BFGS-B polishing.
Use `--select balanced-accuracy` when selecting the regularization setting by local
balanced accuracy rather than the regularized training objective.

### 2. Building Submission Bundles
Submissions are packaged into deterministic, independently uploadable Codabench ZIP files via `scripts/build.py`:

```bash
# Build Main Track submissions
python3 scripts/build.py solutions

# Build Small Models Track submissions (automatically injects small.txt)
python3 scripts/build.py solutions --small
```

Packaged artifacts will appear in `dist/`:
- `dist/solution-heimdall-bifrost.zip` (Main Track)
- `dist/solution-heimdall-bifrost-small.zip` (Small Models Track)
- `dist/solution-heimdall-gjallarhorn.zip` (Main Track)
- `dist/solution-heimdall-gjallarhorn-small.zip` (Small Models Track)

### 3. Running Local Offline Evaluation
Evaluate any built ZIP archive locally using the competition evaluation container:

```bash
# Test Heimdall-Bifröst on Small Models Track with multi-GPU parallelization
CUDA_VISIBLE_DEVICES=0,1 uv run scripts/run_local.py dist/solution-heimdall-bifrost-small.zip \
  --small \
  --input-dir data/train-sample/input \
  --reference-dir data/train-sample/reference
```

---

## Execution Mechanics & Rules Compliance

- **Zero Ground Truth Required**: Completely unsupervised and reference-free at test time.
- **Zero Prompt Paraphrasing**: Operates directly on the unperturbed problem string, avoiding semantic drift and LLM judge latency.
- **Dynamic Multi-GPU Support**: The parallel execution engine automatically discovers available GPUs through `CUDA_VISIBLE_DEVICES` (supports 1, 2, or more GPUs seamlessly).
- **Strict Offline Mode**: Complies fully with competition air-gapping (`HF_HUB_OFFLINE=1`, `local_files_only=True`).
- **Deterministic & Bounded**: All generations run under greedy decoding (`do_sample=False`) with strict response caps.

---

## Evaluation Environment & Docker Reproduction

Submissions are evaluated in an offline container running on an NVIDIA RTX PRO 6000 (Blackwell architecture, `sm_120`) with 96 GB of VRAM. The exact evaluation container environment is pinned in [`Dockerfile.competition`](Dockerfile.competition).

### Pinned Runtime Dependencies
- **Base Image**: `pytorch/pytorch:2.12.1-cuda13.2-cudnn9-runtime` (pinned via manifest digest)
- **PyTorch**: `2.12.1` with CUDA `13.2`
- **Transformers**: `5.13.0`
- **Accelerate**: `1.13.0`
- **Tokenizers**: `0.22.2`
- **Scikit-Learn**: `1.8.0`
- **Hugging Face Hub**: `1.22.0`
- **Joblib**: `1.5.3`
- **Pandas**: `3.0.3`

### Building and Running via Docker
To reproduce the exact evaluation container locally:

```bash
# 1. Build the competition container image
docker build -t aimo-competition -f Dockerfile.competition .

# 2. Score a built submission inside the container
docker run --gpus all --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/dist:/app/dist \
  aimo-competition \
  python3 scripts/run_local.py /app/dist/solution-heimdall-bifrost-small.zip \
    --small \
    --input-dir /app/data/train-sample/input \
    --reference-dir /app/data/train-sample/reference
```

Evaluation containers have networking completely disabled, with offline flags preset:
- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`
- `HF_HOME=/app/data/huggingface`
- `TOKENIZERS_PARALLELISM=false`

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
