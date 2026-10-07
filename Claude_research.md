# Estimating reasoning robustness without ground truth: overview and approach

This is a **correctness/uncertainty estimation problem for reasoning traces**, with white-box access, a hard compute budget, and almost no labels. The literature is mostly under "hallucination detection", "uncertainty quantification", "probing for correctness", and "CoT faithfulness". I'm citing from memory, so verify titles and years before relying on them. I can also search for the challenge page and recent 2025-26 papers if you want.

## 1. Pin down what "robust" means operationally

Your results depend on this more than on any model choice. Plausible label definitions:

- **Sampled accuracy**: the model gets the problem right in most of k samples (robust) vs. rarely or inconsistently (brittle).
- **Perturbation stability**: the answer survives paraphrases, number or name changes, and added distractors.
- **Faithfulness**: the answer is actually caused by the visible reasoning rather than a shortcut or memorized answer.

If you can infer from the 9 labeled cases which definition the organizers used (compare labels against the model's sample agreement, for example), you can generate far more training data yourself. See section 4.

## 2. Signal families and key papers

**A. Behavioral consistency (cheap, strong baseline)**
- Self-consistency (Wang et al., 2022). Agreement among sampled answers is the best-known correctness proxy.
- Semantic entropy (Kuhn et al., 2023; Farquhar et al., *Nature* 2024). Entropy over meaning-equivalent answer clusters. Semantic Entropy Probes (Kossen et al., 2024) distill it into a single-pass probe.
- SelfCheckGPT (Manakul et al., 2023). Consistency across samples as a hallucination signal.
- Perturbation fragility: GSM-Symbolic (Mirzadeh et al., 2024), GSM-Plus, and "LLMs Can Be Easily Distracted by Irrelevant Context" (Shi et al., 2023).

**B. Token-level confidence**
- Kadavath et al. 2022, "Language Models (Mostly) Know What They Know".
- DeepConf, "Deep Think with Confidence" (Fu et al., 2025). Local and group confidence over a trace, and its lowest window, predict failure.
- Self-certainty (Kang et al., 2025) and "Don't Overthink It: Preferring Shorter Thinking Chains" (Hassid et al., 2025). Shorter traces are often more correct.

**C. Internal-state probing (white-box)**
- Azaria & Mitchell 2023, "The Internal State of an LLM Knows When It's Lying".
- Burns et al. 2022 (CCS) and Marks & Tegmark 2023 ("Geometry of Truth"). Truth-like directions exist in activations.
- Orgad et al. 2024, "LLMs Know More Than They Show". Truthfulness information concentrates at specific tokens, especially the exact-answer tokens.
- Zhang et al. 2025, "Reasoning Models Know When They're Right", probes hidden states for answer correctness in reasoning models.
- Afzal et al. 2025, "Knowing Before Saying", shows CoT success is predictable from representations before completion.
- Lookback Lens (Chuang et al., 2024) is an attention-based variant.

**D. Trace structure and dynamics (specific to R1-style models)**
- Overthinking ("Do NOT Think That Much for 2+3=?", Chen et al., 2024) and underthinking ("Thoughts Are All Over the Place", Wang et al., 2025).
- Features: count of "wait/hmm" backtracks, repetition loops, truncation at max length, and answer convergence. For convergence, force an answer at several prefixes of the trace. If the answer is stable early and stays stable, that's a robustness signal. If it flips, that's a brittleness signal.

**E. Faithfulness and spurious reasoning**
- Turpin et al. 2023, Lanham et al. 2023 (corrupt or truncate the CoT and check whether the answer changes), Chen et al. 2025 ("Reasoning Models Don't Always Say What They Think"), Arcuschin et al. 2025, and Thought Anchors (Bogdan et al., 2025).
- **Contamination and spurious success**: "Spurious Rewards" (Shao et al., 2025) and Wu et al. 2025 on Qwen-math contamination. This matters because two of your models are Qwen-based. A model can be right for the wrong reason on memorized problems.
- **Mechanistic work on shortcuts**: Nikankin et al. 2024, "Arithmetic Without Algorithms" (bag of heuristics), and Kantamneni & Tegmark 2025 (trig representations of addition).

**F. Verifiers**
- Let's Verify Step by Step (Lightman et al., 2023) and Math-Shepherd (process reward models). These need another model, so they're expensive under your time limit, but a small PRM or judge is a possible feature.

## 3. Practical pipeline under your constraints

1. **Cheap sampling.** Use vLLM with k = 4-8 samples per problem and a capped token budget. Compute answer agreement, semantic-entropy-style cluster entropy, truncation rate, and length statistics.
2. **One forward pass for internals.** Run the model on the problem plus its greedy trace and extract hidden states at a few layers at the answer token and at the end of the trace. Also extract token-entropy and min-window confidence features.
3. **Small, heavily regularized classifier.** Use 10-30 features, logistic regression, and rank-normalized features. Avoid high-dimensional probes trained on tiny data.
4. **Per-model calibration.** Different architectures have different confidence scales. Normalize within the problem set at test time, since you see all problems together.
5. **Threshold by quantile.** You don't know the prevalence, so predict a ranking and pick the cutoff from a prior or from your pseudo-labeled data.

**Compute sanity check:** 8 GPUs for 3600s is roughly 8 GPU-hours. With batched vLLM, an 8B model can plausibly produce tens of thousands of tokens per second per node. That makes 8 samples × ~8k tokens × a few hundred problems feasible, but benchmark it. The 120B model will need tensor parallelism and a much smaller sample count.

## 4. Escaping data starvation

You don't have to learn from 9 examples. The constraint is **inference time offline**, not development time.

- Build your own labeled corpus now, with internet, from public math sets with ground truth (MATH, GSM8K, past AIME/olympiad problems, and so on). Run each target model on them and label each problem with the same definition the challenge appears to use.
- Train the detector on that corpus. Ship only the small weights or feature code in the Docker image.
- Treat the 9 public cases purely as a **sanity-check validation set**, not training data. If your detector scores well on your synthetic labels but poorly on the 9, your label definition has likely drifted from theirs.
- Keep problem-source distribution in mind. Hidden problems may be harder or differently distributed than MATH. Leave out whole difficulty tiers or sources when validating to test for shift.
- Use leave-one-source-out CV, report bootstrap confidence intervals, and prefer features that behave consistently across all four models (agreement, convergence, and entropy transfer better than raw probe directions).

## 5. Common failure modes

- **Overfitting through feature count**: a probe on 4096-dim states with 9 labels will memorize.
- **Length confounds**: wrong traces are often longer or truncated, so length can look like signal while really measuring difficulty. Check that your features add value beyond length.
- **Label noise at the boundary**: problems with 40-60% sampled accuracy are inherently ambiguous, so expect a ceiling on achievable AUROC.
- **Leaderboard overfitting**: with few labels, tuning against the hidden leaderboard is just noise-fitting. Trust your own larger validation set.

## 6. Suggested reading order

1. Kadavath 2022 → Farquhar 2024 (uncertainty fundamentals)
2. Orgad 2024 → Zhang 2025 → Afzal 2025 (probing for correctness in reasoning models)
3. DeepConf 2025 (confidence from traces)
4. Lanham 2023 → Chen 2025 (faithfulness)
5. Shao 2025 / Wu 2025 (contamination, spurious correctness)

If you tell me what the 9 labeled cases look like (label balance, the models, how the labels relate to sampled accuracy), I can help you work out which definition they used and design the feature set and validation protocol around it. I can also run a literature search for any AIMO-specific write-ups.