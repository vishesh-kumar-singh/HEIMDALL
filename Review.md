# Architectural teardown (no new data assumed)

I'm citing from memory, so verify titles and years. I'm also working only from your description, not the code.

## 1. The core premise inverts the standard definition of robust

In stability theory and adversarial robustness, low sensitivity to perturbation means robust. Your geometric pipeline labels low divergence (a "frozen sink") as False and high divergence as True. That needs an actual argument that spurious reasoning is a noise-insensitive attractor.

There is a partial one. Memorized outputs are produced early and confidently (Haviv et al. 2023, on idiom memorization; Carlini et al. 2021 and 2023, on quantifying memorization). But a correct, easy, confidently solved problem has the same signature, so the signal is confounded with easiness.

The leaderboard inversion you saw (`True, True, False` where you expected the reverse) fits this. A feature whose sign flips between train and test isn't tracking the construct.

## 2. Perturbation by Gaussian embedding noise

- **Random noise is a weak probe.** Fawzi et al. (2016, "Robustness of classifiers: from adversarial to random noise") show that robustness to random noise exceeds adversarial robustness by roughly a √d factor. In 4096 dimensions, isotropic noise is nearly orthogonal to any direction the model cares about. NEFTune (Jain et al. 2023) shows models tolerate embedding noise well enough that it works as a training regularizer.
- **σ doesn't mean the same thing across models.** Embedding norms differ across architectures, and Gemma multiplies embeddings by √d_model. Unless σ is set relative to each model's embedding norm, the same noise is a different effective perturbation for each model. This alone could explain why thresholds don't transfer.
- **Free-running divergence isn't state-space sensitivity.** Once one sampled or argmax token differs, the two trajectories receive different inputs, and everything downstream diverges. That is a token-flip detector, not a Lyapunov-style measurement. Decoding discrete tokens breaks the continuous-dynamics analogy. A clean measurement needs teacher forcing on the clean trace.
- **`div_500` is position-specific.** It is undefined for short traces and mixes length with sensitivity.

## 3. Hidden-state geometry and TDA

- **The path-to-diameter ratio is length-confounded.** If hidden states behave even roughly like a random walk, path length scales as n and diameter as √n, so the ratio grows as √n. The score tracks the token count.
- **PCA and intrinsic dimension (ID) depend on the sample count, and your samples are autocorrelated.** Consecutive tokens are highly correlated, so the effective sample size is far below the token count. Valeriani et al. (2023) and Ansuini et al. (2019) show that ID varies systematically with layer depth, not with correctness.
- **A 1D collapse on both classes is the signature of a few dominant dimensions.** Anisotropy (Ethayarajh 2019), rogue dimensions (Timkey & van Schijndel 2021), and massive activations at BOS and delimiter tokens (Sun et al. 2024) dominate cosine similarity and the first principal component. These are architecture artifacts and vary across the four models.
- **Minimum cosine similarity and velocity spikes** mostly detect special tokens and formatting boundaries such as `</think>` and newlines, not reasoning events.
- **Layer choice is a free hyperparameter.** Probing studies find correctness information concentrated at specific layers and tokens (Orgad et al. 2024). Picking a layer by hand on 9 examples is another degree of freedom.

## 4. Answer-flip pipelines

**Forced-answer checkpoints (truncate, force decode)**
- Truncating mid-`<think>` and appending a closing tag plus "Final answer" is off-distribution. Lanham et al. (2023) found early-answering results are inconsistent across tasks and sensitive to the forcing format.
- An answer that is stable from token 0 means the answer was available before reasoning. That fits memorization, trivial problems, and strong priors equally well.
- At least one flip occurs on any problem where the model had to derive the answer. So ">0 flips" mostly measures "needed computation," meaning difficulty.

**Temperature-sampled flips**
- Flips depend on temperature and top-p. A low temperature gives zero flips everywhere, and a high one gives flips everywhere.
- With small k, binarizing is statistically brutal. If a problem's answers split 90/10, then with k=4 all samples agree about 66% of the time (0.9⁴ + 0.1⁴). Even a 50/50 problem shows zero flips about 12.5% of the time. "0 vs >0 flips" is mostly sampling noise.
- Dichotomizing a continuous quantity throws away information (MacCallum et al. 2002).

**The sign contradiction.** `answer-flips` treats flips as robust, while `stability-submission` treats agreement as robust. In the semantic-entropy and self-consistency literature (Wang et al. 2022; Kuhn et al. 2023; Farquhar et al. 2024), agreement predicts correctness and disagreement predicts error. If flips really mean robust, the labels measure something unrelated to correctness. Either way, the two methods can't both be measuring one construct, and "which one wins on which model" is a symptom.

## 5. The ML layer

- **Probes and trees on 9 examples.** Hewitt & Liang (2019) show probe accuracy reflects probe capacity, not representation. Farquhar et al. (2023, "Challenges with Unsupervised LLM Knowledge Discovery") show truth-probing methods latch onto whichever feature is most prominent, not truth. Varoquaux (2018) shows cross-validation error bars at tiny n are too wide to compare methods.
- **Selection bias.** Feature × layer × threshold × model on 9 points means some combination will separate 2 from 7 by chance. Gelman & Loken (2013, "garden of forking paths") describe this. Each leaderboard submission adds adaptive overfitting (Dwork et al. 2015).
- **The base rate is unknown.** A 7/2 split is 78% True. If the hidden prevalence differs, any fixed threshold or tree is miscalibrated, and your metric (accuracy, F1, or AUROC) changes what's optimal.

## 6. Routing and ensembling

Per-model routing (DeepSeek→flips, Skywork→stability, Olmo→geometry) is model selection on 9 points per model, which means you're effectively fitting the router to the leaderboard. A measurement that only works when you pick it per architecture suggests it isn't measuring the construct. It's also unsupported architecturally: two of your models are long-CoT reasoning models, while Olmo-Instruct and Gemma-3 are non-reasoning instruct models with short traces. `<think>` length, backtrack counts, and checkpoint forcing don't mean the same thing across them. Routing the unknown Qwen model to "pseudo-lean" is a placeholder.

## 7. Judge-based equivalence

LLMs struggle to verify their own reasoning (Huang et al. 2023, "LLMs cannot self-correct reasoning yet") and favor their own outputs (Panickssery et al. 2024). Answer equivalence doesn't need a judge. Use deterministic parsing and symbolic normalization (a math-verify-style library or sympy), which also saves your time budget.

## 8. What to do with no new labels

All of this is label-free or one-bit-fit:

1. **Pick the sign from theory, not the 9 cases.** Use the 9 only to sanity-check the sign, never to fit thresholds.
2. **Use unit-weighted rank aggregation, not fitted models.** Dawes (1979, "The robust beauty of improper linear models") shows equal-weight composites beat fitted weights at small n. Rank each unsupervised signal within the test set and average the ranks. Candidates are answer-cluster entropy, DeepConf-style lowest-window confidence, and answer convergence under prefixes.
3. **Normalize within the test set.** All problems arrive together, so z-score or rank each feature per model. That removes scale differences without any labels.
4. **Residualize on length within the test set.** Regress each score on trace length across the hidden problems and keep the residual. This removes the length confound with no labels.
5. **If geometry stays, fix the measurement.** Use teacher-forced perturbations on the clean trace, σ relative to the embedding norm, fixed token windows rather than whole trajectories, and exclude massive-activation tokens.
6. **Cut by quantile, not absolute threshold,** using a stated prior on prevalence.
7. **Procedural generation, if allowed.** If "no more data" doesn't rule out synthesizing problems offline (templated or sympy-generated with known answers, as in GSM-Symbolic or Reasoning Gym), that gives labels with no external data.

My expectation is that a rank-aggregated behavioral score will beat any threshold-fitted geometric rule on unseen data, but with 9 examples you can't verify that. Treat any result as unconfirmed.