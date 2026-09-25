

# Lean 4 pipeline

Notebook: [`qwen35_4b_generate.ipynb`](qwen35_4b_generate.ipynb) (Kaggle, GPU, Internet on)

For the AIMO Interpretability Challenge, exploring the possibility of proving model robustness by asking it to generate a Lean4 program. Lean4 is a proof assistant programming language and can interface with python. So it becomes relatively easy to check the output, as if the output compiles, the proof is fully robust and we have our answer.
## Steps

1. **Setup.** Upgrade `transformers` and load `Qwen/Qwen3.5-4B`. It uses bf16 if the GPU supports it and fp16 otherwise.
2. **Prompt.** A system prompt tells the model to solve the problem in 5 steps (understand, explore, plan, execute, verify) while thinking. The final output must be a single ` ```lean4 ` block with:
   - `import Mathlib` at the top
   - a `def answer` for the answer
   - a `theorem` that states the problem and proves it
   - no `sorry` or new axioms
3. **Generation (two phases).**
   - **Think:** the model reasons for up to `THINK_BUDGET` (6144) tokens. If it runs out, reasoning is cut off and the model is told to answer.
   - **Answer:** the reply starts pre-filled with ` ```lean4\nimport Mathlib `. The model then writes the program, up to `ANSWER_BUDGET` (2048) tokens, and generation stops at the closing ` ``` `.
   - Sampling uses temperature 0.6, top-p 0.95, top-k 20, because greedy decoding tends to loop.
4. **Extract.** The Lean code is saved to `solution.lean`. The reasoning is hidden unless `SHOW_THINKING = True`.
5. **Verify.** LeanInteract compiles the file against Mathlib (`LEAN_VERSION = v4.19.0`). A proof passes only if it has:
   - no errors
   - no `sorry`(this is a copout that makes the program compile without proof)
   - only the standard axioms (`propext`, `Classical.choice`, `Quot.sound`), checked with `#print axioms` (so the model does not invent new maths to justify its answers)

## Results so far (on full-small-labeled-dedup)

| Problem | Label | Lean program |
|---|---|---|
| 1 |False  | NOT VERIFIED |
| 2| True | VERIFIED |
| 3| True | VERIFIED|
| 4| False | NOT VERIFIED|
| 5| False | NOT VERIFIED|
| 6| False | NOT VERIFIED|
| 7| False | NOT VERIFIED|
| 8| True | VERIFIED*|
| 9| True | NOT VERIFIED|
| 10| False | NOT VERIFIED|

Overall result is 90%, with some caveats
- **False negatives may exist.** In problem 8, it ran out of tokens, but the half-finished proof was on the right track and had the right answer, and when I ran it after increasing the token limit by 2048, it gave the full proof. I cannot say where else this could have happened. It was no where near on the problems labeled false, but maybe with more tokens, it could solve those as well
- **Qwen-3.5-4B is far more limited than the 8B version.** The more advanced model may be able to solve all of these problems, given enough time and compute
- **Average time to answer is around 10 minutes** 
