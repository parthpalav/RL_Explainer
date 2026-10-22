# RL Explainer

Paste a reinforcement learning exam question and watch it get solved step by step. The app shows which formula applies, flies each given value into it, and reduces the arithmetic one operation at a time, the way you would by hand.

All maths is done by deterministic code (`src/solvers`). A language model is only used to read the question.

## Run

```sh
npm install
npm run dev     # http://localhost:5173
npm test        # solver tests against the IA1 answers
npm run verify  # 1200 random problems vs an independent Python reference (needs numpy)
```

### Question extraction (optional)

Needs [Ollama](https://ollama.com) running locally:

```sh
ollama serve
ollama pull qwen2.5:7b                 # default model, ~4.7 GB
OLLAMA=1 npx vitest run src/extract    # check extraction of the IA1 questions
```

Without Ollama, "Detect parts only" splits the sub-questions and guesses each algorithm from keywords. You then type the numbers into the form.

## Using it

1. Paste a question, or pick an example from **Load example…**.
2. Click **Extract with local LLM** (or **Detect parts only** and type the numbers in).
3. Check the extracted data in the form, then click **Solve step by step**.
4. Use ▶ Play, ← → or the step list. Turn on **Predict mode** to test yourself before each calculation is revealed.

## Status

- Phase 1 (done): dynamic programming solvers (policy evaluation, in-place evaluation, exact linear-system evaluation, policy improvement, policy iteration, value iteration, Q from V, greedy and ε-greedy, discounted return), step player with animations, data review form, IA1 examples.
- Phase 2 (done): question extraction with a local LLM (Ollama), keyword cross-check, offline part detection.
- Phase 3 (done): Monte Carlo prediction (first/every-visit, constant-α) and control, TD(0), SARSA, Q-learning, bandits (sample average, incremental, ε-greedy, UCB), A/B test, episode and pull timeline, practice examples.
- Phase 4 (done): n-step TD and SARSA, off-policy n-step SARSA with importance sampling, n-step tree backup, Dyna-Q, DQN targets and loss.
- Phase 5 (done): predict mode (pick the formula and the value before the working is shown, with a score), revision cards for theory questions, phone layout and dark mode.
