# RL Explainer

Paste a reinforcement learning exam question and watch it get solved step by step. The app shows which formula applies, flies each given value into it, and reduces the arithmetic one operation at a time, the way you would by hand.

All maths is done by deterministic code (`src/solvers`). A language model is only used to read the question.

## Run

```sh
npm install
npm run dev     # http://localhost:5173
npm test        # solver tests against the IA1 answers
```

### Question extraction (optional)

Needs [Ollama](https://ollama.com) running locally:

```sh
ollama serve
ollama pull qwen2.5:7b                 # default model, ~4.7 GB
OLLAMA=1 npx vitest run src/extract    # check extraction of the IA1 questions
```

Without Ollama, "Detect parts only" splits the sub-questions and guesses each algorithm from keywords. You then type the numbers into the form.

## Status

- Phase 1 (done): dynamic programming solvers (policy evaluation, in-place evaluation, exact linear-system evaluation, policy improvement, policy iteration, value iteration, Q from V, greedy and ε-greedy, discounted return), step player with animations, data review form, IA1 examples.
- Phase 2 (done): question extraction with a local LLM (Ollama), keyword cross-check, offline part detection.
- Phase 3 (done): Monte Carlo prediction (first/every-visit, constant-α) and control, TD(0), SARSA, Q-learning, bandits (sample average, incremental, ε-greedy, UCB), A/B test, episode and pull timeline, practice examples.
- Phase 4 (done): n-step TD and SARSA, off-policy n-step SARSA with importance sampling, n-step tree backup, Dyna-Q, DQN targets and loss.
- Phase 5: predict-the-step game mode, theory cards.
