# RL Explainer

Paste a reinforcement learning exam question and watch it get solved step by step. The app shows which formula applies, flies each given value into it, and reduces the arithmetic one operation at a time, the way you would by hand.

All maths is done by deterministic code (`src/solvers`). A language model is only used to read the question.

## Run

```sh
npm install
npm run dev     # http://localhost:5173
npm test        # solver tests against the IA1 answers
```

## Status

- Phase 1 (done): dynamic programming solvers (policy evaluation, in-place evaluation, exact linear-system evaluation, policy improvement, policy iteration, value iteration, Q from V, greedy and ε-greedy, discounted return), step player with animations, data review form, IA1 examples.
- Phase 2: question extraction with a local LLM (Ollama).
- Phase 3: Monte Carlo, bandits, TD(0), SARSA, Q-learning.
- Phase 4: n-step methods, tree backup, Dyna-Q, DQN.
- Phase 5: predict-the-step game mode, theory cards.
