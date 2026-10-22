// Turn a pasted question into Problem JSON with a local Ollama model.
// The model only copies numbers into the schema; all maths happens in src/solvers.
import { z } from 'zod';
import { ProblemBase, type Problem } from '../schema';
import { HEADINGS } from '../solvers';
import {
  FEWSHOT, FEWSHOT_BANDIT, FEWSHOT_BANDIT_TEXT, FEWSHOT_DQN, FEWSHOT_DQN_TEXT, FEWSHOT_MF, FEWSHOT_MF_TEXT, FEWSHOT_NSTEP, FEWSHOT_NSTEP_TEXT, FEWSHOT_TEXT,
} from '../fixtures/fewshot';
import { detectTasks } from './keywords';

export const DEFAULT_MODEL = 'qwen2.5:7b';

const TASK_HINTS: Record<string, string> = {
  return_Gt: 'discounted return G_t from a reward sequence (fill part.rewards)',
  bellman_v: 'one Bellman expectation backup V(s) from given values',
  q_from_v: 'compute Q(s,a) from given state values V',
  bellman_optimality: 'one Bellman optimality backup V*(s) = max_a Q(s,a)',
  policy_evaluation: 'iterative policy evaluation sweeps V_k for a fixed policy ("compute V1", "V2", "k iterations")',
  policy_evaluation_inplace: 'in-place / asynchronous / Gauss-Seidel policy evaluation',
  policy_evaluation_linear: 'exact policy evaluation by solving the system of linear Bellman equations',
  policy_improvement: 'one policy improvement step: greedy policy from given V',
  policy_iteration: 'full policy iteration (evaluation + improvement, repeated)',
  value_iteration: 'value iteration sweeps',
  greedy_action: 'pick the greedy action from given Q-values',
  epsilon_greedy_probs: 'action probabilities under an epsilon-greedy policy (needs Q and epsilon)',
  best_state: 'which state has the highest value',
  mc_prediction_first_visit: 'Monte Carlo prediction of V from episodes, first-visit (the default for "Monte Carlo")',
  mc_prediction_every_visit: 'Monte Carlo prediction of V from episodes, every-visit',
  mc_control_epsilon_greedy: 'Monte Carlo control: Q from episode returns, then epsilon-greedy policy improvement',
  td0_prediction: 'TD(0) updates of V from episode steps',
  sarsa: 'SARSA updates of Q (uses the next action actually taken)',
  q_learning: 'Q-learning updates of Q (uses max over next actions)',
  bandit_sample_average: 'bandit arm estimates as the average of each arm\'s rewards',
  bandit_incremental: 'bandit incremental update Q <- Q + (1/n)(R - Q) after each pull',
  bandit_epsilon_greedy: 'probability of choosing each bandit arm under epsilon-greedy',
  bandit_ucb: 'UCB arm selection (needs c, and t if given)',
  ab_test: 'A/B test: conversion rates and z-test (variants are actions; arms.n = visitors, arms.wins = conversions)',
  nstep_td: 'n-step TD updates of V from episodes (needs n, alpha)',
  nstep_sarsa: 'n-step SARSA updates of Q from episodes (needs n, alpha)',
  nstep_offpolicy_is: 'off-policy n-step SARSA with importance sampling ratio (policy = target pi, behavior = b)',
  tree_backup: 'n-step tree backup (policy = target pi)',
  dyna_q: 'Dyna-Q: Q-learning on real steps plus n planning updates from the learned model',
  dqn_target_loss: 'DQN targets y = r + gamma max Q_target(s\', a\') and mean squared loss over a replay minibatch',
  theory: 'explain / compare / define: no calculation',
};

const SYSTEM = `You extract reinforcement learning exam questions into JSON. Copy data exactly. Never compute answers.

Rules:
- states / actions: use the ids written in the question (e.g. S1, A1). Put descriptive words in "name".
- terminal: ids of terminal / absorbing states.
- transitions: one row per (s, a, s2) with probability p and reward r. If a reward is given per (s, a), copy it into every row of that (s, a). Probabilities of each (s, a) must add up to 1.
- policy: rows (s, a, p). A deterministic policy has p = 1. Leave empty if no policy is given.
- V0: initial values if given (V0 = 0 for all states means rows with v = 0). Q: given Q-values only.
- epsilon, alpha (α, step size / learning rate), c (UCB constant), t (UCB time step): copy if they appear anywhere, including inside a sub-question.
- episodes: each step is (s, a, r) = state, action taken, reward received after it. "end" is the state reached after the last step. "endAction" is the action taken there, only if given. Leave "a" empty if the question gives no actions.
- pulls: bandit history, one (a, r) row per pull, in order. arms: per-arm snapshot (q estimate, n pull count / visitors, wins conversions), only if given. Bandit arms are listed as actions; bandit questions have no states.
- n: the n of n-step methods, or the number of Dyna-Q planning steps per real step. behavior: behaviour policy b(a|s) for off-policy methods (then "policy" is the target policy). planning: (s, a) pairs Dyna-Q replays, only if listed.
- DQN: replay = minibatch rows (s, a, r, s2, done). Q = online network predictions, Qtarget = target network values.
- parts.episodes: 1-based episode numbers a part uses ("episode 2" -> [2]); [] means all.
- parts: one per sub-question (a, b, c, ...), in order. "label" is the letter, "text" the sub-question.
  "iterations" is the final iteration index the part asks for: "compute V2" -> 2, "2 iterations" -> 2.
  "states" restricts a part to specific states (e.g. "in S1" -> ["S1"]); otherwise [].
  "rewards" is only for return_Gt.
- task must be one of:
${Object.entries(TASK_HINTS).map(([k, v]) => `  ${k}: ${HEADINGS[k as keyof typeof HEADINGS]}: ${v}`).join('\n')}`;

const SCHEMA = z.toJSONSchema(ProblemBase, { io: 'output' });

const MF = /^(mc_|td0|sarsa|q_learning)/;
const BANDIT = /^(bandit_|ab_test)/;
const NSTEP = /^(nstep_|tree_backup|dyna_q)/;
const DQN = /^dqn/;

/** Show the model the worked example(s) closest to this question, to keep the prompt short. */
function examplesFor(text: string) {
  const found = detectTasks(text);
  const picks = [
    found.some((t) => MF.test(t)) && ([FEWSHOT_MF_TEXT, FEWSHOT_MF] as const),
    found.some((t) => BANDIT.test(t)) && ([FEWSHOT_BANDIT_TEXT, FEWSHOT_BANDIT] as const),
    found.some((t) => NSTEP.test(t)) && ([FEWSHOT_NSTEP_TEXT, FEWSHOT_NSTEP] as const),
    found.some((t) => DQN.test(t)) && ([FEWSHOT_DQN_TEXT, FEWSHOT_DQN] as const),
  ].filter((x) => x !== false);
  return (picks.length ? picks : [[FEWSHOT_TEXT, FEWSHOT] as const]).flatMap(([q, a]) => [
    { role: 'user', content: q },
    { role: 'assistant', content: JSON.stringify(a) },
  ]);
}

export async function listModels(base = '/ollama'): Promise<string[]> {
  const r = await fetch(`${base}/api/tags`);
  if (!r.ok) throw new Error(`Ollama answered ${r.status}`);
  const j = (await r.json()) as { models?: { name: string }[] };
  return (j.models ?? []).map((m) => m.name);
}

export async function extract(text: string, model: string, opts: { base?: string; signal?: AbortSignal } = {}): Promise<Problem> {
  const r = await fetch(`${opts.base ?? '/ollama'}/api/chat`, {
    method: 'POST',
    signal: opts.signal,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model,
      stream: false,
      format: SCHEMA,
      options: { temperature: 0, num_ctx: 8192 },
      messages: [{ role: 'system', content: SYSTEM }, ...examplesFor(text), { role: 'user', content: text }],
    }),
  });
  if (!r.ok) throw new Error(`Ollama answered ${r.status}: ${(await r.text()).slice(0, 200)}`);
  const j = (await r.json()) as { message?: { content?: string } };
  let raw: unknown;
  try {
    raw = JSON.parse(j.message?.content ?? '');
  } catch {
    throw new Error('The model did not return valid JSON. Try again or use a larger model.');
  }
  // Structural check only; cross-field problems are left for the review form to show.
  const parsed = ProblemBase.safeParse(raw);
  if (!parsed.success) throw new Error(`The model returned data in the wrong shape: ${parsed.error.issues[0]?.message}`);
  return clean(parsed.data, text);
}

const numAfter = (text: string, re: RegExp) => {
  const m = re.exec(text);
  return m ? Number(m[1]) : undefined;
};

/** Deterministic fixes for things small models often miss; the question text wins over the model. */
export function clean(p: Problem, text: string): Problem {
  // "γ = 0.9", "discount factor (γ): 0.9", "discount 0.8", "learning rate of 0.5"
  const V = String.raw`\s*(?:\([^)]*\)\s*)?(?:[=:]|of|is)?\s*(\d*\.?\d+)`;
  const gamma = numAfter(text, new RegExp(String.raw`(?:γ|gamma|discount(?:\s+factor)?)` + V, 'i'));
  const epsilon = numAfter(text, new RegExp(String.raw`(?:ε|epsilon)` + V, 'i'));
  const alpha = numAfter(text, new RegExp(String.raw`(?:α|alpha|step[-\s]size|learning\s+rate)` + V, 'i'));
  const n = numAfter(text, /\b(\d+)[-\s]step\b/i) ?? numAfter(text, /\bn\s*=\s*(\d+)/) ?? numAfter(text, /\b(\d+)\s+planning\s+(?:steps?|updates?)/i);
  const ucb = /\bUCB\b|upper\s+confidence/i.test(text);
  const c = ucb ? numAfter(text, /\bc\s*=\s*(\d*\.?\d+)/) : undefined;
  const t = ucb ? numAfter(text, /\bt\s*=\s*(\d+)/) : undefined;
  return {
    ...p,
    gamma: gamma ?? p.gamma,
    epsilon: epsilon ?? p.epsilon,
    alpha: alpha ?? p.alpha,
    c: c ?? p.c,
    n: n ?? p.n,
    t: t ?? p.t,
    policy: p.policy.filter((x) => x.p > 0),
    parts: p.parts.map((pt) => ({ ...pt, iterations: pt.iterations ?? iterationsIn(pt.text) })),
  };
}

/** "compute V2(S1)" -> 2, "2 iterations" / "two sweeps" -> 2. */
function iterationsIn(text: string): number | undefined {
  const words: Record<string, number> = { one: 1, two: 2, three: 3, four: 4, five: 5 };
  const ns = [
    ...[...text.matchAll(/\bV_?\{?(\d+)\}?\s*\(/g)].map((m) => +m[1]),
    ...[...text.matchAll(/\b(\d+|one|two|three|four|five)\s+(?:iterations?|sweeps?|steps? of (?:iterative )?(?:policy evaluation|value iteration))/gi)].map(
      (m) => words[m[1].toLowerCase()] ?? +m[1],
    ),
  ].filter((n) => n >= 1);
  return ns.length ? Math.max(...ns) : undefined;
}
