// Turn a pasted question into Problem JSON with a local Ollama model.
// The model only copies numbers into the schema; all maths happens in src/solvers.
import { z } from 'zod';
import { ProblemBase, type Problem } from '../schema';
import { HEADINGS } from '../solvers';
import { FEWSHOT, FEWSHOT_TEXT } from '../fixtures/fewshot';

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
  theory: 'explain / compare / define: no calculation',
};

const SYSTEM = `You extract reinforcement learning exam questions into JSON. Copy data exactly. Never compute answers.

Rules:
- states / actions: use the ids written in the question (e.g. S1, A1). Put descriptive words in "name".
- terminal: ids of terminal / absorbing states.
- transitions: one row per (s, a, s2) with probability p and reward r. If a reward is given per (s, a), copy it into every row of that (s, a). Probabilities of each (s, a) must add up to 1.
- policy: rows (s, a, p). A deterministic policy has p = 1. Leave empty if no policy is given.
- V0: initial values if given (V0 = 0 for all states means rows with v = 0). Q: given Q-values only.
- epsilon: if ε appears anywhere in the question, including inside a sub-question.
- parts: one per sub-question (a, b, c, ...), in order. "label" is the letter, "text" the sub-question.
  "iterations" is the final iteration index the part asks for: "compute V2" -> 2, "2 iterations" -> 2.
  "states" restricts a part to specific states (e.g. "in S1" -> ["S1"]); otherwise [].
  "rewards" is only for return_Gt.
- task must be one of:
${Object.entries(TASK_HINTS).map(([k, v]) => `  ${k}: ${HEADINGS[k as keyof typeof HEADINGS]}: ${v}`).join('\n')}`;

const SCHEMA = z.toJSONSchema(ProblemBase, { io: 'output' });

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
      messages: [
        { role: 'system', content: SYSTEM },
        { role: 'user', content: FEWSHOT_TEXT },
        { role: 'assistant', content: JSON.stringify(FEWSHOT) },
        { role: 'user', content: text },
      ],
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
  const gamma = numAfter(text, /(?:γ|gamma|discount\s+factor)\s*(?:\(γ\)\s*)?[=:]\s*(\d*\.?\d+)/i);
  const epsilon = numAfter(text, /(?:ε|epsilon)\s*=\s*(\d*\.?\d+)/i);
  return {
    ...p,
    gamma: gamma ?? p.gamma,
    epsilon: epsilon ?? p.epsilon,
    policy: p.policy.filter((x) => x.p > 0),
  };
}
