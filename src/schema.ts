import { z } from 'zod';

// Every task id the extractor may pick. Solvers live in src/solvers/index.ts.
export const TASKS = [
  'return_Gt',
  'bellman_v',
  'q_from_v',
  'bellman_optimality',
  'policy_evaluation',
  'policy_evaluation_inplace',
  'policy_evaluation_linear',
  'policy_improvement',
  'policy_iteration',
  'value_iteration',
  'greedy_action',
  'epsilon_greedy_probs',
  'best_state',
  'mc_prediction_first_visit',
  'mc_prediction_every_visit',
  'mc_control_epsilon_greedy',
  'td0_prediction',
  'sarsa',
  'q_learning',
  'bandit_sample_average',
  'bandit_incremental',
  'bandit_epsilon_greedy',
  'bandit_ucb',
  'ab_test',
  'theory',
] as const;
export type Task = (typeof TASKS)[number];

const Id = z.string().trim().min(1);

export const PartSchema = z.object({
  label: z.string().default(''),
  task: z.enum(TASKS),
  // Final iteration index the part asks for (e.g. "compute V2" -> 2, "2 iterations" -> 2).
  iterations: z.number().int().min(1).max(50).optional(),
  // Restrict the part to these states; empty means all non-terminal states.
  states: z.array(Id).default([]),
  // Reward sequence R1, R2, ... for return_Gt.
  rewards: z.array(z.number()).default([]),
  // 1-based episode numbers this part uses; empty means all episodes.
  episodes: z.array(z.number().int().min(1)).default([]),
  // Free text of the sub-question, shown above its steps.
  text: z.string().default(''),
});
export type Part = z.infer<typeof PartSchema>;

const Named = z.object({ id: Id, name: z.string().default('') });

// Structure only: what the LLM must return. Cross-field checks live in ProblemSchema.
export const ProblemBase = z.object({
    title: z.string().default(''),
    states: z.array(Named).default([]), // bandit questions have no states
    actions: z.array(Named).default([]),
    terminal: z.array(Id).default([]),
    gamma: z.number().min(0).max(1).default(0.9),
    transitions: z.array(z.object({ s: Id, a: Id, s2: Id, p: z.number().min(0).max(1), r: z.number() })).default([]),
    policy: z.array(z.object({ s: Id, a: Id, p: z.number().min(0).max(1) })).default([]),
    V0: z.array(z.object({ s: Id, v: z.number() })).default([]),
    Q: z.array(z.object({ s: Id, a: Id, v: z.number() })).default([]),
    epsilon: z.number().min(0).max(1).optional(),
    alpha: z.number().min(0).max(1).optional(),
    // Sampled episodes: each step is (S_t, A_t, R_{t+1}). "end" is the state after the last step
    // ('' = terminal); "endAction" is A' there, needed by SARSA when the episode is cut short.
    episodes: z
      .array(
        z.object({
          steps: z.array(z.object({ s: Id, a: z.string().trim().default(''), r: z.number() })).min(1),
          end: z.string().trim().default(''),
          endAction: z.string().trim().default(''),
        }),
      )
      .default([]),
    // Bandits: arms are the actions. "pulls" is a history of (arm, reward); "arms" a snapshot of
    // per-arm estimate q, pull count n and successes wins (A/B tests).
    pulls: z.array(z.object({ a: Id, r: z.number() })).default([]),
    arms: z.array(z.object({ a: Id, q: z.number().optional(), n: z.number().min(0).optional(), wins: z.number().min(0).optional() })).default([]),
    c: z.number().min(0).optional(), // UCB exploration constant
    t: z.number().int().min(1).optional(), // UCB time step
    parts: z.array(PartSchema).min(1),
  });

export const ProblemSchema = ProblemBase.superRefine((p, ctx) => {
    const S = new Set(p.states.map((s) => s.id));
    const A = new Set(p.actions.map((a) => a.id));
    const bad = (path: (string | number)[], message: string) => ctx.addIssue({ code: 'custom', path, message });
    const st = (v: string, path: (string | number)[]) => S.has(v) || bad(path, `Unknown state "${v}"`);
    const ac = (v: string, path: (string | number)[]) => A.has(v) || bad(path, `Unknown action "${v}"`);

    p.terminal.forEach((s, i) => st(s, ['terminal', i]));
    p.transitions.forEach((t, i) => {
      st(t.s, ['transitions', i, 's']);
      ac(t.a, ['transitions', i, 'a']);
      st(t.s2, ['transitions', i, 's2']);
    });
    p.policy.forEach((x, i) => {
      st(x.s, ['policy', i, 's']);
      ac(x.a, ['policy', i, 'a']);
    });
    p.V0.forEach((x, i) => st(x.s, ['V0', i, 's']));
    p.Q.forEach((x, i) => {
      st(x.s, ['Q', i, 's']);
      ac(x.a, ['Q', i, 'a']);
    });
    p.parts.forEach((pt, i) => pt.states.forEach((s, j) => st(s, ['parts', i, 'states', j])));
    p.parts.forEach((pt, i) =>
      pt.episodes.forEach((e, j) => e <= p.episodes.length || bad(['parts', i, 'episodes', j], `Episode ${e} does not exist`)),
    );
    p.episodes.forEach((ep, e) => {
      ep.steps.forEach((x, t) => {
        st(x.s, ['episodes', e, 'steps', t, 's']);
        if (x.a) ac(x.a, ['episodes', e, 'steps', t, 'a']);
      });
      if (ep.end) st(ep.end, ['episodes', e, 'end']);
      if (ep.endAction) ac(ep.endAction, ['episodes', e, 'endAction']);
    });
    p.pulls.forEach((x, i) => ac(x.a, ['pulls', i, 'a']));
    p.arms.forEach((x, i) => ac(x.a, ['arms', i, 'a']));

    const sums = new Map<string, number>();
    for (const t of p.transitions) sums.set(`${t.s}, ${t.a}`, (sums.get(`${t.s}, ${t.a}`) ?? 0) + t.p);
    for (const [k, v] of sums)
      if (Math.abs(v - 1) > 1e-6) bad(['transitions'], `Probabilities for (${k}) add up to ${+v.toFixed(6)}, not 1`);

    const pol = new Map<string, number>();
    for (const x of p.policy) pol.set(x.s, (pol.get(x.s) ?? 0) + x.p);
    for (const [s, v] of pol)
      if (Math.abs(v - 1) > 1e-6) bad(['policy'], `Policy probabilities for ${s} add up to ${+v.toFixed(6)}, not 1`);
  });

export type Problem = z.infer<typeof ProblemSchema>;

/** Fill defaults so hand-written problems (fixtures, tests) can omit empty fields. */
export const problem = (x: z.input<typeof ProblemBase>): Problem => ProblemBase.parse(x);
