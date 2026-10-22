import { describe, expect, it } from 'vitest';
import { crossCheck, detectTasks, splitParts } from './keywords';
import { clean, extract } from './ollama';
import { EXAMPLES } from '../fixtures/ia1';
import { solveAll } from '../solvers';
import { ProblemSchema, problem, type Problem } from '../schema';

describe('keywords', () => {
  it('prefers the specific rule', () => {
    expect(detectTasks('Using an ε-greedy policy with ε = 0.2')[0]).toBe('epsilon_greedy_probs');
    expect(detectTasks('Apply policy iteration for 2 iterations')[0]).toBe('policy_iteration');
    expect(detectTasks('Compute V2(S1), V2(S2)')[0]).toBe('policy_evaluation');
  });
  it('splits sub-questions and guesses tasks for every IA1 and few-shot part', () => {
    for (const { text, problem: p } of EXAMPLES)
      expect(splitParts(text).map((x) => [x.label, x.task])).toEqual(p.parts.map((x) => [x.label, x.task]));
  });
  it('every example is valid and solves without errors', () => {
    for (const { problem: p } of EXAMPLES) {
      expect(ProblemSchema.safeParse(p).error?.issues).toBeUndefined();
      expect(solveAll(p).map((t) => t.error).filter(Boolean)).toEqual([]);
    }
  });
  it('flags a task that contradicts the part text', () => {
    const parts = EXAMPLES[1].problem.parts;
    expect(crossCheck(parts)).toEqual([]);
    expect(crossCheck([{ ...parts[0], task: 'greedy_action' }])).toHaveLength(1);
  });
});

describe('clean', () => {
  it('takes γ and ε from the text and drops zero-probability policy rows', () => {
    const p = { ...EXAMPLES[1].problem, gamma: 0.5, epsilon: undefined, policy: [{ s: 'S1', a: 'A2', p: 0 }, { s: 'S1', a: 'A1', p: 1 }] };
    const c = clean(p, 'Discount factor (γ): 0.8. Use ε = 0.2 in part e.');
    expect([c.gamma, c.epsilon, c.policy]).toEqual([0.8, 0.2, [{ s: 'S1', a: 'A1', p: 1 }]]);
  });
  it('reads α, c, t and iteration counts written in words', () => {
    const p = { ...EXAMPLES[0].problem, parts: EXAMPLES[0].problem.parts.map((x) => ({ ...x, iterations: undefined })) };
    const c = clean(p, 'Discount 0.8, learning rate 0.5. UCB with c = 1 at t = 6. Use 3-step TD.');
    expect([c.gamma, c.alpha, c.c, c.t, c.n]).toEqual([0.8, 0.5, 1, 6, 3]);
    expect(clean(p, 'Dyna-Q with 5 planning steps').n).toBe(5);
    expect(c.parts.map((x) => x.iterations)).toEqual([1, 2, undefined, undefined, undefined]);
    expect(clean({ ...p, parts: [{ ...p.parts[0], text: 'Run two sweeps of value iteration' }] }, '').parts[0].iterations).toBe(2);
  });
});

/** The bits that change the maths: ids may be renamed by the model, so compare by value. */
const essence = (p: Problem) => ({
  gamma: p.gamma,
  epsilon: p.epsilon,
  terminal: p.terminal.length,
  transitions: p.transitions.map((t) => [t.p, t.r]).sort(),
  policy: p.policy.map((x) => x.p).sort(),
  Q: p.Q.map((x) => x.v).filter((v) => v !== 0).sort(), // explicit zeros = missing
  alpha: p.alpha,
  episodes: p.episodes.map((e) => e.steps.map((x) => x.r)),
  pulls: p.pulls.map((x) => x.r),
  arms: p.arms.map((x) => [x.n, x.wins]),
  c: p.c,
  n: p.n,
  replay: p.replay.map((x) => [x.r, x.done]),
  Qtarget: p.Qtarget.map((x) => x.v).sort(),
  parts: p.parts.map((x) => [x.task, x.iterations ?? null]),
});

// Held-out questions: not shown to the model as examples.
const HELD_OUT = [
  {
    name: 'model-free',
    text: `An agent has states A, B and terminal state C. Actions: up, down. Discount 0.8, learning rate 0.5. Q-values start at zero.
Episode: A, up, reward 2 → B, down, reward -1 → C.
a) Using Q-learning, update Q for each transition.
b) Using SARSA, update Q for each transition.
c) Using TD(0), update V for each state visited.`,
    want: problem({
      states: [{ id: 'A' }, { id: 'B' }, { id: 'C' }],
      actions: [{ id: 'up' }, { id: 'down' }],
      terminal: ['C'],
      gamma: 0.8,
      alpha: 0.5,
      episodes: [{ steps: [{ s: 'A', a: 'up', r: 2 }, { s: 'B', a: 'down', r: -1 }], end: 'C' }],
      parts: [{ task: 'q_learning' }, { task: 'sarsa' }, { task: 'td0_prediction' }],
    }),
  },
  {
    name: 'bandit',
    text: `Two slot machines M1 and M2 were played in this order: M1 gave 5, M2 gave 3, M2 gave 4, M1 gave 1.
a) Estimate each machine's value using sample averages.
b) Using UCB with c = 1, which machine is played next?`,
    want: problem({
      actions: [{ id: 'M1' }, { id: 'M2' }],
      pulls: [{ a: 'M1', r: 5 }, { a: 'M2', r: 3 }, { a: 'M2', r: 4 }, { a: 'M1', r: 1 }],
      c: 1,
      parts: [{ task: 'bandit_sample_average' }, { task: 'bandit_ucb' }],
    }),
  },
  {
    name: 'A/B test',
    text: `An A/B test of a sign-up page: variant A converted 30 of 400 visitors, variant B converted 45 of 410 visitors.
a) Compute both conversion rates.
b) Is B significantly better at the 5% level?`,
    want: problem({
      actions: [{ id: 'A' }, { id: 'B' }],
      arms: [{ a: 'A', n: 400, wins: 30 }, { a: 'B', n: 410, wins: 45 }],
      parts: [{ task: 'ab_test' }, { task: 'ab_test' }],
    }),
  },
  {
    name: 'n-step',
    text: `Rooms X, Y and Z, where Z is terminal. Actions: left, right. Discount factor 0.8, step size 0.2. Initial V and Q values are zero.
Episode: X, right, reward 3 → Y, right, reward 5 → Z.
a) Perform 3-step TD updates for the episode.
b) Perform 3-step SARSA updates for the episode.`,
    want: problem({
      states: [{ id: 'X' }, { id: 'Y' }, { id: 'Z' }],
      actions: [{ id: 'left' }, { id: 'right' }],
      terminal: ['Z'],
      gamma: 0.8,
      alpha: 0.2,
      n: 3,
      episodes: [{ steps: [{ s: 'X', a: 'right', r: 3 }, { s: 'Y', a: 'right', r: 5 }], end: 'Z' }],
      parts: [{ task: 'nstep_td' }, { task: 'nstep_sarsa' }],
    }),
  },
  {
    name: 'DQN',
    text: `Two transitions are sampled from experience replay: (A, up, reward 2, next state B) and (B, down, reward -1, episode ends). Discount 0.95.
The Q-network outputs Q(A, up) = 1.0 and Q(B, down) = 0.5. The target network gives Q̂(B, up) = 2.0 and Q̂(B, down) = 1.0.
a) Find the TD target for each transition.
b) Find the MSE loss.`,
    want: problem({
      states: [{ id: 'A' }, { id: 'B' }],
      actions: [{ id: 'up' }, { id: 'down' }],
      gamma: 0.95,
      Q: [{ s: 'A', a: 'up', v: 1 }, { s: 'B', a: 'down', v: 0.5 }],
      Qtarget: [{ s: 'B', a: 'up', v: 2 }, { s: 'B', a: 'down', v: 1 }],
      replay: [{ s: 'A', a: 'up', r: 2, s2: 'B' }, { s: 'B', a: 'down', r: -1, done: true }],
      parts: [{ task: 'dqn_target_loss' }, { task: 'dqn_target_loss' }],
    }),
  },
];

// Real model run: OLLAMA=1 npx vitest run src/extract
describe.skipIf(!import.meta.env.OLLAMA)('LLM extraction of the IA1 questions', () => {
  const model = import.meta.env.OLLAMA_MODEL ?? 'qwen2.5:7b';
  for (const ex of EXAMPLES.filter((x) => x.name.startsWith('IA1'))) {
    it(ex.name, async () => {
      const got = await extract(ex.text, model, { base: 'http://localhost:11434' });
      expect(essence(got)).toEqual(essence(ex.problem));
    }, 300_000);
  }
  for (const ex of HELD_OUT) {
    it(`held-out ${ex.name}`, async () => {
      const got = await extract(ex.text, model, { base: 'http://localhost:11434' });
      const { gamma: _g, ...rest } = essence(got); // γ is irrelevant for bandits and defaults differ
      const { gamma: _w, ...want } = essence(ex.want);
      const bandit = ex.name === 'bandit' || ex.name === 'A/B test';
      expect(bandit ? rest : essence(got)).toEqual(bandit ? want : essence(ex.want));
    }, 300_000);
  }
});
