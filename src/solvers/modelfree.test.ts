// Hand-checked textbook-style cases for the sample-based and bandit solvers.
import { describe, expect, it } from 'vitest';
import { solveAll } from './index';
import { ProblemSchema } from '../schema';
import type { Task } from '../schema';
import { idTex as x } from '../mdp';

const run = (x: object, task: Task) => {
  const [pt] = solveAll(ProblemSchema.parse({ ...x, parts: [{ task }] }));
  expect(pt.error).toBeUndefined();
  // last written value per cell, keyed "row|col"
  const out = new Map<string, number | string>();
  const full = new Map<string, number | string>();
  for (const st of pt.steps)
    for (const r of st.results) {
      out.set(`${r.cell.row}|${r.cell.col}`, r.value);
      full.set(`${r.cell.table}#${r.cell.row}|${r.cell.col}`, r.value);
    }
  return { out, full, steps: pt.steps };
};

const AB = { states: [{ id: 'A' }, { id: 'B' }], actions: [{ id: 'L' }, { id: 'R' }] };

describe('TD(0)', () => {
  it('two episodes A -> B -> terminal, α = 0.5, γ = 1', () => {
    const ep = { steps: [{ s: 'A', r: 0 }, { s: 'B', r: 1 }] };
    const { out } = run({ ...AB, gamma: 1, alpha: 0.5, episodes: [ep, ep] }, 'td0_prediction');
    expect(out.get(`V|${x('A')}`)).toBeCloseTo(0.25, 10);
    expect(out.get(`V|${x('B')}`)).toBeCloseTo(0.75, 10);
  });
});

describe('Monte Carlo prediction', () => {
  // A(r=1) B(r=0) A(r=2) terminal, γ = 0.5: G2 = 2, G1 = 1, G0 = 1.5
  const data = { ...AB, gamma: 0.5, episodes: [{ steps: [{ s: 'A', r: 1 }, { s: 'B', r: 0 }, { s: 'A', r: 2 }] }] };
  it('first-visit', () => {
    const { out } = run(data, 'mc_prediction_first_visit');
    expect(out.get('G|t = 0')).toBeCloseTo(1.5, 10);
    expect(out.get(`V|${x('A')}`)).toBeCloseTo(1.5, 10);
    expect(out.get(`V|${x('B')}`)).toBeCloseTo(1, 10);
  });
  it('ignores an α meant for other parts', () => {
    expect(run({ ...data, alpha: 0.5 }, 'mc_prediction_first_visit').out.get(`V|${x('A')}`)).toBeCloseTo(1.5, 10);
  });
  it('every-visit averages both visits to A', () => {
    expect(run(data, 'mc_prediction_every_visit').out.get(`V|${x('A')}`)).toBeCloseTo(1.75, 10);
  });
  it('constant-α', () => {
    // V(A) = 0 + 0.5(1.5 - 0) = 0.75, V(B) = 0.5
    const [pt] = solveAll(ProblemSchema.parse({ ...data, alpha: 0.5, parts: [{ task: 'mc_prediction_first_visit', text: 'constant-α MC' }] }));
    const out = new Map(pt.steps.flatMap((st) => st.results.map((r) => [`${r.cell.row}|${r.cell.col}`, r.value] as const)));
    expect(out.get(`V|${x('A')}`)).toBeCloseTo(0.75, 10);
    expect(out.get(`V|${x('B')}`)).toBeCloseTo(0.5, 10);
  });
});

describe('SARSA vs Q-learning on one transition', () => {
  // Q(S2,R) = 1, Q(S2,L) = 2. From S1 take R, reward 1, land in S2 and take R next.
  const data = {
    states: [{ id: 'S1' }, { id: 'S2' }],
    actions: [{ id: 'L' }, { id: 'R' }],
    gamma: 0.9,
    alpha: 0.5,
    Q: [{ s: 'S2', a: 'R', v: 1 }, { s: 'S2', a: 'L', v: 2 }],
    episodes: [{ steps: [{ s: 'S1', a: 'R', r: 1 }], end: 'S2', endAction: 'R' }],
  };
  it('SARSA uses the action taken next', () => {
    expect(run(data, 'sarsa').out.get(`S_{1}|${x('R')}`)).toBeCloseTo(0.5 * (1 + 0.9 * 1), 10);
  });
  it('Q-learning uses the best next action', () => {
    expect(run(data, 'q_learning').out.get(`S_{1}|${x('R')}`)).toBeCloseTo(0.5 * (1 + 0.9 * 2), 10);
  });
  it('SARSA without A′ explains what is missing', () => {
    const [pt] = solveAll(ProblemSchema.parse({ ...data, episodes: [{ ...data.episodes[0], endAction: '' }], parts: [{ task: 'sarsa' }] }));
    expect(pt.error).toMatch(/next action/);
  });
});

describe('Monte Carlo control', () => {
  it('averages first-visit returns and makes the policy ε-greedy', () => {
    const { full } = run(
      {
        states: [{ id: 'S1' }, { id: 'S2' }],
        actions: [{ id: 'L' }, { id: 'R' }],
        gamma: 1,
        epsilon: 0.1,
        episodes: [{ steps: [{ s: 'S1', a: 'R', r: 1 }, { s: 'S2', a: 'L', r: 2 }] }],
      },
      'mc_control_epsilon_greedy',
    );
    const q = '\\text{Monte Carlo } Q(s,a)#';
    const pi = '\\text{Improved policy } \\pi(a\\mid s)#';
    expect(full.get(`${q}S_{1}|${x('R')}`)).toBe(3);
    expect(full.get(`${q}S_{2}|${x('L')}`)).toBe(2);
    expect(full.get(`${pi}S_{1}|${x('R')}`)).toBeCloseTo(0.95, 10);
    expect(full.get(`${pi}S_{1}|${x('L')}`)).toBeCloseTo(0.05, 10);
  });
});

describe('bandits', () => {
  const pulls = { actions: [{ id: 'A1' }, { id: 'A2' }], pulls: [{ a: 'A1', r: 1 }, { a: 'A2', r: 0 }, { a: 'A1', r: 3 }, { a: 'A2', r: 2 }, { a: 'A1', r: 2 }] };
  it('sample average', () => {
    const { out } = run(pulls, 'bandit_sample_average');
    expect([out.get('Q(a)|A_{1}'), out.get('Q(a)|A_{2}')]).toEqual([2, 1]);
  });
  it('incremental update reaches the same averages', () => {
    const { out, steps } = run(pulls, 'bandit_incremental');
    expect(steps).toHaveLength(6);
    expect(out.get('Q(a)|A_{1}')).toBeCloseTo(2, 10);
    expect(out.get('Q(a)|A_{2}')).toBeCloseTo(1, 10);
  });
  it('UCB with t = total pulls', () => {
    const { out, steps } = run({ ...pulls, c: 2 }, 'bandit_ucb');
    expect(out.get('\\text{UCB}|A_{1}')).toBeCloseTo(2 + 2 * Math.sqrt(Math.log(5) / 3), 10);
    expect(out.get('\\text{UCB}|A_{2}')).toBeCloseTo(1 + 2 * Math.sqrt(Math.log(5) / 2), 10);
    expect(steps.at(-1)!.lines.at(-1)!.rhs).toBe('A_{1}');
  });
  it('UCB picks an unpulled arm first', () => {
    const { steps } = run({ actions: [{ id: 'A1' }, { id: 'A2' }], arms: [{ a: 'A1', q: 5, n: 3 }, { a: 'A2', q: 0, n: 0 }], c: 1 }, 'bandit_ucb');
    expect(steps.at(-1)!.lines.at(-1)!.rhs).toBe('A_{2}');
  });
  it('ε-greedy arm probabilities from given estimates', () => {
    const { out } = run({ actions: [{ id: 'A1' }, { id: 'A2' }], arms: [{ a: 'A1', q: 2 }, { a: 'A2', q: 1 }], epsilon: 0.1 }, 'bandit_epsilon_greedy');
    expect(out.get('P|A_{1}')).toBeCloseTo(0.95, 10);
    expect(out.get('P|A_{2}')).toBeCloseTo(0.05, 10);
  });
  it('A/B test z-statistic', () => {
    const { out, steps } = run({ actions: [{ id: 'A' }, { id: 'B' }], arms: [{ a: 'A', n: 1000, wins: 200 }, { a: 'B', n: 1000, wins: 250 }] }, 'ab_test');
    const se = Math.sqrt(0.225 * 0.775 * (2 / 1000));
    expect(out.get(`\\text{test}|z`)).toBeCloseTo(-0.05 / se, 8);
    expect(steps.at(-1)!.why).toMatch(/significant/);
  });
});
