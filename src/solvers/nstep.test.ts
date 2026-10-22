// Hand-checked cases for n-step methods, tree backup, Dyna-Q and DQN.
import { describe, expect, it } from 'vitest';
import { solveAll } from './index';
import { ProblemSchema, type Task } from '../schema';
import { idTex as x } from '../mdp';

const run = (data: object, task: Task) => {
  const [pt] = solveAll(ProblemSchema.parse({ ...data, parts: [{ task }] }));
  expect(pt.error).toBeUndefined();
  const out = new Map<string, number | string>();
  for (const st of pt.steps) for (const r of st.results) out.set(`${r.cell.table}#${r.cell.row}|${r.cell.col}`, r.value);
  return out;
};

const ABC = { states: [{ id: 'A' }, { id: 'B' }, { id: 'C' }], actions: [{ id: 'L' }, { id: 'R' }] };

describe('n-step TD and SARSA', () => {
  // A r1, B r2, C r3, terminal. γ = 0.5, α = 1 (so values equal the returns), n = 2, all start at 1.
  const base = { ...ABC, gamma: 0.5, alpha: 1, n: 2 };
  it('n-step TD bootstraps only while the episode lasts', () => {
    const out = run(
      { ...base, V0: [{ s: 'A', v: 1 }, { s: 'B', v: 1 }, { s: 'C', v: 1 }], episodes: [{ steps: [{ s: 'A', r: 1 }, { s: 'B', r: 2 }, { s: 'C', r: 3 }] }] },
      'nstep_td',
    );
    const v = (s: string) => out.get(`\\text{n-step TD } V(s)#V|${x(s)}`);
    expect(v('A')).toBeCloseTo(1 + 0.5 * 2 + 0.25 * 1, 10);
    expect(v('B')).toBeCloseTo(2 + 0.5 * 3, 10);
    expect(v('C')).toBeCloseTo(3, 10);
  });
  it('n-step SARSA bootstraps from Q(S_{t+n}, A_{t+n})', () => {
    const Q = ['A', 'B', 'C'].flatMap((s) => ['L', 'R'].map((a) => ({ s, a, v: 1 })));
    const out = run({ ...base, Q, episodes: [{ steps: [{ s: 'A', a: 'R', r: 1 }, { s: 'B', a: 'R', r: 2 }, { s: 'C', a: 'L', r: 3 }] }] }, 'nstep_sarsa');
    expect(out.get(`\\text{n-step SARSA } Q(s,a)#${x('A')}|${x('R')}`)).toBeCloseTo(2.25, 10);
  });
});

describe('off-policy n-step SARSA', () => {
  // π always R; b picks L/R 50/50. A R 1, B R 2, C R 3, terminal. n = 2, α = 0.5, γ = 1, Q0 = 0.
  const pol = (p: number) => ['A', 'B', 'C'].flatMap((s) => [{ s, a: 'R', p }, { s, a: 'L', p: 1 - p }]);
  const data = { ...ABC, gamma: 1, alpha: 0.5, n: 2, policy: pol(1), behavior: pol(0.5) };
  const ep = (last: string) => ({ steps: [{ s: 'A', a: 'R', r: 1 }, { s: 'B', a: 'R', r: 2 }, { s: 'C', a: last, r: 3 }] });
  const T = '\\text{Off-policy n-step SARSA } Q(s,a)#';
  it('weights updates by ρ', () => {
    const out = run({ ...data, episodes: [ep('R')] }, 'nstep_offpolicy_is');
    expect(out.get(`${T}${x('A')}|${x('R')}`)).toBeCloseTo(0.5 * 4 * 3, 10); // ρ = 2·2, G = 1 + 2 + Q(C,R)=0
    expect(out.get(`${T}${x('B')}|${x('R')}`)).toBeCloseTo(0.5 * 2 * 5, 10); // ρ = 2, G = 2 + 3
    expect(out.get(`${T}${x('C')}|${x('R')}`)).toBeCloseTo(0.5 * 1 * 3, 10); // ρ = 1
  });
  it('cancels updates when π would never take a later action', () => {
    const out = run({ ...data, episodes: [ep('L')] }, 'nstep_offpolicy_is');
    expect(out.get(`${T}${x('A')}|${x('R')}`)).toBe(0);
    expect(out.get(`${T}${x('C')}|${x('L')}`)).toBeCloseTo(1.5, 10);
  });
});

describe('tree backup', () => {
  it('mixes expected values of untaken actions with the deeper return', () => {
    const Q = ['A', 'B', 'C'].flatMap((s) => [{ s, a: 'L', v: 1 }, { s, a: 'R', v: 2 }]);
    const policy = ['A', 'B', 'C'].flatMap((s) => [{ s, a: 'L', p: 0.5 }, { s, a: 'R', p: 0.5 }]);
    const out = run(
      { ...ABC, gamma: 1, alpha: 1, n: 2, Q, policy, episodes: [{ steps: [{ s: 'A', a: 'R', r: 1 }, { s: 'B', a: 'R', r: 2 }, { s: 'C', a: 'R', r: 3 }] }] },
      'tree_backup',
    );
    const q = (s: string) => out.get(`\\text{Tree backup } Q(s,a)#${x(s)}|${x('R')}`);
    expect(q('A')).toBeCloseTo(3.25, 10); // 1 + 0.5·1 + 0.5·(2 + 0.5·1 + 0.5·2)
    expect(q('B')).toBeCloseTo(4, 10); // 2 + 0.5·1 + 0.5·3
    expect(q('C')).toBeCloseTo(3, 10);
  });
});

describe('Dyna-Q', () => {
  it('direct update then n planning updates from the model', () => {
    const out = run(
      { states: [{ id: 'S1' }, { id: 'S2' }], actions: [{ id: 'a' }], gamma: 0.9, alpha: 0.5, n: 2, episodes: [{ steps: [{ s: 'S1', a: 'a', r: 1 }], end: 'S2' }] },
      'dyna_q',
    );
    // 0 -> 0.5 (real) -> 0.75 -> 0.875 (planning); Q(S2,·) stays 0
    expect(out.get(`\\text{Dyna-Q } Q(s,a)#S_{1}|${x('a')}`)).toBeCloseTo(0.875, 10);
  });
});

describe('DQN', () => {
  it('targets from the target network and mean squared TD error', () => {
    const out = run(
      {
        states: [{ id: 'S1' }, { id: 'S2' }],
        actions: [{ id: 'A1' }, { id: 'A2' }],
        gamma: 0.9,
        Q: [{ s: 'S1', a: 'A1', v: 0.5 }, { s: 'S2', a: 'A2', v: 4 }],
        Qtarget: [{ s: 'S2', a: 'A1', v: 2 }, { s: 'S2', a: 'A2', v: 3 }],
        replay: [{ s: 'S1', a: 'A1', r: 1, s2: 'S2' }, { s: 'S2', a: 'A2', r: 5, done: true }],
      },
      'dqn_target_loss',
    );
    const T = '\\text{DQN minibatch}#';
    expect(out.get(`${T}i = 1|y_i`)).toBeCloseTo(3.7, 10);
    expect(out.get(`${T}i = 2|y_i`)).toBe(5);
    expect(out.get(`${T}\\text{loss}|L`)).toBeCloseTo((3.2 ** 2 + 1) / 2, 10);
  });
});
