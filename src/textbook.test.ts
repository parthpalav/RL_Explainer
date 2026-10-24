// Published textbook problems with published answers (Sutton & Barto 2nd ed., Silver/Stanford slides).
import { describe, expect, it } from 'vitest';
import { solveAll } from './solvers';
import { ProblemSchema } from './schema';

const solve = (x: object, parts: object[]) => solveAll(ProblemSchema.parse({ ...x, parts }));
/** Value of "row|col" (last write), optionally restricted to table names containing `table`. */
const val = (pt: ReturnType<typeof solveAll>[number], key: string, table = '') => {
  let v: number | string | undefined;
  for (const st of pt.steps)
    for (const r of st.results) if (`${r.cell.row}|${r.cell.col}` === key && r.cell.table.includes(table)) v = r.value;
  return v as number;
};
/** Value written by the n-th result with this key (1-based): replays an update sequence. */
const nth = (pt: ReturnType<typeof solveAll>[number], key: string, n: number) => {
  const all: (number | string)[] = [];
  for (const st of pt.steps) for (const r of st.results) if (`${r.cell.row}|${r.cell.col}` === key) all.push(r.value);
  return all[n - 1] as number;
};

// Example 4.1: 4x4 gridworld, cells 1..14, corners terminal, reward -1, gamma 1, equiprobable.
const cells = Array.from({ length: 14 }, (_, i) => `S${i + 1}`);
const pos = (n: number) => [Math.floor(n / 4), n % 4];
const gridId = (r: number, c: number) => ((r === 0 && c === 0) || (r === 3 && c === 3) ? 'T' : `S${r * 4 + c}`);
const D = [['U', -1, 0], ['D', 1, 0], ['L', 0, -1], ['R', 0, 1]] as const;
const grid = {
  states: [...cells.map((id) => ({ id })), { id: 'T' }],
  actions: D.map(([id]) => ({ id })),
  terminal: ['T'],
  gamma: 1,
  transitions: cells.flatMap((s) => {
    const [r, c] = pos(+s.slice(1));
    return D.map(([a, dr, dc]) => {
      const nr = r + dr, nc = c + dc;
      const ok = nr >= 0 && nr < 4 && nc >= 0 && nc < 4;
      return { s, a, s2: ok ? gridId(nr, nc) : s, p: 1, r: -1 };
    });
  }),
  policy: cells.flatMap((s) => D.map(([a]) => ({ s, a, p: 0.25 }))),
};


describe('textbook', () => {
  it('S&B Ex 3.8: gamma .5, R1..R5 = -1,2,6,3,2 gives G0 = 2', () => {
    const [pt] = solve({ gamma: 0.5 }, [{ task: 'return_Gt', rewards: [-1, 2, 6, 3, 2] }]);
    expect(pt.steps.at(-1)!.lines.at(-1)!.rhs).toContain('{2}');
  });

  it('S&B Ex 4.1 / Fig 4.1: policy evaluation k = 1, 2, 3', () => {
    const [pt] = solve(grid, [{ task: 'policy_evaluation', iterations: 3 }]);
    const v = (k: number, s: string) => val(pt, `k = ${k}|${s}`);
    expect(v(1, 'S_{1}')).toBe(-1);
    expect(v(2, 'S_{1}')).toBeCloseTo(-1.75, 10);
    expect(v(2, 'S_{6}')).toBeCloseTo(-2, 10);
    expect(v(3, 'S_{1}')).toBeCloseTo(-2.4, 1);
    expect(v(3, 'S_{2}')).toBeCloseTo(-2.9, 1);
    expect(v(3, 'S_{3}')).toBeCloseTo(-3.0, 1);
    expect(v(3, 'S_{7}')).toBeCloseTo(-2.9, 1);
  });

  it('S&B Ex 4.1: converged v_pi = -14, -20, -22', () => {
    const [pt] = solve(grid, [{ task: 'policy_evaluation_linear' }]);
    const want: Record<number, number> = { 1: -14, 2: -20, 3: -22, 4: -14, 5: -18, 6: -20, 7: -20, 8: -20, 9: -20, 10: -18, 11: -14, 12: -22, 13: -20, 14: -14 };
    for (const [i, w] of Object.entries(want)) expect(val(pt, `\\pi|S_{${i}}`)).toBeCloseTo(w, 8);
  });

  it('S&B Ex 4.1: value iteration = minus shortest distance to a corner', () => {
    const [pt] = solve(grid, [{ task: 'value_iteration', iterations: 3 }]);
    expect(val(pt, 'k = 3|S_{3}', 'Value iteration } V_k')).toBe(-3);
    expect(val(pt, 'k = 3|S_{5}', 'Value iteration } V_k')).toBe(-2);
    expect(val(pt, 'k = 2|S_{3}', 'Value iteration } V_k')).toBe(-2);
  });

  // S&B Exercise 2.2 data: A1=1 R=-1, A2=2 R=1, A3=2 R=-2, A4=2 R=2, A5=3 R=0
  const bandit = {
    actions: [{ id: 'A1' }, { id: 'A2' }, { id: 'A3' }, { id: 'A4' }],
    epsilon: 0.5,
    pulls: [{ a: 'A1', r: -1 }, { a: 'A2', r: 1 }, { a: 'A2', r: -2 }, { a: 'A2', r: 2 }, { a: 'A3', r: 0 }],
  };
  it('S&B Ex 2.2: sample averages and incremental updates', () => {
    const [pt] = solve(bandit, [{ task: 'bandit_sample_average' }]);
    expect(val(pt, 'Q(a)|A_{1}')).toBe(-1);
    expect(val(pt, 'Q(a)|A_{2}')).toBeCloseTo(1 / 3, 10);
    const [inc] = solve(bandit, [{ task: 'bandit_incremental' }]);
    expect(nth(inc, 'Q(a)|A_{2}', 3)).toBeCloseTo(0.5 - 1, 10); // after R=-2: 1 + (-2-1)/2
    expect(val(inc, 'Q(a)|A_{2}')).toBeCloseTo(1 / 3, 10);
  });
  it('epsilon-greedy: 1 - e + e/k for the greedy arm, e/k for the rest (S&B Ex 2.1: 0.75 for k = 2)', () => {
    const [pt] = solve(bandit, [{ task: 'bandit_epsilon_greedy' }]);
    expect(val(pt, 'P|A_{2}')).toBeCloseTo(0.625, 10);
    expect(val(pt, 'P|A_{1}')).toBeCloseTo(0.125, 10);
    const two = solve({ actions: [{ id: 'A1' }, { id: 'A2' }], epsilon: 0.5, pulls: [{ a: 'A1', r: 1 }, { a: 'A2', r: 0 }] }, [{ task: 'bandit_epsilon_greedy' }]);
    expect(val(two[0], 'P|A_{1}')).toBeCloseTo(0.75, 10);
  });
  it('UCB (S&B eq 2.10): Q + c sqrt(ln t / N), unpulled arm first', () => {
    const [pt] = solve({ ...bandit, c: 2, t: 6 }, [{ task: 'bandit_ucb' }]);
    expect(val(pt, '\\text{UCB}|A_{1}')).toBeCloseTo(-1 + 2 * Math.sqrt(Math.log(6)), 10);
    expect(val(pt, '\\text{UCB}|A_{2}')).toBeCloseTo(1 / 3 + 2 * Math.sqrt(Math.log(6) / 3), 10);
    expect(pt.steps.at(-1)!.lines.at(-1)!.rhs).toBe('A_{4}');
  });

  // S&B Ex 6.4 (and Silver / CS234 slides): A,0,B,0 then B,1 x6 then B,0 -> batch MC V(A)=0, V(B)=0.75
  it('S&B Ex 6.4 "You are the predictor": MC gives V(A) = 0, V(B) = 0.75', () => {
    const one = { steps: [{ s: 'B', r: 1 }] };
    const ab = {
      states: [{ id: 'A' }, { id: 'B' }], actions: [{ id: 'x' }], gamma: 1,
      episodes: [{ steps: [{ s: 'A', r: 0 }, { s: 'B', r: 0 }] }, ...Array(6).fill(one), { steps: [{ s: 'B', r: 0 }] }],
    };
    const [pt] = solve(ab, [{ task: 'mc_prediction_first_visit' }]);
    expect(val(pt, 'V|\\text{A}')).toBe(0);
    expect(val(pt, 'V|\\text{B}')).toBeCloseTo(0.75, 10);
  });

  // Utah CS 6300 "Q-Learning Practice" (users.cs.utah.edu/~dsbrown/classes/cs6300/practice/Q-learn_solution.pdf)
  const st = (s: string, a: string, r: number) => ({ s, a, r });
  const utah = {
    states: ['X31', 'X21', 'X22', 'X23'].map((id) => ({ id })), actions: ['N', 'S', 'E'].map((id) => ({ id })), gamma: 1, alpha: 0.5,
    episodes: [
      [st('X31', 'S', 0), st('X21', 'E', 0), st('X22', 'E', 0), st('X23', 'N', 50)],
      [st('X31', 'S', 0), st('X21', 'E', 0), st('X22', 'S', -100)],
      [st('X31', 'S', 0), st('X21', 'E', 0), st('X22', 'E', 0), st('X23', 'S', 30)],
      [st('X31', 'S', 0), st('X21', 'E', 0), st('X22', 'E', 0), st('X23', 'N', 50)],
      [st('X31', 'S', 0), st('X21', 'E', 0), st('X22', 'E', 0), st('X23', 'S', 30)],
    ].map((steps) => ({ steps })),
  };
  // nth(cell, 1 + k) is the k-th visit of that cell (write 1 is the initial table).
  it('Utah Q-learning key: Q22E 12.5 (ep3), Q22S -50, Q23N 25, Q23S 15, Q21E 6.25 at (ep4, trial 2)', () => {
    const [pt] = solve(utah, [{ task: 'q_learning' }]);
    expect(nth(pt, 'X_{22}|\\text{E}', 1 + 2)).toBeCloseTo(12.5, 10);
    expect(nth(pt, 'X_{22}|\\text{S}', 1 + 1)).toBe(-50);
    expect(nth(pt, 'X_{23}|\\text{N}', 1 + 1)).toBe(25);
    expect(nth(pt, 'X_{23}|\\text{S}', 1 + 1)).toBe(15);
    expect(nth(pt, 'X_{21}|\\text{E}', 1 + 3)).toBe(0); // ep3 still 0
    expect(nth(pt, 'X_{21}|\\text{E}', 1 + 4)).toBeCloseTo(6.25, 10); // (4,2)
  });
  it('Utah SARSA key: Q22E stays 0 in ep3, 12.5 in ep4; Q23N 37.5; Q21E 6.25 at (ep5, trial 2)', () => {
    const [pt] = solve(utah, [{ task: 'sarsa' }]);
    expect(nth(pt, 'X_{22}|\\text{E}', 1 + 2)).toBe(0);
    expect(nth(pt, 'X_{22}|\\text{E}', 1 + 3)).toBeCloseTo(12.5, 10);
    expect(nth(pt, 'X_{23}|\\text{N}', 1 + 2)).toBeCloseTo(37.5, 10);
    expect(nth(pt, 'X_{21}|\\text{E}', 1 + 4)).toBe(0);
    expect(nth(pt, 'X_{21}|\\text{E}', 1 + 5)).toBeCloseTo(6.25, 10); // (5,2)
  });

  it('A/B test: 40/2000 vs 55/2000 gives pooled 0.02375, SE 0.00482, |z| 1.56, not significant', () => {
    const [pt] = solve({ actions: [{ id: 'A' }, { id: 'B' }], arms: [{ a: 'A', n: 2000, wins: 40 }, { a: 'B', n: 2000, wins: 55 }] }, [{ task: 'ab_test' }]);
    expect(val(pt, '\\text{test}|\\hat{p}_{\\text{pool}}')).toBeCloseTo(0.02375, 10);
    expect(val(pt, '\\text{test}|SE')).toBeCloseTo(0.00482, 5);
    expect(Math.abs(val(pt, '\\text{test}|z'))).toBeCloseTo(1.56, 2);
  });
});
