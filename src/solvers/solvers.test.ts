import { describe, expect, it } from 'vitest';
import { solveAll } from './index';
import { ProblemSchema, type Problem } from '../schema';
import { BATCH1, BATCH2 } from '../fixtures/ia1';
import { add, chain, fmt, mul, num, toLatex } from '../math/expr';
import type { PartTrace } from './types';

/** All results keyed "table|row|col". */
const results = (parts: PartTrace[]) => {
  const out = new Map<string, number | string>();
  for (const pt of parts) for (const st of pt.steps) for (const r of st.results) out.set(`${r.cell.row}|${r.cell.col}`, r.value);
  return out;
};
const cellsOf = (pt: PartTrace) => {
  const out = new Map<string, number | string>();
  for (const st of pt.steps) for (const r of st.results) out.set(`${r.cell.table}#${r.cell.row}|${r.cell.col}`, r.value);
  return out;
};
const lastLine = (pt: PartTrace) => pt.steps.at(-1)!.lines.at(-1)!.rhs;

describe('expr', () => {
  it('reduces one level per line, like hand working', () => {
    // 0.2(-8 + 0.9(-5.6)) + 0.8(-5 + 0.9(0.2))
    const e = add(mul(num(0.2), add(num(-8), mul(num(0.9), num(-5.6)))), mul(num(0.8), add(num(-5), mul(num(0.9), num(0.2)))));
    const lines = chain(e);
    expect(lines.map((l) => toLatex(l).replace(/\\htmlClass\{fresh\}\{([^}]*)\}/g, '$1'))).toEqual([
      '0.2 \\times \\left(-8 + 0.9 \\times \\left(-5.6\\right)\\right) + 0.8 \\times \\left(-5 + 0.9 \\times 0.2\\right)',
      '0.2 \\times \\left(-8 - 5.04\\right) + 0.8 \\times \\left(-5 + 0.18\\right)',
      '0.2 \\times \\left(-13.04\\right) + 0.8 \\times \\left(-4.82\\right)',
      '-2.608 - 3.856',
      '-6.464',
    ]);
  });
  it('formats floats cleanly', () => {
    expect(fmt(0.1 + 0.2)).toBe('0.3');
    expect(fmt(9.4 / 0.84)).toBe('11.1905');
    expect(fmt(1e-12)).toBe('0');
    expect(fmt(0.00006975)).toBe('0.00006975'); // small values keep their significant digits
    expect(fmt(2.6 / 0.9 + 0.1)).toBe('2.98889');
  });
});

describe('IA1 Batch 1', () => {
  const parts = solveAll(ProblemSchema.parse(BATCH1));
  const r = results(parts);
  it('has no errors', () => parts.forEach((p) => expect(p.error).toBeUndefined()));
  it('V1 = (-5.6, 0.2, 4.2)', () => {
    expect(r.get('k = 1|S_{1}')).toBeCloseTo(-5.6, 10);
    expect(r.get('k = 1|S_{2}')).toBeCloseTo(0.2, 10);
    expect(r.get('k = 1|S_{3}')).toBeCloseTo(4.2, 10);
  });
  it('part b continues from part a: V2 = (-6.464, -1.186, 6.54)', () => {
    expect(parts[1].steps).toHaveLength(3);
    expect(r.get('k = 2|S_{1}')).toBeCloseTo(-6.464, 10);
    expect(r.get('k = 2|S_{2}')).toBeCloseTo(-1.186, 10);
    expect(r.get('k = 2|S_{3}')).toBeCloseTo(6.54, 10);
  });
  it('c) best state is S3', () => expect(lastLine(parts[2])).toBe('S_{3}'));
  it('d) greedy action is A2', () => expect(lastLine(parts[3])).toBe('A_{2}'));
  it('e) epsilon-greedy 0.9 / 0.1', () => {
    expect(r.get('S_{1}|A_{2}')).toBeCloseTo(0.9, 10);
    expect(r.get('S_{1}|A_{1}')).toBeCloseTo(0.1, 10);
  });
});

describe('IA1 Batch 2', () => {
  const parts = solveAll(ProblemSchema.parse(BATCH2));
  it('has no errors', () => parts.forEach((p) => expect(p.error).toBeUndefined()));
  it('policy iteration matches the solution', () => {
    const c = cellsOf(parts[0]);
    const pv = '\\text{Policy iteration } V^{\\pi_i}(s)';
    const pp = '\\text{Policy iteration } \\pi_i(s)';
    expect(c.get(`${pv}#\\pi_{0}|S_{1}`)).toBeCloseTo(10, 10);
    expect(c.get(`${pv}#\\pi_{0}|S_{2}`)).toBeCloseTo(4, 10);
    expect(c.get(`\\text{Policy iteration } Q^{\\pi_{0}}(s,a)#S_{1}|A_{2}`)).toBeCloseTo(7.6, 10);
    expect(c.get(`\\text{Policy iteration } Q^{\\pi_{0}}(s,a)#S_{2}|A_{2}`)).toBeCloseTo(10.04, 10);
    expect(c.get(`${pp}#\\pi_{1}|S_{1}`)).toBe('A_{1}');
    expect(c.get(`${pp}#\\pi_{1}|S_{2}`)).toBe('A_{2}');
    expect(c.get(`${pv}#\\pi_{1}|S_{2}`)).toBeCloseTo(9.4 / 0.84, 10);
    expect(c.get(`\\text{Policy iteration } Q^{\\pi_{1}}(s,a)#S_{1}|A_{2}`)).toBeCloseTo(10.47619, 4);
    expect(c.get(`${pp}#\\pi_{2}|S_{1}`)).toBe('A_{2}');
    expect(c.get(`${pp}#\\pi_{2}|S_{2}`)).toBe('A_{2}');
  });
  it('value iteration matches the solution', () => {
    const c = cellsOf(parts[1]);
    const v = '\\text{Value iteration } V_k(s)';
    const g = '\\text{Value iteration greedy } \\pi_k(s)';
    expect(c.get(`${v}#k = 1|S_{1}`)).toBe(10);
    expect(c.get(`${v}#k = 1|S_{2}`)).toBe(4);
    expect(c.get(`${v}#k = 2|S_{1}`)).toBe(10);
    expect(c.get(`${v}#k = 2|S_{2}`)).toBeCloseTo(10.04, 10);
    expect(c.get(`${g}#\\pi_{2}|S_{1}`)).toBe('A_{1}');
    expect(c.get(`${g}#\\pi_{2}|S_{2}`)).toBe('A_{2}');
  });
});

describe('linear solve with coupled unknowns', () => {
  it('solves a 2-state loop by elimination and back-substitution', () => {
    // V1 = 1 + 0.9 V2,  V2 = 2 + 0.9 (0.5 V1 + 0.5 V2)
    const p: Problem = ProblemSchema.parse({
      states: [{ id: 'S1' }, { id: 'S2' }],
      actions: [{ id: 'A1' }],
      gamma: 0.9,
      transitions: [
        { s: 'S1', a: 'A1', s2: 'S2', p: 1, r: 1 },
        { s: 'S2', a: 'A1', s2: 'S1', p: 0.5, r: 2 },
        { s: 'S2', a: 'A1', s2: 'S2', p: 0.5, r: 2 },
      ],
      policy: [{ s: 'S1', a: 'A1', p: 1 }, { s: 'S2', a: 'A1', p: 1 }],
      parts: [{ task: 'policy_evaluation_linear' }],
    });
    const [pt] = solveAll(p);
    expect(pt.error).toBeUndefined();
    const r = results([pt]);
    // V2 = 2 + 0.45(1 + 0.9 V2) + 0.45 V2 -> V2 (1 - 0.855) = 2.45
    const v2 = 2.45 / 0.145;
    expect(r.get('\\pi|S_{2}')).toBeCloseTo(v2, 10);
    expect(r.get('\\pi|S_{1}')).toBeCloseTo(1 + 0.9 * v2, 10);
  });
});
