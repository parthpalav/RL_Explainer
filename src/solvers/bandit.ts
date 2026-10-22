// Multi-armed bandits and A/B tests. Arms are the problem's actions.
import type { Problem } from '../schema';
import type { Cell, Solver, Step } from './types';
import { REF_ALPHA, REF_C, REF_T, cellRef, focusPull, idTex as sT, refArm, refPull } from '../mdp';
import { add, div, evaluate, fmt, ln, mul, num, sqrt, sub, type Expr } from '../math/expr';
import { EPS_WHY, argmaxLines, epsProb, exprLines } from './dp';
import { need } from './modelfree';

const F = {
  avg: String.raw`Q(a) = \frac{R_1 + R_2 + \cdots + R_{N(a)}}{N(a)}`,
  inc: String.raw`Q_{n+1} = Q_n + \frac{1}{n}\big[R_n - Q_n\big]`,
  incA: String.raw`Q_{n+1} = Q_n + \alpha\big[R_n - Q_n\big]`,
  greedy: String.raw`A^{*} = \arg\max_{a}\, Q(a)`,
  eps: String.raw`P(a) = \begin{cases} 1 - \varepsilon + \frac{\varepsilon}{k} & a = A^{*} \\ \frac{\varepsilon}{k} & \text{otherwise} \end{cases}`,
  ucb: String.raw`A_t = \arg\max_{a} \Big[ Q_t(a) + c\,\sqrt{\frac{\ln t}{N_t(a)}} \Big]`,
  rate: String.raw`\hat{p} = \frac{\text{conversions}}{\text{visitors}}`,
  pool: String.raw`\hat{p}_{\text{pool}} = \frac{x_A + x_B}{n_A + n_B}`,
  se: String.raw`SE = \sqrt{\hat{p}_{\text{pool}}\,(1 - \hat{p}_{\text{pool}})\Big(\frac{1}{n_A} + \frac{1}{n_B}\Big)}`,
  z: String.raw`z = \frac{\hat{p}_A - \hat{p}_B}{SE}`,
};

const T = String.raw`\text{Bandit estimates}`;
const qCell = (a: string): Cell => ({ table: T, row: 'Q(a)', col: sT(a) });
const nCell = (a: string): Cell => ({ table: T, row: 'N(a)', col: sT(a) });

function arms(p: Problem): string[] {
  const ids = p.actions.length ? p.actions.map((a) => a.id) : [...new Set([...p.arms.map((x) => x.a), ...p.pulls.map((x) => x.a)])];
  if (!ids.length) throw new Error('This part needs the list of arms (actions)');
  return ids;
}
const snap = (p: Problem, a: string) => p.arms.find((x) => x.a === a);

/** Q(a) = average reward of each arm's pulls. */
function sampleAverage(p: Problem): { steps: Step[]; Q: Map<string, { v: number; src?: string }>; N: Map<string, number> } {
  if (!p.pulls.length) throw new Error('This part needs the history of pulls (arm, reward)');
  const steps: Step[] = [];
  const Q = new Map<string, { v: number; src?: string }>();
  const N = new Map<string, number>();
  for (const a of arms(p)) {
    const mine = p.pulls.map((x, i) => ({ ...x, i })).filter((x) => x.a === a);
    N.set(a, mine.length);
    if (!mine.length) {
      Q.set(a, { v: 0 });
      steps.push({ title: `$Q(${sT(a)})$: never pulled`, why: `${a} has no rewards yet, so its estimate stays at the initial value 0.`, lines: [{ lhs: `Q(${sT(a)})`, rhs: '0' }], results: [{ cell: qCell(a), value: 0 }, { cell: nCell(a), value: 0 }] });
      continue;
    }
    const ex: Expr = mine.length === 1 ? num(mine[0].r, refPull(mine[0].i)) : div(add(...mine.map((x) => num(x.r, refPull(x.i)))), num(mine.length));
    const v = evaluate(ex);
    Q.set(a, { v, src: cellRef(qCell(a)) });
    steps.push({
      title: `Sample average $Q(${sT(a)})$`,
      why: `${a} was pulled ${mine.length} time${mine.length > 1 ? 's' : ''}. Its value estimate is the average of the rewards it gave.`,
      formula: F.avg,
      lines: exprLines(`Q(${sT(a)})`, ex),
      results: [{ cell: qCell(a), value: v }, { cell: nCell(a), value: mine.length }],
    });
  }
  return { steps, Q, N };
}

/** Estimates from the arms snapshot when given, otherwise worked out from the pulls. */
function estimates(p: Problem) {
  const ids = arms(p);
  if (ids.every((a) => snap(p, a)?.q !== undefined)) {
    return {
      steps: [] as Step[],
      Q: new Map(ids.map((a) => [a, { v: snap(p, a)!.q!, src: refArm(a, 'q') }])),
      N: new Map(ids.map((a) => [a, snap(p, a)?.n ?? NaN])),
      nSrc: (a: string) => refArm(a, 'n'),
    };
  }
  return { ...sampleAverage(p), nSrc: (a: string) => cellRef(nCell(a)) };
}

const bSampleAverage: Solver = (p) => sampleAverage(p).steps;

const bIncremental: Solver = (p) => {
  if (!p.pulls.length) throw new Error('This part needs the history of pulls (arm, reward)');
  const ids = arms(p);
  const Q = new Map(ids.map((a) => [a, snap(p, a)?.q ?? 0]));
  const N = new Map(ids.map((a) => [a, snap(p, a)?.n ?? 0]));
  const steps: Step[] = [{
    title: 'Given: starting estimates',
    why: 'Each arm starts with these estimates and pull counts (0 if none are given).',
    lines: ids.map((a) => ({ lhs: `Q(${sT(a)}),\\ N(${sT(a)})`, rhs: `${fmt(Q.get(a)!)},\\ ${N.get(a)}` })),
    results: ids.flatMap((a) => [{ cell: qCell(a), value: Q.get(a)! }, { cell: nCell(a), value: N.get(a)! }]),
  }];
  p.pulls.forEach((x, i) => {
    const n = N.get(x.a)! + 1;
    N.set(x.a, n);
    const old = num(Q.get(x.a)!, cellRef(qCell(x.a)));
    const step = p.alpha !== undefined ? num(p.alpha, REF_ALPHA) : div(num(1), num(n));
    const ex = add(old, mul(step, sub(num(x.r, refPull(i)), old)));
    Q.set(x.a, evaluate(ex));
    steps.push({
      title: `Pull ${i + 1}: ${x.a} gives reward ${fmt(x.r)}`,
      why: p.alpha !== undefined
        ? `Constant step size α: move Q(${x.a}) a fraction α toward the new reward.`
        : `This is pull number ${n} of ${x.a}, so the step size is 1/${n}. This keeps Q equal to the running average without storing every reward.`,
      formula: p.alpha !== undefined ? F.incA : F.inc,
      lines: exprLines(`Q(${sT(x.a)})`, ex),
      results: [{ cell: qCell(x.a), value: Q.get(x.a)! }, { cell: nCell(x.a), value: n }],
      focus: focusPull(i),
    });
  });
  return steps;
};

const bEpsGreedy: Solver = (p) => {
  const eps = need(p.epsilon, 'ε (epsilon)');
  const { steps, Q } = estimates(p);
  const ids = arms(p);
  const items = ids.map((a) => ({ id: a, v: Q.get(a)!.v, src: Q.get(a)!.src ?? '' }));
  const g = argmaxLines('A^{*}', '\\arg\\max_a Q(a)', items);
  steps.push({ title: 'Greedy arm $A^{*}$', why: `ε-greedy first finds the arm with the highest estimate. There are k = ${ids.length} arms.`, formula: F.greedy, lines: g.lines, results: [] });
  for (const a of ids) {
    const ex = epsProb(a === g.pick, eps, ids.length);
    steps.push({
      title: `$P(${sT(a)})$: ${a === g.pick ? 'greedy' : 'non-greedy'} arm`,
      why: a === g.pick ? EPS_WHY.greedy : EPS_WHY.other(a),
      formula: F.eps,
      lines: exprLines(`P(${sT(a)})`, ex),
      results: [{ cell: { table: String.raw`\varepsilon\text{-greedy } P(a)`, row: 'P', col: sT(a) }, value: evaluate(ex) }],
    });
  }
  return steps;
};

const bUcb: Solver = (p) => {
  const c = need(p.c, 'c (UCB exploration constant)');
  const { steps, Q, N, nSrc } = estimates(p);
  const ids = arms(p);
  if (ids.some((a) => !Number.isFinite(N.get(a)))) throw new Error('UCB needs the pull count N(a) of every arm');
  const t = p.t ?? ids.reduce((s, a) => s + N.get(a)!, 0);
  const ucbT = String.raw`\text{UCB scores}`;
  const items: { id: string; v: number; src: string }[] = [];
  for (const a of ids) {
    const n = N.get(a)!;
    const cell: Cell = { table: ucbT, row: '\\text{UCB}', col: sT(a) };
    if (n === 0) {
      items.push({ id: a, v: Infinity, src: cellRef(cell) });
      steps.push({ title: `UCB score of $${sT(a)}$`, why: `${a} has never been pulled (N = 0), so its bonus is infinite: UCB always tries every arm once first.`, lines: [{ lhs: `\\text{UCB}(${sT(a)})`, rhs: '\\infty' }], results: [{ cell, value: '\\infty' }] });
      continue;
    }
    const q = Q.get(a)!;
    const ex = add(num(q.v, q.src), mul(num(c, REF_C), sqrt(div(ln(num(t, p.t ? REF_T : undefined)), num(n, nSrc(a))))));
    const v = evaluate(ex);
    items.push({ id: a, v, src: cellRef(cell) });
    steps.push({
      title: `UCB score of $${sT(a)}$`,
      why: `Estimate plus an exploration bonus. The bonus shrinks as ${a} is pulled more (N = ${n}) and grows slowly with time t = ${t}${p.t ? '' : ' (total pulls so far)'}.`,
      formula: F.ucb,
      lines: exprLines(`\\text{UCB}(${sT(a)})`, ex),
      results: [{ cell, value: v }],
    });
  }
  const g = argmaxLines('A_t', '\\arg\\max_a \\text{UCB}(a)', items.map((x) => ({ ...x, v: Number.isFinite(x.v) ? x.v : Number.MAX_VALUE })));
  g.lines[1].rhs = g.lines[1].rhs.replace(fmt(Number.MAX_VALUE), '\\infty');
  steps.push({ title: 'Arm chosen by UCB', why: 'Pull the arm with the largest UCB score.', formula: F.ucb, lines: g.lines, results: [] });
  return steps;
};

const abTest: Solver = (p) => {
  const ids = arms(p);
  const rows = ids.map((a) => {
    const s = snap(p, a);
    if (s?.n === undefined || s.wins === undefined) throw new Error(`A/B test needs visitors n and conversions for ${a}`);
    return { a, n: s.n, w: s.wins };
  });
  const rT = String.raw`\text{A/B test}`;
  const steps: Step[] = rows.map(({ a, n, w }) => {
    const ex = div(num(w, refArm(a, 'wins')), num(n, refArm(a, 'n')));
    return {
      title: `Conversion rate of $${sT(a)}$`,
      why: `Fraction of ${a}'s visitors who converted.`,
      formula: F.rate,
      lines: exprLines(`\\hat{p}_{${sT(a)}}`, ex),
      results: [{ cell: { table: rT, row: '\\hat{p}', col: sT(a) }, value: evaluate(ex) }],
    };
  });
  if (rows.length !== 2) {
    const g = argmaxLines('\\text{best}', '\\arg\\max_a \\hat{p}_a', rows.map((r) => ({ id: r.a, v: r.w / r.n, src: cellRef({ table: rT, row: '\\hat{p}', col: sT(r.a) }) })));
    steps.push({ title: 'Best variant', why: 'With more than two variants, compare the conversion rates directly.', lines: g.lines, results: [] });
    return steps;
  }
  const [A, B] = rows;
  const rate = (r: typeof A) => num(r.w / r.n, cellRef({ table: rT, row: '\\hat{p}', col: sT(r.a) }));
  const poolEx = div(add(num(A.w, refArm(A.a, 'wins')), num(B.w, refArm(B.a, 'wins'))), add(num(A.n, refArm(A.a, 'n')), num(B.n, refArm(B.a, 'n'))));
  const pool = evaluate(poolEx);
  const poolCell: Cell = { table: rT, row: '\\text{test}', col: '\\hat{p}_{\\text{pool}}' };
  const pn = () => num(pool, cellRef(poolCell));
  const seEx = sqrt(mul(pn(), sub(num(1), pn()), add(div(num(1), num(A.n, refArm(A.a, 'n'))), div(num(1), num(B.n, refArm(B.a, 'n'))))));
  const se = evaluate(seEx);
  const seCell: Cell = { table: rT, row: '\\text{test}', col: 'SE' };
  const zEx = div(sub(rate(A), rate(B)), num(se, cellRef(seCell)));
  const z = evaluate(zEx);
  steps.push(
    { title: 'Pooled conversion rate', why: 'Assume for now that both variants convert equally (the null hypothesis) and pool all the data.', formula: F.pool, lines: exprLines('\\hat{p}_{\\text{pool}}', poolEx), results: [{ cell: poolCell, value: pool }] },
    { title: 'Standard error of the difference', why: 'How much the difference in rates would wobble by chance alone.', formula: F.se, lines: exprLines('SE', seEx), results: [{ cell: seCell, value: se }] },
    { title: 'z-statistic', why: 'How many standard errors apart the two rates are.', formula: F.z, lines: exprLines('z', zEx), results: [{ cell: { table: rT, row: '\\text{test}', col: 'z' }, value: z }] },
    {
      title: 'Conclusion at the 5% level',
      why: Math.abs(z) > 1.96
        ? `|z| > 1.96, so the difference is statistically significant: ${z < 0 ? B.a : A.a} converts better.`
        : '|z| ≤ 1.96, so the difference could be chance. Keep testing or treat the variants as equal.',
      lines: [{ lhs: '|z|', rhs: `${fmt(Math.abs(z))} ${Math.abs(z) > 1.96 ? '>' : '\\le'} 1.96` }],
      results: [],
    },
  );
  return steps;
};

export const BANDIT = {
  bandit_sample_average: bSampleAverage,
  bandit_incremental: bIncremental,
  bandit_epsilon_greedy: bEpsGreedy,
  bandit_ucb: bUcb,
  ab_test: abTest,
};
