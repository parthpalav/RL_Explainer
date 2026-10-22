// Module 1-2 solvers: Bellman backups, policy evaluation / improvement / iteration,
// value iteration, greedy and epsilon-greedy action selection.
import type { Problem } from '../schema';
import type { Cell, Ctx, Line, Solver, Step } from './types';
import {
  REF_EPS, REF_GAMMA, actionsIn, cellRef, edgesOf, idTex, initialV, label, nonTerminal, policyAt, qExpr, refPi, refQ, targetStates,
} from '../mdp';
import { add, chain, div, evaluate, fmt, mul, num, pow, max, safeRef, sub, toLatex, type Expr } from '../math/expr';
import { solveLinear } from './linear';

export const F = {
  peDet: String.raw`V_{k+1}(s) = \sum_{s'} P(s'\mid s,\pi(s))\,\big[R(s,\pi(s),s') + \gamma\,V_k(s')\big]`,
  peSto: String.raw`V_{k+1}(s) = \sum_{a}\pi(a\mid s)\sum_{s'} P(s'\mid s,a)\,\big[R(s,a,s') + \gamma\,V_k(s')\big]`,
  q: String.raw`Q(s,a) = \sum_{s'} P(s'\mid s,a)\,\big[R(s,a,s') + \gamma\,V(s')\big]`,
  vi: String.raw`V_{k+1}(s) = \max_{a}\, Q_{k+1}(s,a)`,
  greedy: String.raw`\pi(s) = \arg\max_{a}\, Q(s,a)`,
  eps: String.raw`P(a^{*}) = 1 - \varepsilon + \frac{\varepsilon}{|A|}, \qquad P(a \ne a^{*}) = \frac{\varepsilon}{|A|}`,
  ret: String.raw`G_t = R_{t+1} + \gamma R_{t+2} + \gamma^{2} R_{t+3} + \cdots`,
};

const T = {
  pe: String.raw`\text{Policy evaluation } V_k(s)`,
  ip: String.raw`\text{In-place evaluation } V_k(s)`,
  vi: String.raw`\text{Value iteration } V_k(s)`,
  viQ: (k: number) => String.raw`\text{Value iteration } Q_{${k}}(s,a)`,
  viPi: String.raw`\text{Value iteration greedy } \pi_k(s)`,
  piV: String.raw`\text{Policy iteration } V^{\pi_i}(s)`,
  piQ: (i: number) => String.raw`\text{Policy iteration } Q^{\pi_{${i}}}(s,a)`,
  pi: String.raw`\text{Policy iteration } \pi_i(s)`,
  q: String.raw`Q(s,a)`,
  pol: String.raw`\text{Improved policy } \pi'(s)`,
  eps: String.raw`\varepsilon\text{-greedy } P(a\mid s)`,
};

const sT = idTex;
const nm = (p: Problem, id: string) => (label(p, id) ? ` (${label(p, id)})` : '');
const vCell = (table: string, row: string, s: string): Cell => ({ table, row, col: sT(s) });
const qCell = (table: string, s: string, a: string): Cell => ({ table, row: sT(s), col: sT(a) });

export const exprLines = (lhs: string, e: Expr): Line[] => chain(e).map((x, i) => ({ lhs: i ? undefined : lhs, rhs: toLatex(x) }));

const tagged = (v: number, src: string) => `\\htmlData{src=${safeRef(src)}}{${fmt(v)}}`;

/** argmax written so each number is labelled with its action underneath. */
export function argmaxLines(lhs: string, rhsSym: string, items: { id: string; v: number; src: string }[], prefer?: string) {
  const best = Math.max(...items.map((x) => x.v));
  const winners = items.filter((x) => Math.abs(x.v - best) < 1e-9);
  const pick = winners.find((x) => x.id === prefer) ?? winners[0];
  const lines: Line[] = [
    { lhs, rhs: rhsSym },
    { rhs: `\\arg\\max\\left(${items.map((x) => `\\underset{${sT(x.id)}}{${tagged(x.v, x.src)}}`).join(',\\ ')}\\right)` },
    { rhs: sT(pick.id) },
  ];
  return { lines, pick: pick.id, tie: winners.length > 1 };
}

function givenV(p: Problem, V: Map<string, number>, table: string, row: string, sym: string, why: string): Step {
  return {
    title: `Given: starting values $${sym}$`,
    why,
    lines: p.states.map((s) => ({ lhs: `${sym}(${sT(s.id)})`, rhs: fmt(V.get(s.id)!) })),
    results: p.states.map((s) => ({ cell: vCell(table, row, s.id), value: V.get(s.id)! })),
  };
}

const terminalCells = (p: Problem, table: string, row: string) =>
  p.terminal.map((t) => ({ cell: vCell(table, row, t), value: 0 }));

function evaluation(inplace: boolean, defaultIters = 1): Solver {
  return (p, part, ctx) => {
    const table = inplace ? T.ip : T.pe;
    const target = part.iterations ?? defaultIters;
    const steps: Step[] = [];
    let k = 0;
    let V: Map<string, number>;
    if (ctx.pe && ctx.pe.inplace === inplace && ctx.pe.k < target) ({ k, V } = ctx.pe);
    else {
      V = initialV(p);
      steps.push(givenV(p, V, table, 'k = 0', 'V_0', 'Policy evaluation starts from the initial values given in the question (0 if none are given).'));
    }
    for (; k < target; k++) {
      const next = new Map(V);
      const row = `k = ${k + 1}`;
      const updated = new Set<string>();
      nonTerminal(p).forEach((s, idx) => {
        const from = inplace ? next : V;
        const vSrc = (s2: string) => cellRef(vCell(table, inplace && updated.has(s2) ? row : `k = ${k}`, s2));
        const pol = policyAt(p, s);
        const det = pol.length === 1 && pol[0].p === 1;
        const e = det
          ? qExpr(p, s, pol[0].a, from, vSrc)
          : add(...pol.map((x) => mul(num(x.p, refPi(s, x.a)), qExpr(p, s, x.a, from, vSrc))));
        const v = evaluate(e);
        const lhs = `V_{${k + 1}}(${sT(s)})`;
        steps.push({
          title: `Sweep ${k + 1}: $${lhs}$${nm(p, s)}`,
          why:
            (det
              ? `The policy picks ${pol[0].a}${nm(p, pol[0].a)} in ${s}. For each place ${pol[0].a} can lead, multiply its probability by (reward + γ × value of that next state), then add them up.`
              : `The policy is stochastic in ${s}: work out the bracket for each action, weight it by π(a|s), then add.`) +
            (inplace
              ? ' In-place: states already updated in this sweep use their new value.'
              : ` Next-state values come from the previous sweep, V${k}.`),
          formula: det ? F.peDet : F.peSto,
          lines: exprLines(lhs, e),
          results: [{ cell: vCell(table, row, s), value: v }, ...(idx === 0 ? terminalCells(p, table, row) : [])],
          edges: pol.flatMap((x) => edgesOf(p, s, x.a)),
        });
        next.set(s, v);
        updated.add(s);
      });
      V = next;
    }
    ctx.pe = { k, V, inplace };
    const row = `k = ${k}`;
    ctx.lastV = { V, label: `V_{${k}}`, src: (s) => cellRef(vCell(table, row, s)) };
    return steps;
  };
}

/** Q(s,a) for every action in `states`, then the greedy pick per state. */
function improve(
  p: Problem,
  states: string[],
  V: Map<string, number>,
  vSrc: (s: string) => string | undefined,
  qTable: string,
  vSym: string,
  piOut: { table: string; row: string; sym: string } | null,
  prev?: Map<string, string>,
) {
  const steps: Step[] = [];
  const policy = new Map<string, string>();
  for (const s of states) {
    const items: { id: string; v: number; src: string }[] = [];
    for (const a of actionsIn(p, s)) {
      const e = qExpr(p, s, a, V, vSrc);
      const v = evaluate(e);
      const cell = qCell(qTable, s, a);
      items.push({ id: a, v, src: cellRef(cell) });
      steps.push({
        title: `$Q(${sT(s)}, ${sT(a)})$${nm(p, a)}`,
        why: `Value of taking ${a} in ${s} and then following the current values ${vSym}: probability × (reward + γ × ${vSym} of the next state), summed over outcomes.`,
        formula: F.q,
        lines: exprLines(`Q(${sT(s)}, ${sT(a)})`, e),
        results: [{ cell, value: v }],
        edges: edgesOf(p, s, a),
      });
    }
    if (!items.length) throw new Error(`No transitions given for state ${s}`);
    if (!piOut) continue;
    const lhs = `${piOut.sym}(${sT(s)})`;
    const { lines, pick, tie } = argmaxLines(lhs, `\\arg\\max_a Q(${sT(s)}, a)`, items, prev?.get(s));
    policy.set(s, pick);
    steps.push({
      title: `Greedy action in $${sT(s)}$`,
      why: `Pick the action with the largest Q-value.${tie ? ' Tie: keep the current action so the policy does not flip back and forth.' : ''}`,
      formula: F.greedy,
      lines,
      results: [{ cell: { table: piOut.table, row: piOut.row, col: sT(s) }, value: sT(pick) }],
    });
  }
  return { steps, policy };
}

const currentV = (p: Problem, ctx: Ctx) =>
  ctx.lastV ?? { V: initialV(p), label: 'V', src: undefined as ((s: string) => string) | undefined };

const qFromV: Solver = (p, part, ctx) => {
  const { V, label: l, src } = currentV(p, ctx);
  return improve(p, targetStates(p, part.states), V, (s) => src?.(s), T.q, l, null).steps;
};

const policyImprovement: Solver = (p, part, ctx) => {
  const { V, label: l, src } = currentV(p, ctx);
  return improve(p, targetStates(p, part.states), V, (s) => src?.(s), T.q, l, { table: T.pol, row: "\\pi'", sym: "\\pi'" }).steps;
};

function initialPolicy(p: Problem): Map<string, string> {
  const pi = new Map<string, string>();
  for (const s of nonTerminal(p)) {
    const xs = p.policy.filter((x) => x.s === s).sort((a, b) => b.p - a.p);
    const a = xs[0]?.a ?? actionsIn(p, s)[0];
    if (!a) throw new Error(`No actions available in state ${s}`);
    pi.set(s, a);
  }
  return pi;
}

const policyEvaluationLinear: Solver = (p, _part, ctx) => {
  const pi = new Map(nonTerminal(p).map((s) => [s, policyAt(p, s)]));
  const { steps, V } = solveLinear(p, pi, T.piV, '\\pi', '\\pi');
  ctx.lastV = { V, label: 'V^{\\pi}', src: (s) => cellRef(vCell(T.piV, '\\pi', s)) };
  return steps;
};

const policyIteration: Solver = (p, part, ctx) => {
  const maxIters = part.iterations ?? 20;
  let pi = initialPolicy(p);
  const steps: Step[] = [
    {
      title: 'Given: initial policy $\\pi_0$',
      why: 'Policy iteration starts from the policy given in the question (or the first action in each state if none is given).',
      lines: [...pi].map(([s, a]) => ({ lhs: `\\pi_0(${sT(s)})`, rhs: sT(a) })),
      results: [...pi].map(([s, a]) => ({ cell: { table: T.pi, row: '\\pi_0', col: sT(s) }, value: sT(a) })),
    },
  ];
  for (let i = 0; i < maxIters; i++) {
    const row = `\\pi_{${i}}`;
    const det = new Map([...pi].map(([s, a]) => [s, [{ a, p: 1 }]]));
    const ev = solveLinear(p, det, T.piV, row, row);
    ev.steps[0].title = `Iteration ${i + 1}, evaluation: ${ev.steps[0].title}`;
    steps.push(...ev.steps);
    const imp = improve(p, nonTerminal(p), ev.V, (s) => cellRef(vCell(T.piV, row, s)), T.piQ(i), `V^{${row}}`,
      { table: T.pi, row: `\\pi_{${i + 1}}`, sym: `\\pi_{${i + 1}}` }, pi);
    imp.steps[0].title = `Iteration ${i + 1}, improvement: ${imp.steps[0].title}`;
    steps.push(...imp.steps);
    ctx.lastV = { V: ev.V, label: `V^{${row}}`, src: (s) => cellRef(vCell(T.piV, row, s)) };
    const stable = [...pi].every(([s, a]) => imp.policy.get(s) === a);
    pi = imp.policy;
    if (stable) {
      steps.push({
        title: 'Policy is stable: stop',
        why: `π${i + 1} is the same as π${i}, so improving again changes nothing. This policy is optimal.`,
        lines: [...pi].map(([s, a]) => ({ lhs: `\\pi^{*}(${sT(s)})`, rhs: sT(a) })),
        results: [],
      });
      break;
    }
  }
  return steps;
};

function valueIteration(defaultIters = 1): Solver {
  return (p, part, ctx) => {
    const target = part.iterations ?? defaultIters;
    const steps: Step[] = [];
    let k = 0;
    let V: Map<string, number>;
    if (ctx.vi && ctx.vi.k < target) ({ k, V } = ctx.vi);
    else {
      V = initialV(p);
      steps.push(givenV(p, V, T.vi, 'k = 0', 'V_0', 'Value iteration starts from the initial values given in the question (0 if none are given).'));
    }
    for (; k < target; k++) {
      const next = new Map(V);
      const row = `k = ${k + 1}`;
      const qT = T.viQ(k + 1);
      nonTerminal(p).forEach((s, idx) => {
        const imp = improve(p, [s], V, (s2) => cellRef(vCell(T.vi, `k = ${k}`, s2)), qT, `V_{${k}}`, null);
        imp.steps.forEach((st) => {
          st.title = `Sweep ${k + 1}: ${st.title.replace('Q(', `Q_{${k + 1}}(`)}`;
          st.lines[0].lhs = st.lines[0].lhs!.replace('Q(', `Q_{${k + 1}}(`);
        });
        steps.push(...imp.steps);
        const acts = actionsIn(p, s);
        const qs = acts.map((a) => imp.steps[acts.indexOf(a)].results[0].value as number);
        const e = max(...acts.map((a, i) => num(qs[i], cellRef(qCell(qT, s, a)))));
        const v = evaluate(e);
        const best = acts[qs.indexOf(Math.max(...qs))];
        const lhs = `V_{${k + 1}}(${sT(s)})`;
        steps.push({
          title: `Sweep ${k + 1}: $${lhs}$${nm(p, s)}`,
          why: `Value iteration keeps the best action's Q-value as the new state value. The best action (${best}) is the greedy policy for now.`,
          formula: F.vi,
          lines: [...exprLines(lhs, e), { lhs: `\\pi_{${k + 1}}(${sT(s)})`, rhs: sT(best) }],
          results: [
            { cell: vCell(T.vi, row, s), value: v },
            { cell: { table: T.viPi, row: `\\pi_{${k + 1}}`, col: sT(s) }, value: sT(best) },
            ...(idx === 0 ? terminalCells(p, T.vi, row) : []),
          ],
        });
        next.set(s, v);
      });
      V = next;
    }
    ctx.vi = { k, V };
    const row = `k = ${k}`;
    ctx.lastV = { V, label: `V_{${k}}`, src: (s) => cellRef(vCell(T.vi, row, s)) };
    return steps;
  };
}

function qTable(p: Problem, states: string[]) {
  const S = states.length ? states : [...new Set(p.Q.map((x) => x.s))];
  if (!S.length) throw new Error('This part needs a Q-table (Q(s,a) values) in the data');
  return S.map((s) => {
    const items = p.Q.filter((x) => x.s === s).map((x) => ({ id: x.a, v: x.v, src: refQ(s, x.a) }));
    if (!items.length) throw new Error(`No Q-values given for state ${s}`);
    return { s, items };
  });
}

/** 1 - ε + ε/|A| for the greedy action, ε/|A| otherwise. */
export const epsProb = (greedy: boolean, eps: number, nA: number): Expr =>
  greedy ? add(sub(num(1), num(eps, REF_EPS)), div(num(eps, REF_EPS), num(nA))) : div(num(eps, REF_EPS), num(nA));

export const EPS_WHY = {
  greedy: 'With probability 1 − ε we exploit (pick a*). With probability ε we explore uniformly, and a* can be picked then too, adding ε/|A|.',
  other: (a: string) => `${a} is only picked while exploring: probability ε spread evenly over |A| actions.`,
};

const greedyAction: Solver = (p, part) =>
  qTable(p, part.states).map(({ s, items }) => {
    const { lines } = argmaxLines(`\\pi_{\\text{greedy}}(${sT(s)})`, `\\arg\\max_a Q(${sT(s)}, a)`, items);
    return {
      title: `Greedy action in $${sT(s)}$${nm(p, s)}`,
      why: 'A greedy policy always exploits: it picks the action with the largest Q-value.',
      formula: F.greedy,
      lines,
      results: [],
    };
  });

const epsilonGreedy: Solver = (p, part) => {
  if (p.epsilon === undefined) throw new Error('ε (epsilon) is missing from the data');
  const eps = p.epsilon;
  const steps: Step[] = [];
  for (const { s, items } of qTable(p, part.states)) {
    const g = argmaxLines(`a^{*}`, `\\arg\\max_a Q(${sT(s)}, a)`, items);
    const nA = items.length;
    steps.push({
      title: `Find the greedy action $a^{*}$ in $${sT(s)}$`,
      why: `ε-greedy first finds the greedy action. There are |A| = ${nA} actions.`,
      formula: F.greedy,
      lines: g.lines,
      results: [],
    });
    for (const { id } of items) {
      const isG = id === g.pick;
      const e = epsProb(isG, eps, nA);
      steps.push({
        title: `$P(${sT(id)} \\mid ${sT(s)})$: ${isG ? 'greedy' : 'non-greedy'} action`,
        why: isG ? EPS_WHY.greedy : EPS_WHY.other(id),
        formula: F.eps,
        lines: exprLines(`P(${sT(id)} \\mid ${sT(s)})`, e),
        results: [{ cell: qCell(T.eps, s, id), value: evaluate(e) }],
      });
    }
  }
  return steps;
};

const bestState: Solver = (p, part, ctx) => {
  const { V, label: l, src } = currentV(p, ctx);
  const S = targetStates(p, part.states);
  const items = S.map((s) => ({ id: s, v: V.get(s) ?? 0, src: src?.(s) ?? '' }));
  const { lines } = argmaxLines('s^{*}', `\\arg\\max_s ${l}(s)`, items);
  return [{
    title: 'State with the highest value',
    why: `Compare the latest values ${l}. The largest one has the highest expected long-term return.`,
    lines,
    results: [],
  }];
};

const returnGt: Solver = (p, part) => {
  if (!part.rewards.length) throw new Error('This part needs the reward sequence R1, R2, ...');
  const e = add(
    ...part.rewards.map((r, i) =>
      i === 0 ? num(r) : mul(i === 1 ? num(p.gamma, REF_GAMMA) : pow(num(p.gamma, REF_GAMMA), num(i)), num(r)),
    ),
  );
  return [{
    title: 'Discounted return $G_t$',
    why: 'Each later reward is multiplied by one more factor of γ, because rewards further away are worth less.',
    formula: F.ret,
    lines: exprLines('G_t', e),
    results: [],
  }];
};

const theory: Solver = (_p, part) => [{
  title: 'Theory question',
  why: `This part asks for an explanation, not a calculation.${part.text ? ` "${part.text}"` : ''}`,
  lines: [],
  results: [],
}];

export const DP = {
  return_Gt: returnGt,
  bellman_v: evaluation(false),
  q_from_v: qFromV,
  bellman_optimality: valueIteration(),
  policy_evaluation: evaluation(false),
  policy_evaluation_inplace: evaluation(true),
  policy_evaluation_linear: policyEvaluationLinear,
  policy_improvement: policyImprovement,
  policy_iteration: policyIteration,
  value_iteration: valueIteration(),
  greedy_action: greedyAction,
  epsilon_greedy_probs: epsilonGreedy,
  best_state: bestState,
  theory,
};
