// Module 4-5: n-step TD / SARSA, off-policy n-step SARSA with importance sampling,
// n-step tree backup, Dyna-Q planning and DQN targets / loss.
import type { Problem } from '../schema';
import type { Cell, Solver, Step } from './types';
import {
  REF_ALPHA, REF_GAMMA, cellRef, focusEp, idTex as sT, initialV, refB, refEpR, refPi, refQ, refQT, refReplay,
} from '../mdp';
import { add, div, evaluate, fmt, max, mul, num, pow, sub, type Expr } from '../math/expr';
import { exprLines } from './dp';
import { actionIds, episodes, givenQ, initialQ, need, qKey, type Ep } from './modelfree';

const F = {
  ret: String.raw`G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^{n}\,V(S_{t+n})`,
  retQ: String.raw`G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^{n}\,Q(S_{t+n}, A_{t+n})`,
  v: String.raw`V(S_t) \leftarrow V(S_t) + \alpha\big[G_{t:t+n} - V(S_t)\big]`,
  q: String.raw`Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha\big[G_{t:t+n} - Q(S_t,A_t)\big]`,
  rho: String.raw`\rho_{t+1:t+n} = \prod_{k=t+1}^{\min(t+n,\,T-1)} \frac{\pi(A_k \mid S_k)}{b(A_k \mid S_k)}`,
  qIs: String.raw`Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha\,\rho_{t+1:t+n}\big[G_{t:t+n} - Q(S_t,A_t)\big]`,
  tb: String.raw`G_{k} = R_{k+1} + \gamma \sum_{a \ne A_{k+1}} \pi(a \mid S_{k+1})\,Q(S_{k+1},a) + \gamma\,\pi(A_{k+1} \mid S_{k+1})\,G_{k+1}`,
  tbLeaf: String.raw`G_{k} = R_{k+1} + \gamma \sum_{a} \pi(a \mid S_{k+1})\,Q(S_{k+1},a)`,
  ql: String.raw`Q(S,A) \leftarrow Q(S,A) + \alpha\big[R + \gamma\,\max_{a} Q(S',a) - Q(S,A)\big]`,
  y: String.raw`y = \begin{cases} r & \text{if } s' \text{ is terminal} \\ r + \gamma \max_{a'} \hat{Q}(s', a') & \text{otherwise} \end{cases}`,
  loss: String.raw`L = \frac{1}{N}\sum_{i=1}^{N} \big(y_i - Q(s_i, a_i)\big)^2`,
};

const gammaPow = (p: Problem, k: number): Expr => (k === 1 ? num(p.gamma, REF_GAMMA) : pow(num(p.gamma, REF_GAMMA), num(k)));

/** State at time k of an episode; null if the episode is over (terminal). */
const stateAt = (p: Problem, ep: Ep, k: number): string | null => {
  const s = k < ep.steps.length ? ep.steps[k].s : k === ep.steps.length ? ep.end : '';
  return !s || p.terminal.includes(s) ? null : s;
};
const actionAt = (ep: Ep, k: number) => (k < ep.steps.length ? ep.steps[k].a : ep.endAction);

/** R_{t+1} + gamma R_{t+2} + ... (+ gamma^n * boot): the n-step return as one expression. */
function nReturn(p: Problem, ep: Ep, e: number, t: number, n: number, boot: (s: string, k: number) => Expr | null) {
  const T = ep.steps.length;
  const h = Math.min(t + n, T);
  const terms: Expr[] = [];
  for (let i = t; i < h; i++) {
    const r = num(ep.steps[i].r, refEpR(e, i));
    terms.push(i === t ? r : mul(gammaPow(p, i - t), r));
  }
  const s = t + n <= T ? stateAt(p, ep, t + n) : null;
  const b = s ? boot(s, t + n) : null;
  if (b) terms.push(mul(gammaPow(p, n), b));
  return { ex: add(...terms), bootstrapped: !!b, s };
}

const returnWhy = (n: number, bootstrapped: boolean, what: string) =>
  bootstrapped
    ? `Add up the next ${n} reward${n > 1 ? 's' : ''}, each discounted one more time, then bootstrap: add γ^n × ${what} of the state reached after n steps.`
    : 'The episode ends before n steps, so this is just the discounted sum of the remaining rewards (no bootstrap).';

function nstep(kind: 'v' | 'sarsa'): Solver {
  return (p, part) => {
    const n = need(p.n, 'n (number of steps)');
    const alpha = need(p.alpha, 'α (step size)');
    const steps: Step[] = [];
    const V = initialV(p);
    const Q = initialQ(p);
    const vT = String.raw`\text{n-step TD } V(s)`;
    const qT = String.raw`\text{n-step SARSA } Q(s,a)`;
    const vCell = (s: string): Cell => ({ table: vT, row: 'V', col: sT(s) });
    const qCell = (s: string, a: string): Cell => ({ table: qT, row: sT(s), col: sT(a) });
    if (kind === 'v')
      steps.push({
        title: 'Given: starting values $V(s)$',
        why: 'n-step TD starts from the values given in the question (0 if none are given).',
        lines: [...V].map(([s, x]) => ({ lhs: `V(${sT(s)})`, rhs: fmt(x) })),
        results: [...V].map(([s, x]) => ({ cell: vCell(s), value: x })),
      });
    else steps.push(givenQ(p, Q, qCell, 'n-step SARSA starts from these Q-values (0 if none are given).'));

    for (const { ep, e } of episodes(p, part)) {
      const gT = `\\text{${kind === 'v' ? 'n-step TD' : 'n-step SARSA'}, episode ${e + 1} } G_{t:t+${n}}`;
      ep.steps.forEach((x, t) => {
        const { ex, bootstrapped } = nReturn(p, ep, e, t, n, (s, k) => {
          if (kind === 'v') return num(V.get(s) ?? 0, cellRef(vCell(s)));
          const a = actionAt(ep, k);
          if (!a) throw new Error(`n-step SARSA needs the action taken at time ${k} of episode ${e + 1}`);
          return num(Q.get(qKey(s, a)) ?? 0, cellRef(qCell(s, a)));
        });
        const G = evaluate(ex);
        const gCell: Cell = { table: gT, row: 'G', col: `t = ${t}` };
        steps.push({
          title: `Episode ${e + 1}, t = ${t}: return $G_{${t}:${t + n}}$`,
          why: returnWhy(n, bootstrapped, kind === 'v' ? 'V' : 'Q'),
          formula: kind === 'v' ? F.ret : F.retQ,
          lines: exprLines(`G_{${t}:${t + n}}`, ex),
          results: [{ cell: gCell, value: G }],
          focus: focusEp(e, t),
        });
        if (kind === 'v') {
          const old = num(V.get(x.s) ?? 0, cellRef(vCell(x.s)));
          const up = add(old, mul(num(alpha, REF_ALPHA), sub(num(G, cellRef(gCell)), old)));
          V.set(x.s, evaluate(up));
          steps.push({
            title: `Episode ${e + 1}, t = ${t}: update $V(${sT(x.s)})$`,
            why: `Move V(${x.s}) a fraction α toward the ${n}-step return.`,
            formula: F.v,
            lines: exprLines(`V(${sT(x.s)})`, up),
            results: [{ cell: vCell(x.s), value: V.get(x.s)! }],
            focus: focusEp(e, t),
          });
        } else {
          if (!x.a) throw new Error(`Episode ${e + 1}, step ${t}: the action taken is missing`);
          const old = num(Q.get(qKey(x.s, x.a)) ?? 0, cellRef(qCell(x.s, x.a)));
          const up = add(old, mul(num(alpha, REF_ALPHA), sub(num(G, cellRef(gCell)), old)));
          Q.set(qKey(x.s, x.a), evaluate(up));
          steps.push({
            title: `Episode ${e + 1}, t = ${t}: update $Q(${sT(x.s)}, ${sT(x.a)})$`,
            why: `Move Q(${x.s}, ${x.a}) a fraction α toward the ${n}-step return.`,
            formula: F.q,
            lines: exprLines(`Q(${sT(x.s)}, ${sT(x.a)})`, up),
            results: [{ cell: qCell(x.s, x.a), value: Q.get(qKey(x.s, x.a))! }],
            focus: focusEp(e, t),
          });
        }
      });
    }
    return steps;
  };
}

/** pi(a|s) or b(a|s) from a policy table; 0 for actions not listed in a listed state. */
function prob(p: Problem, which: 'policy' | 'behavior', s: string, a: string): number {
  const rows = p[which].filter((x) => x.s === s);
  if (!rows.length)
    throw new Error(`${which === 'policy' ? 'Target policy π' : 'Behaviour policy b'} is not given for state ${s}`);
  return rows.find((x) => x.a === a)?.p ?? 0;
}

const nstepOffPolicy: Solver = (p, part) => {
  const n = need(p.n, 'n (number of steps)');
  const alpha = need(p.alpha, 'α (step size)');
  if (!p.behavior.length) throw new Error('Off-policy learning needs the behaviour policy b(a|s)');
  const Q = initialQ(p);
  const qT = String.raw`\text{Off-policy n-step SARSA } Q(s,a)`;
  const qCell = (s: string, a: string): Cell => ({ table: qT, row: sT(s), col: sT(a) });
  const steps: Step[] = [givenQ(p, Q, qCell, 'Learning starts from these Q-values (0 if none are given).')];
  for (const { ep, e } of episodes(p, part)) {
    const T = ep.steps.length;
    const tag = `\\text{Off-policy, episode ${e + 1}}`;
    ep.steps.forEach((x, t) => {
      if (!x.a) throw new Error(`Episode ${e + 1}, step ${t}: the action taken is missing`);
      const ratios: Expr[] = [];
      for (let k = t + 1; k <= Math.min(t + n, T - 1); k++) {
        const { s, a } = ep.steps[k];
        const bp = prob(p, 'behavior', s, a);
        if (bp === 0) throw new Error(`b(${a}|${s}) = 0, but the behaviour policy took ${a} in ${s}`);
        ratios.push(div(num(prob(p, 'policy', s, a), refPi(s, a)), num(bp, refB(s, a))));
      }
      const rhoEx = ratios.length ? mul(...ratios) : num(1);
      const rho = evaluate(rhoEx);
      const rCell: Cell = { table: `${tag}\\ \\rho`, row: '\\rho', col: `t = ${t}` };
      steps.push({
        title: `Episode ${e + 1}, t = ${t}: importance ratio $\\rho_{${t + 1}:${t + n}}$`,
        why: ratios.length
          ? 'The data came from b, but we learn about π. Each later action is re-weighted by how much more (or less) likely π was to take it than b. A ratio of 0 means π would never act like this, so the update is cancelled.'
          : 'No later actions in this window need correcting (A_t itself is being evaluated), so ρ = 1.',
        formula: F.rho,
        lines: exprLines(`\\rho_{${t + 1}:${t + n}}`, rhoEx),
        results: [{ cell: rCell, value: rho }],
        focus: focusEp(e, t),
      });
      const { ex, bootstrapped } = nReturn(p, ep, e, t, n, (s, k) => {
        const a = actionAt(ep, k);
        if (!a) throw new Error(`Need the action taken at time ${k} of episode ${e + 1} to bootstrap`);
        return num(Q.get(qKey(s, a)) ?? 0, cellRef(qCell(s, a)));
      });
      const G = evaluate(ex);
      const gCell: Cell = { table: `${tag}\\ G_{t:t+${n}}`, row: 'G', col: `t = ${t}` };
      steps.push({
        title: `Episode ${e + 1}, t = ${t}: return $G_{${t}:${t + n}}$`,
        why: returnWhy(n, bootstrapped, 'Q'),
        formula: F.retQ,
        lines: exprLines(`G_{${t}:${t + n}}`, ex),
        results: [{ cell: gCell, value: G }],
        focus: focusEp(e, t),
      });
      const old = num(Q.get(qKey(x.s, x.a)) ?? 0, cellRef(qCell(x.s, x.a)));
      const up = add(old, mul(num(alpha, REF_ALPHA), num(rho, cellRef(rCell)), sub(num(G, cellRef(gCell)), old)));
      Q.set(qKey(x.s, x.a), evaluate(up));
      steps.push({
        title: `Episode ${e + 1}, t = ${t}: update $Q(${sT(x.s)}, ${sT(x.a)})$`,
        why: 'Ordinary n-step SARSA update, scaled by the importance ratio ρ.',
        formula: F.qIs,
        lines: exprLines(`Q(${sT(x.s)}, ${sT(x.a)})`, up),
        results: [{ cell: qCell(x.s, x.a), value: Q.get(qKey(x.s, x.a))! }],
        focus: focusEp(e, t),
      });
    });
  }
  return steps;
};

const treeBackup: Solver = (p, part) => {
  const n = need(p.n, 'n (number of steps)');
  const alpha = need(p.alpha, 'α (step size)');
  const A = actionIds(p);
  const Q = initialQ(p);
  const qT = String.raw`\text{Tree backup } Q(s,a)`;
  const qCell = (s: string, a: string): Cell => ({ table: qT, row: sT(s), col: sT(a) });
  const q = (s: string, a: string) => num(Q.get(qKey(s, a)) ?? 0, cellRef(qCell(s, a)));
  const pi = (s: string, a: string) => num(prob(p, 'policy', s, a), refPi(s, a));
  const steps: Step[] = [givenQ(p, Q, qCell, 'Tree backup starts from these Q-values (0 if none are given).')];
  for (const { ep, e } of episodes(p, part)) {
    const T = ep.steps.length;
    ep.steps.forEach((x, t) => {
      if (!x.a) throw new Error(`Episode ${e + 1}, step ${t}: the action taken is missing`);
      const h = Math.min(t + n, T);
      const gT = `\\text{Episode ${e + 1} tree-backup returns}`;
      const gCell = (k: number): Cell => ({ table: gT, row: `t = ${t}`, col: `G_{${k}}` });
      let G = 0;
      for (let k = h - 1; k >= t; k--) {
        const r = num(ep.steps[k].r, refEpR(e, k));
        const s1 = stateAt(p, ep, k + 1);
        let ex: Expr;
        let why: string;
        if (!s1) {
          ex = r;
          why = 'The episode ends here, so the return is just the last reward.';
        } else if (k === h - 1) {
          ex = add(r, mul(num(p.gamma, REF_GAMMA), add(...A.map((a) => mul(pi(s1, a), q(s1, a))))));
          why = `Leaf of the tree: after R${k + 1}, average Q over every action in ${s1}, weighted by the target policy π.`;
        } else {
          const a1 = actionAt(ep, k + 1);
          const others = A.filter((a) => a !== a1);
          ex = add(
            r,
            ...(others.length ? [mul(num(p.gamma, REF_GAMMA), add(...others.map((a) => mul(pi(s1, a), q(s1, a)))))] : []),
            mul(num(p.gamma, REF_GAMMA), pi(s1, a1), num(G, cellRef(gCell(k + 1)))),
          );
          why = `Actions not taken in ${s1} contribute their current Q-value (weighted by π). The action actually taken, ${a1}, contributes the return from deeper in the tree instead.`;
        }
        G = evaluate(ex);
        steps.push({
          title: `Episode ${e + 1}, t = ${t}: tree-backup return $G_{${k}}$`,
          why,
          formula: !s1 ? undefined : k === h - 1 ? F.tbLeaf : F.tb,
          lines: exprLines(`G_{${k}}`, ex),
          results: [{ cell: gCell(k), value: G }],
          focus: focusEp(e, k),
        });
      }
      const old = q(x.s, x.a);
      const up = add(old, mul(num(alpha, REF_ALPHA), sub(num(G, cellRef(gCell(t))), old)));
      Q.set(qKey(x.s, x.a), evaluate(up));
      steps.push({
        title: `Episode ${e + 1}, t = ${t}: update $Q(${sT(x.s)}, ${sT(x.a)})$`,
        why: 'Move Q a fraction α toward the tree-backup return. No importance sampling is needed: π is used directly as weights.',
        formula: F.q,
        lines: exprLines(`Q(${sT(x.s)}, ${sT(x.a)})`, up),
        results: [{ cell: qCell(x.s, x.a), value: Q.get(qKey(x.s, x.a))! }],
        focus: focusEp(e, t),
      });
    });
  }
  return steps;
};

const dynaQ: Solver = (p, part) => {
  const alpha = need(p.alpha, 'α (step size)');
  const n = p.n ?? p.planning.length;
  const A = actionIds(p);
  const Q = initialQ(p);
  const qT = String.raw`\text{Dyna-Q } Q(s,a)`;
  const mT = String.raw`\text{Dyna-Q model}`;
  const qCell = (s: string, a: string): Cell => ({ table: qT, row: sT(s), col: sT(a) });
  const q = (s: string, a: string) => num(Q.get(qKey(s, a)) ?? 0, cellRef(qCell(s, a)));
  const model = new Map<string, { s: string; a: string; r: number; s2: string | null }>();
  const steps: Step[] = [givenQ(p, Q, qCell, 'Dyna-Q starts from these Q-values (0 if none are given) and an empty model.')];
  let cursor = 0;

  const update = (s: string, a: string, r: Expr, s2: string | null) => {
    const ex = add(q(s, a), mul(num(alpha, REF_ALPHA), sub(add(r, mul(num(p.gamma, REF_GAMMA), s2 ? max(...A.map((b) => q(s2, b))) : num(0))), q(s, a))));
    Q.set(qKey(s, a), evaluate(ex));
    return ex;
  };

  for (const { ep, e } of episodes(p, part)) {
    ep.steps.forEach((x, t) => {
      if (!x.a) throw new Error(`Episode ${e + 1}, step ${t}: the action taken is missing`);
      const s2 = stateAt(p, ep, t + 1);
      const ex = update(x.s, x.a, num(x.r, refEpR(e, t)), s2);
      steps.push({
        title: `Real step ${e + 1}.${t}: direct RL on $Q(${sT(x.s)}, ${sT(x.a)})$`,
        why: `Real experience ${x.s} --${x.a}--> ${s2 ?? 'terminal'}, reward ${fmt(x.r)}: an ordinary Q-learning update.`,
        formula: F.ql,
        lines: exprLines(`Q(${sT(x.s)}, ${sT(x.a)})`, ex),
        results: [{ cell: qCell(x.s, x.a), value: Q.get(qKey(x.s, x.a))! }],
        focus: focusEp(e, t),
      });
      model.set(qKey(x.s, x.a), { s: x.s, a: x.a, r: x.r, s2 });
      const mc = { table: mT, row: `(${sT(x.s)}, ${sT(x.a)})`, col: "R,\\ S'" };
      steps.push({
        title: `Model learning: remember $(${sT(x.s)}, ${sT(x.a)})$`,
        why: 'Dyna-Q stores what happened so it can replay it later without acting in the real world.',
        lines: [{ lhs: `\\text{Model}(${sT(x.s)}, ${sT(x.a)})`, rhs: `(${fmt(x.r)},\\ ${s2 ? sT(s2) : '\\text{terminal}'})` }],
        results: [{ cell: mc, value: `${fmt(x.r)},\\ ${s2 ? sT(s2) : '\\text{end}'}` }],
        focus: focusEp(e, t),
      });
      for (let i = 0; i < n; i++) {
        const keys = [...model.keys()];
        const pick = p.planning.length ? p.planning[cursor] : model.get(keys[cursor % keys.length]);
        cursor++;
        if (!pick) throw new Error('Ran out of planning (s, a) pairs listed in the question');
        const m = model.get(qKey(pick.s, pick.a));
        if (!m) throw new Error(`Planning picks (${pick.s}, ${pick.a}), but the model has not seen that pair yet`);
        const ex2 = update(m.s, m.a, num(m.r, cellRef({ table: mT, row: `(${sT(m.s)}, ${sT(m.a)})`, col: "R,\\ S'" })), m.s2);
        steps.push({
          title: `Planning ${i + 1}/${n}: $Q(${sT(m.s)}, ${sT(m.a)})$`,
          why: `Simulated experience from the model: (${m.s}, ${m.a}) gave reward ${fmt(m.r)} and led to ${m.s2 ?? 'terminal'}. Same Q-learning update, no real step taken.${p.planning.length ? '' : ' (Real Dyna-Q picks a random remembered pair; here they are replayed in order.)'}`,
          formula: F.ql,
          lines: exprLines(`Q(${sT(m.s)}, ${sT(m.a)})`, ex2),
          results: [{ cell: qCell(m.s, m.a), value: Q.get(qKey(m.s, m.a))! }],
        });
      }
    });
  }
  return steps;
};

const dqn: Solver = (p) => {
  if (!p.replay.length) throw new Error('This part needs the replay minibatch (s, a, r, s′, done)');
  const useTarget = p.Qtarget.length > 0;
  const tgt = useTarget ? p.Qtarget : p.Q;
  const steps: Step[] = [];
  const T = String.raw`\text{DQN minibatch}`;
  const yCell = (i: number): Cell => ({ table: T, row: `i = ${i + 1}`, col: 'y_i' });
  const qs: Expr[] = [];
  p.replay.forEach((x, i) => {
    let ex: Expr = num(x.r, refReplay(i));
    if (!x.done) {
      const next = tgt.filter((y) => y.s === x.s2);
      if (!next.length) throw new Error(`No ${useTarget ? 'target-network' : ''} Q-values given for next state ${x.s2}`);
      ex = add(ex, mul(num(p.gamma, REF_GAMMA), max(...next.map((y) => num(y.v, useTarget ? refQT(y.s, y.a) : refQ(y.s, y.a))))));
    }
    const y = evaluate(ex);
    const qv = p.Q.find((y2) => y2.s === x.s && y2.a === x.a);
    if (!qv) throw new Error(`The network's prediction Q(${x.s}, ${x.a}) is not given`);
    qs.push(sub(num(y, cellRef(yCell(i))), num(qv.v, refQ(x.s, x.a))));
    steps.push({
      title: `Target $y_{${i + 1}}$ for $(${sT(x.s)}, ${sT(x.a)})$`,
      why: x.done
        ? 'The episode ended at this transition, so there is no future value: the target is just the reward.'
        : `Reward plus γ times the best next-state value, read from the ${useTarget ? 'target network Q̂ (a frozen copy, which keeps targets stable)' : 'network (no separate target network given)'}.`,
      formula: F.y,
      lines: exprLines(`y_{${i + 1}}`, ex),
      results: [{ cell: yCell(i), value: y }],
    });
  });
  const loss = div(add(...qs.map((d) => pow(d, num(2)))), num(qs.length));
  steps.push({
    title: 'Minibatch loss',
    why: 'Mean squared TD error between the targets and the online network’s predictions Q(s, a). Gradient descent on this loss trains the network.',
    formula: F.loss,
    lines: exprLines('L', loss),
    results: [{ cell: { table: T, row: '\\text{loss}', col: 'L' }, value: evaluate(loss) }],
  });
  return steps;
};

export const NSTEP = {
  nstep_td: nstep('v'),
  nstep_sarsa: nstep('sarsa'),
  nstep_offpolicy_is: nstepOffPolicy,
  tree_backup: treeBackup,
  dyna_q: dynaQ,
  dqn_target_loss: dqn,
};
