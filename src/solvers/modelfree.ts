// Module 2-3 sample-based solvers: Monte Carlo prediction / control, TD(0), SARSA, Q-learning.
// They learn from the episodes given in the question, one update at a time.
import type { Part, Problem } from '../schema';
import type { Cell, Solver, Step } from './types';
import { REF_ALPHA, REF_GAMMA, cellRef, focusEp, idTex as sT, initialV, refEpR } from '../mdp';
import { add, div, evaluate, fmt, max, mul, num, sub, type Expr } from '../math/expr';
import { EPS_WHY, argmaxLines, epsProb, exprLines } from './dp';

const F = {
  td: String.raw`V(S_t) \leftarrow V(S_t) + \alpha\big[R_{t+1} + \gamma\,V(S_{t+1}) - V(S_t)\big]`,
  ret: String.raw`G_t = R_{t+1} + \gamma\,G_{t+1}`,
  mcAvg: String.raw`V(s) = \frac{1}{N(s)} \sum \text{Returns}(s)`,
  mcAlpha: String.raw`V(S_t) \leftarrow V(S_t) + \alpha\big[G_t - V(S_t)\big]`,
  mcQ: String.raw`Q(s,a) = \frac{1}{N(s,a)} \sum \text{Returns}(s,a)`,
  sarsa: String.raw`Q(S,A) \leftarrow Q(S,A) + \alpha\big[R + \gamma\,Q(S',A') - Q(S,A)\big]`,
  ql: String.raw`Q(S,A) \leftarrow Q(S,A) + \alpha\big[R + \gamma\,\max_{a} Q(S',a) - Q(S,A)\big]`,
  greedy: String.raw`\pi(s) = \arg\max_{a}\, Q(s,a)`,
  eps: String.raw`\pi(a\mid s) = \begin{cases} 1 - \varepsilon + \frac{\varepsilon}{|A|} & a = a^{*} \\ \frac{\varepsilon}{|A|} & \text{otherwise} \end{cases}`,
};

export function need<T>(v: T | undefined, what: string): T {
  if (v === undefined) throw new Error(`${what} is missing from the data`);
  return v;
}

/** The episodes a part uses, with their original index so refs match the timeline. */
const episodes = (p: Problem, part: Part) => {
  if (!p.episodes.length) throw new Error('This part needs at least one episode (sequence of states, actions and rewards)');
  const all = p.episodes.map((ep, e) => ({ ep, e }));
  return part.episodes.length ? all.filter(({ e }) => part.episodes.includes(e + 1)) : all;
};

type Ep = Problem['episodes'][number];
/** State after step t, or null when the episode ends in a terminal state. */
const nextOf = (p: Problem, ep: Ep, t: number): string | null => {
  const s = t + 1 < ep.steps.length ? ep.steps[t + 1].s : ep.end;
  return !s || p.terminal.includes(s) ? null : s;
};

/** Discounted returns G_t of one episode, worked backwards, one step each. */
function returnsSteps(p: Problem, ep: Ep, e: number) {
  const table = `\\text{Episode ${e + 1} returns } G_t`;
  const cell = (t: number): Cell => ({ table, row: 'G', col: `t = ${t}` });
  const G: number[] = [];
  const steps: Step[] = [];
  const T = ep.steps.length;
  for (let t = T - 1; t >= 0; t--) {
    const r = num(ep.steps[t].r, refEpR(e, t));
    const ex = t === T - 1 ? r : add(r, mul(num(p.gamma, REF_GAMMA), num(G[t + 1], cellRef(cell(t + 1)))));
    G[t] = evaluate(ex);
    steps.push({
      title: `Episode ${e + 1}: return $G_{${t}}$`,
      why: t === T - 1
        ? 'Work backwards from the end of the episode. The last return is just the final reward.'
        : 'Work backwards: each return is the next reward plus γ times the return one step later.',
      formula: F.ret,
      lines: exprLines(`G_{${t}}`, ex),
      results: [{ cell: cell(t), value: G[t] }],
      focus: focusEp(e, t),
    });
  }
  return { G, steps, src: (t: number) => cellRef(cell(t)) };
}

const setTex = (xs: number[]) => `\\{${xs.map(fmt).join(',\\ ')}\\}`;

function mcPrediction(firstVisit: boolean): Solver {
  return (p, part) => {
    const table = String.raw`\text{Monte Carlo } V(s)`;
    const cell = (s: string): Cell => ({ table, row: 'V', col: sT(s) });
    // α in the data may belong to another part (e.g. TD); MC only uses it when this part asks for it.
    const alpha = /α|alpha|constant[-\s]?(?:step|α)|step[-\s]size|learning\s+rate/i.test(part.text) ? need(p.alpha, 'α (step size)') : undefined;
    const V = initialV(p);
    const returns = new Map<string, { v: number; src: string }[]>();
    const steps: Step[] = [];
    if (alpha !== undefined)
      steps.push({
        title: 'Given: starting values $V(s)$',
        why: 'Constant-α Monte Carlo nudges each value toward the observed return, starting from these values.',
        lines: [...V].map(([s, v]) => ({ lhs: `V(${sT(s)})`, rhs: fmt(v) })),
        results: [...V].map(([s, v]) => ({ cell: cell(s), value: v })),
      });
    episodes(p, part).forEach(({ ep, e }) => {
      const R = returnsSteps(p, ep, e);
      steps.push(...R.steps);
      const seen = new Set<string>();
      const used = ep.steps.flatMap((x, t) => (firstVisit && seen.has(x.s) ? [] : (seen.add(x.s), [t])));
      if (alpha !== undefined) {
        for (const t of used) {
          const s = ep.steps[t].s;
          const old = num(V.get(s) ?? 0, cellRef(cell(s)));
          const ex = add(old, mul(num(alpha, REF_ALPHA), sub(num(R.G[t], R.src(t)), old)));
          V.set(s, evaluate(ex));
          steps.push({
            title: `Episode ${e + 1}, t = ${t}: update $V(${sT(s)})$`,
            why: `${firstVisit ? 'First' : 'A'} visit to ${s}: move its value a fraction α toward the return G${t} seen from here.`,
            formula: F.mcAlpha,
            lines: exprLines(`V(${sT(s)})`, ex),
            results: [{ cell: cell(s), value: V.get(s)! }],
            focus: focusEp(e, t),
          });
        }
        return;
      }
      for (const t of used) {
        const s = ep.steps[t].s;
        returns.set(s, [...(returns.get(s) ?? []), { v: R.G[t], src: R.src(t) }]);
      }
      const visited = [...new Set(used.map((t) => ep.steps[t].s))];
      steps.push({
        title: `Episode ${e + 1}: record the returns`,
        why: firstVisit
          ? 'First-visit MC: only the return from the first time each state appears in this episode counts.'
          : 'Every-visit MC: the return from every occurrence of a state counts.',
        lines: visited.map((s) => ({ lhs: `\\text{Returns}(${sT(s)})`, rhs: setTex(returns.get(s)!.map((x) => x.v)) })),
        results: [],
      });
      for (const s of visited) {
        const rs = returns.get(s)!;
        const ex: Expr = rs.length === 1 ? num(rs[0].v, rs[0].src) : div(add(...rs.map((x) => num(x.v, x.src))), num(rs.length));
        V.set(s, evaluate(ex));
        steps.push({
          title: `$V(${sT(s)})$ after episode ${e + 1}`,
          why: `The estimate is the average of every return recorded for ${s} so far (${rs.length}).`,
          formula: F.mcAvg,
          lines: exprLines(`V(${sT(s)})`, ex),
          results: [{ cell: cell(s), value: V.get(s)! }],
        });
      }
    });
    return steps;
  };
}

const actionIds = (p: Problem) => {
  if (!p.actions.length) throw new Error('This part needs the list of actions');
  return p.actions.map((a) => a.id);
};
const qKey = (s: string, a: string) => `${s}|${a}`;
const initialQ = (p: Problem) => new Map(p.Q.map((x) => [qKey(x.s, x.a), x.v]));

/** The "Given" step for a Q-table: all non-terminal states x actions, missing entries 0. */
function givenQ(p: Problem, Q: Map<string, number>, cell: (s: string, a: string) => Cell, why: string): Step {
  const S = p.states.map((s) => s.id).filter((s) => !p.terminal.includes(s));
  const A = actionIds(p);
  const all = S.flatMap((s) => A.map((a) => [s, a] as const));
  const missing = all.some(([s, a]) => !Q.has(qKey(s, a)));
  return {
    title: 'Given: starting values $Q(s,a)$',
    why,
    lines: [
      ...p.Q.map((x) => ({ lhs: `Q(${sT(x.s)}, ${sT(x.a)})`, rhs: fmt(x.v) })),
      ...(missing ? [{ lhs: p.Q.length ? '\\text{all other } Q(s,a)' : 'Q(s,a)', rhs: '0' }] : []),
    ],
    results: all.map(([s, a]) => ({ cell: cell(s, a), value: Q.get(qKey(s, a)) ?? 0 })),
  };
}

function tdControl(kind: 'sarsa' | 'q'): Solver {
  return (p, part) => {
    const alpha = need(p.alpha, 'α (step size)');
    const A = actionIds(p);
    const table = kind === 'sarsa' ? String.raw`\text{SARSA } Q(s,a)` : String.raw`\text{Q-learning } Q(s,a)`;
    const cell = (s: string, a: string): Cell => ({ table, row: sT(s), col: sT(a) });
    const Q = initialQ(p);
    const q = (s: string, a: string) => num(Q.get(qKey(s, a)) ?? 0, cellRef(cell(s, a)));
    const steps: Step[] = [givenQ(p, Q, cell, 'Learning starts from the Q-values given in the question (0 if none are given).')];
    episodes(p, part).forEach(({ ep, e }) =>
      ep.steps.forEach((x, t) => {
        if (!x.a) throw new Error(`Episode ${e + 1}, step ${t}: the action taken is missing`);
        const s2 = nextOf(p, ep, t);
        let next: Expr = num(0);
        let why: string;
        if (kind === 'sarsa') {
          const a2 = t + 1 < ep.steps.length ? ep.steps[t + 1].a : ep.endAction;
          if (s2 && !a2) throw new Error(`SARSA needs the next action A′ taken in ${s2} (episode ${e + 1}, after step ${t})`);
          if (s2) next = q(s2, a2);
          why = s2
            ? `On-policy: the target uses the action actually taken next, A′ = ${a2} in ${s2}.`
            : `The next state is terminal, so Q(S′, A′) = 0.`;
        } else {
          if (s2) next = max(...A.map((a) => q(s2, a)));
          why = s2
            ? `Off-policy: the target uses the best next action, max over a of Q(${s2}, a), whatever the agent actually does next.`
            : `The next state is terminal, so max Q(S′, a) = 0.`;
        }
        const old = q(x.s, x.a);
        const ex = add(old, mul(num(alpha, REF_ALPHA), sub(add(num(x.r, refEpR(e, t)), mul(num(p.gamma, REF_GAMMA), next)), old)));
        const v = evaluate(ex);
        Q.set(qKey(x.s, x.a), v);
        steps.push({
          title: `Episode ${e + 1}, t = ${t}: update $Q(${sT(x.s)}, ${sT(x.a)})$`,
          why: `${x.s} --${x.a}--> ${s2 ?? 'terminal'} with reward ${fmt(x.r)}. ${why} Move Q a fraction α toward the target.`,
          formula: kind === 'sarsa' ? F.sarsa : F.ql,
          lines: exprLines(`Q(${sT(x.s)}, ${sT(x.a)})`, ex),
          results: [{ cell: cell(x.s, x.a), value: v }],
          focus: focusEp(e, t),
        });
      }),
    );
    return steps;
  };
}

const mcControl: Solver = (p, part) => {
  const A = actionIds(p);
  const eps = p.epsilon;
  const table = String.raw`\text{Monte Carlo } Q(s,a)`;
  const cell = (s: string, a: string): Cell => ({ table, row: sT(s), col: sT(a) });
  const pt = String.raw`\text{Improved policy } \pi(a\mid s)`;
  const Q = initialQ(p);
  const returns = new Map<string, { v: number; src: string }[]>();
  const steps: Step[] = [givenQ(p, Q, cell, 'Monte Carlo control starts from these action values (0 if none are given).')];
  episodes(p, part).forEach(({ ep, e }) => {
    const R = returnsSteps(p, ep, e);
    steps.push(...R.steps);
    const seen = new Set<string>();
    ep.steps.forEach((x, t) => {
      if (!x.a) throw new Error(`Episode ${e + 1}, step ${t}: the action taken is missing`);
      const k = qKey(x.s, x.a);
      if (seen.has(k)) return;
      seen.add(k);
      returns.set(k, [...(returns.get(k) ?? []), { v: R.G[t], src: R.src(t) }]);
      const rs = returns.get(k)!;
      const ex: Expr = rs.length === 1 ? num(rs[0].v, rs[0].src) : div(add(...rs.map((r) => num(r.v, r.src))), num(rs.length));
      Q.set(k, evaluate(ex));
      steps.push({
        title: `Episode ${e + 1}: $Q(${sT(x.s)}, ${sT(x.a)})$`,
        why: `First visit to the pair (${x.s}, ${x.a}) at t = ${t}: record G${t}, then average all returns recorded for this pair.`,
        formula: F.mcQ,
        lines: exprLines(`Q(${sT(x.s)}, ${sT(x.a)})`, ex),
        results: [{ cell: cell(x.s, x.a), value: Q.get(k)! }],
        focus: focusEp(e, t),
      });
    });
    for (const s of [...new Set(ep.steps.map((x) => x.s))]) {
      const items = A.map((a) => ({ id: a, v: Q.get(qKey(s, a)) ?? 0, src: cellRef(cell(s, a)) }));
      const g = argmaxLines(`a^{*}`, `\\arg\\max_a Q(${sT(s)}, a)`, items);
      steps.push({
        title: `Improve the policy in $${sT(s)}$`,
        why: eps === undefined ? 'Make the policy greedy with respect to the new Q-values.' : `First find the greedy action, then give every action an ε-greedy probability (ε = ${fmt(eps)}, |A| = ${A.length}).`,
        formula: F.greedy,
        lines: g.lines,
        results: eps === undefined ? [{ cell: { table: pt, row: sT(s), col: 'a^{*}' }, value: sT(g.pick) }] : [],
      });
      if (eps === undefined) continue;
      for (const a of A) {
        const ex = epsProb(a === g.pick, eps, A.length);
        steps.push({
          title: `$\\pi(${sT(a)} \\mid ${sT(s)})$ after episode ${e + 1}`,
          why: a === g.pick ? EPS_WHY.greedy : EPS_WHY.other(a),
          formula: F.eps,
          lines: exprLines(`\\pi(${sT(a)} \\mid ${sT(s)})`, ex),
          results: [{ cell: { table: pt, row: sT(s), col: sT(a) }, value: evaluate(ex) }],
        });
      }
    }
  });
  return steps;
};

const td0: Solver = (p, part) => {
  const alpha = need(p.alpha, 'α (step size)');
  const table = String.raw`\text{TD(0) } V(s)`;
  const cell = (s: string): Cell => ({ table, row: 'V', col: sT(s) });
  const V = initialV(p);
  const v = (s: string) => num(V.get(s) ?? 0, cellRef(cell(s)));
  const steps: Step[] = [{
    title: 'Given: starting values $V(s)$',
    why: 'TD(0) starts from the values given in the question (0 if none are given).',
    lines: [...V].map(([s, x]) => ({ lhs: `V(${sT(s)})`, rhs: fmt(x) })),
    results: [...V].map(([s, x]) => ({ cell: cell(s), value: x })),
  }];
  episodes(p, part).forEach(({ ep, e }) =>
    ep.steps.forEach((x, t) => {
      const s2 = nextOf(p, ep, t);
      const ex = add(v(x.s), mul(num(alpha, REF_ALPHA), sub(add(num(x.r, refEpR(e, t)), mul(num(p.gamma, REF_GAMMA), s2 ? v(s2) : num(0))), v(x.s))));
      V.set(x.s, evaluate(ex));
      steps.push({
        title: `Episode ${e + 1}, t = ${t}: update $V(${sT(x.s)})$`,
        why: `${x.s} → ${s2 ?? 'terminal'} with reward ${fmt(x.r)}. TD target = R + γ·V(S′)${s2 ? '' : ', and V(terminal) = 0'}. Move V(${x.s}) a fraction α toward it, without waiting for the episode to end.`,
        formula: F.td,
        lines: exprLines(`V(${sT(x.s)})`, ex),
        results: [{ cell: cell(x.s), value: V.get(x.s)! }],
        focus: focusEp(e, t),
      });
    }),
  );
  return steps;
};

export const MODEL_FREE = {
  mc_prediction_first_visit: mcPrediction(true),
  mc_prediction_every_visit: mcPrediction(false),
  mc_control_epsilon_greedy: mcControl,
  td0_prediction: td0,
  sarsa: tdControl('sarsa'),
  q_learning: tdControl('q'),
};

