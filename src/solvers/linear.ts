// Exact policy evaluation: write one Bellman equation per state, then solve the
// linear system by substitution / elimination, the way it is done on paper.
import type { Problem } from '../schema';
import type { Cell, Line, Step } from './types';
import { REF_GAMMA, cellRef, edgesOf, idTex, nonTerminal, outcomes, refP, refPi, refR } from '../mdp';
import { add, div, fmt, mul, noteLines, num, safeRef } from '../math/expr';

type LF = { c: number; co: Map<string, number> }; // c + sum co[s] V(s)

const EPS = 1e-12;
const vt = (s: string) => `V(${idTex(s)})`;
const n = (v: number, src?: string) => (src ? `\\htmlData{src=${safeRef(src)}}{${fmt(v)}}` : fmt(v));
const signed = (v: number, first: boolean) => (first ? fmt(v) : v < 0 ? ` - ${fmt(-v)}` : ` + ${fmt(v)}`);

function coefTex(c: number, body: string, first: boolean) {
  const mag = Math.abs(c);
  const k = Math.abs(mag - 1) < EPS ? '' : `${fmt(mag)}\\,`;
  if (first) return `${c < 0 ? '-' : ''}${k}${body}`;
  return `${c < 0 ? ' - ' : ' + '}${k}${body}`;
}

function lfTex(lf: LF, sub?: { x: string; tex: string }): string {
  let out = '';
  if (Math.abs(lf.c) > EPS) out += signed(lf.c, true);
  for (const [s, c] of lf.co) {
    if (Math.abs(c) < EPS) continue;
    const body = sub && s === sub.x ? sub.tex : vt(s);
    out += coefTex(c, body, out === '');
  }
  return out || '0';
}

const unknowns = (lf: LF, except?: string) => [...lf.co].filter(([s, c]) => s !== except && Math.abs(c) > EPS).map(([s]) => s);

function substitute(lf: LF, x: string, by: LF): LF {
  const k = lf.co.get(x) ?? 0;
  const co = new Map(lf.co);
  co.delete(x);
  for (const [s, c] of by.co) co.set(s, (co.get(s) ?? 0) + k * c);
  return { c: lf.c + k * by.c, co };
}

const scale = (lf: LF, f: number): LF => ({ c: lf.c * f, co: new Map([...lf.co].map(([s, c]) => [s, c * f])) });

/** Solve V^pi exactly. `pi` maps state -> action probabilities. Values are written to `table` / `row`. */
export function solveLinear(
  p: Problem,
  pi: Map<string, { a: string; p: number }[]>,
  table: string,
  row: string,
  piTex: string,
): { steps: Step[]; V: Map<string, number> } {
  const steps: Step[] = [];
  const cell = (s: string): Cell => ({ table, row, col: idTex(s) });
  const S = nonTerminal(p);
  const eqs = new Map<string, LF>();

  for (const s of S) {
    const acts = pi.get(s);
    if (!acts?.length) throw new Error(`No policy action for state ${s}`);
    const stoch = acts.length > 1 || acts[0].p !== 1;
    const lf: LF = { c: 0, co: new Map() };
    const written: string[] = [];
    const expanded: string[] = [];
    for (const { a, p: w } of acts) {
      const outs = outcomes(p, s, a);
      if (!outs.length) throw new Error(`No transitions given for state ${s}, action ${a}`);
      const inner = outs
        .map((t) => {
          const term = p.terminal.includes(t.s2) ? '0' : vt(t.s2);
          return `${n(t.p, refP(s, a, t.s2))}\\left(${n(t.r, refR(s, a, t.s2))} + ${n(p.gamma, REF_GAMMA)}\\,${term}\\right)`;
        })
        .join(' + ');
      written.push(stoch ? `${n(w, refPi(s, a))}\\left[${inner}\\right]` : inner);
      for (const t of outs) {
        lf.c += w * t.p * t.r;
        expanded.push(signed(w * t.p * t.r, expanded.length === 0));
        if (!p.terminal.includes(t.s2)) {
          const k = w * t.p * p.gamma;
          lf.co.set(t.s2, (lf.co.get(t.s2) ?? 0) + k);
          expanded.push(coefTex(k, vt(t.s2), false));
        }
      }
    }
    eqs.set(s, lf);
    const lines: Line[] = [{ lhs: vt(s), rhs: written.join(' + ') }];
    const exp = expanded.join('');
    const col = lfTex(lf);
    lines.push({ rhs: exp });
    if (col !== exp) lines.push({ rhs: col });
    const solvedNow = unknowns(lf).length === 0;
    steps.push({
      title: `Write the Bellman equation for $${vt(s)}$ under $${piTex}$`,
      why: stoch
        ? `Under ${piTex.replace(/\\/g, '')}, each action is weighted by its policy probability. Every outcome adds probability × (reward + γ × value of the next state). Terminal states have value 0. Unknown values stay as symbols.`
        : `The policy picks ${acts[0].a} in ${s}. Each outcome adds probability × (reward + γ × value of the next state). Terminal states have value 0. Unknown values stay as symbols because we solve all equations together.`,
      formula: String.raw`V^{\pi}(s) = \sum_{a}\pi(a\mid s)\sum_{s'} P(s'\mid s,a)\,\big[R(s,a,s') + \gamma\,V^{\pi}(s')\big]`,
      lines,
      results: solvedNow ? [{ cell: cell(s), value: lf.c }] : [],
      edges: acts.flatMap(({ a }) => edgesOf(p, s, a)),
    });
  }

  const remaining = new Set(S);
  const iso = new Map<string, LF>();
  const order: string[] = [];
  const V = new Map<string, number>(p.terminal.map((t) => [t, 0]));

  while (remaining.size) {
    const x = [...remaining].sort((a, b) => unknowns(eqs.get(a)!, a).length - unknowns(eqs.get(b)!, b).length)[0];
    const lf = eqs.get(x)!;
    const a = lf.co.get(x) ?? 0;
    const rest: LF = { c: lf.c, co: new Map([...lf.co].filter(([s]) => s !== x)) };
    remaining.delete(x);
    order.push(x);

    if (Math.abs(a) > EPS) {
      if (Math.abs(1 - a) < 1e-9) throw new Error(`Cannot isolate ${vt(x)}: coefficient 1 - ${fmt(a)} = 0 (no unique solution)`);
      const res = scale(rest, 1 / (1 - a));
      const numeric = unknowns(rest).length === 0;
      const lines: Line[] = [
        { lhs: vt(x), rhs: lfTex(lf) },
        { lhs: `${vt(x)} - ${fmt(a)}\\,${vt(x)}`, rhs: lfTex(rest) },
        { lhs: `${fmt(1 - a)}\\,${vt(x)}`, rhs: lfTex(rest) },
      ];
      if (numeric) {
        lines.push(...noteLines(vt(x), div(num(rest.c), num(1 - a))));
      } else {
        lines.push({ lhs: vt(x), rhs: `\\dfrac{${lfTex(rest)}}{${fmt(1 - a)}}` }, { rhs: lfTex(res) });
      }
      steps.push({
        title: `Isolate $${vt(x)}$`,
        why: `${vt(x).replace(/[\\{}_]/g, '')} appears on both sides. Move it to the left, factor it out, then divide by its coefficient.`,
        lines,
        results: numeric ? [{ cell: cell(x), value: res.c }] : [],
      });
      iso.set(x, res);
    } else {
      iso.set(x, rest);
    }

    const ix = iso.get(x)!;
    const numeric = unknowns(ix).length === 0;
    if (numeric) V.set(x, ix.c);
    for (const y of remaining) {
      const ey = eqs.get(y)!;
      if (Math.abs(ey.co.get(x) ?? 0) < EPS) continue;
      const by = numeric ? n(ix.c, cellRef(cell(x))) : `\\left[${lfTex(ix)}\\right]`;
      const next = substitute(ey, x, ix);
      eqs.set(y, next);
      const k = ey.co.get(x)!;
      const subTex = numeric ? `\\left(${by}\\right)` : by;
      const lines: Line[] = [
        { lhs: vt(y), rhs: lfTex(ey, { x, tex: subTex }).replace(`${fmt(Math.abs(k))}\\,${subTex}`, `${fmt(Math.abs(k))} \\times ${subTex}`) },
        { rhs: lfTex(next) },
      ];
      const done = unknowns(next).length === 0;
      steps.push({
        title: `Substitute $${vt(x)}$ into the equation for $${vt(y)}$`,
        why: numeric
          ? `We now know ${vt(x).replace(/[\\{}_]/g, '')} = ${fmt(ix.c)}. Plug it in, then multiply and add the numbers.`
          : `Replace ${vt(x).replace(/[\\{}_]/g, '')} by its expression so this equation has one unknown fewer.`,
        lines,
        results: done ? [{ cell: cell(y), value: next.c }] : [],
      });
    }
  }

  for (const x of [...order].reverse()) {
    if (V.has(x)) continue;
    const ix = iso.get(x)!;
    const e = add(num(ix.c), ...[...ix.co].filter(([, c]) => Math.abs(c) > EPS).map(([s, c]) => mul(num(c), num(V.get(s)!, cellRef(cell(s))))));
    const v = ix.c + [...ix.co].reduce((acc, [s, c]) => acc + c * V.get(s)!, 0);
    V.set(x, v);
    steps.push({
      title: `Back-substitute to get $${vt(x)}$`,
      why: 'Every value this expression depends on is now known. Plug them in.',
      lines: [{ lhs: vt(x), rhs: lfTex(ix) }, ...noteLines(undefined, e)],
      results: [{ cell: cell(x), value: v }],
    });
  }

  steps.at(-1)?.results.push(...p.terminal.map((t) => ({ cell: cell(t), value: 0 })));
  return { steps, V };
}
