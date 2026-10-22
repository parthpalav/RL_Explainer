import type { Problem } from './schema';
import type { Cell } from './solvers/types';
import { add, mul, num, safeRef, type Expr } from './math/expr';

/** "S1" -> S_{1}, "A2" -> A_{2}, anything else -> \text{...} */
export function idTex(id: string): string {
  const m = /^([A-Za-z]+)_?(\d+)$/.exec(id);
  return m ? `${m[1]}_{${m[2]}}` : `\\text{${id.replace(/[\\{}$&#^_%~]/g, '')}}`;
}

export const label = (p: Problem, id: string) =>
  p.states.find((s) => s.id === id)?.name || p.actions.find((a) => a.id === id)?.name || '';

// Refs tie a number in a formula to where it came from, for the fly-in animation.
export const refP = (s: string, a: string, s2: string) => `P|${s}|${a}|${s2}`;
export const refR = (s: string, a: string, s2: string) => `R|${s}|${a}|${s2}`;
export const refQ = (s: string, a: string) => `Q|${s}|${a}`;
export const refPi = (s: string, a: string) => `pi|${s}|${a}`;
export const REF_GAMMA = 'gamma';
export const REF_EPS = 'epsilon';
export const REF_ALPHA = 'alpha';
export const REF_C = 'c';
export const REF_T = 't';
export const REF_N = 'n';
export const refB = (s: string, a: string) => `b|${s}|${a}`;
export const refQT = (s: string, a: string) => `QT|${s}|${a}`;
export const refReplay = (i: number) => `rep|${i}|r`;
export const refEpR = (e: number, t: number) => `ep|${e}|${t}|r`;
export const refPull = (i: number) => `pull|${i}`;
export const refArm = (a: string, f: 'q' | 'n' | 'wins') => `arm|${a}|${f}`;
// Timeline items a step can point at.
export const focusEp = (e: number, t: number) => `ep|${e}|${t}`;
export const focusPull = (i: number) => `pull|${i}`;
export const cellRef = (c: Cell) => `cell|${c.table}|${c.row}|${c.col}`;
export const domRef = (r: string) => safeRef(r);

/** What a referenced number is, as LaTeX, so a flying value can say where it came from. */
export function refTex(r: string): string {
  const [k, ...x] = r.split('|');
  const i = (j: number) => idTex(x[j]);
  switch (k) {
    case 'P': return `P(${i(2)} \\mid ${i(0)}, ${i(1)})`;
    case 'R': return `R(${i(0)}, ${i(1)}, ${i(2)})`;
    case 'Q': return `Q(${i(0)}, ${i(1)})`;
    case 'QT': return `\\hat{Q}(${i(0)}, ${i(1)})`;
    case 'pi': return `\\pi(${i(1)} \\mid ${i(0)})`;
    case 'b': return `b(${i(1)} \\mid ${i(0)})`;
    case 'gamma': return '\\gamma';
    case 'epsilon': return '\\varepsilon';
    case 'alpha': return '\\alpha';
    case 'c': case 't': case 'n': return k;
    case 'rep': return `r_{${+x[0] + 1}}`;
    case 'ep': return `R_{${+x[1] + 1}}\\ (\\text{ep } ${+x[0] + 1})`;
    case 'pull': return `r_{\\#${+x[0] + 1}}`;
    case 'arm': return `${{ q: 'Q', n: 'N', wins: '\\text{wins}' }[x[1]] ?? x[1]}(${i(0)})`;
    case 'cell': return `${x.at(-1)}\\ (${x.at(-2)})`; // column (row), e.g. S_1 (k = 1)
    default: return '';
  }
}

export const edgeKey = (s: string, a: string, s2: string) => `${s}|${a}|${s2}`;

export const nonTerminal = (p: Problem) => p.states.map((s) => s.id).filter((s) => !p.terminal.includes(s));

export const targetStates = (p: Problem, states: string[]) => (states.length ? states : nonTerminal(p));

export const outcomes = (p: Problem, s: string, a: string) => p.transitions.filter((t) => t.s === s && t.a === a);

export const actionsIn = (p: Problem, s: string) => {
  const used = new Set(p.transitions.filter((t) => t.s === s).map((t) => t.a));
  return p.actions.map((a) => a.id).filter((a) => used.has(a));
};

export function initialV(p: Problem): Map<string, number> {
  const V = new Map(p.states.map((s) => [s.id, 0]));
  for (const x of p.V0) V.set(x.s, x.v);
  for (const t of p.terminal) V.set(t, 0);
  return V;
}

/** pi(a|s) as a list; errors name the missing state so the user can fix the form. */
export function policyAt(p: Problem, s: string): { a: string; p: number }[] {
  const xs = p.policy.filter((x) => x.s === s && x.p > 0);
  if (!xs.length) throw new Error(`No policy given for state ${s}`);
  return xs;
}

/** sum_{s'} P(s'|s,a) [R + gamma V(s')] with every number tagged by its source. */
export function qExpr(p: Problem, s: string, a: string, V: Map<string, number>, vSrc: (s2: string) => string | undefined): Expr {
  const outs = outcomes(p, s, a);
  if (!outs.length) throw new Error(`No transitions given for state ${s}, action ${a}`);
  return add(
    ...outs.map((t) =>
      mul(
        num(t.p, refP(s, a, t.s2)),
        add(num(t.r, refR(s, a, t.s2)), mul(num(p.gamma, REF_GAMMA), num(V.get(t.s2) ?? 0, vSrc(t.s2)))),
      ),
    ),
  );
}

export const edgesOf = (p: Problem, s: string, a: string) => outcomes(p, s, a).map((t) => edgeKey(s, a, t.s2));
