// Tiny arithmetic expression tree that can be reduced one level at a time,
// so every solver gets hand-solving style working lines for free.

export type Expr =
  | { k: 'num'; v: number; src?: string; fresh?: boolean }
  | { k: 'add'; xs: Expr[] }
  | { k: 'mul'; xs: Expr[] }
  | { k: 'sub'; a: Expr; b: Expr }
  | { k: 'div'; a: Expr; b: Expr }
  | { k: 'pow'; a: Expr; b: Expr }
  | { k: 'max'; xs: Expr[] }
  | { k: 'sqrt'; x: Expr }
  | { k: 'ln'; x: Expr };

export const num = (v: number, src?: string): Expr => ({ k: 'num', v, src });
export const add = (...xs: Expr[]): Expr => (xs.length === 1 ? xs[0] : { k: 'add', xs });
export const mul = (...xs: Expr[]): Expr => (xs.length === 1 ? xs[0] : { k: 'mul', xs });
export const sub = (a: Expr, b: Expr): Expr => ({ k: 'sub', a, b });
export const div = (a: Expr, b: Expr): Expr => ({ k: 'div', a, b });
export const pow = (a: Expr, b: Expr): Expr => ({ k: 'pow', a, b });
export const max = (...xs: Expr[]): Expr => (xs.length === 1 ? xs[0] : { k: 'max', xs });
export const sqrt = (x: Expr): Expr => ({ k: 'sqrt', x });
export const ln = (x: Expr): Expr => ({ k: 'ln', x });

const kids = (e: Expr): Expr[] => {
  switch (e.k) {
    case 'num': return [];
    case 'add': case 'mul': case 'max': return e.xs;
    case 'sub': case 'div': case 'pow': return [e.a, e.b];
    case 'sqrt': case 'ln': return [e.x];
  }
};

const nv = (e: Expr) => (e as { v: number }).v;

function apply(e: Expr, c: number[]): number {
  switch (e.k) {
    case 'num': return e.v;
    case 'add': return c.reduce((s, x) => s + x, 0);
    case 'mul': return c.reduce((s, x) => s * x, 1);
    case 'max': return Math.max(...c);
    case 'sub': return c[0] - c[1];
    case 'div': return c[0] / c[1];
    case 'pow': return c[0] ** c[1];
    case 'sqrt': return Math.sqrt(c[0]);
    case 'ln': return Math.log(c[0]);
  }
}

export const evaluate = (e: Expr): number => apply(e, kids(e).map(evaluate));

const rebuild = (e: Expr, c: Expr[]): Expr => {
  switch (e.k) {
    case 'num': return e;
    case 'add': case 'mul': case 'max': return { ...e, xs: c };
    case 'sub': case 'div': case 'pow': return { ...e, a: c[0], b: c[1] };
    case 'sqrt': case 'ln': return { ...e, x: c[0] };
  }
};

/** Evaluate every node whose children are all numbers (one hand-written line). */
export function reduceOnce(e: Expr): Expr {
  if (e.k === 'num') return { k: 'num', v: e.v };
  const c = kids(e);
  if (c.every((x) => x.k === 'num')) return { k: 'num', v: apply(e, c.map(nv)), fresh: true };
  return rebuild(e, c.map(reduceOnce));
}

/** Substituted expression followed by each reduction until a single number. */
export function chain(e: Expr): Expr[] {
  const out = [e];
  while (e.k !== 'num') out.push((e = reduceOnce(e)));
  return out;
}

export function fmt(v: number): string {
  const r = Math.round(v * 1e4) / 1e4;
  return (Object.is(r, -0) ? 0 : r).toString();
}

/** Refs end up inside KaTeX \htmlData and HTML attributes: keep them to [A-Za-z0-9-]. */
export const safeRef = (s: string) =>
  s.replace(/[^A-Za-z0-9]/g, (c) => '-' + c.charCodeAt(0).toString(36) + '-');

const PREC: Record<Expr['k'], number> = { add: 1, sub: 1, mul: 2, div: 2, pow: 3, num: 4, max: 4, sqrt: 4, ln: 4 };

function numTex(e: Extract<Expr, { k: 'num' }>, v = e.v): string {
  let t = fmt(v);
  if (e.src) t = `\\htmlData{src=${safeRef(e.src)}}{${t}}`;
  if (e.fresh) t = `\\htmlClass{fresh}{${t}}`;
  return t;
}

const paren = (t: string) => `\\left(${t}\\right)`;

export function toLatex(e: Expr, parent = 0): string {
  const self = PREC[e.k];
  let t: string;
  switch (e.k) {
    case 'num':
      t = numTex(e);
      return e.v < 0 && parent > 0 ? paren(t) : t;
    case 'add':
      // a + (-b) is written a - b, like on paper
      t = e.xs
        .map((x, i) => {
          if (i === 0) return x.k === 'num' ? numTex(x) : toLatex(x, 1);
          if (x.k === 'num' && x.v < 0) return ` - ${numTex(x, -x.v)}`;
          return ` + ${toLatex(x, 1)}`;
        })
        .join('');
      break;
    case 'sub':
      t = `${toLatex(e.a, 1)} - ${toLatex(e.b, 2)}`;
      break;
    case 'mul':
      t = e.xs.map((x) => toLatex(x, 2)).join(' \\times ');
      break;
    case 'div':
      return `\\dfrac{${toLatex(e.a)}}{${toLatex(e.b)}}`;
    case 'pow':
      t = `${toLatex(e.a, 4)}^{${toLatex(e.b)}}`;
      break;
    case 'max':
      return `\\max\\left(${e.xs.map((x) => toLatex(x)).join(',\\ ')}\\right)`;
    case 'sqrt':
      return `\\sqrt{${toLatex(e.x)}}`;
    case 'ln':
      return `\\ln${paren(toLatex(e.x))}`;
  }
  return self < parent ? paren(t) : t;
}
