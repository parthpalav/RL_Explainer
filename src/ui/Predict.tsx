import { useMemo, useState } from 'react';
import type { Step } from '../solvers/types';
import { CARDS } from '../content/theory';
import { fmt } from '../math/expr';
import { Rich, Tex } from './Tex';

/** A step can be quizzed when it has a formula and a numeric answer worked out over several lines. */
export const quizzable = (s: Step) => !!s.formula && s.lines.length > 1 && s.results.some((r) => typeof r.value === 'number');

// Stable pseudo-random order, so options do not reshuffle on every render.
const hash = (s: string) => [...s].reduce((h, c) => (h * 31 + c.charCodeAt(0)) | 0, 7);

export const close = (got: number, want: number) => Math.abs(got - want) <= Math.max(0.01, Math.abs(want) * 1e-3);

interface Props {
  step: Step;
  pool: string[]; // other formulas to use as wrong options
  onDone: (correct: boolean) => void;
}

/** Game mode: pick the formula and work out the number before the solution is shown. */
export function Predict({ step, pool, onDone }: Props) {
  const right = step.formula!;
  const answer = step.results.find((r) => typeof r.value === 'number')!.value as number;
  const options = useMemo(() => {
    const wrong = [...new Set(pool.filter((f) => f !== right))].sort((a, b) => hash(a + right) - hash(b + right)).slice(0, 2);
    return [right, ...wrong].sort((a, b) => hash(a + step.title) - hash(b + step.title));
  }, [pool, right, step.title]);
  const [choice, setChoice] = useState<string | null>(null);
  const [val, setVal] = useState('');
  const [result, setResult] = useState<{ f: boolean; v: boolean } | null>(null);
  const lhs = step.lines[0].lhs ?? 'x';

  return (
    <div className="predict">
      <p className="why">Your turn: pick the formula, then work out the value on paper.</p>
      <fieldset disabled={!!result}>
        <legend>1. Which formula applies?</legend>
        {options.map((f) => (
          <label key={f} className={`opt ${result && f === right ? 'right' : ''} ${result && f === choice && f !== right ? 'wrong' : ''}`}>
            <input type="radio" name="formula" checked={choice === f} onChange={() => setChoice(f)} />
            <Tex tex={f} display />
          </label>
        ))}
      </fieldset>
      <fieldset disabled={!!result}>
        <legend>2. What is the result?</legend>
        <label className="ans">
          <Tex tex={`${lhs} =`} />
          <input type="number" step="any" value={val} onChange={(e) => setVal(e.target.value)} aria-label="Your answer" />
        </label>
      </fieldset>
      {!result ? (
        <button className="primary" disabled={!choice || val === ''} onClick={() => setResult({ f: choice === right, v: close(+val, answer) })}>
          Check
        </button>
      ) : (
        <div className="verdict" role="status">
          <p className={result.f ? 'ok' : 'bad'}>{result.f ? '✓ Right formula.' : '✗ Not that formula: the right one is highlighted.'}</p>
          <p className={result.v ? 'ok' : 'bad'}>
            {result.v ? `✓ Correct: ${fmt(answer)}.` : `✗ The answer is ${fmt(answer)}. Watch the working to see where it differs.`}
          </p>
          <button className="primary" onClick={() => onDone(result.f && result.v)}>Show the working ▶</button>
        </div>
      )}
    </div>
  );
}

/** Revision card for a theory question. */
export function TheoryCards({ ids }: { ids: string[] }) {
  return (
    <>
      {CARDS.filter((c) => ids.includes(c.id)).map((c) => (
        <section className="theory" key={c.id}>
          <h4>{c.title}</h4>
          {c.intro && <p><Rich text={c.intro} /></p>}
          {c.table && (
            <table>
              <thead>
                <tr>{c.table.head.map((h) => <th key={h}>{h}</th>)}</tr>
              </thead>
              <tbody>
                {c.table.rows.map((r) => (
                  <tr key={r[0]}>
                    {r.map((cell, i) => (i ? <td key={i}><Rich text={cell} /></td> : <th key={i}>{cell}</th>))}
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {c.points && <ul>{c.points.map((x) => <li key={x}><Rich text={x} /></li>)}</ul>}
        </section>
      ))}
    </>
  );
}
