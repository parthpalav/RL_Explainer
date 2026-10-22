import type { Problem } from '../schema';
import type { Cell } from '../solvers/types';
import { REF_EPS, REF_GAMMA, cellRef, domRef, edgeKey, idTex, refP, refPi, refQ, refR } from '../mdp';
import { fmt } from '../math/expr';
import { Tex } from './Tex';
import { MdpGraph } from './MdpGraph';

const Num = ({ r, v }: { r: string; v: number }) => (
  <span className="val" data-ref={domRef(r)}>
    {fmt(v)}
  </span>
);

/** Everything given in the question. Each number carries data-ref so formulas can fly values out of it. */
export function DataPanel({ p, active }: { p: Problem; active: string[] }) {
  const on = new Set(active);
  return (
    <aside className="panel data">
      <h2>Given</h2>
      <MdpGraph p={p} active={active} />
      <div className="chips">
        <span className="chip">
          <Tex tex="\gamma" /> = <Num r={REF_GAMMA} v={p.gamma} />
        </span>
        {p.epsilon !== undefined && (
          <span className="chip">
            <Tex tex="\varepsilon" /> = <Num r={REF_EPS} v={p.epsilon} />
          </span>
        )}
        {p.terminal.length > 0 && <span className="chip">terminal: {p.terminal.join(', ')}</span>}
      </div>
      {p.transitions.length > 0 && (
        <table>
          <caption>Transitions</caption>
          <thead>
            <tr>
              <th><Tex tex="s" /></th>
              <th><Tex tex="a" /></th>
              <th><Tex tex="s'" /></th>
              <th><Tex tex="P(s'\mid s,a)" /></th>
              <th><Tex tex="R" /></th>
            </tr>
          </thead>
          <tbody>
            {p.transitions.map((t) => (
              <tr key={edgeKey(t.s, t.a, t.s2)} className={on.has(edgeKey(t.s, t.a, t.s2)) ? 'on' : ''}>
                <td><Tex tex={idTex(t.s)} /></td>
                <td><Tex tex={idTex(t.a)} /></td>
                <td><Tex tex={idTex(t.s2)} /></td>
                <td><Num r={refP(t.s, t.a, t.s2)} v={t.p} /></td>
                <td><Num r={refR(t.s, t.a, t.s2)} v={t.r} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {p.policy.length > 0 && (
        <table>
          <caption>Policy <Tex tex="\pi(a\mid s)" /></caption>
          <tbody>
            {p.policy.map((x) => (
              <tr key={`${x.s}|${x.a}`}>
                <td><Tex tex={`\\pi(${idTex(x.a)}\\mid ${idTex(x.s)})`} /></td>
                <td><Num r={refPi(x.s, x.a)} v={x.p} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {p.Q.length > 0 && (
        <table>
          <caption>Given Q-values</caption>
          <tbody>
            {p.Q.map((x) => (
              <tr key={`${x.s}|${x.a}`}>
                <td><Tex tex={`Q(${idTex(x.s)}, ${idTex(x.a)})`} /></td>
                <td><Num r={refQ(x.s, x.a)} v={x.v} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {(p.states.some((s) => s.name) || p.actions.some((a) => a.name)) && (
        <dl className="legend">
          {[...p.states, ...p.actions].filter((x) => x.name).map((x) => (
            <div key={x.id}>
              <dt><Tex tex={idTex(x.id)} /></dt>
              <dd>{x.name}</dd>
            </div>
          ))}
        </dl>
      )}
    </aside>
  );
}

export type Shown = { cell: Cell; value: number | string; fresh: boolean };

/** Running tables of every value computed so far. */
export function ResultsPanel({ shown }: { shown: Shown[] }) {
  const tables = new Map<string, { rows: string[]; cols: string[]; vals: Map<string, Shown> }>();
  for (const s of shown) {
    const t = tables.get(s.cell.table) ?? { rows: [] as string[], cols: [] as string[], vals: new Map<string, Shown>() };
    tables.set(s.cell.table, t);
    if (!t.rows.includes(s.cell.row)) t.rows.push(s.cell.row);
    if (!t.cols.includes(s.cell.col)) t.cols.push(s.cell.col);
    t.vals.set(`${s.cell.row}\u0000${s.cell.col}`, s);
  }
  return (
    <aside className="panel results">
      <h2>Worked values</h2>
      {!tables.size && <p className="muted">Values appear here as each step is solved.</p>}
      {[...tables].map(([name, t]) => (
        <table key={name}>
          <caption><Tex tex={name} /></caption>
          <thead>
            <tr>
              <th />
              {t.cols.map((c) => <th key={c}><Tex tex={c} /></th>)}
            </tr>
          </thead>
          <tbody>
            {t.rows.map((r) => (
              <tr key={r}>
                <th><Tex tex={r} /></th>
                {t.cols.map((c) => {
                  const s = t.vals.get(`${r}\u0000${c}`);
                  return (
                    <td key={c} className={s?.fresh ? 'fresh-cell' : ''}>
                      {s && (
                        <span className="val" data-ref={domRef(cellRef(s.cell))}>
                          {typeof s.value === 'number' ? fmt(s.value) : <Tex tex={s.value} />}
                        </span>
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      ))}
    </aside>
  );
}
