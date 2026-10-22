import type { ReactNode } from 'react';
import type { Problem } from '../schema';
import type { Cell } from '../solvers/types';
import {
  REF_ALPHA, REF_C, REF_EPS, REF_GAMMA, REF_N, REF_T, cellRef, refB, refQT, refReplay, domRef, edgeKey, focusEp, focusPull, idTex, refArm, refEpR, refP, refPi, refPull, refQ, refR,
} from '../mdp';
import { fmt } from '../math/expr';
import { Tex } from './Tex';
import { MdpGraph } from './MdpGraph';

const Num = ({ r, v }: { r: string; v: number }) => (
  <span className="val" data-ref={domRef(r)}>
    {fmt(v)}
  </span>
);

/** Two-column table of labelled given numbers; renders nothing when empty. */
function KV({ caption, rows }: { caption: ReactNode; rows: [string, string, number][] }) {
  if (!rows.length) return null;
  return (
    <table>
      <caption>{caption}</caption>
      <tbody>
        {rows.map(([tex, r, v]) => (
          <tr key={r}>
            <td><Tex tex={tex} /></td>
            <td><Num r={r} v={v} /></td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** Everything given in the question. Each number carries data-ref so formulas can fly values out of it. */
export function DataPanel({ p, active, focus }: { p: Problem; active: string[]; focus?: string }) {
  const on = new Set(active);
  const chip = (tex: string, r: string, v: number | undefined) =>
    v !== undefined && (
      <span className="chip">
        <Tex tex={tex} /> = <Num r={r} v={v} />
      </span>
    );
  return (
    <aside className="panel data">
      <h2>Given</h2>
      {p.transitions.length > 0 && <MdpGraph p={p} active={active} />}
      <div className="chips">
        {(p.transitions.length > 0 || p.episodes.length > 0) && chip('\\gamma', REF_GAMMA, p.gamma)}
        {chip('\\varepsilon', REF_EPS, p.epsilon)}
        {chip('\\alpha', REF_ALPHA, p.alpha)}
        {chip('c', REF_C, p.c)}
        {chip('t', REF_T, p.t)}
        {chip('n', REF_N, p.n)}
        {p.terminal.length > 0 && <span className="chip">terminal: {p.terminal.join(', ')}</span>}
      </div>
      {p.episodes.length > 0 && (
        <div className="timeline">
          <h3>Episodes <span className="muted">(state, action, reward received)</span></h3>
          {p.episodes.map((ep, e) => (
            <div key={e} className="episode">
              <span className="ep-label">Ep {e + 1}</span>
              {ep.steps.map((x, t) => (
                <span key={t} className={`ep-step ${focus === focusEp(e, t) ? 'on' : ''}`}>
                  <span className="st"><Tex tex={idTex(x.s)} /></span>
                  <span className="tr">
                    {x.a && <Tex tex={idTex(x.a)} />} r=<Num r={refEpR(e, t)} v={x.r} /> →
                  </span>
                </span>
              ))}
              <span className="st end">{ep.end ? <Tex tex={idTex(ep.end)} /> : 'end'}</span>
              {ep.endAction && <span className="tr"><Tex tex={idTex(ep.endAction)} /></span>}
            </div>
          ))}
        </div>
      )}
      {p.pulls.length > 0 && (
        <div className="timeline">
          <h3>Pulls</h3>
          <div className="pulls">
            {p.pulls.map((x, i) => (
              <span key={i} className={`pull ${focus === focusPull(i) ? 'on' : ''}`}>
                <span className="muted">#{i + 1}</span> <Tex tex={idTex(x.a)} /> r=<Num r={refPull(i)} v={x.r} />
              </span>
            ))}
          </div>
        </div>
      )}
      {p.arms.length > 0 && (
        <table>
          <caption>Arms</caption>
          <thead>
            <tr>
              <th>arm</th>
              {p.arms.some((x) => x.q !== undefined) && <th><Tex tex="Q(a)" /></th>}
              {p.arms.some((x) => x.n !== undefined) && <th><Tex tex="N(a)" /></th>}
              {p.arms.some((x) => x.wins !== undefined) && <th>conversions</th>}
            </tr>
          </thead>
          <tbody>
            {p.arms.map((x) => (
              <tr key={x.a}>
                <td><Tex tex={idTex(x.a)} /></td>
                {p.arms.some((y) => y.q !== undefined) && <td>{x.q !== undefined && <Num r={refArm(x.a, 'q')} v={x.q} />}</td>}
                {p.arms.some((y) => y.n !== undefined) && <td>{x.n !== undefined && <Num r={refArm(x.a, 'n')} v={x.n} />}</td>}
                {p.arms.some((y) => y.wins !== undefined) && <td>{x.wins !== undefined && <Num r={refArm(x.a, 'wins')} v={x.wins} />}</td>}
              </tr>
            ))}
          </tbody>
        </table>
      )}
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
      <KV caption={<>Policy <Tex tex="\pi(a\mid s)" /></>}
        rows={p.policy.map((x) => [`\\pi(${idTex(x.a)}\\mid ${idTex(x.s)})`, refPi(x.s, x.a), x.p])} />
      <KV caption={<>Behaviour policy <Tex tex="b(a\mid s)" /></>}
        rows={p.behavior.map((x) => [`b(${idTex(x.a)}\\mid ${idTex(x.s)})`, refB(x.s, x.a), x.p])} />
      <KV caption={p.Qtarget.length ? 'Online network Q(s,a)' : 'Given Q-values'}
        rows={p.Q.map((x) => [`Q(${idTex(x.s)}, ${idTex(x.a)})`, refQ(x.s, x.a), x.v])} />
      <KV caption="Target network Q̂(s,a)"
        rows={p.Qtarget.map((x) => [`\\hat{Q}(${idTex(x.s)}, ${idTex(x.a)})`, refQT(x.s, x.a), x.v])} />
      {p.replay.length > 0 && (
        <table>
          <caption>Replay minibatch</caption>
          <thead>
            <tr><th>i</th><th><Tex tex="s" /></th><th><Tex tex="a" /></th><th><Tex tex="r" /></th><th><Tex tex="s'" /></th></tr>
          </thead>
          <tbody>
            {p.replay.map((x, i) => (
              <tr key={i}>
                <td>{i + 1}</td>
                <td><Tex tex={idTex(x.s)} /></td>
                <td><Tex tex={idTex(x.a)} /></td>
                <td><Num r={refReplay(i)} v={x.r} /></td>
                <td>{x.done ? 'terminal' : <Tex tex={idTex(x.s2)} />}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {p.planning.length > 0 && (
        <p className="muted">Planning order: {p.planning.map((x) => `(${x.s}, ${x.a})`).join(', ')}</p>
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
