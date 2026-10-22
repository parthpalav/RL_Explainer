import type { ReactNode } from 'react';
import { TASKS, type Problem } from '../schema';
import { HEADINGS } from '../solvers';
import { Icon } from './Icon';

type Col<T> = { key: keyof T & string; label: string; kind: 'state' | 'action' | 'number' | 'optnum' | 'text' | 'bool' };

interface RowsProps<T> {
  title: ReactNode;
  rows: T[];
  cols: Col<T>[];
  blank: T;
  states: string[];
  actions: string[];
  onChange: (rows: T[]) => void;
}

/** Generic editable table: one input per cell, add / remove rows. */
function Rows<T extends Record<string, unknown>>({ title, rows, cols, blank, states, actions, onChange }: RowsProps<T>) {
  const set = (i: number, k: keyof T, v: unknown) => onChange(rows.map((r, j) => (j === i ? { ...r, [k]: v } : r)));
  return (
    <details className="sec" open>
      <summary><span className="sec-title">{title}</span><span className="count">{rows.length}</span><Icon name="chevron" size={16} /></summary>
      <div className="table-wrap">
      <table className="edit">
        <thead>
          <tr>
            {cols.map((c) => <th key={c.key}>{c.label}</th>)}
            <th />
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              {cols.map((c) => (
                <td key={c.key}>
                  {c.kind === 'state' || c.kind === 'action' ? (
                    <select value={String(r[c.key] ?? '')} onChange={(e) => set(i, c.key, e.target.value)}>
                      <option value="" />
                      {(c.kind === 'state' ? states : actions).map((x) => <option key={x}>{x}</option>)}
                    </select>
                  ) : c.kind === 'number' || c.kind === 'optnum' ? (
                    <input
                      type="number"
                      step="any"
                      value={Number.isFinite(r[c.key]) ? String(r[c.key]) : ''}
                      // empty: NaN fails validation for required numbers, undefined means "not given"
                      onChange={(e) => set(i, c.key, e.target.value === '' ? (c.kind === 'number' ? NaN : undefined) : e.target.valueAsNumber)}
                    />
                  ) : c.kind === 'bool' ? (
                    <input type="checkbox" checked={!!r[c.key]} onChange={(e) => set(i, c.key, e.target.checked)} />
                  ) : (
                    <input value={String(r[c.key] ?? '')} onChange={(e) => set(i, c.key, e.target.value)} />
                  )}
                </td>
              ))}
              <td>
                <button className="icon" onClick={() => onChange(rows.filter((_, j) => j !== i))} aria-label="Remove row"><Icon name="x" size={14} /></button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      </div>
      <button className="add" onClick={() => onChange([...rows, { ...blank }])}><Icon name="plus" size={14} /> Add row</button>
    </details>
  );
}

const list = (s: string) => s.split(',').map((x) => x.trim()).filter(Boolean);

interface Props {
  draft: Problem;
  setDraft: (p: Problem) => void;
  issues: string[];
  warnings: string[];
}

/** Every extracted value is editable here; nothing is solved until the data validates. */
export function ReviewForm({ draft: d, setDraft, issues, warnings }: Props) {
  const S = d.states.map((s) => s.id);
  const A = d.actions.map((a) => a.id);
  const up = (patch: Partial<Problem>) => setDraft({ ...d, ...patch });
  const common = { states: S, actions: A };
  const states = d.states.map((s) => ({ ...s, terminal: d.terminal.includes(s.id) }));

  // Optional data stays hidden until it has rows, so the form only shows what this question uses.
  const has = (k: keyof typeof EXTRA) => (d[k] as unknown[]).length > 0;
  const missing = (Object.keys(EXTRA) as (keyof typeof EXTRA)[]).filter((k) => !has(k));

  return (
    <div className="form">
      {issues.length > 0 && (
        <div className="banner bad" role="alert">
          <b>Fix these before solving</b>
          <ul>{issues.map((x) => <li key={x}>{x}</li>)}</ul>
        </div>
      )}
      {warnings.length > 0 && (
        <div className="banner warn">
          <b>Double-check</b>
          <ul>{warnings.map((x) => <li key={x}>{x}</li>)}</ul>
        </div>
      )}

      <section className="sec static">
        <div className="sec-head"><span className="sec-title">What each part asks</span><span className="count">{d.parts.length}</span></div>
        <div className="parts-edit">
          {d.parts.map((pt, i) => {
            const setPart = (patch: Partial<typeof pt>) => up({ parts: d.parts.map((x, j) => (j === i ? { ...x, ...patch } : x)) });
            return (
              <div className="part-card" key={i}>
                <div className="part-top">
                  <input className="lbl" value={pt.label} onChange={(e) => setPart({ label: e.target.value })} aria-label="Part label" placeholder="a" />
                  <select className="task" value={pt.task} onChange={(e) => setPart({ task: e.target.value as (typeof TASKS)[number] })} aria-label="Method">
                    {TASKS.map((t) => <option key={t} value={t}>{HEADINGS[t]}</option>)}
                  </select>
                  <button className="icon" onClick={() => up({ parts: d.parts.filter((_, j) => j !== i) })} aria-label="Remove part"><Icon name="x" size={14} /></button>
                </div>
                <input className="txt" value={pt.text} placeholder="Sub-question text" onChange={(e) => setPart({ text: e.target.value })} aria-label="Sub-question text" />
                <div className="part-opts">
                  <label>up to k
                    <input type="number" min={1} value={pt.iterations ?? ''} placeholder="—"
                      onChange={(e) => setPart({ iterations: e.target.value === '' ? undefined : e.target.valueAsNumber })} />
                  </label>
                  <label>states <input value={pt.states.join(', ')} placeholder="all" onChange={(e) => setPart({ states: list(e.target.value) })} /></label>
                  <label>rewards <input value={pt.rewards.join(', ')} placeholder="for Gₜ"
                    onChange={(e) => setPart({ rewards: list(e.target.value).map(Number) })} /></label>
                  <label>episodes <input value={pt.episodes.join(', ')} placeholder="all"
                    onChange={(e) => setPart({ episodes: list(e.target.value).map(Number) })} /></label>
                </div>
              </div>
            );
          })}
        </div>
        <button className="add" onClick={() => up({ parts: [...d.parts, { label: '', task: 'policy_evaluation', states: [], rewards: [], episodes: [], text: '' }] })}>
          <Icon name="plus" size={14} /> Add part
        </button>
      </section>

      <section className="sec static">
        <div className="sec-head"><span className="sec-title">Parameters</span></div>
        <div className="params">
          <label><span>Title</span> <input value={d.title} onChange={(e) => up({ title: e.target.value })} /></label>
          <label><span>γ discount</span>
            <input type="number" step="any" value={Number.isFinite(d.gamma) ? d.gamma : ''}
              onChange={(e) => up({ gamma: e.target.value === '' ? NaN : e.target.valueAsNumber })} />
          </label>
          {(['epsilon', 'alpha', 'n', 'c', 't'] as const).map((k) => (
            <label key={k}><span>{{ epsilon: 'ε exploration', alpha: 'α step size', n: 'n steps / planning', c: 'c (UCB)', t: 't (UCB time)' }[k]}</span>
              <input type="number" step="any" value={d[k] ?? ''} placeholder="not given"
                onChange={(e) => up({ [k]: e.target.value === '' ? undefined : e.target.valueAsNumber })} />
            </label>
          ))}
        </div>
      </section>

      <div className="two">
        <Rows title="States" rows={states} {...common} blank={{ id: '', name: '', terminal: false }}
          cols={[{ key: 'id', label: 'Id', kind: 'text' }, { key: 'name', label: 'Name', kind: 'text' }, { key: 'terminal', label: 'Terminal', kind: 'bool' }]}
          onChange={(rows) => up({ states: rows.map(({ id, name }) => ({ id, name })), terminal: rows.filter((r) => r.terminal).map((r) => r.id) })} />
        <Rows title="Actions" rows={d.actions} {...common} blank={{ id: '', name: '' }}
          cols={[{ key: 'id', label: 'Id', kind: 'text' }, { key: 'name', label: 'Name', kind: 'text' }]}
          onChange={(actions) => up({ actions })} />
      </div>
      {has('transitions') && (
        <Rows title="Transitions P(s′|s,a), R" rows={d.transitions} {...common} blank={{ s: '', a: '', s2: '', p: NaN, r: NaN }}
          cols={[{ key: 's', label: 's', kind: 'state' }, { key: 'a', label: 'a', kind: 'action' }, { key: 's2', label: "s'", kind: 'state' },
            { key: 'p', label: 'P', kind: 'number' }, { key: 'r', label: 'R', kind: 'number' }]}
          onChange={(transitions) => up({ transitions })} />
      )}
      {has('policy') && (
        <Rows title="Policy π(a|s)" rows={d.policy} {...common} blank={{ s: '', a: '', p: 1 }}
          cols={[{ key: 's', label: 's', kind: 'state' }, { key: 'a', label: 'a', kind: 'action' }, { key: 'p', label: 'π(a|s)', kind: 'number' }]}
          onChange={(policy) => up({ policy })} />
      )}
      {has('V0') && (
        <Rows title="Initial values V₀(s) (missing = 0)" rows={d.V0} {...common} blank={{ s: '', v: 0 }}
          cols={[{ key: 's', label: 's', kind: 'state' }, { key: 'v', label: 'V₀', kind: 'number' }]}
          onChange={(V0) => up({ V0 })} />
      )}
      {has('Q') && (
        <Rows title="Given Q-values" rows={d.Q} {...common} blank={{ s: '', a: '', v: NaN }}
          cols={[{ key: 's', label: 's', kind: 'state' }, { key: 'a', label: 'a', kind: 'action' }, { key: 'v', label: 'Q(s,a)', kind: 'number' }]}
          onChange={(Q) => up({ Q })} />
      )}
      {d.episodes.map((ep, e) => {
        const setEp = (patch: Partial<typeof ep>) => up({ episodes: d.episodes.map((x, j) => (j === e ? { ...x, ...patch } : x)) });
        return (
          <div key={e} className="episode-edit">
            <Rows title={<>Episode {e + 1}: steps (S<sub>t</sub>, A<sub>t</sub>, R<sub>t+1</sub>)</>} rows={ep.steps} {...common} blank={{ s: '', a: '', r: NaN }}
              cols={[{ key: 's', label: 'S', kind: 'state' }, { key: 'a', label: 'A (optional)', kind: 'action' }, { key: 'r', label: 'R', kind: 'number' }]}
              onChange={(steps) => setEp({ steps })} />
            <div className="ep-end">
              <label>then ends in
                <select value={ep.end} onChange={(ev) => setEp({ end: ev.target.value })}>
                  <option value="">terminal</option>
                  {S.map((x) => <option key={x}>{x}</option>)}
                </select>
              </label>
              <label>next action A′ (SARSA)
                <select value={ep.endAction} onChange={(ev) => setEp({ endAction: ev.target.value })}>
                  <option value="" />
                  {A.map((x) => <option key={x}>{x}</option>)}
                </select>
              </label>
              <button className="ghost small" onClick={() => up({ episodes: d.episodes.filter((_, j) => j !== e) })}>Remove episode</button>
            </div>
          </div>
        );
      })}
      {has('episodes') && (
        <button className="add" onClick={() => up({ episodes: [...d.episodes, structuredClone(EXTRA.episodes.blank)] })}>
          <Icon name="plus" size={14} /> Add episode
        </button>
      )}
      {has('pulls') && (
        <Rows title="Bandit pulls (arm, reward)" rows={d.pulls} {...common} blank={{ a: '', r: NaN }}
          cols={[{ key: 'a', label: 'arm', kind: 'action' }, { key: 'r', label: 'reward', kind: 'number' }]}
          onChange={(pulls) => up({ pulls })} />
      )}
      {has('behavior') && (
        <Rows title="Behaviour policy b(a|s) (off-policy)" rows={d.behavior} {...common} blank={{ s: '', a: '', p: NaN }}
          cols={[{ key: 's', label: 's', kind: 'state' }, { key: 'a', label: 'a', kind: 'action' }, { key: 'p', label: 'b(a|s)', kind: 'number' }]}
          onChange={(behavior) => up({ behavior })} />
      )}
      {has('planning') && (
        <Rows title="Dyna-Q planning order (s, a)" rows={d.planning} {...common} blank={{ s: '', a: '' }}
          cols={[{ key: 's', label: 's', kind: 'state' }, { key: 'a', label: 'a', kind: 'action' }]}
          onChange={(planning) => up({ planning })} />
      )}
      {has('replay') && (
        <Rows title="DQN replay minibatch" rows={d.replay} {...common} blank={{ s: '', a: '', r: NaN, s2: '', done: false }}
          cols={[{ key: 's', label: 's', kind: 'state' }, { key: 'a', label: 'a', kind: 'action' }, { key: 'r', label: 'r', kind: 'number' },
            { key: 's2', label: "s'", kind: 'state' }, { key: 'done', label: 'terminal', kind: 'bool' }]}
          onChange={(replay) => up({ replay })} />
      )}
      {has('Qtarget') && (
        <Rows title="DQN target network Q̂(s,a)" rows={d.Qtarget} {...common} blank={{ s: '', a: '', v: NaN }}
          cols={[{ key: 's', label: 's', kind: 'state' }, { key: 'a', label: 'a', kind: 'action' }, { key: 'v', label: 'Q̂(s,a)', kind: 'number' }]}
          onChange={(Qtarget) => up({ Qtarget })} />
      )}
      {has('arms') && (
        <Rows title="Arm snapshot (optional fields)" rows={d.arms} {...common} blank={{ a: '' }}
          cols={[{ key: 'a', label: 'arm', kind: 'action' }, { key: 'q', label: 'Q(a)', kind: 'optnum' }, { key: 'n', label: 'N(a) / visitors', kind: 'optnum' },
            { key: 'wins', label: 'conversions', kind: 'optnum' }]}
          onChange={(arms) => up({ arms })} />
      )}

      {missing.length > 0 && (
        <div className="more">
          <span className="muted">Question gives more data?</span>
          {missing.map((k) => (
            <button key={k} className="pill" onClick={() => up({ [k]: [structuredClone(EXTRA[k].blank)] })}>
              <Icon name="plus" size={13} /> {EXTRA[k].label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

/** Sections hidden while empty; "blank" is the first row added when the user asks for one. */
const EXTRA = {
  transitions: { label: 'Transitions', blank: { s: '', a: '', s2: '', p: NaN, r: NaN } },
  policy: { label: 'Policy', blank: { s: '', a: '', p: 1 } },
  V0: { label: 'Initial values', blank: { s: '', v: 0 } },
  Q: { label: 'Q-values', blank: { s: '', a: '', v: NaN } },
  episodes: { label: 'Episodes', blank: { steps: [{ s: '', a: '', r: NaN }], end: '', endAction: '' } },
  pulls: { label: 'Bandit pulls', blank: { a: '', r: NaN } },
  arms: { label: 'Arm snapshot', blank: { a: '' } },
  behavior: { label: 'Behaviour policy', blank: { s: '', a: '', p: NaN } },
  planning: { label: 'Dyna-Q planning', blank: { s: '', a: '' } },
  replay: { label: 'Replay minibatch', blank: { s: '', a: '', r: NaN, s2: '', done: false } },
  Qtarget: { label: 'Target network', blank: { s: '', a: '', v: NaN } },
};
