import type { ReactNode } from 'react';
import { TASKS, type Problem } from '../schema';
import { HEADINGS } from '../solvers';

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
    <fieldset>
      <legend>{title}</legend>
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
                <button className="icon" onClick={() => onChange(rows.filter((_, j) => j !== i))} aria-label="Remove row">✕</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <button className="add" onClick={() => onChange([...rows, { ...blank }])}>+ Add row</button>
    </fieldset>
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

  return (
    <div className="form">
      {issues.length > 0 && (
        <div className="issues" role="alert">
          <b>Fix these before solving:</b>
          <ul>{issues.map((x) => <li key={x}>{x}</li>)}</ul>
        </div>
      )}
      {warnings.length > 0 && (
        <div className="warnings">
          <b>Double-check:</b>
          <ul>{warnings.map((x) => <li key={x}>{x}</li>)}</ul>
        </div>
      )}
      <fieldset>
        <legend>Basics</legend>
        <label>Title <input value={d.title} onChange={(e) => up({ title: e.target.value })} /></label>
        <label>γ (discount)
          <input type="number" step="any" value={Number.isFinite(d.gamma) ? d.gamma : ''}
            onChange={(e) => up({ gamma: e.target.value === '' ? NaN : e.target.valueAsNumber })} />
        </label>
        {(['epsilon', 'alpha', 'c', 't'] as const).map((k) => (
          <label key={k}>{{ epsilon: 'ε', alpha: 'α (step size)', c: 'c (UCB)', t: 't (UCB time)' }[k]}
            <input type="number" step="any" value={d[k] ?? ''} placeholder="—"
              onChange={(e) => up({ [k]: e.target.value === '' ? undefined : e.target.valueAsNumber })} />
          </label>
        ))}
      </fieldset>
      <Rows title="States" rows={states} {...common} blank={{ id: '', name: '', terminal: false }}
        cols={[{ key: 'id', label: 'Id', kind: 'text' }, { key: 'name', label: 'Name', kind: 'text' }, { key: 'terminal', label: 'Terminal', kind: 'bool' }]}
        onChange={(rows) => up({ states: rows.map(({ id, name }) => ({ id, name })), terminal: rows.filter((r) => r.terminal).map((r) => r.id) })} />
      <Rows title="Actions" rows={d.actions} {...common} blank={{ id: '', name: '' }}
        cols={[{ key: 'id', label: 'Id', kind: 'text' }, { key: 'name', label: 'Name', kind: 'text' }]}
        onChange={(actions) => up({ actions })} />
      <Rows title="Transitions P(s′|s,a), R" rows={d.transitions} {...common} blank={{ s: '', a: '', s2: '', p: NaN, r: NaN }}
        cols={[{ key: 's', label: 's', kind: 'state' }, { key: 'a', label: 'a', kind: 'action' }, { key: 's2', label: "s'", kind: 'state' },
          { key: 'p', label: 'P', kind: 'number' }, { key: 'r', label: 'R', kind: 'number' }]}
        onChange={(transitions) => up({ transitions })} />
      <Rows title="Policy π(a|s)" rows={d.policy} {...common} blank={{ s: '', a: '', p: 1 }}
        cols={[{ key: 's', label: 's', kind: 'state' }, { key: 'a', label: 'a', kind: 'action' }, { key: 'p', label: 'π(a|s)', kind: 'number' }]}
        onChange={(policy) => up({ policy })} />
      <Rows title="Initial values V₀(s) (missing = 0)" rows={d.V0} {...common} blank={{ s: '', v: 0 }}
        cols={[{ key: 's', label: 's', kind: 'state' }, { key: 'v', label: 'V₀', kind: 'number' }]}
        onChange={(V0) => up({ V0 })} />
      <Rows title="Given Q-values" rows={d.Q} {...common} blank={{ s: '', a: '', v: NaN }}
        cols={[{ key: 's', label: 's', kind: 'state' }, { key: 'a', label: 'a', kind: 'action' }, { key: 'v', label: 'Q(s,a)', kind: 'number' }]}
        onChange={(Q) => up({ Q })} />
      {d.episodes.map((ep, e) => {
        const setEp = (patch: Partial<typeof ep>) => up({ episodes: d.episodes.map((x, j) => (j === e ? { ...x, ...patch } : x)) });
        return (
          <div key={e}>
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
              <button className="icon" onClick={() => up({ episodes: d.episodes.filter((_, j) => j !== e) })}>Remove episode</button>
            </div>
          </div>
        );
      })}
      <button className="add" onClick={() => up({ episodes: [...d.episodes, { steps: [{ s: '', a: '', r: NaN }], end: '', endAction: '' }] })}>
        + Add episode
      </button>
      <Rows title="Bandit pulls (arm, reward)" rows={d.pulls} {...common} blank={{ a: '', r: NaN }}
        cols={[{ key: 'a', label: 'arm', kind: 'action' }, { key: 'r', label: 'reward', kind: 'number' }]}
        onChange={(pulls) => up({ pulls })} />
      <Rows title="Arm snapshot (optional fields)" rows={d.arms} {...common} blank={{ a: '' }}
        cols={[{ key: 'a', label: 'arm', kind: 'action' }, { key: 'q', label: 'Q(a)', kind: 'optnum' }, { key: 'n', label: 'N(a) / visitors', kind: 'optnum' },
          { key: 'wins', label: 'conversions', kind: 'optnum' }]}
        onChange={(arms) => up({ arms })} />

      <fieldset>
        <legend>Question parts: what each part asks</legend>
        {d.parts.map((pt, i) => {
          const setPart = (patch: Partial<typeof pt>) => up({ parts: d.parts.map((x, j) => (j === i ? { ...x, ...patch } : x)) });
          return (
            <div className="part-row" key={i}>
              <input className="lbl" value={pt.label} onChange={(e) => setPart({ label: e.target.value })} aria-label="Part label" />
              <select value={pt.task} onChange={(e) => setPart({ task: e.target.value as (typeof TASKS)[number] })}>
                {TASKS.map((t) => <option key={t} value={t}>{HEADINGS[t]}</option>)}
              </select>
              <label>up to k =
                <input type="number" min={1} value={pt.iterations ?? ''}
                  onChange={(e) => setPart({ iterations: e.target.value === '' ? undefined : e.target.valueAsNumber })} />
              </label>
              <label>states <input value={pt.states.join(', ')} placeholder="all" onChange={(e) => setPart({ states: list(e.target.value) })} /></label>
              <label>rewards <input value={pt.rewards.join(', ')} placeholder="for G_t"
                onChange={(e) => setPart({ rewards: list(e.target.value).map(Number) })} /></label>
              <label>episodes <input value={pt.episodes.join(', ')} placeholder="all"
                onChange={(e) => setPart({ episodes: list(e.target.value).map(Number) })} /></label>
              <input className="txt" value={pt.text} placeholder="Sub-question text" onChange={(e) => setPart({ text: e.target.value })} />
              <button className="icon" onClick={() => up({ parts: d.parts.filter((_, j) => j !== i) })} aria-label="Remove part">✕</button>
            </div>
          );
        })}
        <button className="add" onClick={() => up({ parts: [...d.parts, { label: '', task: 'policy_evaluation', states: [], rewards: [], episodes: [], text: '' }] })}>
          + Add part
        </button>
      </fieldset>
    </div>
  );
}
