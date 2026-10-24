import type { ReactNode } from 'react';
import { TASKS, type Problem } from '../schema';
import { HEADINGS } from '../solvers';
import { Icon } from './Icon';

type Col<T> = {
  key: keyof T & string;
  label: ReactNode;
  kind: 'state' | 'action' | 'number' | 'optnum' | 'text' | 'bool';
  /** prob: draw a bar for a 0..1 value; signed: tint by sign (rewards, values). */
  tone?: 'prob' | 'signed';
  /** Draw an arrow before this column (s, a -> s'). */
  arrow?: boolean;
  /** Part of the group key: repeats inside a group are dimmed. */
  group?: boolean;
};

interface RowsProps<T> {
  title: ReactNode;
  rows: T[];
  cols: Col<T>[];
  blank: T;
  /** Rows with the same key sit together in one card. */
  groupKey?: (r: T) => string;
  /** Per-group sanity check shown on the first row of each group. */
  check?: (group: T[]) => { ok: boolean; text: string };
  states: string[];
  actions: string[];
  onChange: (rows: T[]) => void;
}

const sign = (v: unknown) => (typeof v !== 'number' || !Number.isFinite(v) ? '' : v > 0 ? 'pos' : v < 0 ? 'neg' : 'zero');

/** Generic editable table: one input per cell, add / remove rows. */
function Rows<T extends Record<string, unknown>>({ title, rows, cols, blank, groupKey, check, states, actions, onChange }: RowsProps<T>) {
  const set = (i: number, k: keyof T, v: unknown) => onChange(rows.map((r, j) => (j === i ? { ...r, [k]: v } : r)));
  const span = cols.length + cols.filter((c) => c.arrow).length + (check ? 4 : 3);
  return (
    <details className="sec" open>
      <summary><span className="sec-title">{title}</span><span className="count">{rows.length}</span><Icon name="chevron" size={16} /></summary>
      <div className="table-wrap">
      <table className="edit">
        <thead>
          <tr>
            <th className="ix">#</th>
            {cols.map((c) => [
              c.arrow && <th key={`${c.key}-arr`} className="arrow" />,
              <th key={c.key} className={`k-${c.kind}${c.tone ? ` t-${c.tone}` : ''}`}>{c.label}</th>,
            ])}
            {check && <th className="chk">Check</th>}
            <th className="fill" />
            <th className="rm" />
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => {
            const g = groupKey?.(r);
            const first = g !== undefined && i > 0 && groupKey!(rows[i - 1]) !== g;
            const repeat = g !== undefined && i > 0 && groupKey!(rows[i - 1]) === g;
            const res = check && g !== undefined && !repeat ? check(rows.filter((x) => groupKey!(x) === g)) : null;
            return [
              first && <tr key={`gap${i}`} className="gap"><td colSpan={span} /></tr>,
              <tr key={i} className={repeat ? 'repeat' : undefined}>
                <td className="ix">{i + 1}</td>
                {cols.map((c) => {
                  const v = r[c.key];
                  return [
                    c.arrow && <td key={`${c.key}-arr`} className="arrow" aria-hidden="true">→</td>,
                    <td key={c.key} className={`k-${c.kind}${c.tone ? ` t-${c.tone}` : ''}${c.group ? ' g' : ''}`} data-sign={c.tone === 'signed' ? sign(v) : undefined}>
                      {c.kind === 'state' || c.kind === 'action' ? (
                        <select value={String(v ?? '')} onChange={(e) => set(i, c.key, e.target.value)} className={v ? undefined : 'empty'}>
                          <option value="">—</option>
                          {(c.kind === 'state' ? states : actions).map((x) => <option key={x}>{x}</option>)}
                        </select>
                      ) : c.kind === 'number' || c.kind === 'optnum' ? (
                        <div className="num">
                          <input
                            type="number"
                            step="any"
                            value={Number.isFinite(v) ? String(v) : ''}
                            placeholder={c.kind === 'optnum' ? '—' : '?'}
                            // empty: NaN fails validation for required numbers, undefined means "not given"
                            onChange={(e) => set(i, c.key, e.target.value === '' ? (c.kind === 'number' ? NaN : undefined) : e.target.valueAsNumber)}
                          />
                          {c.tone === 'prob' && (
                            <span className="pbar" aria-hidden="true"><i style={{ width: `${Math.max(0, Math.min(1, Number(v) || 0)) * 100}%` }} /></span>
                          )}
                        </div>
                      ) : c.kind === 'bool' ? (
                        <input type="checkbox" className="switch" checked={!!v} onChange={(e) => set(i, c.key, e.target.checked)} />
                      ) : (
                        <input value={String(v ?? '')} onChange={(e) => set(i, c.key, e.target.value)} placeholder="—" />
                      )}
                    </td>,
                  ];
                })}
                {check && <td className="chk">{res && <span className={`badge ${res.ok ? 'ok' : 'bad'}`}>{res.text}</span>}</td>}
                <td className="fill" />
                <td className="rm">
                  <button className="icon" onClick={() => onChange(rows.filter((_, j) => j !== i))} aria-label="Remove row"><Icon name="x" size={14} /></button>
                </td>
              </tr>,
            ];
          })}
        </tbody>
      </table>
      </div>
      <button className="add" onClick={() => onChange([...rows, { ...blank }])}><Icon name="plus" size={14} /> Add row</button>
    </details>
  );
}

/** Plain-language names for the single numbers a question can give. */
const PARAMS = {
  gamma: { sym: 'γ', name: 'Discount factor', hint: 'How much a future reward is worth now (0 to 1).' },
  alpha: { sym: 'α', name: 'Learning rate', hint: 'Step size of each update.', start: 0.1 },
  epsilon: { sym: 'ε', name: 'Exploration rate', hint: 'Chance of a random action.', start: 0.1 },
  n: { sym: 'n', name: 'Steps', hint: 'n-step length, or Dyna-Q planning steps per real step.', start: 2 },
  c: { sym: 'c', name: 'UCB constant', hint: 'Weight of the exploration bonus.', start: 2 },
  t: { sym: 't', name: 'UCB time step', hint: 'Time step used in ln t (default: pulls so far).', start: 1 },
} as const;
const PARAM_KEYS = ['alpha', 'epsilon', 'n', 'c', 't'] as const;

const sumCheck = (g: { p: number }[]) => {
  const sum = g.reduce((x, t) => x + (Number.isFinite(t.p) ? t.p : 0), 0);
  const ok = Math.abs(sum - 1) < 1e-6;
  return { ok, text: `Σ = ${+sum.toFixed(6)}${ok ? ' ✓' : ' ≠ 1'}` };
};

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
  const shownParams = PARAM_KEYS.filter((k) => d[k] !== undefined);
  const hiddenParams = PARAM_KEYS.filter((k) => d[k] === undefined);
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
        <div className="sec-head"><span className="sec-title">Parameters</span><span className="count">{1 + shownParams.length}</span></div>
        <div className="params">
          <label className="title"><span>Title</span> <input value={d.title} onChange={(e) => up({ title: e.target.value })} /></label>
          <label className="param">
            <span><b>γ</b> Discount factor</span>
            <input type="number" step="any" value={Number.isFinite(d.gamma) ? d.gamma : ''}
              onChange={(e) => up({ gamma: e.target.value === '' ? NaN : e.target.valueAsNumber })} />
            <small>{PARAMS.gamma.hint}</small>
          </label>
          {shownParams.map((k) => (
            <label className="param" key={k}>
              <span><b>{PARAMS[k].sym}</b> {PARAMS[k].name}
                <button className="icon" onClick={(e) => { e.preventDefault(); up({ [k]: undefined }); }} aria-label={`Remove ${PARAMS[k].name}`}><Icon name="x" size={12} /></button>
              </span>
              <input type="number" step="any" value={d[k] ?? ''} onChange={(e) => up({ [k]: e.target.value === '' ? undefined : e.target.valueAsNumber })} />
              <small>{PARAMS[k].hint}</small>
            </label>
          ))}
        </div>
        {hiddenParams.length > 0 && (
          <div className="more">
            <span className="muted">Question gives another number?</span>
            {hiddenParams.map((k) => (
              <button key={k} className="pill" onClick={() => up({ [k]: PARAMS[k].start })}><Icon name="plus" size={13} /> {PARAMS[k].sym} {PARAMS[k].name}</button>
            ))}
          </div>
        )}
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
        <Rows title="Transitions: probability P(s′|s,a) and reward R" rows={d.transitions} {...common} blank={{ s: '', a: '', s2: '', p: NaN, r: NaN }}
          groupKey={(t) => `${t.s}|${t.a}`}
          check={(g) => { const sum = g.reduce((x, t) => x + (Number.isFinite(t.p) ? t.p : 0), 0); const ok = Math.abs(sum - 1) < 1e-6; return { ok, text: `ΣP = ${+sum.toFixed(6)}${ok ? ' ✓' : ' ≠ 1'}` }; }}
          cols={[{ key: 's', label: 'From s', kind: 'state', group: true }, { key: 'a', label: 'Action a', kind: 'action', group: true },
            { key: 's2', label: "To s′", kind: 'state', arrow: true }, { key: 'p', label: 'Probability', kind: 'number', tone: 'prob' },
            { key: 'r', label: 'Reward', kind: 'number', tone: 'signed' }]}
          onChange={(transitions) => up({ transitions })} />
      )}
      {has('policy') && (
        <Rows title="Policy π(a|s)" rows={d.policy} {...common} blank={{ s: '', a: '', p: 1 }}
          groupKey={(t) => t.s} check={sumCheck}
          cols={[{ key: 's', label: 'State', kind: 'state', group: true }, { key: 'a', label: 'Action', kind: 'action' }, { key: 'p', label: 'π(a|s)', kind: 'number', tone: 'prob' }]}
          onChange={(policy) => up({ policy })} />
      )}
      {has('V0') && (
        <Rows title="Initial values V₀(s) (missing = 0)" rows={d.V0} {...common} blank={{ s: '', v: 0 }}
          cols={[{ key: 's', label: 'State', kind: 'state' }, { key: 'v', label: 'V₀(s)', kind: 'number', tone: 'signed' }]}
          onChange={(V0) => up({ V0 })} />
      )}
      {has('Q') && (
        <Rows title="Given Q-values" rows={d.Q} {...common} blank={{ s: '', a: '', v: NaN }}
          cols={[{ key: 's', label: 'State', kind: 'state' }, { key: 'a', label: 'Action', kind: 'action' }, { key: 'v', label: 'Q(s,a)', kind: 'number', tone: 'signed' }]}
          onChange={(Q) => up({ Q })} />
      )}
      {d.episodes.map((ep, e) => {
        const setEp = (patch: Partial<typeof ep>) => up({ episodes: d.episodes.map((x, j) => (j === e ? { ...x, ...patch } : x)) });
        return (
          <div key={e} className="episode-edit">
            <Rows title={<>Episode {e + 1}: steps (S<sub>t</sub>, A<sub>t</sub>, R<sub>t+1</sub>)</>} rows={ep.steps} {...common} blank={{ s: '', a: '', r: NaN }}
              cols={[{ key: 's', label: 'S', kind: 'state' }, { key: 'a', label: 'A (optional)', kind: 'action' }, { key: 'r', label: 'Reward', kind: 'number', tone: 'signed' }]}
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
          cols={[{ key: 'a', label: 'Arm', kind: 'action' }, { key: 'r', label: 'Reward', kind: 'number', tone: 'signed' }]}
          onChange={(pulls) => up({ pulls })} />
      )}
      {has('behavior') && (
        <Rows title="Behaviour policy b(a|s) (off-policy)" rows={d.behavior} {...common} blank={{ s: '', a: '', p: NaN }}
          groupKey={(t) => t.s} check={sumCheck}
          cols={[{ key: 's', label: 'State', kind: 'state', group: true }, { key: 'a', label: 'Action', kind: 'action' }, { key: 'p', label: 'b(a|s)', kind: 'number', tone: 'prob' }]}
          onChange={(behavior) => up({ behavior })} />
      )}
      {has('planning') && (
        <Rows title="Dyna-Q planning order (s, a)" rows={d.planning} {...common} blank={{ s: '', a: '' }}
          cols={[{ key: 's', label: 's', kind: 'state' }, { key: 'a', label: 'a', kind: 'action' }]}
          onChange={(planning) => up({ planning })} />
      )}
      {has('replay') && (
        <Rows title="DQN replay minibatch" rows={d.replay} {...common} blank={{ s: '', a: '', r: NaN, s2: '', done: false }}
          cols={[{ key: 's', label: 's', kind: 'state' }, { key: 'a', label: 'a', kind: 'action' }, { key: 'r', label: 'Reward', kind: 'number', tone: 'signed' },
            { key: 's2', label: "s′", kind: 'state', arrow: true }, { key: 'done', label: 'Terminal', kind: 'bool' }]}
          onChange={(replay) => up({ replay })} />
      )}
      {has('Qtarget') && (
        <Rows title="DQN target network Q̂(s,a)" rows={d.Qtarget} {...common} blank={{ s: '', a: '', v: NaN }}
          cols={[{ key: 's', label: 's', kind: 'state' }, { key: 'a', label: 'a', kind: 'action' }, { key: 'v', label: 'Q̂(s,a)', kind: 'number', tone: 'signed' }]}
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
