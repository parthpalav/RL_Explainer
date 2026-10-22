import { useMemo, useState } from 'react';
import { ProblemSchema, type Problem } from './schema';
import { solveAll } from './solvers';
import { EXAMPLES } from './fixtures/ia1';
import { ReviewForm } from './ui/ReviewForm';
import { ExtractBar } from './ui/ExtractBar';
import { crossCheck } from './extract/keywords';
import { StepPlayer, flatten } from './ui/StepPlayer';
import { DataPanel, ResultsPanel, type Shown } from './ui/Panels';
import { Icon } from './ui/Icon';

export default function App() {
  const [question, setQuestion] = useState(EXAMPLES[0].text);
  const [draft, setDraft] = useState<Problem>(EXAMPLES[0].problem);
  const [solved, setSolved] = useState<Problem | null>(null);
  const [index, setIndex] = useState(0);
  const [revealed, setRevealed] = useState(false);

  const parsed = useMemo(() => ProblemSchema.safeParse(draft), [draft]);
  const issues = parsed.success
    ? []
    : parsed.error.issues.map((i) => (i.path.length ? `${i.path.join(' › ')}: ${i.message}` : i.message));
  const warnings = crossCheck(draft.parts);
  const traces = useMemo(() => (solved ? solveAll(solved) : []), [solved]);
  const flat = useMemo(() => flatten(traces), [traces]);

  const shown: Shown[] = flat.flatMap((f, i) =>
    i < index || (i === index && revealed) ? f.step.results.map((r) => ({ ...r, fresh: i === index })) : [],
  );

  const loadExample = (i: number) => {
    setQuestion(EXAMPLES[i].text);
    setDraft(EXAMPLES[i].problem);
    setSolved(null);
  };
  const solve = () => {
    if (!parsed.success) return;
    setIndex(0);
    setSolved(parsed.data);
  };

  const stage = solved ? 2 : 1;
  const STAGES = ['Paste question', 'Check data', 'Solve'];

  return (
    <div className="app">
      <header className="top">
        <div className="brand">
          <span className="logo" aria-hidden="true">
            <svg viewBox="0 0 32 32" width="28" height="28">
              <defs>
                <linearGradient id="lg" x1="0" y1="0" x2="1" y2="1">
                  <stop offset="0" stopColor="#8b95ff" />
                  <stop offset="1" stopColor="#5fd4c4" />
                </linearGradient>
              </defs>
              <rect width="32" height="32" rx="9" fill="url(#lg)" />
              <path d="M9 22V10h5.5a3.5 3.5 0 0 1 0 7H9m5.5 0L20 22" fill="none" stroke="#0a0c11" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
              <circle cx="23" cy="10.5" r="2" fill="#0a0c11" />
            </svg>
          </span>
          <div>
            <h1>RL Explainer</h1>
            <p className="tagline">Every formula, every number, one step at a time</p>
          </div>
        </div>

        <ol className="stepper" aria-label="Progress">
          {STAGES.map((s, i) => (
            <li key={s} className={i < stage ? 'done' : i === stage ? 'on' : ''} aria-current={i === stage ? 'step' : undefined}>
              <span className="dot">{i < stage ? <Icon name="check" size={13} /> : i + 1}</span>
              <span className="lbl">{s}</span>
            </li>
          ))}
        </ol>

        <div className="top-actions">
          {solved ? (
            <button className="ghost" onClick={() => setSolved(null)}>
              <Icon name="edit" size={16} /> Edit data
            </button>
          ) : (
            <button className="primary cta" onClick={solve} disabled={!parsed.success}
              title={parsed.success ? 'Solve every part step by step' : 'Fix the highlighted data first'}>
              Solve step by step <Icon name="next" size={16} />
            </button>
          )}
        </div>
      </header>

      {!solved ? (
        <main className="edit-layout">
          <section className="panel question">
            <div className="panel-head">
              <h2><span className="num">1</span> Question</h2>
              <select className="examples" aria-label="Load example" value="" onChange={(e) => e.target.value && loadExample(+e.target.value)}>
                <option value="">Try an example…</option>
                {EXAMPLES.map((x, i) => <option key={x.name} value={i}>{x.name}</option>)}
              </select>
            </div>
            <textarea value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="Paste the full question here, including every sub-part (a, b, c…)" spellCheck={false} />
            <ExtractBar question={question} onProblem={setDraft} onParts={(parts) => setDraft({ ...draft, parts })} />
            <p className="note">
              <Icon name="info" size={15} />
              <span>The model only copies numbers out of the question. All maths is done by the app, so check the data on the right before solving.</span>
            </p>
          </section>
          <section className="panel review">
            <div className="panel-head">
              <h2><span className="num">2</span> Extracted data</h2>
              <span className={`status ${parsed.success ? 'ok' : 'bad'}`}>
                {parsed.success ? <><Icon name="check" size={14} /> Ready to solve</> : <><Icon name="x" size={14} /> {issues.length} to fix</>}
              </span>
            </div>
            <ReviewForm draft={draft} setDraft={setDraft} issues={issues} warnings={warnings} />
          </section>
        </main>
      ) : (
        <main className="solve-layout">
          <DataPanel p={solved} active={flat[index]?.step.edges ?? []} focus={flat[index]?.step.focus} />
          <StepPlayer traces={traces} flat={flat} index={index} setIndex={setIndex} onRevealed={setRevealed} />
          <ResultsPanel shown={shown} />
        </main>
      )}
    </div>
  );
}
