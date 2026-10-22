import { useMemo, useState } from 'react';
import { ProblemSchema, type Problem } from './schema';
import { solveAll } from './solvers';
import { EXAMPLES } from './fixtures/ia1';
import { ReviewForm } from './ui/ReviewForm';
import { ExtractBar } from './ui/ExtractBar';
import { crossCheck } from './extract/keywords';
import { StepPlayer, flatten } from './ui/StepPlayer';
import { DataPanel, ResultsPanel, type Shown } from './ui/Panels';

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

  return (
    <div className="app">
      <header className="top">
        <h1>RL Explainer</h1>
        <select aria-label="Load example" value="" onChange={(e) => e.target.value && loadExample(+e.target.value)}>
          <option value="">Load example…</option>
          {EXAMPLES.map((x, i) => <option key={x.name} value={i}>{x.name}</option>)}
        </select>
        <div className="grow" />
        {solved ? (
          <button onClick={() => setSolved(null)}>✎ Edit data</button>
        ) : (
          <button className="primary" onClick={solve} disabled={!parsed.success}>Solve step by step ▶</button>
        )}
      </header>

      {!solved ? (
        <main className="edit-layout">
          <section className="panel">
            <h2>Question</h2>
            <textarea value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="Paste the question here" />
            <ExtractBar question={question} onProblem={setDraft} onParts={(parts) => setDraft({ ...draft, parts })} />
            <p className="muted">The model only copies the data. Check it on the right before solving: all maths is done by the app, not the model.</p>
          </section>
          <section className="panel">
            <h2>Extracted data</h2>
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
