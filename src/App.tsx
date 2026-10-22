import { useLayoutEffect, useMemo, useRef, useState } from 'react';
import gsap from 'gsap';
import { Flip } from 'gsap/Flip';
import { ProblemSchema, problem, type Problem } from './schema';
import { solveAll } from './solvers';
import { EXAMPLES } from './fixtures/ia1';
import { ReviewForm } from './ui/ReviewForm';
import { ExtractBar } from './ui/ExtractBar';
import { crossCheck } from './extract/keywords';
import { StepPlayer, flatten } from './ui/StepPlayer';
import { DataPanel, ResultsPanel, type Shown } from './ui/Panels';
import { Icon } from './ui/Icon';
import { Welcome } from './ui/Welcome';

gsap.registerPlugin(Flip);

type Stage = 'welcome' | 'paste' | 'review' | 'solve';
const STAGES = ['Paste question', 'Check data', 'Solve'];
const reduced = () => matchMedia('(prefers-reduced-motion: reduce)').matches;

export default function App() {
  const [stage, setStage] = useState<Stage>('welcome');
  const [question, setQuestion] = useState('');
  const [draft, setDraft] = useState<Problem>(EXAMPLES[0].problem);
  const [solved, setSolved] = useState<Problem | null>(null);
  const [index, setIndex] = useState(0);
  const [revealed, setRevealed] = useState(false);
  const [reading, setReading] = useState(false);
  const main = useRef<HTMLElement>(null);
  const flip = useRef<Flip.FlipState | null>(null);

  const parsed = useMemo(() => ProblemSchema.safeParse(draft), [draft]);
  const issues = parsed.success
    ? []
    : parsed.error.issues.map((i) => (i.path.length ? `${i.path.join(' › ')}: ${i.message}` : i.message));
  const warnings = crossCheck(draft.parts);
  const traces = useMemo(() => (solved ? solveAll(solved) : []), [solved]);
  const flat = useMemo(() => flatten(traces), [traces]);
  const known = EXAMPLES.find((x) => x.text === question)?.problem;

  const shown: Shown[] = flat.flatMap((f, i) =>
    i < index || (i === index && revealed) ? f.step.results.map((r) => ({ ...r, fresh: i === index })) : [],
  );

  /** Change stage; with morph, the question box glides into its place on the next screen. */
  const go = (next: Stage, morph = false) => {
    flip.current = morph && !reduced() ? Flip.getState('[data-flip-id]') : null;
    setReading(false);
    setStage(next);
  };

  useLayoutEffect(() => {
    const el = main.current;
    const state = flip.current;
    flip.current = null;
    if (!el || reduced() || stage === 'welcome') return;
    window.scrollTo(0, 0);
    const ctx = gsap.context(() => {
      if (state) {
        // paste -> review: the big box shrinks into the question panel, then what was deduced slides in
        Flip.from(state, { targets: '[data-flip-id]', duration: 0.85, ease: 'power3.inOut', absolute: true });
        gsap.from('.question > :not(textarea)', { autoAlpha: 0, duration: 0.4, delay: 0.6 });
        gsap.from('.review', { autoAlpha: 0, x: 80, duration: 0.8, delay: 0.3, ease: 'power3.out' });
        gsap.from('.found > *, .form > *', { autoAlpha: 0, y: 18, duration: 0.45, stagger: 0.05, delay: 0.65, ease: 'power2.out' });
        gsap.from('.found b', { textContent: 0, snap: { textContent: 1 }, duration: 1, delay: 0.65, ease: 'power1.out' });
      } else {
        gsap.from(el.children, { autoAlpha: 0, y: 18, duration: 0.55, stagger: 0.08, ease: 'power2.out' });
      }
    }, el);
    return () => ctx.revert();
  }, [stage]);

  const loadExample = (i: number) => {
    setQuestion(EXAMPLES[i].text);
    setDraft(EXAMPLES[i].problem);
    setSolved(null);
  };
  const onProblem = (p: Problem) => {
    setDraft(p);
    if (stage === 'paste') go('review', true);
  };
  const solve = () => {
    if (!parsed.success) return;
    setIndex(0);
    setSolved(parsed.data);
    go('solve');
  };

  const found = (
    [
      [draft.parts.length, 'parts'], [draft.states.length, 'states'], [draft.actions.length, 'actions'],
      [draft.transitions.length, 'transitions'], [draft.policy.length, 'policy entries'], [draft.Q.length, 'Q-values'],
      [draft.episodes.length, 'episodes'], [draft.pulls.length, 'pulls'], [draft.arms.length, 'arms'], [draft.replay.length, 'samples'],
    ] as const
  ).filter(([n]) => n > 0);

  if (stage === 'welcome')
    return <Welcome onStart={() => setStage('paste')} onExample={() => { loadExample(0); setStage('paste'); }} />;

  const at = { paste: 0, review: 1, solve: 2 }[stage];
  const back: (Stage | null)[] = ['paste', 'review', null];

  return (
    <div className="app">
      <header className="top">
        <button className="brand" onClick={() => setStage('welcome')} title="Back to the start">
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
          <span className="brand-text">
            <span className="brand-name">RL Explainer</span>
            <span className="tagline">Every formula, every number, one step at a time</span>
          </span>
        </button>

        <ol className="stepper" aria-label="Progress">
          {STAGES.map((s, i) => {
            const target = i < at ? back[i] : null;
            return (
              <li key={s} className={i < at ? 'done' : i === at ? 'on' : ''} aria-current={i === at ? 'step' : undefined}>
                <button disabled={!target} onClick={() => target && go(target)}>
                  <span className="dot">{i < at ? <Icon name="check" size={13} /> : i + 1}</span>
                  <span className="lbl">{s}</span>
                </button>
              </li>
            );
          })}
        </ol>

        <div className="top-actions">
          {stage === 'paste' && (
            <button className="ghost" onClick={() => go('review')} title="Skip extraction and type the data in yourself">
              Enter data by hand
            </button>
          )}
          {stage === 'review' && (
            <button className="primary cta" onClick={solve} disabled={!parsed.success}
              title={parsed.success ? 'Solve every part step by step' : 'Fix the highlighted data first'}>
              Solve step by step <Icon name="next" size={16} />
            </button>
          )}
          {stage === 'solve' && (
            <button className="ghost" onClick={() => go('review')}>
              <Icon name="edit" size={16} /> Edit data
            </button>
          )}
        </div>
      </header>

      {stage === 'paste' && (
        <main className="paste-layout" ref={main}>
          <div className="paste-intro">
            <h2>Paste your question</h2>
            <p>Include every sub-part (a, b, c…) and all the numbers. The question is read on your Mac and never uploaded.</p>
          </div>
          <div className={`paste-box ${reading ? 'reading' : ''}`} data-flip-id="question">
            <textarea
              autoFocus
              value={question}
              readOnly={reading}
              onChange={(e) => setQuestion(e.target.value)}
              spellCheck={false}
              aria-label="Question"
              placeholder={'A robot moves between states S1, S2, S3 with γ = 0.9…\n\na) Using iterative policy evaluation, compute V1(S1)…\nb) Which action is greedy in S1?'}
            />
            <div className="scan" aria-hidden="true" />
            <div className="paste-foot">
              <select className="examples" aria-label="Load example" value="" onChange={(e) => e.target.value && loadExample(+e.target.value)}>
                <option value="">Try an example…</option>
                {EXAMPLES.map((x, i) => <option key={x.name} value={i}>{x.name}</option>)}
              </select>
              <span className="muted small">{reading ? 'Reading the question…' : `${question.trim() ? question.trim().split(/\s+/).length : 0} words`}</span>
            </div>
          </div>
          <div className="paste-actions">
            <ExtractBar question={question} known={known} onBusy={setReading} onProblem={onProblem}
              onParts={(parts) => { setDraft(problem({ parts })); go('review', true); }} />
          </div>
        </main>
      )}

      {stage === 'review' && (
        <main className="edit-layout" ref={main}>
          <section className="panel question" data-flip-id="question">
            <div className="panel-head">
              <h2><span className="num">1</span> Question</h2>
              <select className="examples" aria-label="Load example" value="" onChange={(e) => e.target.value && loadExample(+e.target.value)}>
                <option value="">Try an example…</option>
                {EXAMPLES.map((x, i) => <option key={x.name} value={i}>{x.name}</option>)}
              </select>
            </div>
            <textarea value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="Paste the full question here, including every sub-part (a, b, c…)" spellCheck={false} />
            <ExtractBar question={question} known={known} onProblem={onProblem} onParts={(parts) => setDraft({ ...draft, parts })} />
            <p className="note">
              <Icon name="info" size={15} />
              <span>The model only copies numbers out of the question. All maths is done by the app, so check the data on the right before solving.</span>
            </p>
          </section>
          <section className="panel review">
            <div className="panel-head">
              <h2><span className="num">2</span> What was found</h2>
              <span className={`status ${parsed.success ? 'ok' : 'bad'}`}>
                {parsed.success ? <><Icon name="check" size={14} /> Ready to solve</> : <><Icon name="x" size={14} /> {issues.length} to fix</>}
              </span>
            </div>
            {found.length > 0 && (
              <div className="found">
                {found.map(([n, l]) => <span key={l}><b>{n}</b> {l}</span>)}
                {Number.isFinite(draft.gamma) && <span><i>γ</i> = {draft.gamma}</span>}
              </div>
            )}
            <ReviewForm draft={draft} setDraft={setDraft} issues={issues} warnings={warnings} />
          </section>
        </main>
      )}

      {stage === 'solve' && solved && (
        <main className="solve-layout" ref={main}>
          <DataPanel p={solved} active={flat[index]?.step.edges ?? []} focus={flat[index]?.step.focus} />
          <StepPlayer traces={traces} flat={flat} index={index} setIndex={setIndex} onRevealed={setRevealed} />
          <ResultsPanel shown={shown} />
        </main>
      )}
    </div>
  );
}
