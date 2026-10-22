import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import gsap from 'gsap';
import type { PartTrace, Step } from '../solvers/types';
import { Rich, Tex } from './Tex';
import { Predict, TheoryCards, quizzable } from './Predict';
import { F } from '../solvers/dp';

export type Flat = { pi: number; step: Step };

export function flatten(traces: PartTrace[]): Flat[] {
  return traces.flatMap((t, pi) =>
    t.error
      ? [{ pi, step: { title: 'Could not solve this part', why: t.error, lines: [], results: [] } }]
      : t.steps.map((step) => ({ pi, step })),
  );
}

interface Props {
  traces: PartTrace[];
  flat: Flat[];
  index: number;
  setIndex: (i: number) => void;
  onRevealed: (done: boolean) => void;
}

const SPEEDS = [0.5, 1, 1.5, 2];

export function StepPlayer({ traces, flat, index, setIndex, onRevealed }: Props) {
  const card = useRef<HTMLDivElement>(null);
  const tl = useRef<gsap.core.Timeline | null>(null);
  const instant = useRef(false);
  const [playing, setPlaying] = useState(false);
  const [done, setDone] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [predict, setPredict] = useState(false);
  const [answered, setAnswered] = useState<Set<number>>(new Set());
  const [score, setScore] = useState({ right: 0, total: 0, streak: 0 });
  const cur = flat[index];
  const part = cur ? traces[cur.pi] : undefined;
  const waiting = predict && !!cur && quizzable(cur.step) && !answered.has(index);
  const pool = useMemo(() => [...new Set([...flat.map((f) => f.step.formula), ...Object.values(F)].filter((f): f is string => !!f))], [flat]);

  useEffect(() => setAnswered(new Set()), [flat]);

  const go = (i: number, skipAnim = false) => {
    if (i < 0 || i >= flat.length) return;
    instant.current = skipAnim;
    setIndex(i);
  };

  useLayoutEffect(() => {
    const root = card.current;
    if (!root || !cur) return;
    setDone(false);
    onRevealed(false);
    if (waiting) return; // quiz first: keep the answer out of the results table
    const chips: HTMLElement[] = [];
    const t = gsap.timeline({
      onComplete: () => {
        setDone(true);
        onRevealed(true);
      },
    });
    tl.current = t;
    const q = (s: string, el: ParentNode = root) => [...el.querySelectorAll<HTMLElement>(s)];

    t.from(q('.why, .formula'), { autoAlpha: 0, y: 8, duration: 0.45, stagger: 0.25 });
    t.to({}, { duration: 0.01 }); // keeps the timeline valid when a step has no lines
    const lines = q('.line');
    if (lines[0]) {
      const targets = q('[data-src]', lines[0]);
      t.from(lines[0], { autoAlpha: 0, duration: 0.35 }, '+=0.2');
      if (targets.length) t.set(targets, { opacity: 0 }, '<');
      targets.forEach((el, i) => {
        const src = document.querySelector<HTMLElement>(`[data-ref="${el.dataset.src}"]`);
        const at = `>${i ? -0.35 : 0.1}`;
        if (!src) {
          t.to(el, { opacity: 1, duration: 0.25 }, at);
          return;
        }
        const a = src.getBoundingClientRect();
        const b = el.getBoundingClientRect();
        const chip = document.createElement('div');
        chip.className = 'fly-chip';
        chip.textContent = src.textContent;
        Object.assign(chip.style, { left: `${a.left}px`, top: `${a.top}px`, minWidth: `${a.width}px`, height: `${a.height}px` });
        document.body.appendChild(chip);
        chips.push(chip);
        gsap.set(chip, { autoAlpha: 0 });
        t.call(() => src.classList.add('src-flash'), [], at)
          .set(chip, { autoAlpha: 1 }, '<')
          .to(chip, { x: b.left - a.left, y: b.top - a.top, duration: 0.7, ease: 'power2.inOut' }, '<')
          .set(el, { opacity: 1 })
          .to(chip, { autoAlpha: 0, duration: 0.15 }, '<')
          .call(() => src.classList.remove('src-flash'));
      });
    }
    lines.slice(1).forEach((line) => {
      t.from(line, { autoAlpha: 0, y: 10, duration: 0.4 }, '+=0.45');
      const fresh = q('.fresh', line);
      if (fresh.length) t.from(fresh, { scale: 1.35, duration: 0.45, ease: 'back.out(3)' }, '<0.1');
    });

    t.timeScale(speed);
    if (instant.current) t.progress(1);
    instant.current = false;
    return () => {
      t.revert(); // restores KaTeX's own inline styles, unlike clearProps
      chips.forEach((c) => c.remove());
      document.querySelectorAll('.src-flash').forEach((e) => e.classList.remove('src-flash'));
    };
    // speed is applied separately below so changing it does not restart the step
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [index, flat, waiting]);

  useEffect(() => {
    tl.current?.timeScale(speed);
  }, [speed]);

  useEffect(() => {
    if (!playing || !done || waiting) return;
    if (index >= flat.length - 1) {
      setPlaying(false);
      return;
    }
    const id = setTimeout(() => go(index + 1), 1100 / speed);
    return () => clearTimeout(id);
  });

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      if (el.closest?.('input, textarea, select') || (e.key === ' ' && el.closest?.('button'))) return;
      if (e.key === 'ArrowRight') go(index + 1);
      else if (e.key === 'ArrowLeft') go(index - 1, true);
      else if (e.key === ' ') {
        e.preventDefault();
        setPlaying((x) => !x);
      } else return;
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  if (!cur || !part) return <section className="player"><p className="muted">Nothing to solve.</p></section>;
  const partSteps = flat.map((f, i) => ({ ...f, i })).filter((f) => f.pi === cur.pi);
  const { step } = cur;
  const posInPart = partSteps.findIndex((f) => f.i === index);

  return (
    <section className="player">
      <nav className="parts">
        {traces.map((t, pi) => {
          const first = flat.findIndex((f) => f.pi === pi);
          return (
            <button key={pi} className={pi === cur.pi ? 'on' : ''} onClick={() => go(first, true)} disabled={first < 0}>
              <b>{t.part.label ? `${t.part.label})` : `Part ${pi + 1}`}</b> {t.heading}
            </button>
          );
        })}
      </nav>
      {part.part.text && <p className="part-text">{part.part.text}</p>}

      <div className="card" ref={card} key={`${index}-${waiting}`}>
        <header>
          <span className="count">
            Step {posInPart + 1} / {partSteps.length}
          </span>
          <h3><Rich text={step.title} /></h3>
        </header>
        {waiting ? (
          <Predict
            step={step}
            pool={pool}
            onDone={(ok) => {
              setScore((x) => ({ right: x.right + +ok, total: x.total + 1, streak: ok ? x.streak + 1 : 0 }));
              setAnswered((x) => new Set(x).add(index));
            }}
          />
        ) : (
          <>
        <p className="why">{step.why}</p>
        {step.formula && (
          <div className="formula">
            <span className="tag">Formula</span>
            <Tex tex={step.formula} display />
          </div>
        )}
        {step.lines.length > 0 ? (
          <div className="lines">
            {step.lines.map((l, i) => (
              <div className="line" key={i}>
                <span className="lhs">{l.lhs && <Tex tex={l.lhs} />}</span>
                <span className="eq"><Tex tex="=" /></span>
                <span className="rhs"><Tex tex={l.rhs} /></span>
              </div>
            ))}
          </div>
        ) : null}
        {step.cards && <TheoryCards ids={step.cards} />}
          </>
        )}
      </div>

      <div className="controls">
        <button onClick={() => go(index - 1, true)} disabled={index === 0} title="Previous step (←)">◀ Prev</button>
        <button onClick={() => setPlaying((x) => !x)} className="primary" title="Play / pause (space)">
          {playing ? '❚❚ Pause' : '▶ Play'}
        </button>
        <button onClick={() => go(index + 1)} disabled={index === flat.length - 1} title="Next step (→)">Next ▶</button>
        <button onClick={() => tl.current?.restart()} title="Replay this step">↻ Replay</button>
        <label className="toggle" title="Guess the formula and the value before each calculation is shown">
          <input type="checkbox" checked={predict} onChange={(e) => setPredict(e.target.checked)} /> Predict mode
        </label>
        {predict && score.total > 0 && (
          <span className="score" aria-live="polite">
            {score.right}/{score.total} correct{score.streak > 1 ? ` · streak ${score.streak} 🔥` : ''}
          </span>
        )}
        <label>
          Speed{' '}
          <select value={speed} onChange={(e) => setSpeed(+e.target.value)}>
            {SPEEDS.map((s) => <option key={s} value={s}>{s}×</option>)}
          </select>
        </label>
      </div>

      <ol className="steps">
        {partSteps.map((f) => (
          <li key={f.i} className={f.i === index ? 'on' : f.i < index ? 'past' : ''}>
            <button onClick={() => go(f.i, true)}><Rich text={f.step.title} /></button>
          </li>
        ))}
      </ol>
    </section>
  );
}
