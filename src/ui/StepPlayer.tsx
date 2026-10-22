import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import gsap from 'gsap';
import type { PartTrace, Step } from '../solvers/types';
import { Rich, Tex } from './Tex';
import { Predict, TheoryCards, quizzable } from './Predict';
import { F } from '../solvers/dp';
import { Icon } from './Icon';

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
  const pct = ((posInPart + 1) / partSteps.length) * 100;

  return (
    <section className="player">
      <nav className="parts" aria-label="Question parts">
        {traces.map((t, pi) => {
          const first = flat.findIndex((f) => f.pi === pi);
          const state = pi === cur.pi ? 'on' : pi < cur.pi ? 'done' : '';
          return (
            <button key={pi} className={`part-tab ${state}`} onClick={() => go(first, true)} disabled={first < 0}
              aria-current={pi === cur.pi ? 'true' : undefined}>
              <span className="badge">{t.part.label || pi + 1}</span>
              <span className="h">{t.heading}</span>
            </button>
          );
        })}
      </nav>

      <div className="stage">
        {part.part.text && (
          <p className="part-text">
            <span className="eyebrow">Question {part.part.label ? `(${part.part.label})` : ''}</span>
            {part.part.text}
          </p>
        )}
        <div className="progress" role="progressbar" aria-valuemin={1} aria-valuemax={partSteps.length} aria-valuenow={posInPart + 1}>
          <div className="bar" style={{ width: `${pct}%` }} />
        </div>

        <div className="card" ref={card} key={`${index}-${waiting}`}>
          <header>
            <span className="eyebrow">Step {posInPart + 1} of {partSteps.length}</span>
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
              <p className="why"><Icon name="info" size={16} /><span>{step.why}</span></p>
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
              {step.lines.length > 0 && (
                <footer className="key">
                  <span><i className="sw given" /> value plugged in from the question or an earlier step</span>
                  <span><i className="sw fresh" /> just calculated</span>
                </footer>
              )}
            </>
          )}
        </div>
      </div>

      <div className="transport">
        <div className="tgroup">
          <button className="round" onClick={() => go(index - 1, true)} disabled={index === 0} title="Previous step (←)" aria-label="Previous step">
            <Icon name="prev" />
          </button>
          <button className="round play" onClick={() => setPlaying((x) => !x)} title="Play / pause (space)" aria-label={playing ? 'Pause' : 'Play'}>
            <Icon name={playing ? 'pause' : 'play'} size={20} />
          </button>
          <button className="round" onClick={() => go(index + 1)} disabled={index === flat.length - 1} title="Next step (→)" aria-label="Next step">
            <Icon name="next" />
          </button>
          <button className="round subtle" onClick={() => tl.current?.restart()} title="Replay this step" aria-label="Replay this step">
            <Icon name="replay" size={17} />
          </button>
        </div>
        <div className="seg" role="radiogroup" aria-label="Speed">
          {SPEEDS.map((s) => (
            <button key={s} role="radio" aria-checked={speed === s} className={speed === s ? 'on' : ''} onClick={() => setSpeed(s)}>{s}×</button>
          ))}
        </div>
        <label className="switch" title="Guess the formula and the value before each calculation is shown">
          <input type="checkbox" checked={predict} onChange={(e) => setPredict(e.target.checked)} />
          <span className="track"><span className="thumb" /></span>
          Predict mode
        </label>
        {predict && score.total > 0 && (
          <span className="score" aria-live="polite">
            <Icon name="target" size={15} /> {score.right}/{score.total}{score.streak > 1 ? ` · streak ${score.streak}` : ''}
          </span>
        )}
        <span className="keys" aria-hidden="true"><kbd>←</kbd><kbd>→</kbd><kbd>space</kbd></span>
      </div>

      <ol className="steps" aria-label="Steps in this part">
        {partSteps.map((f, k) => (
          <li key={f.i} className={f.i === index ? 'on' : f.i < index ? 'past' : ''}>
            <button onClick={() => go(f.i, true)}>
              <span className="n">{f.i < index ? <Icon name="check" size={12} /> : k + 1}</span>
              <span className="t"><Rich text={f.step.title} /></span>
            </button>
          </li>
        ))}
      </ol>
    </section>
  );
}
