import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import gsap from 'gsap';
import type { PartTrace, Step } from '../solvers/types';
import { Rich, Tex, texHtml } from './Tex';
import { Predict, TheoryCards, quizzable } from './Predict';
import { F } from '../solvers/dp';
import { Icon } from './Icon';
import { ABOUT } from '../solvers';
import { ValueTables, type Shown } from './Panels';

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
const PACE = 1.4; // 1× plays the timeline 1.4× faster than its written durations, so long Bellman steps stay brisk

export function StepPlayer({ traces, flat, index, setIndex, onRevealed }: Props) {
  const card = useRef<HTMLDivElement>(null);
  const tl = useRef<gsap.core.Timeline | null>(null);
  const instant = useRef(false);
  const flyResult = useRef(false);
  const [playing, setPlaying] = useState(false);
  const [done, setDone] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [predict, setPredict] = useState(false);
  const [answered, setAnswered] = useState<Set<number>>(new Set());
  const [score, setScore] = useState({ right: 0, total: 0, streak: 0 });
  const [answers, setAnswers] = useState(false);
  const cur = flat[index];
  const part = cur ? traces[cur.pi] : undefined;
  const waiting = !answers && predict && !!cur && quizzable(cur.step) && !answered.has(index);
  const pool = useMemo(() => [...new Set([...flat.map((f) => f.step.formula), ...Object.values(F)].filter((f): f is string => !!f))], [flat]);

  useEffect(() => setAnswered(new Set()), [flat]);

  const go = (i: number, skipAnim = false) => {
    if (i < 0 || i >= flat.length) return;
    instant.current = skipAnim;
    setAnswers(false);
    setIndex(i);
  };
  /** Own pace: the first press finishes the step being animated, the next one moves on. */
  const forward = () => {
    if (answers) return;
    if (!done && tl.current) tl.current.progress(1);
    else go(index + 1);
  };
  const skipToEnd = () => {
    setPlaying(false);
    instant.current = true;
    flyResult.current = false;
    setAnswers(true);
    setIndex(flat.length - 1);
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
    flyResult.current = !instant.current && !answers;
    const q = (s: string, el: ParentNode = root) => [...el.querySelectorAll<HTMLElement>(s)];
    const lines = q('.line');
    const kids = (line: HTMLElement) => [...line.children].filter((c) => !c.querySelector(':scope > .note')) as HTMLElement[];

    // Measure everything before any tween moves it. Chips inside the card are positioned relative to
    // the card, so scrolling the page mid-animation does not knock them off target.
    const cardBox = root.getBoundingClientRect();
    const local = (el: Element) => {
      const r = el.getBoundingClientRect();
      return { x: r.left - cardBox.left + root.scrollLeft, y: r.top - cardBox.top + root.scrollTop, w: r.width, h: r.height };
    };
    const firstSrc = lines[0] ? q('[data-src]', lines[0]).map((el) => ({ el, to: el.getBoundingClientRect(), from: document.querySelector<HTMLElement>(`[data-ref="${el.dataset.src}"]`) })) : [];
    const morphs = lines.map((line, i) =>
      i === 0
        ? []
        : q('[data-fk]', line).flatMap((f) => {
            const op = lines[i - 1].querySelector<HTMLElement>(`[data-op="${f.dataset.fk}"]`);
            return op ? [{ f, op, a: local(op), b: local(f) }] : [];
          }),
    );

    t.from(q('.why, .formula'), { autoAlpha: 0, y: 8, duration: 0.45, stagger: 0.25 });
    t.to({}, { duration: 0.01 }); // keeps the timeline valid when a step has no lines

    if (lines[0]) {
      // 1. Substitute: every given value flies out of the data panel, labelled with what it is.
      t.from([...kids(lines[0]), ...q('.note', lines[0])], { autoAlpha: 0, duration: 0.35 }, '+=0.2');
      if (firstSrc.length) t.set(firstSrc.map((x) => x.el), { opacity: 0 }, '<');
      firstSrc.forEach(({ el, to: b, from: src }, i) => {
        const at = `>${i ? -0.35 : 0.1}`;
        if (!src) {
          t.to(el, { opacity: 1, duration: 0.25 }, at);
          return;
        }
        const a = src.getBoundingClientRect();
        const chip = document.createElement('div');
        chip.className = 'fly-chip';
        chip.innerHTML = `${src.dataset.tex ? `<span class="chip-tag">${texHtml(src.dataset.tex)}</span>` : ''}<span>${src.textContent}</span>`;
        Object.assign(chip.style, { left: `${a.left}px`, top: `${a.top}px`, minWidth: `${a.width}px`, height: `${a.height}px` });
        document.body.appendChild(chip);
        chips.push(chip);
        gsap.set(chip, { autoAlpha: 0 });
        t.call(() => src.classList.add('src-flash'), [], at)
          .set(chip, { autoAlpha: 1 }, '<')
          .to(chip, { x: b.left - a.left, y: b.top - a.top, duration: 0.8, ease: 'power2.inOut' }, '<')
          .set(el, { opacity: 1 })
          .to(chip, { autoAlpha: 0, duration: 0.15 }, '<')
          .call(() => src.classList.remove('src-flash'));
      });
    }

    lines.slice(1).forEach((line, j) => {
      const pairs = morphs[j + 1];
      const label = `line${j + 1}`;
      t.addLabel(label, '+=0.3');
      if (!pairs.length) {
        // Not an arithmetic chain (e.g. a pick or a rearranged equation): just reveal it.
        t.from([...kids(line), ...q('.note', line)], { autoAlpha: 0, y: 10, duration: 0.4 }, label);
        const fresh = q('.fresh', line);
        if (fresh.length) t.from(fresh, { scale: 1.35, duration: 0.45, ease: 'back.out(3)' }, '<0.1');
        return;
      }
      // 2. Point at the work: box each operation that is ready to be worked out, and say what it is.
      const ops = pairs.map((p) => p.op);
      t.from(q('.note', line), { autoAlpha: 0, x: -6, duration: 0.3 }, label)
        .to(ops, { backgroundColor: 'rgba(255,180,107,0.16)', boxShadow: '0 0 0 1.5px rgba(255,180,107,0.8)', duration: 0.3, stagger: 0.1 }, label)
        .to({}, { duration: 0.25 });
      // 3. Work it out: each boxed operation shrinks into the number it becomes on the next line.
      const results = pairs.map((p) => p.f);
      t.from(kids(line), { autoAlpha: 0, y: 6, duration: 0.35 }).set(results, { opacity: 0 }, '<');
      const base = t.duration() - 0.1;
      pairs.forEach(({ f, a, b }, k) => {
        const chip = document.createElement('div');
        chip.className = 'morph-chip';
        chip.textContent = f.textContent;
        Object.assign(chip.style, { left: `${a.x}px`, top: `${a.y}px`, width: `${a.w}px`, height: `${a.h}px` });
        root.appendChild(chip);
        chips.push(chip);
        gsap.set(chip, { autoAlpha: 0 });
        const s0 = base + k * 0.15; // chips overlap a little so long lines do not drag
        t.set(chip, { autoAlpha: 1 }, s0)
          .from(chip, { scale: 0.9, duration: 0.2 }, s0)
          .to(chip, { x: b.x - a.x, y: b.y - a.y, width: b.w, height: b.h, duration: 0.6, ease: 'power2.inOut' }, s0 + 0.15)
          .set(f, { opacity: 1 }, s0 + 0.75)
          .to(chip, { autoAlpha: 0, duration: 0.2 }, s0 + 0.75)
          .from(f.querySelector('.fresh') ?? f, { scale: 1.3, duration: 0.4, ease: 'back.out(3)' }, s0 + 0.75);
      });
      t.to(ops, { backgroundColor: 'rgba(255,180,107,0)', boxShadow: '0 0 0 0px rgba(255,180,107,0)', duration: 0.3 });
    });

    t.timeScale(speed * PACE);
    if (instant.current) t.progress(1);
    instant.current = false;
    return () => {
      t.revert(); // restores KaTeX's own inline styles, unlike clearProps
      chips.forEach((c) => c.remove());
      document.querySelectorAll('.src-flash').forEach((e) => e.classList.remove('src-flash'));
    };
    // speed is applied separately below so changing it does not restart the step
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [index, flat, waiting, answers]);

  useEffect(() => {
    tl.current?.timeScale(speed * PACE);
  }, [speed]);

  // 4. Record: once the working is done, the answer flies from the last line into the worked-values table.
  useEffect(() => {
    if (!done || !flyResult.current) return;
    flyResult.current = false;
    const from = card.current?.querySelector('.line:last-child .rhs .katex');
    const cells = [...document.querySelectorAll<HTMLElement>('.results td.fresh-cell .val')];
    if (!from || !cells.length || matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const t = gsap.timeline();
    const chips: HTMLElement[] = [];
    cells.forEach((cell, i) => {
      const panel = cell.closest<HTMLElement>('.results');
      if (panel && getComputedStyle(panel).position === 'sticky') cell.scrollIntoView({ block: 'nearest' });
      const a = from.getBoundingClientRect();
      const b = cell.getBoundingClientRect();
      if (b.bottom < 0 || b.top > innerHeight) return; // table is off screen (phones): nothing to see
      const chip = document.createElement('div');
      chip.className = 'fly-chip result';
      chip.textContent = cell.textContent;
      Object.assign(chip.style, { left: `${a.left}px`, top: `${a.top + a.height / 2 - b.height / 2}px`, minWidth: `${b.width}px`, height: `${b.height}px` });
      document.body.appendChild(chip);
      chips.push(chip);
      const td = cell.parentElement!;
      t.fromTo(chip, { autoAlpha: 0, scale: 1.2 }, { autoAlpha: 1, scale: 1, duration: 0.2 }, i * 0.1)
        .to(chip, { x: b.left - a.left, y: b.top - (a.top + a.height / 2 - b.height / 2), duration: 0.75, ease: 'power2.inOut' }, '>')
        .call(() => td.classList.add('land'))
        .to(chip, { autoAlpha: 0, duration: 0.2 }, '<');
    });
    t.timeScale(speed * PACE);
    return () => {
      t.kill();
      chips.forEach((c) => c.remove());
      document.querySelectorAll('.results td.land').forEach((e) => e.classList.remove('land'));
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [done]);

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
      if (e.key === 'ArrowRight') forward();
      else if (e.key === 'ArrowLeft') go(answers ? index : index - 1, true);
      else if (e.key === ' ') {
        e.preventDefault();
        setPlaying((x) => !x);
      } else if (e.key === 'End') skipToEnd();
      else return;
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
        <AlgoBanner part={part} />
        {part.part.text && (
          <p className="part-text">
            <span className="eyebrow">Question {part.part.label ? `(${part.part.label})` : ''}</span>
            {part.part.text}
          </p>
        )}
        <div className="progress" role="progressbar" aria-valuemin={1} aria-valuemax={partSteps.length} aria-valuenow={posInPart + 1}>
          <div className="bar" style={{ width: `${pct}%` }} />
        </div>

        <div className={`card${answers ? ' answers' : ''}`} ref={card} key={`${index}-${waiting}-${answers}`}>
          {answers ? <Answers traces={traces} onOpen={(pi) => go(flat.findIndex((f) => f.pi === pi), true)} /> : <>
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
                      {/* continuation lines have no lhs, so their margin note sits there, like on paper */}
                      <span className="lhs">{l.lhs ? <Tex tex={l.lhs} /> : l.note && <span className="note">{l.note}</span>}</span>
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
                  <span><i className="sw box" /> being worked out</span>
                  <span><i className="sw fresh" /> just calculated</span>
                </footer>
              )}
            </>
          )}
          </>}
        </div>
      </div>

      <div className="transport">
        <div className="tgroup">
          <button className="round play" onClick={() => setPlaying((x) => !x)} disabled={answers} title="Play / pause (space)" aria-label={playing ? 'Pause' : 'Play'}>
            <Icon name={playing ? 'pause' : 'play'} size={20} />
          </button>
          <button className="round subtle" onClick={() => tl.current?.restart()} disabled={answers} title="Replay this step" aria-label="Replay this step">
            <Icon name="replay" size={17} />
          </button>
        </div>
        <div className="pace">
          <button className="pace-btn" onClick={() => go(answers ? index : index - 1, true)} disabled={index === 0 && !answers} title="Back one step (←)">
            <Icon name="prev" size={16} /> Back
          </button>
          <div className="seg" role="radiogroup" aria-label="Speed">
            {SPEEDS.map((s) => (
              <button key={s} role="radio" aria-checked={speed === s} className={speed === s ? 'on' : ''} onClick={() => setSpeed(s)}>{s}×</button>
            ))}
          </div>
          <button className="pace-btn" onClick={forward} disabled={answers || (index === flat.length - 1 && done)} title="Forward one step; finishes the current animation first (→)">
            {done ? 'Next' : 'Finish step'} <Icon name="next" size={16} />
          </button>
          <button className={`pace-btn skip${answers ? ' on' : ''}`} onClick={skipToEnd} title="Skip to the end and show only the answers (End)">
            Answers <Icon name="skip" size={16} />
          </button>
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
        <span className="keys" aria-hidden="true"><kbd>←</kbd><kbd>→</kbd><kbd>space</kbd><kbd>End</kbd></span>
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

/** Which algorithm this part uses, in plain words, with its update rule. */
function AlgoBanner({ part }: { part: PartTrace }) {
  const about = ABOUT[part.part.task];
  // the update rule this part applies most often (a pick step like argmax can come first)
  const count = new Map<string, number>();
  for (const s of part.steps) if (s.formula) count.set(s.formula, (count.get(s.formula) ?? 0) + 1);
  const formula = [...count].sort((x, y) => y[1] - x[1])[0]?.[0];
  return (
    <div className="algo">
      <div className="algo-head">
        <span className="algo-family">{about.family}</span>
        <h2 className="algo-name">{part.heading}</h2>
      </div>
      <p className="algo-what">{about.what}</p>
      {formula && <div className="algo-formula"><Tex tex={formula} /></div>}
    </div>
  );
}

/** The final value of every cell each part computed, or its conclusion when it computes none. */
function Answers({ traces, onOpen }: { traces: PartTrace[]; onOpen: (pi: number) => void }) {
  return (
    <div className="answer-list">
      <header>
        <span className="eyebrow">Skipped to the end</span>
        <h3>Answers</h3>
      </header>
      {traces.map((t, pi) => {
        const last = new Map<string, Shown>();
        for (const st of t.steps) for (const r of st.results) last.set(`${r.cell.table}\u0000${r.cell.row}\u0000${r.cell.col}`, { ...r, fresh: false });
        const end = t.steps.at(-1);
        const concl = end && !end.results.length ? end.lines.at(-1) : undefined;
        const lhs = [...(end?.lines ?? [])].reverse().find((l) => l.lhs)?.lhs;
        return (
          <section className="answer" key={pi}>
            <div className="answer-head">
              <span className="badge">{t.part.label || pi + 1}</span>
              <span className="h">{t.heading}</span>
              <button className="ghost small" onClick={() => onOpen(pi)}>Show working <Icon name="next" size={14} /></button>
            </div>
            {t.part.text && <p className="muted">{t.part.text}</p>}
            {t.error && <p className="answer-err">{t.error}</p>}
            {concl && <div className="answer-line"><Tex tex={`${lhs ? `${lhs} = ` : ''}${concl.rhs}`} /></div>}
            {!concl && end && !last.size && <p className="muted">{end.title}</p>}
            {last.size > 0 && <div className="answer-tables"><ValueTables shown={[...last.values()]} /></div>}
          </section>
        );
      })}
    </div>
  );
}
