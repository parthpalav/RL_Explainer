import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import gsap from 'gsap';
import { add, chain, mul, num, toLatex } from '../math/expr';
import { Tex } from './Tex';
import { Icon } from './Icon';

// The teaser is the app's own working for V1(S1) in the IA1 irrigation question, reduced line by line.
const DEMO = chain(add(mul(num(0.2), add(num(-8), mul(num(0.9), num(0)))), mul(num(0.8), add(num(-5), mul(num(0.9), num(0))))));

/** A faint state graph drifting behind the hero: dots run along the edges like probability flowing. */
function Backdrop() {
  const nodes = [
    { x: 160, y: 170, l: 'S₁' }, { x: 520, y: 90, l: 'S₂' }, { x: 860, y: 230, l: 'S₃' },
    { x: 300, y: 470, l: 'S₄' }, { x: 720, y: 520, l: 'S₅' }, { x: 1040, y: 430, l: 'S₆' },
  ];
  const edges = [[0, 1], [1, 2], [0, 3], [3, 4], [4, 2], [2, 5], [4, 5], [1, 4]];
  const path = (a: number, b: number) => {
    const p = nodes[a];
    const q = nodes[b];
    const mx = (p.x + q.x) / 2 - (q.y - p.y) * 0.18;
    const my = (p.y + q.y) / 2 + (q.x - p.x) * 0.18;
    return `M${p.x},${p.y} Q${mx},${my} ${q.x},${q.y}`;
  };
  return (
    <svg className="backdrop" viewBox="0 0 1200 640" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
      {edges.map(([a, b], i) => (
        <g key={i}>
          <path d={path(a, b)} className="bd-edge" />
          <circle r="3" className="bd-dot">
            <animateMotion dur={`${4 + (i % 3) * 1.5}s`} begin={`${i * 0.7}s`} repeatCount="indefinite" path={path(a, b)} />
          </circle>
        </g>
      ))}
      {nodes.map((n) => (
        <g key={n.l} className="bd-node">
          <circle cx={n.x} cy={n.y} r="26" />
          <text x={n.x} y={n.y + 6} textAnchor="middle">{n.l}</text>
        </g>
      ))}
    </svg>
  );
}

/** Landing page: what the app does, and one button into it. */
export function Welcome({ onStart, onExample }: { onStart: () => void; onExample: () => void }) {
  const root = useRef<HTMLDivElement>(null);
  const [line, setLine] = useState(0);
  const lines = useMemo(() => DEMO.map((e) => toLatex(e)), []);

  useLayoutEffect(() => {
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const ctx = gsap.context(() => {
      gsap.timeline({ defaults: { ease: 'power3.out' } })
        .from('.hero-badge', { autoAlpha: 0, y: 12, duration: 0.5 })
        .from('.hero h1', { autoAlpha: 0, y: 28, duration: 0.8 }, '-=0.2')
        .from('.hero .lede', { autoAlpha: 0, y: 18, duration: 0.6 }, '-=0.5')
        .from('.hero-actions > *', { autoAlpha: 0, y: 14, duration: 0.5, stagger: 0.1 }, '-=0.35')
        .from('.demo', { autoAlpha: 0, y: 24, scale: 0.98, duration: 0.7 }, '-=0.3')
        .from('.feature', { autoAlpha: 0, y: 20, duration: 0.5, stagger: 0.1 }, '-=0.4');
    }, root);
    return () => ctx.revert();
  }, []);

  // cycle the teaser: each line is one more hand-solving step, then start over
  useEffect(() => {
    const id = setInterval(() => setLine((i) => (i + 1) % (lines.length + 1)), 1500);
    return () => clearInterval(id);
  }, [lines.length]);

  const leave = (then: () => void) => {
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) return then();
    gsap.to(root.current!.querySelectorAll('.hero > *, .demo, .feature'), {
      autoAlpha: 0, y: -24, duration: 0.4, stagger: 0.04, ease: 'power2.in', onComplete: then,
    });
  };

  return (
    <div className="welcome" ref={root}>
      <Backdrop />
      <section className="hero">
        <span className="hero-badge"><Icon name="spark" size={14} /> Reinforcement learning, solved by hand</span>
        <h1>RL Explainer</h1>
        <p className="lede">
          Paste an exam question. It finds the method, pulls out every number and then works the answer the way you
          would on paper, one line at a time.
        </p>
        <div className="hero-actions">
          <button className="primary hero-cta" onClick={() => leave(onStart)}>
            Paste a question <Icon name="next" size={18} />
          </button>
          <button className="link-btn" onClick={() => leave(onExample)}>or try an IA1 question</button>
        </div>
      </section>

      <div className="demo" aria-label="Example of the working the app shows">
        <div className="demo-head">
          <span className="dots"><i /><i /><i /></span>
          <span>Iterative policy evaluation · <Tex tex="V_1(S_1)" /></span>
        </div>
        <div className="demo-body">
          {lines.map((l, i) => (
            <div key={i} className={`demo-line ${i < line ? 'on' : ''}`}>
              <span className="lhs">{i === 0 && <Tex tex="V_1(S_1)" />}</span>
              <Tex tex="=" />
              <Tex tex={l} />
            </div>
          ))}
        </div>
      </div>

      <ul className="features">
        <li className="feature">
          <Icon name="target" size={20} />
          <b>Spots the method</b>
          <span>Bellman, value iteration, Monte Carlo, TD, SARSA, Q-learning, UCB, n-step, Dyna, DQN.</span>
        </li>
        <li className="feature">
          <Icon name="spark" size={20} />
          <b>Pulls out the numbers</b>
          <span>A local model reads the question on your Mac. You check every value before solving.</span>
        </li>
        <li className="feature">
          <Icon name="check" size={20} />
          <b>Solves it by hand</b>
          <span>Watch each value get plugged in and every multiply and add written out.</span>
        </li>
      </ul>
    </div>
  );
}
