import type { Problem } from '../schema';
import { edgeKey } from '../mdp';
import { fmt } from '../math/expr';

const W = 420;
const H = 300;
const R = 30;

/** States on a circle; transitions as curved arrows. Labels only on highlighted edges to keep it readable. */
export function MdpGraph({ p, active = [] }: { p: Problem; active?: string[] }) {
  const on = new Set(active);
  const n = p.states.length;
  const pos = new Map(
    p.states.map((s, i) => {
      const t = (i / n) * Math.PI * 2 - Math.PI / 2;
      const rad = n === 1 ? 0 : Math.min(W, H) / 2 - R - 50; // room for self-loops
      return [s.id, { x: W / 2 + rad * Math.cos(t), y: H / 2 + rad * Math.sin(t) }];
    }),
  );
  const actIdx = new Map(p.actions.map((a, i) => [a.id, i]));
  const edges = p.transitions.map((t) => {
    const a = pos.get(t.s)!;
    const b = pos.get(t.s2)!;
    const k = edgeKey(t.s, t.a, t.s2);
    const i = actIdx.get(t.a) ?? 0;
    let d: string;
    let lx: number;
    let ly: number;
    if (t.s === t.s2) {
      // self loop, pushed outward from the centre; one loop size per action
      const ang = Math.atan2(a.y - H / 2, a.x - W / 2) || -Math.PI / 2;
      const L = R + 26 + i * 16;
      const sp = 0.45;
      const p1 = { x: a.x + R * Math.cos(ang - sp), y: a.y + R * Math.sin(ang - sp) };
      const p2 = { x: a.x + R * Math.cos(ang + sp), y: a.y + R * Math.sin(ang + sp) };
      const c1 = { x: a.x + L * 1.6 * Math.cos(ang - sp * 1.6), y: a.y + L * 1.6 * Math.sin(ang - sp * 1.6) };
      const c2 = { x: a.x + L * 1.6 * Math.cos(ang + sp * 1.6), y: a.y + L * 1.6 * Math.sin(ang + sp * 1.6) };
      d = `M${p1.x},${p1.y} C${c1.x},${c1.y} ${c2.x},${c2.y} ${p2.x},${p2.y}`;
      lx = a.x + L * 1.25 * Math.cos(ang);
      ly = a.y + L * 1.25 * Math.sin(ang);
    } else {
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const len = Math.hypot(dx, dy);
      const ux = dx / len;
      const uy = dy / len;
      const bend = 22 + i * 18; // separate arrows of different actions
      const mx = (a.x + b.x) / 2 - uy * bend;
      const my = (a.y + b.y) / 2 + ux * bend;
      const s0 = { x: a.x + ux * R, y: a.y + uy * R };
      const e0 = { x: b.x - ux * (R + 6), y: b.y - uy * (R + 6) };
      d = `M${s0.x},${s0.y} Q${mx},${my} ${e0.x},${e0.y}`;
      lx = (s0.x + 2 * mx + e0.x) / 4;
      ly = (s0.y + 2 * my + e0.y) / 4;
    }
    return { t, k, d, lx, ly };
  });

  return (
    <svg className="graph" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="State transition diagram">
      <defs>
        <marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" className="arrowhead" />
        </marker>
        <marker id="arr-on" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" className="arrowhead on" />
        </marker>
      </defs>
      {edges.map(({ k, d }) => (
        <path key={k} d={d} className={`edge ${on.has(k) ? 'on' : ''}`} markerEnd={`url(#${on.has(k) ? 'arr-on' : 'arr'})`} />
      ))}
      {edges
        .filter(({ k }) => on.has(k))
        .map(({ k, d }) => (
          // a dot travelling s -> s' shows which way the probability flows
          <circle key={`dot${k}`} r={4} className="pulse">
            <animateMotion dur="1.6s" repeatCount="indefinite" path={d} />
          </circle>
        ))}
      {p.states.map((s) => {
        const { x, y } = pos.get(s.id)!;
        const term = p.terminal.includes(s.id);
        return (
          <g key={s.id} className="node">
            <circle cx={x} cy={y} r={R} />
            {term && <circle cx={x} cy={y} r={R - 5} />}
            <text x={x} y={s.name ? y - 2 : y + 5} textAnchor="middle" className="nid">{s.id}</text>
            {s.name && <text x={x} y={y + 13} textAnchor="middle" className="nname">{s.name.slice(0, 10)}</text>}
          </g>
        );
      })}
      {edges
        .filter(({ k }) => on.has(k))
        .map(({ t, k, lx, ly }) => (
          <g key={`l${k}`} className="elabel">
            <rect x={lx - 50} y={ly - 11} width={100} height={22} rx={6} />
            <text x={lx} y={ly + 4} textAnchor="middle">{`${t.a}  p=${fmt(t.p)}  r=${fmt(t.r)}`}</text>
          </g>
        ))}
    </svg>
  );
}
