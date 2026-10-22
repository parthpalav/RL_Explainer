"""Run with `npm run verify`. Independent reference implementations (Sutton & Barto definitions) + random problem generator
+ checker that compares the app's solver output and re-evaluates every displayed line of working."""
import json, math, random, re, sys
import numpy as np

def tex(i):
    m = re.fullmatch(r'([A-Za-z]+)_?(\d+)', i)
    return f'{m[1]}_{{{m[2]}}}' if m else '\\text{' + re.sub(r'[\\{}$&#^_%~]', '', i) + '}'

# ---------- problem helpers ----------
def nonterm(p): return [s['id'] for s in p['states'] if s['id'] not in p.get('terminal', [])]
def outs(p, s, a): return [t for t in p['transitions'] if t['s'] == s and t['a'] == a]
def acts_in(p, s):
    used = {t['a'] for t in p['transitions'] if t['s'] == s}
    return [a['id'] for a in p['actions'] if a['id'] in used]
def V0(p):
    V = {s['id']: 0.0 for s in p['states']}
    for x in p.get('V0', []): V[x['s']] = x['v']
    for t in p.get('terminal', []): V[t] = 0.0
    return V
def pol(p, s): return [(x['a'], x['p']) for x in p['policy'] if x['s'] == s and x['p'] > 0]
def q_of(p, s, a, V): return sum(t['p'] * (t['r'] + p['gamma'] * V.get(t['s2'], 0.0)) for t in outs(p, s, a))
def argmax_first(items, prefer=None):
    best = max(v for _, v in items)
    win = [k for k, v in items if abs(v - best) < 1e-9]
    return prefer if prefer in win else win[0]

T = dict(
    pe=r'\text{Policy evaluation } V_k(s)', ip=r'\text{In-place evaluation } V_k(s)',
    vi=r'\text{Value iteration } V_k(s)', viPi=r'\text{Value iteration greedy } \pi_k(s)',
    piV=r'\text{Policy iteration } V^{\pi_i}(s)', pi=r'\text{Policy iteration } \pi_i(s)',
    q='Q(s,a)', pol=r"\text{Improved policy } \pi'(s)", eps=r'\varepsilon\text{-greedy } P(a\mid s)',
)

def expect_part(p, part, ctx):
    """Return (cells: {(table,row,col): value}, last_rhs: list or None)."""
    task = part['task']; cells = {}; last = None; g = p['gamma']
    it = part.get('iterations')
    if task in ('policy_evaluation', 'bellman_v', 'policy_evaluation_inplace'):
        inplace = task == 'policy_evaluation_inplace'; table = T['ip'] if inplace else T['pe']
        V = V0(p)
        for k in range(it or 1):
            new = dict(V)
            for s in nonterm(p):
                src = new if inplace else V
                new[s] = sum(w * q_of(p, s, a, src) for a, w in pol(p, s))
                cells[(table, f'k = {k+1}', tex(s))] = new[s]
            V = new
        ctx['lastV'] = V
    elif task in ('value_iteration', 'bellman_optimality'):
        V = V0(p)
        for k in range(it or 1):
            new = dict(V)
            for s in nonterm(p):
                qs = [(a, q_of(p, s, a, V)) for a in acts_in(p, s)]
                for a, q in qs: cells[(rf'\text{{Value iteration }} Q_{{{k+1}}}(s,a)', tex(s), tex(a))] = q
                new[s] = max(q for _, q in qs)
                cells[(T['vi'], f'k = {k+1}', tex(s))] = new[s]
                cells[(T['viPi'], f'\\pi_{{{k+1}}}', tex(s))] = tex(argmax_first(qs))
            V = new
        ctx['lastV'] = V
    elif task in ('policy_evaluation_linear', 'policy_iteration'):
        S = nonterm(p)
        def solve(pi):
            n = len(S); A = np.eye(n); b = np.zeros(n)
            for i, s in enumerate(S):
                for a, w in pi[s]:
                    for t in outs(p, s, a):
                        b[i] += w * t['p'] * t['r']
                        if t['s2'] in S: A[i, S.index(t['s2'])] -= w * t['p'] * g
            x = np.linalg.solve(A, b)
            V = {s: 0.0 for s in [st['id'] for st in p['states']]}
            for i, s in enumerate(S): V[s] = float(x[i])
            return V
        if task == 'policy_evaluation_linear':
            V = solve({s: pol(p, s) for s in S})
            for s in S: cells[(T['piV'], r'\pi', tex(s))] = V[s]
            ctx['lastV'] = V
        else:
            pi = {}
            for s in S:
                xs = sorted([x for x in p['policy'] if x['s'] == s], key=lambda x: -x['p'])
                pi[s] = xs[0]['a'] if xs else acts_in(p, s)[0]
            for i in range(it or 20):
                V = solve({s: [(pi[s], 1)] for s in S})
                for s in S: cells[(T['piV'], f'\\pi_{{{i}}}', tex(s))] = V[s]
                new = {}
                for s in S:
                    qs = [(a, q_of(p, s, a, V)) for a in acts_in(p, s)]
                    for a, q in qs: cells[(rf'\text{{Policy iteration }} Q^{{\pi_{{{i}}}}}(s,a)', tex(s), tex(a))] = q
                    new[s] = argmax_first(qs, pi[s])
                    cells[(T['pi'], f'\\pi_{{{i+1}}}', tex(s))] = tex(new[s])
                ctx['lastV'] = V
                stable = new == pi; pi = new
                if stable: break
    elif task in ('q_from_v', 'policy_improvement'):
        V = ctx.get('lastV') or V0(p)
        for s in (part.get('states') or nonterm(p)):
            qs = [(a, q_of(p, s, a, V)) for a in acts_in(p, s)]
            for a, q in qs: cells[(T['q'], tex(s), tex(a))] = q
            if task == 'policy_improvement': cells[(T['pol'], "\\pi'", tex(s))] = tex(argmax_first(qs))
    elif task == 'best_state':
        V = ctx.get('lastV') or V0(p)
        last = [tex(argmax_first([(s, V.get(s, 0)) for s in (part.get('states') or nonterm(p))]))]
    elif task == 'greedy_action':
        S = part.get('states') or list(dict.fromkeys(x['s'] for x in p['Q']))
        last = [tex(argmax_first([(x['a'], x['v']) for x in p['Q'] if x['s'] == s])) for s in S]
    elif task == 'epsilon_greedy_probs':
        S = part.get('states') or list(dict.fromkeys(x['s'] for x in p['Q']))
        e = p['epsilon']
        for s in S:
            items = [(x['a'], x['v']) for x in p['Q'] if x['s'] == s]
            gA = argmax_first(items)
            for a, _ in items: cells[(T['eps'], tex(s), tex(a))] = (1 - e + e / len(items)) if a == gA else e / len(items)
    elif task == 'return_Gt':
        last = [sum(g ** i * r for i, r in enumerate(part['rewards']))]
    else:
        return expect_mf(p, part, ctx)
    return cells, last

# ---------- sample-based ----------
def eps_of(p, part):
    idx = part.get('episodes') or list(range(1, len(p['episodes']) + 1))
    return [(e - 1, p['episodes'][e - 1]) for e in idx]
def nxt(p, ep, t):
    s = ep['steps'][t + 1]['s'] if t + 1 < len(ep['steps']) else ep.get('end', '')
    return None if not s or s in p.get('terminal', []) else s
def state_at(p, ep, k):
    T_ = len(ep['steps'])
    s = ep['steps'][k]['s'] if k < T_ else (ep.get('end', '') if k == T_ else '')
    return None if not s or s in p.get('terminal', []) else s
def act_at(ep, k): return ep['steps'][k].get('a', '') if k < len(ep['steps']) else ep.get('endAction', '')
def Qinit(p):
    Q = {}
    for s in nonterm(p):
        for a in p['actions']: Q[(s, a['id'])] = 0.0
    for x in p['Q']: Q[(x['s'], x['a'])] = x['v']
    return Q
def prob(p, which, s, a):
    rows = [x for x in p[which] if x['s'] == s]
    if not rows: raise ValueError('no policy')
    return next((x['p'] for x in rows if x['a'] == a), 0.0)

def expect_mf(p, part, ctx):
    task = part['task']; cells = {}; last = None; g = p['gamma']; A = [a['id'] for a in p['actions']]
    if task in ('mc_prediction_first_visit', 'mc_prediction_every_visit'):
        first = task.endswith('first_visit'); rets = {}
        for e, ep in eps_of(p, part):
            T_ = len(ep['steps']); G = [0.0] * (T_ + 1)
            for t in range(T_ - 1, -1, -1):
                G[t] = ep['steps'][t]['r'] + (g * G[t + 1] if t < T_ - 1 else 0)
                cells[(rf'\text{{Episode {e+1} returns }} G_t', 'G', f't = {t}')] = G[t]
            seen = set()
            for t, x in enumerate(ep['steps']):
                if first and x['s'] in seen: continue
                seen.add(x['s']); rets.setdefault(x['s'], []).append(G[t])
            for s in dict.fromkeys(x['s'] for x in ep['steps']):
                cells[(r'\text{Monte Carlo } V(s)', 'V', tex(s))] = sum(rets[s]) / len(rets[s])
    elif task == 'mc_control_epsilon_greedy':
        Q = Qinit(p); rets = {}
        for e, ep in eps_of(p, part):
            T_ = len(ep['steps']); G = [0.0] * (T_ + 1)
            for t in range(T_ - 1, -1, -1): G[t] = ep['steps'][t]['r'] + (g * G[t + 1] if t < T_ - 1 else 0)
            seen = set()
            for t, x in enumerate(ep['steps']):
                k = (x['s'], x['a'])
                if k in seen: continue
                seen.add(k); rets.setdefault(k, []).append(G[t]); Q[k] = sum(rets[k]) / len(rets[k])
            for s in dict.fromkeys(x['s'] for x in ep['steps']):
                gA = argmax_first([(a, Q.get((s, a), 0.0)) for a in A])
                if p.get('epsilon') is None:
                    cells[(r'\text{Improved policy } \pi(a\mid s)', tex(s), 'a^{*}')] = tex(gA)
                else:
                    e_ = p['epsilon']
                    for a in A: cells[(r'\text{Improved policy } \pi(a\mid s)', tex(s), tex(a))] = (1 - e_ + e_ / len(A)) if a == gA else e_ / len(A)
        for (s, a), v in Q.items(): cells[(r'\text{Monte Carlo } Q(s,a)', tex(s), tex(a))] = v
    elif task == 'td0_prediction':
        V = V0(p); al = p['alpha']
        for e, ep in eps_of(p, part):
            for t, x in enumerate(ep['steps']):
                s2 = nxt(p, ep, t)
                V[x['s']] += al * (x['r'] + g * (V.get(s2, 0.0) if s2 else 0) - V[x['s']])
        for s, v in V.items(): cells[(r'\text{TD(0) } V(s)', 'V', tex(s))] = v
    elif task in ('sarsa', 'q_learning'):
        Q = Qinit(p); al = p['alpha']
        for e, ep in eps_of(p, part):
            for t, x in enumerate(ep['steps']):
                s2 = nxt(p, ep, t)
                if task == 'sarsa':
                    a2 = act_at(ep, t + 1)
                    nq = Q.get((s2, a2), 0.0) if s2 else 0.0
                else:
                    nq = max(Q.get((s2, a), 0.0) for a in A) if s2 else 0.0
                k = (x['s'], x['a']); Q[k] = Q.get(k, 0.0) + al * (x['r'] + g * nq - Q.get(k, 0.0))
        name = r'\text{SARSA } Q(s,a)' if task == 'sarsa' else r'\text{Q-learning } Q(s,a)'
        for (s, a), v in Q.items(): cells[(name, tex(s), tex(a))] = v
    elif task in ('nstep_td', 'nstep_sarsa', 'nstep_offpolicy_is', 'tree_backup'):
        n = p['n']; al = p['alpha']; V = V0(p); Q = Qinit(p)
        for e, ep in eps_of(p, part):
            T_ = len(ep['steps'])
            for tau in range(T_):
                if task == 'tree_backup':
                    h = min(tau + n, T_); G = None
                    for k in range(h - 1, tau - 1, -1):
                        r = ep['steps'][k]['r']; s1 = state_at(p, ep, k + 1)
                        if not s1: G = r
                        elif k == h - 1: G = r + g * sum(prob(p, 'policy', s1, a) * Q.get((s1, a), 0.0) for a in A)
                        else:
                            a1 = act_at(ep, k + 1)
                            G = r + g * sum(prob(p, 'policy', s1, a) * Q.get((s1, a), 0.0) for a in A if a != a1) + g * prob(p, 'policy', s1, a1) * G
                else:
                    h = min(tau + n, T_)
                    G = sum(g ** (i - tau) * ep['steps'][i]['r'] for i in range(tau, h))
                    if tau + n <= T_:
                        sb = state_at(p, ep, tau + n)
                        if sb:
                            G += g ** n * (V.get(sb, 0.0) if task == 'nstep_td' else Q.get((sb, act_at(ep, tau + n)), 0.0))
                x = ep['steps'][tau]
                if task == 'nstep_td':
                    V[x['s']] += al * (G - V[x['s']])
                else:
                    rho = 1.0
                    if task == 'nstep_offpolicy_is':
                        for k in range(tau + 1, min(tau + n, T_ - 1) + 1):
                            st = ep['steps'][k]; rho *= prob(p, 'policy', st['s'], st['a']) / prob(p, 'behavior', st['s'], st['a'])
                    kk = (x['s'], x['a']); Q[kk] = Q.get(kk, 0.0) + al * rho * (G - Q.get(kk, 0.0))
        if task == 'nstep_td':
            for s, v in V.items(): cells[(r'\text{n-step TD } V(s)', 'V', tex(s))] = v
        else:
            name = {'nstep_sarsa': r'\text{n-step SARSA } Q(s,a)', 'nstep_offpolicy_is': r'\text{Off-policy n-step SARSA } Q(s,a)', 'tree_backup': r'\text{Tree backup } Q(s,a)'}[task]
            for (s, a), v in Q.items(): cells[(name, tex(s), tex(a))] = v
    elif task == 'dyna_q':
        Q = Qinit(p); al = p['alpha']; n = p.get('n') or len(p.get('planning', [])); model = {}; cur = 0
        def upd(s, a, r, s2):
            nq = max(Q.get((s2, b), 0.0) for b in A) if s2 else 0.0
            Q[(s, a)] = Q.get((s, a), 0.0) + al * (r + g * nq - Q.get((s, a), 0.0))
        for e, ep in eps_of(p, part):
            for t, x in enumerate(ep['steps']):
                s2 = nxt(p, ep, t); upd(x['s'], x['a'], x['r'], s2); model[(x['s'], x['a'])] = (x['r'], s2)
                for _ in range(n):
                    keys = list(model)
                    k = (p['planning'][cur]['s'], p['planning'][cur]['a']) if p.get('planning') else keys[cur % len(keys)]
                    cur += 1; r, s2_ = model[k]; upd(k[0], k[1], r, s2_)
        for (s, a), v in Q.items(): cells[(r'\text{Dyna-Q } Q(s,a)', tex(s), tex(a))] = v
    elif task == 'dqn_target_loss':
        tgt = p['Qtarget'] or p['Q']; errs = []
        for i, x in enumerate(p['replay']):
            y = x['r'] if x.get('done') else x['r'] + g * max(q['v'] for q in tgt if q['s'] == x['s2'])
            cells[(r'\text{DQN minibatch}', f'i = {i+1}', 'y_i')] = y
            qv = next(q['v'] for q in p['Q'] if q['s'] == x['s'] and q['a'] == x['a']); errs.append((y - qv) ** 2)
        cells[(r'\text{DQN minibatch}', r'\text{loss}', 'L')] = sum(errs) / len(errs)
    elif task in ('bandit_sample_average', 'bandit_incremental', 'bandit_epsilon_greedy', 'bandit_ucb', 'ab_test'):
        arms = [a['id'] for a in p['actions']]; B = r'\text{Bandit estimates}'
        snap = {x['a']: x for x in p.get('arms', [])}
        def sample_avg():
            Qa = {}; N = {}
            for a in arms:
                rs = [x['r'] for x in p['pulls'] if x['a'] == a]; N[a] = len(rs); Qa[a] = sum(rs) / len(rs) if rs else 0.0
            return Qa, N
        if task == 'bandit_sample_average':
            Qa, N = sample_avg()
            for a in arms: cells[(B, 'Q(a)', tex(a))] = Qa[a]; cells[(B, 'N(a)', tex(a))] = N[a]
        elif task == 'bandit_incremental':
            Qa = {a: snap.get(a, {}).get('q', 0) or 0 for a in arms}; N = {a: snap.get(a, {}).get('n', 0) or 0 for a in arms}
            for x in p['pulls']:
                N[x['a']] += 1; step = p['alpha'] if p.get('alpha') is not None else 1 / N[x['a']]
                Qa[x['a']] += step * (x['r'] - Qa[x['a']])
            for a in arms: cells[(B, 'Q(a)', tex(a))] = Qa[a]; cells[(B, 'N(a)', tex(a))] = N[a]
        elif task in ('bandit_epsilon_greedy', 'bandit_ucb'):
            if all(snap.get(a, {}).get('q') is not None for a in arms):
                Qa = {a: snap[a]['q'] for a in arms}; N = {a: snap[a].get('n') for a in arms}
            else: Qa, N = sample_avg()
            if task == 'bandit_epsilon_greedy':
                e_ = p['epsilon']; gA = argmax_first([(a, Qa[a]) for a in arms])
                for a in arms: cells[(r'\varepsilon\text{-greedy } P(a)', 'P', tex(a))] = (1 - e_ + e_ / len(arms)) if a == gA else e_ / len(arms)
            else:
                t = p.get('t') or sum(N.values()); sc = []
                for a in arms:
                    if N[a] == 0: sc.append((a, float('inf'))); continue
                    v = Qa[a] + p['c'] * math.sqrt(math.log(t) / N[a]); sc.append((a, v))
                    cells[(r'\text{UCB scores}', r'\text{UCB}', tex(a))] = v
                last = [tex(argmax_first([(a, 1e308 if v == float('inf') else v) for a, v in sc]))]
        else:
            rows = [(a, snap[a]['n'], snap[a]['wins']) for a in arms]
            for a, n_, w in rows: cells[(r'\text{A/B test}', r'\hat{p}', tex(a))] = w / n_
            if len(rows) == 2:
                (a1, n1, w1), (a2, n2, w2) = rows; pool = (w1 + w2) / (n1 + n2)
                se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
                cells[(r'\text{A/B test}', r'\text{test}', r'\hat{p}_{\text{pool}}')] = pool
                cells[(r'\text{A/B test}', r'\text{test}', 'SE')] = se
                cells[(r'\text{A/B test}', r'\text{test}', 'z')] = (w1 / n1 - w2 / n2) / se
    elif task == 'theory':
        pass
    else:
        raise ValueError(task)
    return cells, last

# ---------- LaTeX line -> python expression ----------
def unwrap(s, cmd):
    # \cmd{arg}{body} -> body, brace-matched so bodies may hold \dfrac{..}{..} etc.
    head = re.compile(re.escape(cmd) + r'\{[^{}]*\}\{')
    while (m := head.search(s)):
        depth, j = 1, m.end()
        while depth:
            depth += {'{': 1, '}': -1}.get(s[j], 0)
            j += 1
        s = s[:m.start()] + s[m.end():j - 1] + s[j:]
    return s
def latex_value(rhs):
    s = unwrap(unwrap(rhs, r'\htmlData'), r'\htmlClass')
    if re.search(r'[A-Za-z]_|V\(|Q\(|\\arg|\\underset|\\infty|\\text|\\\{|\\le|>|\\hat|\\pi', s): return None
    s = s.replace(r'\left', '').replace(r'\right', '').replace(r'\times', '*').replace(r',\ ', ',').replace(r'\ ', '')
    s = s.replace(r'\max', 'max').replace(r'\ln', 'math.log').replace(r'\,', '')
    for _ in range(6):
        s = re.sub(r'\\dfrac\{([^{}]*)\}\{([^{}]*)\}', r'((\1)/(\2))', s)
        s = re.sub(r'\\sqrt\{([^{}]*)\}', r'math.sqrt(\1)', s)
        s = re.sub(r'\^\{([^{}]*)\}', r'**(\1)', s)
    if re.search(r'[\\{}]', s): return None
    try: return float(eval(s, {'math': math, 'max': max}))
    except Exception: return None

def fmt(v):
    r = round(v * 1e4) / 1e4
    return ('%.10g' % (0 if r == 0 else r))

# ---------- random problems ----------
def rnd_probs(rng, ids, k=None):
    k = k or rng.randint(1, min(3, len(ids)))
    chosen = rng.sample(ids, k); cuts = sorted(rng.sample(range(1, 10), k - 1)); parts = [b - a for a, b in zip([0] + cuts, cuts + [10])]
    return [(c, w / 10) for c, w in zip(chosen, parts)]

def gen(rng, task):
    nS = rng.randint(2, 4); S = [f'S{i+1}' for i in range(nS)]; A = ['A1', 'A2'] + (['A3'] if rng.random() < 0.3 else [])
    term = [S[-1]] if rng.random() < 0.5 else []
    NT = [s for s in S if s not in term]
    gamma = rng.choice([0.5, 0.8, 0.9])
    p = dict(states=[{'id': s} for s in S], actions=[{'id': a} for a in A], terminal=term, gamma=gamma,
             transitions=[], policy=[], V0=[], Q=[], parts=[])
    for s in NT:
        for a in A:
            for s2, pr in rnd_probs(rng, S): p['transitions'].append(dict(s=s, a=a, s2=s2, p=pr, r=rng.randint(-10, 10)))
        if rng.random() < 0.6: p['policy'].append(dict(s=s, a=rng.choice(A), p=1))
        else:
            for a, pr in rnd_probs(rng, A, 2): p['policy'].append(dict(s=s, a=a, p=pr))
    if rng.random() < 0.5: p['V0'] = [dict(s=s, v=rng.randint(-3, 3)) for s in NT]
    p['Q'] = [dict(s=s, a=a, v=rng.randint(-5, 5)) for s in NT for a in A] if rng.random() < 0.6 else []
    p['epsilon'] = rng.choice([0.1, 0.2, 0.3]); p['alpha'] = rng.choice([0.1, 0.5, 1]); p['n'] = rng.randint(1, 3)
    p['episodes'] = []
    for _ in range(rng.randint(1, 2)):
        steps = [dict(s=rng.choice(NT), a=rng.choice(A), r=rng.randint(-5, 10)) for _ in range(rng.randint(1, 5))]
        end = rng.choice(term + [''] + NT) if rng.random() < 0.7 else ''
        ep = dict(steps=steps, end=end, endAction=rng.choice(A) if end in NT else '')
        p['episodes'].append(ep)
    p['behavior'] = [dict(s=s, a=a, p=pr) for s in NT for a, pr in zip(A, [round(1 / len(A), 10)] * len(A))]
    if len(A) == 3: p['behavior'] = [dict(s=s, a=a, p=pr) for s in NT for a, pr in zip(A, [0.4, 0.3, 0.3])]
    part = dict(task=task, iterations=rng.randint(1, 3))
    if task in ('greedy_action', 'epsilon_greedy_probs') and not p['Q']: p['Q'] = [dict(s=s, a=a, v=rng.randint(-5, 5)) for s in NT for a in A]
    if task == 'return_Gt': part['rewards'] = [rng.randint(-5, 10) for _ in range(rng.randint(1, 5))]
    if task == 'policy_iteration':
        p['policy'] = [dict(s=s, a=rng.choice(A), p=1) for s in NT]
    if task.startswith('bandit') or task == 'ab_test':
        p = dict(actions=[{'id': a} for a in A], parts=[], epsilon=p['epsilon'], alpha=p['alpha'] if rng.random() < 0.3 else None,
                 c=rng.choice([1, 2]), pulls=[dict(a=rng.choice(A), r=rng.randint(0, 5)) for _ in range(rng.randint(len(A), 8))], arms=[])
        if p['alpha'] is None: del p['alpha']
        if task == 'ab_test': p['pulls'] = []; p['arms'] = [dict(a=a, n=rng.randint(100, 2000), wins=rng.randint(5, 90)) for a in A[:2]]; p['actions'] = p['actions'][:2]
        elif task in ('bandit_ucb', 'bandit_epsilon_greedy') and rng.random() < 0.4:
            p['arms'] = [dict(a=a, q=rng.randint(0, 5), n=rng.randint(0, 6)) for a in A]
        if task == 'bandit_ucb' and rng.random() < 0.5: p['t'] = rng.randint(10, 20)
    if task == 'dqn_target_loss':
        p['Q'] = [dict(s=s, a=a, v=rng.randint(-3, 3)) for s in S for a in A]
        p['Qtarget'] = [dict(s=s, a=a, v=rng.randint(-3, 3)) for s in S for a in A] if rng.random() < 0.7 else []
        p['replay'] = []
        for _ in range(rng.randint(1, 4)):
            done = rng.random() < 0.3
            p['replay'].append(dict(s=rng.choice(S), a=rng.choice(A), r=rng.randint(-2, 5), s2='' if done else rng.choice(S), done=done))
    if task == 'tree_backup':
        p['policy'] = [dict(s=s, a=a, p=pr) for s in NT for a, pr in rnd_probs(rng, A, len(A))]
    if task == 'nstep_offpolicy_is':
        p['policy'] = [dict(s=s, a=a, p=pr) for s in NT for a, pr in rnd_probs(rng, A, rng.randint(1, len(A)))]
    p['parts'] = [part]
    return p

DP_TASKS = ['bellman_v', 'q_from_v', 'bellman_optimality', 'policy_evaluation', 'policy_evaluation_inplace', 'policy_evaluation_linear',
            'policy_improvement', 'policy_iteration', 'value_iteration', 'greedy_action', 'epsilon_greedy_probs', 'best_state', 'return_Gt']
MF_TASKS = ['mc_prediction_first_visit', 'mc_prediction_every_visit', 'mc_control_epsilon_greedy', 'td0_prediction', 'sarsa', 'q_learning',
            'bandit_sample_average', 'bandit_incremental', 'bandit_epsilon_greedy', 'bandit_ucb', 'ab_test',
            'nstep_td', 'nstep_sarsa', 'nstep_offpolicy_is', 'tree_backup', 'dyna_q', 'dqn_target_loss']

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'gen':
        rng = random.Random(int(sys.argv[3])); out = []
        for i in range(int(sys.argv[4])):
            out.append(gen(rng, (DP_TASKS + MF_TASKS)[i % len(DP_TASKS + MF_TASKS)]))
        import os; os.makedirs(os.path.dirname(sys.argv[2]) or '.', exist_ok=True)
        json.dump(out, open(sys.argv[2], 'w'))
    else:  # check problems.json dump.json
        probs = json.load(open(sys.argv[2])); dump = json.load(open(sys.argv[3]))
        stats = dict(problems=0, invalid=0, errors=0, cells=0, cell_bad=0, lines=0, line_bad=0, result_bad=0)
        shown = 0
        for p, d in zip(probs, dump):
            if 'invalid' in d: stats['invalid'] += 1; continue
            stats['problems'] += 1; ctx = {}
            p.setdefault('Q', []); p.setdefault('policy', []); p.setdefault('Qtarget', []); p.setdefault('replay', []); p.setdefault('planning', []); p.setdefault('behavior', [])
            p.setdefault('gamma', 0.9); p.setdefault('states', []); p.setdefault('transitions', []); p.setdefault('terminal', [])
            for part, got in zip(p['parts'], d['parts']):
                try:
                    want, last = expect_part(p, part, ctx)
                except (ValueError, np.linalg.LinAlgError, ZeroDivisionError, KeyError, StopIteration) as ex:
                    want, last = None, None
                if got.get('error'):
                    if want is not None:
                        stats['errors'] += 1
                        if shown < 15: print('APP ERROR but reference solved:', part['task'], got['error']); shown += 1
                    continue
                if want is None:
                    if shown < 15: print('reference failed, app solved:', part['task']); shown += 1
                    continue
                gotc = {}
                for st in got['steps']:
                    for r in st['results']: gotc[(r['cell']['table'], r['cell']['row'], r['cell']['col'])] = r['value']
                for k, v in want.items():
                    stats['cells'] += 1
                    gv = gotc.get(k)
                    ok = (gv == v) if isinstance(v, str) else (isinstance(gv, (int, float)) and abs(gv - v) < 1e-7 * max(1, abs(v)))
                    if not ok:
                        stats['cell_bad'] += 1
                        if True: print('MISMATCH', part['task'], k, 'app=', gv, 'ref=', v); shown += 1
                if last is not None:
                    lasts = [st['lines'][-1]['rhs'] for st in got['steps'] if st['lines']]
                    tail = lasts[-len(last):]
                    for a, b in zip(tail, last):
                        stats['cells'] += 1
                        ok = a == b if isinstance(b, str) else abs(float(unwrap(unwrap(a, r'\htmlData'), r'\htmlClass')) - b) < 1e-3
                        if not ok:
                            stats['cell_bad'] += 1
                            if shown < 15: print('LAST MISMATCH', part['task'], a, b); shown += 1
                # every displayed line must equal the step's final value
                for st in got['steps']:
                    vals = []
                    for ln in st['lines']:
                        v = latex_value(ln['rhs'])
                        if ln.get('lhs') is not None and vals: vals = []  # new equation block
                        if v is not None: vals.append((v, ln['rhs']))
                    if len(vals) >= 2:
                        final = vals[-1][0]
                        for v, rhs in vals:
                            stats['lines'] += 1
                            if abs(v - final) > 1e-4 * abs(final) + 1e-9:  # displayed numbers carry 6 significant digits
                                stats['line_bad'] += 1
                                if shown < 40: print('LINE INCONSISTENT', part['task'], st['title'], rhs, '=', v, 'final', final); shown += 1
                    nums = [r['value'] for r in st['results'] if isinstance(r['value'], (int, float))]
                    if vals and len(nums) == 1 and st['lines'][-1].get('lhs') is None:
                        if abs(vals[-1][0] - nums[0]) > 1e-4 * max(1, abs(nums[0])):
                            stats['result_bad'] += 1
                            if shown < 15: print('RESULT != LAST LINE', part['task'], st['title'], vals[-1][0], nums[0]); shown += 1
        print(json.dumps(stats))
        sys.exit(1 if stats['invalid'] or stats['errors'] or stats['cell_bad'] or stats['line_bad'] or stats['result_bad'] else 0)
