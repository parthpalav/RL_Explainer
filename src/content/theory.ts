// Short revision cards for explain / compare questions. Text may contain $...$ math.

export interface Card {
  id: string;
  title: string;
  match: RegExp; // matched against the sub-question text
  intro?: string;
  table?: { head: string[]; rows: string[][] };
  points?: string[];
}

export const CARDS: Card[] = [
  {
    id: 'pi_vs_vi',
    title: 'Policy iteration vs value iteration',
    match: /policy\s+iteration[\s\S]*value\s+iteration|value\s+iteration[\s\S]*policy\s+iteration/i,
    intro: 'Both are dynamic programming methods: they need the full model $P(s\'|s,a)$ and $R$, and both converge to the optimal policy.',
    table: {
      head: ['', 'Policy iteration', 'Value iteration'],
      rows: [
        ['Core idea', 'Alternate full policy evaluation and greedy policy improvement until the policy stops changing.', 'Repeatedly apply the Bellman optimality backup $V(s) \\leftarrow \\max_a \\sum_{s\'} P\\,[R + \\gamma V(s\')]$.'],
        ['Work per iteration', 'High: evaluation solves a linear system (or many sweeps).', 'Low: one sweep with a max over actions.'],
        ['Iterations to converge', 'Few: often a handful of improvements.', 'Many: values converge gradually.'],
        ['Policy', 'Explicit policy kept and improved every iteration.', 'Policy is read off at the end by acting greedily on $V$.'],
        ['Stopping rule', 'Policy is stable (unchanged after improvement).', 'Largest value change in a sweep is below a small threshold.'],
      ],
    },
  },
  {
    id: 'mc_vs_td',
    title: 'Monte Carlo vs temporal-difference learning',
    match: /monte\s*carlo[\s\S]*\bTD\b|\bTD\b[\s\S]*monte\s*carlo|\bMC\b[\s\S]*\bTD\b|advantages?\s+of\s+TD|temporal[-\s]difference/i,
    table: {
      head: ['', 'Monte Carlo', 'TD(0)'],
      rows: [
        ['Update target', 'Actual return $G_t$', '$R_{t+1} + \\gamma V(S_{t+1})$ (bootstraps on its own estimate)'],
        ['When it updates', 'Only after the episode ends', 'After every step (online)'],
        ['Tasks', 'Episodic only', 'Episodic and continuing'],
        ['Bias / variance', 'Unbiased, high variance', 'Biased (by the estimate), lower variance'],
        ['Uses the Markov property', 'No', 'Yes, which usually makes it learn faster'],
      ],
    },
    points: ['Neither needs a model of the environment: both learn from sampled experience.'],
  },
  {
    id: 'on_vs_off',
    title: 'On-policy vs off-policy (SARSA vs Q-learning)',
    match: /on[-\s]policy|off[-\s]policy|sarsa[\s\S]*q[-\s]?learning|q[-\s]?learning[\s\S]*sarsa/i,
    table: {
      head: ['', 'SARSA (on-policy)', 'Q-learning (off-policy)'],
      rows: [
        ['Target', '$R + \\gamma Q(S\', A\')$, with $A\'$ the action actually taken next', '$R + \\gamma \\max_a Q(S\', a)$'],
        ['Learns the value of', 'The policy it follows (e.g. ε-greedy)', 'The greedy (optimal) policy, whatever it follows'],
        ['Behaviour while learning', 'Safer: accounts for its own exploration (cliff walking: takes the safe path)', 'Riskier: learns the optimal path but explores next to danger'],
        ['Data', 'Must come from the current policy', 'Can come from any behaviour policy, e.g. old data or a human'],
      ],
    },
  },
  {
    id: 'explore',
    title: 'Exploration–exploitation dilemma',
    match: /explor|exploit|dilemma/i,
    intro: 'Exploiting picks the action that looks best now. Exploring tries other actions to find out whether they are better. Doing only one fails: pure exploitation can lock onto a bad action, pure exploration never cashes in.',
    points: [
      'ε-greedy: exploit with probability $1-\\varepsilon$, pick uniformly at random with probability $\\varepsilon$. ε is often decayed over time.',
      'Optimistic initial values: start estimates high so every action gets tried.',
      'UCB: pick $\\arg\\max_a \\big[Q(a) + c\\sqrt{\\ln t / N(a)}\\big]$: the bonus favours rarely tried actions.',
      'Softmax / gradient bandits: choose actions with probabilities that grow with their estimated value.',
    ],
  },
  {
    id: 'elements',
    title: 'Elements of reinforcement learning',
    match: /elements?|components?|limitations?|scope|agent[\s\S]*environment/i,
    points: [
      'Agent and environment interact in a loop: the agent sees state $S_t$, takes action $A_t$, gets reward $R_{t+1}$ and next state $S_{t+1}$.',
      'Policy $\\pi$: how the agent chooses actions in each state.',
      'Reward signal: the immediate goal. The agent maximises total reward over time.',
      'Value function: expected long-term return from a state (or state-action pair). This is what the agent actually optimises.',
      'Model (optional): predicts the next state and reward. Model-based methods plan with it; model-free methods learn from experience only.',
      'Limitations: needs many interactions, depends on how states and rewards are designed, can be unsafe while exploring, and struggles with very large state spaces without function approximation.',
    ],
  },
  {
    id: 'mdp',
    title: 'MDPs and Bellman equations',
    match: /\bMDP\b|markov|bellman|value\s+functions?|returns?\b|discount/i,
    points: [
      'An MDP is $(S, A, P, R, \\gamma)$: states, actions, transition probabilities $P(s\'|s,a)$, rewards and a discount factor.',
      'Markov property: the next state and reward depend only on the current state and action, not on the history.',
      'Return: $G_t = R_{t+1} + \\gamma R_{t+2} + \\gamma^2 R_{t+3} + \\cdots$. γ near 0 is short-sighted, near 1 far-sighted.',
      'Bellman expectation: $V^\\pi(s) = \\sum_a \\pi(a|s) \\sum_{s\'} P(s\'|s,a)\\,[R + \\gamma V^\\pi(s\')]$.',
      'Bellman optimality: $V^*(s) = \\max_a \\sum_{s\'} P(s\'|s,a)\\,[R + \\gamma V^*(s\')]$.',
    ],
  },
  {
    id: 'gpi',
    title: 'Generalized policy iteration, asynchronous DP and efficiency',
    match: /generali[sz]ed\s+policy|\bGPI\b|asynchronous|efficien/i,
    points: [
      'GPI: any interleaving of policy evaluation (make $V$ match $\\pi$) and policy improvement (make $\\pi$ greedy on $V$). Almost every RL method fits this pattern.',
      'Asynchronous DP: update states in any order, in place, using the newest values. No full sweeps are needed, so effort can focus on relevant states.',
      'Efficiency: DP finds an optimal policy in time polynomial in $|S|$ and $|A|$, far better than searching all $|A|^{|S|}$ policies. It is still limited by the curse of dimensionality, because $|S|$ grows exponentially with the number of state variables.',
    ],
  },
  {
    id: 'nstep',
    title: 'n-step bootstrapping',
    match: /n[-\s]step|tree\s+backup|importance\s+sampling|eligibility/i,
    points: [
      'n-step methods sit between TD(0) ($n = 1$) and Monte Carlo ($n = \\infty$): they use $n$ real rewards, then bootstrap from the estimate.',
      'An intermediate $n$ often learns fastest: less bias than TD(0), less variance than MC.',
      'Off-policy n-step learning corrects for the behaviour policy with the importance ratio $\\rho = \\prod \\pi(A_k|S_k)/b(A_k|S_k)$, which can have high variance.',
      'Tree backup avoids importance sampling: actions not taken contribute their expected value under $\\pi$.',
    ],
  },
  {
    id: 'dyna',
    title: 'Models and planning: Dyna',
    match: /dyna|planning|model[-\s]based|model[-\s]free/i,
    points: [
      'Model-based methods learn or are given a model and plan with it. Model-free methods learn values directly from experience.',
      'Dyna-Q combines both. Each real step does direct RL (a Q-learning update), model learning (store $(s,a) \\to (r, s\')$), and $n$ planning updates on remembered pairs.',
      'Planning reuses experience, so far fewer real interactions are needed.',
      'Dyna-Q+ adds an exploration bonus for pairs not tried in a long time, to cope with environments that change.',
    ],
  },
  {
    id: 'dqn',
    title: 'Deep Q-learning (DQN)',
    match: /\bDQN\b|deep\s+q|experience\s+replay|target\s+network|deep\s+learning/i,
    points: [
      'A neural network approximates $Q(s, a; \\theta)$, so huge state spaces such as images are possible.',
      'Experience replay: store transitions in a buffer and train on random minibatches. This breaks correlation between consecutive samples and reuses data.',
      'Target network: a periodically copied, frozen network $\\hat{Q}$ computes targets $y = r + \\gamma \\max_{a\'} \\hat{Q}(s\', a\')$, which keeps the targets stable.',
      'Loss: mean squared error $(y - Q(s,a;\\theta))^2$, minimised by gradient descent.',
    ],
  },
  {
    id: 'lifecycle',
    title: 'RL project life cycle',
    match: /life\s*cycle|project|problem\s+definition|debug|future|market|research/i,
    points: [
      'Define the problem: states, actions, reward, episode boundaries and what success means.',
      'Build or choose an environment or simulator, and check the reward cannot be gamed.',
      'Pick an algorithm to match the problem: tabular or function approximation, on- or off-policy, model-based or model-free.',
      'Train, tune hyperparameters (α, γ, ε, network size) and track learning curves.',
      'Evaluate on held-out scenarios, then deploy and monitor for drift.',
      'Debug with simple baselines, check reward scaling, and visualise the agent’s behaviour.',
    ],
  },
  {
    id: 'ab',
    title: 'A/B testing vs bandit algorithms',
    match: /A\/B|bandit/i,
    points: [
      'A/B testing splits traffic in fixed proportions for the whole test, then picks a winner with a significance test.',
      'Bandits adapt the split while running: more traffic goes to the better variant, so less is lost to the worse one (lower regret).',
      'A/B testing gives cleaner statistical conclusions. Bandits are better when exploring is costly or conditions change.',
    ],
  },
];

/** Cards whose topic appears in the sub-question; the first match is the most specific. */
export function cardsFor(text: string): Card[] {
  return CARDS.filter((c) => c.match.test(text)).slice(0, 2);
}
