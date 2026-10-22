// Worked extraction example shown to the LLM. Deliberately not an IA1 question,
// so the IA1 questions stay a fair test of extraction.
import { problem } from '../schema';

export const FEWSHOT_TEXT = `A rescue robot has battery states S1 (High), S2 (Low) and S3 (Dead, terminal). Actions: A1 (Search), A2 (Wait). γ = 0.9.
From High: Search stays High with probability 0.7 and drops to Low with probability 0.3; reward +5. Wait stays High with probability 1, reward +1.
From Low: Search goes to Dead with probability 0.4, reward -10, and stays Low with probability 0.6, reward +5. Wait stays Low with probability 1, reward +1.
The robot follows a random policy: each action with probability 0.5.
a) Starting from V0 = 0, perform one step of iterative policy evaluation.
b) The robot receives rewards 5, 5, -10. Find the discounted return G0.
c) Perform one sweep of value iteration.
d) Given Q(S2, A1) = 3.5 and Q(S2, A2) = 4.1, find the action probabilities in S2 under an ε-greedy policy with ε = 0.1.
e) Explain the exploration-exploitation dilemma.`;

export const FEWSHOT = problem({
  title: 'Rescue robot battery',
  states: [
    { id: 'S1', name: 'High' },
    { id: 'S2', name: 'Low' },
    { id: 'S3', name: 'Dead' },
  ],
  actions: [
    { id: 'A1', name: 'Search' },
    { id: 'A2', name: 'Wait' },
  ],
  terminal: ['S3'],
  gamma: 0.9,
  transitions: [
    { s: 'S1', a: 'A1', s2: 'S1', p: 0.7, r: 5 },
    { s: 'S1', a: 'A1', s2: 'S2', p: 0.3, r: 5 },
    { s: 'S1', a: 'A2', s2: 'S1', p: 1, r: 1 },
    { s: 'S2', a: 'A1', s2: 'S3', p: 0.4, r: -10 },
    { s: 'S2', a: 'A1', s2: 'S2', p: 0.6, r: 5 },
    { s: 'S2', a: 'A2', s2: 'S2', p: 1, r: 1 },
  ],
  policy: [
    { s: 'S1', a: 'A1', p: 0.5 },
    { s: 'S1', a: 'A2', p: 0.5 },
    { s: 'S2', a: 'A1', p: 0.5 },
    { s: 'S2', a: 'A2', p: 0.5 },
  ],
  V0: [
    { s: 'S1', v: 0 },
    { s: 'S2', v: 0 },
    { s: 'S3', v: 0 },
  ],
  Q: [
    { s: 'S2', a: 'A1', v: 3.5 },
    { s: 'S2', a: 'A2', v: 4.1 },
  ],
  epsilon: 0.1,
  parts: [
    { label: 'a', task: 'policy_evaluation', iterations: 1, states: [], rewards: [], text: 'One step of iterative policy evaluation from V0 = 0.' },
    { label: 'b', task: 'return_Gt', states: [], rewards: [5, 5, -10], text: 'Discounted return for rewards 5, 5, -10.' },
    { label: 'c', task: 'value_iteration', iterations: 1, states: [], rewards: [], text: 'One sweep of value iteration.' },
    { label: 'd', task: 'epsilon_greedy_probs', states: ['S2'], rewards: [], text: 'ε-greedy action probabilities in S2 with ε = 0.1.' },
    { label: 'e', task: 'theory', states: [], rewards: [], text: 'Explain the exploration-exploitation dilemma.' },
  ],
});

export const FEWSHOT_MF_TEXT = `A delivery robot moves between rooms S1, S2, S3; S4 is the terminal charging dock. Actions: L (left), R (right). γ = 0.9, α = 0.1. All state values and Q-values start at 0.
Episode 1: S1, R, +1 → S2, R, +2 → S3, R, +10 → S4.
Episode 2: S2, L, -1 → S1, R, +1 → S2, R, +2 → S3, R, +10 → S4.
a) Estimate V(S1), V(S2), V(S3) with first-visit Monte Carlo using both episodes.
b) Apply TD(0) updates for every step of episode 1.
c) Apply SARSA updates for every step of episode 2.
d) Apply Q-learning updates for every step of episode 1.
e) Why is Q-learning called off-policy?`;

export const FEWSHOT_MF = problem({
  title: 'Delivery robot',
  states: [{ id: 'S1' }, { id: 'S2' }, { id: 'S3' }, { id: 'S4', name: 'Charging dock' }],
  actions: [{ id: 'L', name: 'Left' }, { id: 'R', name: 'Right' }],
  terminal: ['S4'],
  gamma: 0.9,
  alpha: 0.1,
  V0: [{ s: 'S1', v: 0 }, { s: 'S2', v: 0 }, { s: 'S3', v: 0 }, { s: 'S4', v: 0 }],
  episodes: [
    { steps: [{ s: 'S1', a: 'R', r: 1 }, { s: 'S2', a: 'R', r: 2 }, { s: 'S3', a: 'R', r: 10 }], end: 'S4' },
    { steps: [{ s: 'S2', a: 'L', r: -1 }, { s: 'S1', a: 'R', r: 1 }, { s: 'S2', a: 'R', r: 2 }, { s: 'S3', a: 'R', r: 10 }], end: 'S4' },
  ],
  parts: [
    { label: 'a', task: 'mc_prediction_first_visit', text: 'First-visit Monte Carlo estimates of V from both episodes.' },
    { label: 'b', task: 'td0_prediction', episodes: [1], text: 'TD(0) updates for every step of episode 1.' },
    { label: 'c', task: 'sarsa', episodes: [2], text: 'SARSA updates for every step of episode 2.' },
    { label: 'd', task: 'q_learning', episodes: [1], text: 'Q-learning updates for every step of episode 1.' },
    { label: 'e', task: 'theory', text: 'Why is Q-learning called off-policy?' },
  ],
});

export const FEWSHOT_BANDIT_TEXT = `A 3-armed bandit has arms A1, A2, A3. The agent pulled: A1 → reward 1, A2 → reward 0, A1 → reward 2, A3 → reward 4, A2 → reward 1.
a) Compute the sample-average estimate of each arm.
b) Show the incremental update of the estimates after each pull.
c) With ε-greedy (ε = 0.3), give the probability of selecting each arm next.
d) Using UCB with c = 2 at t = 6, which arm is selected?`;

export const FEWSHOT_BANDIT = problem({
  title: '3-armed bandit',
  actions: [{ id: 'A1' }, { id: 'A2' }, { id: 'A3' }],
  pulls: [{ a: 'A1', r: 1 }, { a: 'A2', r: 0 }, { a: 'A1', r: 2 }, { a: 'A3', r: 4 }, { a: 'A2', r: 1 }],
  epsilon: 0.3,
  c: 2,
  t: 6,
  parts: [
    { label: 'a', task: 'bandit_sample_average', text: 'Sample-average estimate of each arm.' },
    { label: 'b', task: 'bandit_incremental', text: 'Incremental update after each pull.' },
    { label: 'c', task: 'bandit_epsilon_greedy', text: 'ε-greedy (ε = 0.3) probability of selecting each arm.' },
    { label: 'd', task: 'bandit_ucb', text: 'UCB with c = 2 at t = 6: which arm is selected?' },
  ],
});
