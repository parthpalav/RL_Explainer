// Worked extraction example shown to the LLM. Deliberately not an IA1 question,
// so the IA1 questions stay a fair test of extraction.
import type { Problem } from '../schema';

export const FEWSHOT_TEXT = `A rescue robot has battery states S1 (High), S2 (Low) and S3 (Dead, terminal). Actions: A1 (Search), A2 (Wait). γ = 0.9.
From High: Search stays High with probability 0.7 and drops to Low with probability 0.3; reward +5. Wait stays High with probability 1, reward +1.
From Low: Search goes to Dead with probability 0.4, reward -10, and stays Low with probability 0.6, reward +5. Wait stays Low with probability 1, reward +1.
The robot follows a random policy: each action with probability 0.5.
a) Starting from V0 = 0, perform one step of iterative policy evaluation.
b) The robot receives rewards 5, 5, -10. Find the discounted return G0.
c) Perform one sweep of value iteration.
d) Given Q(S2, A1) = 3.5 and Q(S2, A2) = 4.1, find the action probabilities in S2 under an ε-greedy policy with ε = 0.1.
e) Explain the exploration-exploitation dilemma.`;

export const FEWSHOT: Problem = {
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
};
