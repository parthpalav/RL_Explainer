// IA1 example questions. The original question papers were not in the docx files,
// so the text is rebuilt from the given data in the solutions.
import { problem } from '../schema';

export const BATCH1_TEXT = `A smart irrigation agent manages soil moisture. States: S1 (Dry), S2 (Moderate), S3 (Wet). Actions: A1, A2 (Irrigate 20 mm). Discount factor γ = 0.9. Initial values V0(S1) = V0(S2) = V0(S3) = 0. The agent follows policy π(S1) = A1, π(S2) = A1, π(S3) = A1.
Under A1:
From S1: to S1 with probability 0.2 and reward -8; to S2 with probability 0.8 and reward -5.
From S2: to S1 with probability 0.3 and reward -4; to S2 with probability 0.7 and reward 2.
From S3: to S2 with probability 0.4 and reward 3; to S3 with probability 0.6 and reward 5.
a) Using iterative policy evaluation, compute V1(S1), V1(S2), V1(S3).
b) Compute V2(S1), V2(S2), V2(S3).
c) Which soil state gives the highest expected long-term agricultural benefit?
d) Given Q(S1, A1) = 14.2 and Q(S1, A2) = 18.7, which irrigation action does a greedy policy recommend in S1?
e) Using an ε-greedy policy with ε = 0.2, find the probability of each action in S1.`;

export const BATCH2_TEXT = `An image classification agent decides whether to classify an image or improve its quality first. States: S1 (Good quality), S2 (Poor quality), S3 (Terminal). Actions: A1 (Classify), A2 (Improve image quality). Discount factor γ = 0.8. V(S3) = 0.
From S1: A1 goes to S3 with probability 1.0, reward +10. A2 goes to S1 with probability 0.5 and to S2 with probability 0.5, reward +2 each.
From S2: A1 goes to S3 with probability 1.0, reward +4. A2 goes to S1 with probability 0.8 and to S2 with probability 0.2, reward +3 each.
a) Apply policy iteration for 2 iterations starting from π0(S1) = A1, π0(S2) = A1.
b) Apply value iteration for 2 iterations starting from V0 = 0.
c) Differentiate between policy iteration and value iteration.`;

export const BATCH1 = problem({
  title: 'IA1 Batch 1: Smart irrigation',
  states: [
    { id: 'S1', name: 'Dry' },
    { id: 'S2', name: 'Moderate' },
    { id: 'S3', name: 'Wet' },
  ],
  actions: [
    { id: 'A1', name: '' },
    { id: 'A2', name: 'Irrigate 20 mm' },
  ],
  terminal: [],
  gamma: 0.9,
  transitions: [
    { s: 'S1', a: 'A1', s2: 'S1', p: 0.2, r: -8 },
    { s: 'S1', a: 'A1', s2: 'S2', p: 0.8, r: -5 },
    { s: 'S2', a: 'A1', s2: 'S1', p: 0.3, r: -4 },
    { s: 'S2', a: 'A1', s2: 'S2', p: 0.7, r: 2 },
    { s: 'S3', a: 'A1', s2: 'S2', p: 0.4, r: 3 },
    { s: 'S3', a: 'A1', s2: 'S3', p: 0.6, r: 5 },
  ],
  policy: [
    { s: 'S1', a: 'A1', p: 1 },
    { s: 'S2', a: 'A1', p: 1 },
    { s: 'S3', a: 'A1', p: 1 },
  ],
  V0: [
    { s: 'S1', v: 0 },
    { s: 'S2', v: 0 },
    { s: 'S3', v: 0 },
  ],
  Q: [
    { s: 'S1', a: 'A1', v: 14.2 },
    { s: 'S1', a: 'A2', v: 18.7 },
  ],
  epsilon: 0.2,
  parts: [
    { label: 'a', task: 'policy_evaluation', iterations: 1, states: [], rewards: [], text: 'Compute V1(S1), V1(S2), V1(S3).' },
    { label: 'b', task: 'policy_evaluation', iterations: 2, states: [], rewards: [], text: 'Compute V2(S1), V2(S2), V2(S3).' },
    { label: 'c', task: 'best_state', states: [], rewards: [], text: 'Which state has the highest expected long-term benefit?' },
    { label: 'd', task: 'greedy_action', states: ['S1'], rewards: [], text: 'Greedy action in S1.' },
    { label: 'e', task: 'epsilon_greedy_probs', states: ['S1'], rewards: [], text: 'ε-greedy action probabilities in S1 (ε = 0.2).' },
  ],
});

export const BATCH2 = problem({
  title: 'IA1 Batch 2: Image quality',
  states: [
    { id: 'S1', name: 'Good quality' },
    { id: 'S2', name: 'Poor quality' },
    { id: 'S3', name: 'Terminal' },
  ],
  actions: [
    { id: 'A1', name: 'Classify' },
    { id: 'A2', name: 'Improve image quality' },
  ],
  terminal: ['S3'],
  gamma: 0.8,
  transitions: [
    { s: 'S1', a: 'A1', s2: 'S3', p: 1, r: 10 },
    { s: 'S1', a: 'A2', s2: 'S1', p: 0.5, r: 2 },
    { s: 'S1', a: 'A2', s2: 'S2', p: 0.5, r: 2 },
    { s: 'S2', a: 'A1', s2: 'S3', p: 1, r: 4 },
    { s: 'S2', a: 'A2', s2: 'S1', p: 0.8, r: 3 },
    { s: 'S2', a: 'A2', s2: 'S2', p: 0.2, r: 3 },
  ],
  policy: [
    { s: 'S1', a: 'A1', p: 1 },
    { s: 'S2', a: 'A1', p: 1 },
  ],
  V0: [],
  Q: [],
  parts: [
    { label: 'a', task: 'policy_iteration', iterations: 2, states: [], rewards: [], text: 'Policy iteration, 2 iterations.' },
    { label: 'b', task: 'value_iteration', iterations: 2, states: [], rewards: [], text: 'Value iteration, 2 iterations.' },
    { label: 'c', task: 'theory', states: [], rewards: [], text: 'Differentiate between policy iteration and value iteration.' },
  ],
});

import {
  FEWSHOT, FEWSHOT_BANDIT, FEWSHOT_BANDIT_TEXT, FEWSHOT_DQN, FEWSHOT_DQN_TEXT, FEWSHOT_MF, FEWSHOT_MF_TEXT, FEWSHOT_NSTEP, FEWSHOT_NSTEP_TEXT, FEWSHOT_TEXT,
} from './fewshot';

export const EXAMPLES = [
  { name: 'IA1 Batch 1: Smart irrigation', text: BATCH1_TEXT, problem: BATCH1 },
  { name: 'IA1 Batch 2: Image quality', text: BATCH2_TEXT, problem: BATCH2 },
  { name: 'Practice: DP basics (robot battery)', text: FEWSHOT_TEXT, problem: FEWSHOT },
  { name: 'Practice: MC, TD, SARSA, Q-learning', text: FEWSHOT_MF_TEXT, problem: FEWSHOT_MF },
  { name: 'Practice: bandits', text: FEWSHOT_BANDIT_TEXT, problem: FEWSHOT_BANDIT },
  { name: 'Practice: n-step, tree backup, Dyna-Q', text: FEWSHOT_NSTEP_TEXT, problem: FEWSHOT_NSTEP },
  { name: 'Practice: DQN targets and loss', text: FEWSHOT_DQN_TEXT, problem: FEWSHOT_DQN },
];
