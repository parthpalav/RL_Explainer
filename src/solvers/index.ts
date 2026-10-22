import type { Problem, Task } from '../schema';
import type { Ctx, PartTrace, Solver } from './types';
import { DP } from './dp';
import { MODEL_FREE } from './modelfree';
import { BANDIT } from './bandit';

export const SOLVERS: Record<Task, Solver> = { ...DP, ...MODEL_FREE, ...BANDIT };

export const HEADINGS: Record<Task, string> = {
  return_Gt: 'Discounted return',
  bellman_v: 'Bellman expectation backup',
  q_from_v: 'Action values from state values',
  bellman_optimality: 'Bellman optimality backup',
  policy_evaluation: 'Iterative policy evaluation',
  policy_evaluation_inplace: 'In-place (asynchronous) policy evaluation',
  policy_evaluation_linear: 'Exact policy evaluation (linear system)',
  policy_improvement: 'Policy improvement',
  policy_iteration: 'Policy iteration',
  value_iteration: 'Value iteration',
  greedy_action: 'Greedy action selection',
  epsilon_greedy_probs: 'ε-greedy action probabilities',
  best_state: 'Compare state values',
  mc_prediction_first_visit: 'First-visit Monte Carlo prediction',
  mc_prediction_every_visit: 'Every-visit Monte Carlo prediction',
  mc_control_epsilon_greedy: 'Monte Carlo control (ε-greedy)',
  td0_prediction: 'TD(0) prediction',
  sarsa: 'SARSA (on-policy TD control)',
  q_learning: 'Q-learning (off-policy TD control)',
  bandit_sample_average: 'Bandit: sample-average estimates',
  bandit_incremental: 'Bandit: incremental update',
  bandit_epsilon_greedy: 'Bandit: ε-greedy arm probabilities',
  bandit_ucb: 'Bandit: UCB arm selection',
  ab_test: 'A/B test',
  theory: 'Theory',
};

/** Solve parts in order; later parts may continue from earlier ones (e.g. V1 then V2). */
export function solveAll(p: Problem): PartTrace[] {
  const ctx: Ctx = {};
  return p.parts.map((part) => {
    const heading = HEADINGS[part.task];
    try {
      return { part, heading, steps: SOLVERS[part.task](p, part, ctx) };
    } catch (e) {
      return { part, heading, steps: [], error: (e as Error).message };
    }
  });
}
