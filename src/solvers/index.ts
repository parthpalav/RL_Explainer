import type { Problem, Task } from '../schema';
import type { Ctx, PartTrace, Solver } from './types';
import { DP } from './dp';
import { MODEL_FREE } from './modelfree';
import { BANDIT } from './bandit';
import { NSTEP } from './nstep';

export const SOLVERS: Record<Task, Solver> = { ...DP, ...MODEL_FREE, ...BANDIT, ...NSTEP };

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
  nstep_td: 'n-step TD prediction',
  nstep_sarsa: 'n-step SARSA',
  nstep_offpolicy_is: 'Off-policy n-step SARSA (importance sampling)',
  tree_backup: 'n-step tree backup',
  dyna_q: 'Dyna-Q (planning + learning)',
  dqn_target_loss: 'DQN targets and loss',
  theory: 'Theory',
};

/** Which family of methods each task belongs to, and what it does in one sentence (shown above the steps). */
export const ABOUT: Record<Task, { family: string; what: string }> = {
  return_Gt: { family: 'Returns', what: 'Adds up the rewards that follow, each discounted by γ once per step of delay.' },
  bellman_v: { family: 'Dynamic programming', what: 'One Bellman expectation backup: the value of each state under the policy, from the current estimates.' },
  q_from_v: { family: 'Dynamic programming', what: 'Turns state values into action values with a one-step lookahead through the model.' },
  bellman_optimality: { family: 'Dynamic programming', what: 'One Bellman optimality backup: each state takes the value of its best action.' },
  policy_evaluation: { family: 'Dynamic programming', what: 'Sweeps the Bellman expectation backup over all states, using only last sweep’s values.' },
  policy_evaluation_inplace: { family: 'Dynamic programming', what: 'Sweeps the Bellman expectation backup, using new values as soon as they are computed.' },
  policy_evaluation_linear: { family: 'Dynamic programming', what: 'Solves the Bellman equations as a linear system to get the exact value of the policy.' },
  policy_improvement: { family: 'Dynamic programming', what: 'Makes the policy greedy with respect to the current value function.' },
  policy_iteration: { family: 'Dynamic programming', what: 'Alternates exact policy evaluation and greedy improvement until the policy stops changing.' },
  value_iteration: { family: 'Dynamic programming', what: 'Repeats the Bellman optimality backup, then reads off the greedy policy.' },
  greedy_action: { family: 'Action selection', what: 'Picks the action with the highest Q-value.' },
  epsilon_greedy_probs: { family: 'Action selection', what: 'Gives the greedy action 1 − ε + ε/|A| and every other action ε/|A|.' },
  best_state: { family: 'Comparison', what: 'Compares state values and picks the largest.' },
  mc_prediction_first_visit: { family: 'Monte Carlo', what: 'Averages the return that follows the first visit to each state in every episode.' },
  mc_prediction_every_visit: { family: 'Monte Carlo', what: 'Averages the return that follows every visit to each state.' },
  mc_control_epsilon_greedy: { family: 'Monte Carlo', what: 'Averages returns per (state, action), then makes the policy ε-greedy on them.' },
  td0_prediction: { family: 'Temporal difference', what: 'Moves V(s) a step α toward the one-step target R + γV(s′) after every transition.' },
  sarsa: { family: 'Temporal difference', what: 'On-policy control: the target uses the action actually taken next, R + γQ(s′, a′).' },
  q_learning: { family: 'Temporal difference', what: 'Off-policy control: the target uses the best next action, R + γ maxₐ Q(s′, a).' },
  bandit_sample_average: { family: 'Multi-armed bandit', what: 'Estimates each arm as the average of the rewards it has paid out.' },
  bandit_incremental: { family: 'Multi-armed bandit', what: 'Updates an arm’s average after each pull without storing past rewards.' },
  bandit_epsilon_greedy: { family: 'Multi-armed bandit', what: 'Exploits the best arm with probability 1 − ε and explores at random otherwise.' },
  bandit_ucb: { family: 'Multi-armed bandit', what: 'Picks the arm with the highest estimate plus an uncertainty bonus.' },
  ab_test: { family: 'Statistics', what: 'Two-proportion z-test: is the difference in conversion rates bigger than chance?' },
  nstep_td: { family: 'n-step bootstrapping', what: 'Uses n real rewards, then bootstraps from V of the state n steps later.' },
  nstep_sarsa: { family: 'n-step bootstrapping', what: 'Uses n real rewards, then bootstraps from Q of the (state, action) n steps later.' },
  nstep_offpolicy_is: { family: 'n-step bootstrapping', what: 'n-step SARSA from behaviour-policy data, reweighted by the importance-sampling ratio π/b.' },
  tree_backup: { family: 'n-step bootstrapping', what: 'Off-policy without importance sampling: untaken actions contribute their expected value under π.' },
  dyna_q: { family: 'Planning', what: 'Q-learning on each real step, plus extra Q-learning updates replayed from a learned model.' },
  dqn_target_loss: { family: 'Deep RL', what: 'Builds targets r + γ maxₐ Q̂(s′, a) from the target network and the squared error against Q.' },
  theory: { family: 'Theory', what: 'A written answer, with the key ideas laid out.' },
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
