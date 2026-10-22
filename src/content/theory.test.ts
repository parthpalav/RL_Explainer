import { describe, expect, it } from 'vitest';
import { cardsFor } from './theory';
import { close } from '../ui/Predict';

describe('theory cards', () => {
  it('match the theory parts of the examples', () => {
    expect(cardsFor('Differentiate between policy iteration and value iteration.')[0].id).toBe('pi_vs_vi');
    expect(cardsFor('Explain the exploration-exploitation dilemma.')[0].id).toBe('explore');
    expect(cardsFor('Why is Q-learning called off-policy?')[0].id).toBe('on_vs_off');
    expect(cardsFor('What are the advantages of TD prediction methods?')[0].id).toBe('mc_vs_td');
    expect(cardsFor('Explain experience replay in DQN')[0].id).toBe('dqn');
  });
});

describe('predict mode answer check', () => {
  it('accepts hand-rounded answers only', () => {
    expect(close(11.19, 9.4 / 0.84)).toBe(true);
    expect(close(0.9, 0.9000001)).toBe(true);
    expect(close(10.2, 10)).toBe(false);
    expect(close(-6.46, -6.464)).toBe(true);
  });
});
