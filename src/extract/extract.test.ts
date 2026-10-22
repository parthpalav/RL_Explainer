import { describe, expect, it } from 'vitest';
import { crossCheck, detectTasks, splitParts } from './keywords';
import { clean, extract } from './ollama';
import { EXAMPLES } from '../fixtures/ia1';
import { FEWSHOT, FEWSHOT_TEXT } from '../fixtures/fewshot';
import type { Problem } from '../schema';

describe('keywords', () => {
  it('prefers the specific rule', () => {
    expect(detectTasks('Using an ε-greedy policy with ε = 0.2')[0]).toBe('epsilon_greedy_probs');
    expect(detectTasks('Apply policy iteration for 2 iterations')[0]).toBe('policy_iteration');
    expect(detectTasks('Compute V2(S1), V2(S2)')[0]).toBe('policy_evaluation');
  });
  it('splits sub-questions and guesses tasks for every IA1 and few-shot part', () => {
    for (const [text, p] of [...EXAMPLES.map((e) => [e.text, e.problem] as const), [FEWSHOT_TEXT, FEWSHOT] as const]) {
      expect(splitParts(text).map((x) => [x.label, x.task])).toEqual(p.parts.map((x) => [x.label, x.task]));
    }
  });
  it('flags a task that contradicts the part text', () => {
    const parts = EXAMPLES[1].problem.parts;
    expect(crossCheck(parts)).toEqual([]);
    expect(crossCheck([{ ...parts[0], task: 'greedy_action' }])).toHaveLength(1);
  });
});

describe('clean', () => {
  it('takes γ and ε from the text and drops zero-probability policy rows', () => {
    const p = { ...EXAMPLES[1].problem, gamma: 0.5, epsilon: undefined, policy: [{ s: 'S1', a: 'A2', p: 0 }, { s: 'S1', a: 'A1', p: 1 }] };
    const c = clean(p, 'Discount factor (γ): 0.8. Use ε = 0.2 in part e.');
    expect([c.gamma, c.epsilon, c.policy]).toEqual([0.8, 0.2, [{ s: 'S1', a: 'A1', p: 1 }]]);
  });
});

/** The bits that change the maths: ids may be renamed by the model, so compare by value. */
const essence = (p: Problem) => ({
  gamma: p.gamma,
  epsilon: p.epsilon,
  terminal: p.terminal.length,
  transitions: p.transitions.map((t) => [t.p, t.r]).sort(),
  policy: p.policy.map((x) => x.p).sort(),
  Q: p.Q.map((x) => x.v).sort(),
  parts: p.parts.map((x) => [x.task, x.iterations ?? null]),
});

// Real model run: OLLAMA=1 npx vitest run src/extract
describe.skipIf(!import.meta.env.OLLAMA)('LLM extraction of the IA1 questions', () => {
  const model = import.meta.env.OLLAMA_MODEL ?? 'qwen2.5:7b';
  for (const ex of EXAMPLES) {
    it(ex.name, async () => {
      const got = await extract(ex.text, model, { base: 'http://localhost:11434' });
      expect(essence(got)).toEqual(essence(ex.problem));
    }, 300_000);
  }
});
