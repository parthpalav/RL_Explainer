// Keyword detection: cross-checks the LLM's task choice and splits sub-questions offline.
import type { Part, Task } from '../schema';
import { HEADINGS } from '../solvers';

// Most specific first: "ε-greedy" must win over "greedy", "policy iteration" over "policy".
const RULES: [Task, RegExp][] = [
  ['policy_iteration', /policy\s+iteration/i],
  ['value_iteration', /value\s+iteration/i],
  ['policy_evaluation_inplace', /in[-\s]?place|asynchronous|gauss[-\s]?seidel/i],
  ['policy_evaluation_linear', /linear\s+(system|equations)|system\s+of\s+(linear\s+)?equations|solve\s+exactly/i],
  ['policy_improvement', /policy\s+improvement|improve\s+the\s+policy/i],
  ['policy_evaluation', /policy\s+evaluation|evaluate\s+the\s+policy|\bV_?\{?\d\}?\s*\(/i],
  ['epsilon_greedy_probs', /(ε|epsilon|e)\s*-?\s*greedy/i],
  ['greedy_action', /greedy/i],
  ['bellman_optimality', /bellman\s+optimality|optimal\s+value/i],
  ['bellman_v', /bellman\s+(expectation\s+)?equation/i],
  ['q_from_v', /action[-\s]values?|\bQ\s*\(\s*s/i],
  ['return_Gt', /discounted\s+(return|sum)|\breturn\b|\bG_?\{?t\}?\b|\bG_?\d\b/i],
  ['best_state', /highest|which\s+state|best\s+state/i],
  ['theory', /differentiate|difference|explain|define|describe|what\s+is|why\b/i],
];

// Tasks that answer the same kind of question; picking one when keywords suggest another is fine.
const SAME: Task[][] = [
  ['policy_evaluation', 'bellman_v', 'policy_evaluation_inplace', 'policy_evaluation_linear'],
  ['value_iteration', 'bellman_optimality'],
  ['policy_improvement', 'q_from_v', 'greedy_action'],
];

// "Differentiate policy iteration and value iteration" names algorithms but asks for theory.
const THEORY_FIRST = /^\s*(differentiate|distinguish|explain|define|describe|what\s+is|why\b)/i;

export function detectTasks(text: string): Task[] {
  const found = RULES.filter(([, re]) => re.test(text)).map(([t]) => t);
  return THEORY_FIRST.test(text) ? ['theory', ...found.filter((t) => t !== 'theory')] : found;
}

const related = (a: Task, b: Task) => a === b || SAME.some((g) => g.includes(a) && g.includes(b));

/** Warnings for parts whose task disagrees with the keywords in their own text. */
export function crossCheck(parts: Part[]): string[] {
  return parts.flatMap((p, i) => {
    const found = detectTasks(p.text);
    if (!found.length || found.some((t) => related(t, p.task))) return [];
    return [`Part ${p.label || i + 1}: marked "${HEADINGS[p.task]}" but its text mentions "${HEADINGS[found[0]]}". Check the task.`];
  });
}

/** Split "a) ... b) ..." / "(a) ..." / "1. ..." sub-questions and guess each task from keywords. */
export function splitParts(text: string): Part[] {
  const re = /(?:^|\n)\s*\(?([a-h]|[1-9])[).]\s+/g;
  const marks = [...text.matchAll(re)];
  return marks.map((m, i) => {
    const body = text.slice(m.index! + m[0].length, marks[i + 1]?.index ?? text.length).trim();
    return { label: m[1], task: detectTasks(body)[0] ?? 'theory', states: [], rewards: [], text: body };
  });
}
