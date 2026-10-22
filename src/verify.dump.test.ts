// Used by `npm run verify`: solves the random problems written by scripts/verify/ref.py and dumps every step
// so the independent Python reference can check the answers and re-evaluate each displayed line.
import { describe, it } from 'vitest';
import { readFileSync, writeFileSync } from 'node:fs';
import { ProblemSchema } from './schema';
import { solveAll } from './solvers';

describe.skipIf(!import.meta.env.DUMP_IN)('verify dump', () => {
  it('dump', () => {
    const probs: unknown[] = JSON.parse(readFileSync(import.meta.env.DUMP_IN as string, 'utf8'));
    const out = probs.map((raw) => {
      const r = ProblemSchema.safeParse(raw);
      if (!r.success) return { invalid: r.error.issues.map((i) => `${i.path.join('.')}: ${i.message}`) };
      return {
        parts: solveAll(r.data).map((t) => ({
          task: t.part.task,
          error: t.error,
          steps: t.steps.map((s) => ({ title: s.title, lines: s.lines, results: s.results })),
        })),
      };
    });
    writeFileSync(import.meta.env.DUMP_OUT as string, JSON.stringify(out));
  });
});
