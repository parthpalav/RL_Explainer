import type { Problem, Part } from '../schema';

/** A cell in a running results table, e.g. table "V_k(s)", row "k = 1", column "S_{1}" (all LaTeX). */
export interface Cell {
  table: string;
  row: string;
  col: string;
}

/** One written line of working: `lhs = rhs`, or `= rhs` continuing the line above. */
export interface Line {
  lhs?: string;
  rhs: string;
}

export interface Step {
  title: string; // LaTeX allowed between $...$
  why: string; // plain English: why this formula / what is happening
  formula?: string; // generic formula in LaTeX
  lines: Line[]; // first line may carry \htmlData{src=...} refs for the fly-in animation
  results: { cell: Cell; value: number | string }[];
  edges?: string[]; // transitions to highlight, see edgeKey()
  focus?: string; // episode step or bandit pull to highlight, see focusEp() / focusPull()
}

export interface PartTrace {
  part: Part;
  heading: string;
  steps: Step[];
  error?: string;
}

/** State carried between parts so "b) compute V2" continues from "a) compute V1". */
export interface Ctx {
  pe?: { k: number; V: Map<string, number>; inplace: boolean };
  vi?: { k: number; V: Map<string, number> };
  lastV?: { V: Map<string, number>; label: string; src?: (s: string) => string };
}

export type Solver = (p: Problem, part: Part, ctx: Ctx) => Step[];
