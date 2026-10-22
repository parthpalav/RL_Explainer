import katex from 'katex';
import { useMemo } from 'react';

const OPTS: katex.KatexOptions = {
  throwOnError: false,
  strict: false,
  // Only our own data/class hooks are allowed; no links or styles from user data.
  trust: (c) => c.command === '\\htmlData' || c.command === '\\htmlClass',
};

export function Tex({ tex, display = false, className = '' }: { tex: string; display?: boolean; className?: string }) {
  const html = useMemo(() => katex.renderToString(tex, { ...OPTS, displayMode: display }), [tex, display]);
  return <span className={`tex ${className}`} dangerouslySetInnerHTML={{ __html: html }} />;
}

/** Plain text with inline $...$ math. */
export function Rich({ text }: { text: string }) {
  return (
    <>
      {text.split('$').map((seg, i) => (i % 2 ? <Tex key={i} tex={seg} /> : <span key={i}>{seg}</span>))}
    </>
  );
}
