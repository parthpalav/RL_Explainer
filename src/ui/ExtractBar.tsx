import { useEffect, useRef, useState } from 'react';
import type { Part, Problem } from '../schema';
import { DEFAULT_MODEL, extract, listModels } from '../extract/ollama';
import { splitParts } from '../extract/keywords';
import { Icon } from './Icon';

const KEY = 'rl-explainer-model';
const load = () => {
  try {
    return localStorage.getItem(KEY) || DEFAULT_MODEL;
  } catch {
    return DEFAULT_MODEL;
  }
};

interface Props {
  question: string;
  onProblem: (p: Problem) => void;
  onParts: (parts: Part[]) => void;
}

/** Ollama status, model picker and the extract button. */
export function ExtractBar({ question, onProblem, onParts }: Props) {
  const [models, setModels] = useState<string[] | null>(null);
  const [offline, setOffline] = useState(false);
  const [model, setModel] = useState(load);
  const [busy, setBusy] = useState(false);
  const [secs, setSecs] = useState(0);
  const [error, setError] = useState('');
  const abort = useRef<AbortController | null>(null);

  const refresh = () =>
    listModels()
      .then((m) => {
        setModels(m);
        setOffline(false);
      })
      .catch(() => setOffline(true));
  useEffect(() => {
    refresh();
  }, []);

  useEffect(() => {
    if (!busy) return;
    const t0 = Date.now();
    const id = setInterval(() => setSecs(Math.round((Date.now() - t0) / 1000)), 500);
    return () => clearInterval(id);
  }, [busy]);

  const pick = (m: string) => {
    setModel(m);
    try {
      localStorage.setItem(KEY, m);
    } catch {
      /* private mode: just don't remember */
    }
  };

  const run = async () => {
    setError('');
    setBusy(true);
    setSecs(0);
    abort.current = new AbortController();
    try {
      onProblem(await extract(question, model, { signal: abort.current.signal }));
    } catch (e) {
      const err = e as Error;
      if (err.name !== 'AbortError') setError(err.message.includes('Failed to fetch') ? 'Cannot reach Ollama. Is it running?' : err.message);
    } finally {
      setBusy(false);
    }
  };

  const missing = models && !models.includes(model);
  return (
    <div className="extract">
      <div className="extract-row">
        <span className={`conn ${offline ? 'off' : models ? 'on' : ''}`} title={offline ? 'Ollama is not reachable' : 'Ollama is running'}>
          <i /> {offline ? 'Ollama offline' : models ? 'Ollama' : 'Connecting…'}
        </span>
        {models?.length ? (
          <select className="model" value={model} onChange={(e) => pick(e.target.value)} aria-label="Model">
            {missing && <option value={model}>{model} (not installed)</option>}
            {models.map((m) => <option key={m}>{m}</option>)}
          </select>
        ) : (
          <input className="model" value={model} onChange={(e) => pick(e.target.value)} aria-label="Model" />
        )}
      </div>
      <div className="extract-row">
        {busy ? (
          <button className="primary grow busy" onClick={() => abort.current?.abort()}>
            <span className="spinner" aria-hidden="true" /> Reading question… {secs}s · Stop
          </button>
        ) : (
          <button className="primary grow" onClick={run} disabled={!question.trim() || offline || !!missing}>
            <Icon name="spark" size={16} /> Extract data with local LLM
          </button>
        )}
        <button className="ghost" onClick={() => onParts(splitParts(question))} disabled={busy} title="Split sub-questions and guess tasks from keywords, without the LLM">
          Detect parts only
        </button>
      </div>
      {offline && (
        <p className="warn">
          Ollama is not running. Start it with <code>ollama serve</code>, then <button className="link" onClick={refresh}>retry</button>.
        </p>
      )}
      {missing && <p className="warn">Model not installed. Run <code>ollama pull {model}</code>.</p>}
      {busy && <p className="muted small">A 7B model takes about 30–90 s on an M4.</p>}
      {error && <p className="err" role="alert">{error}</p>}
    </div>
  );
}
