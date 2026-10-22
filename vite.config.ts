import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    // Ollama has no CORS headers by default, so route it through the dev server.
    proxy: { '/ollama': { target: 'http://localhost:11434', rewrite: (p) => p.replace(/^\/ollama/, '') } },
  },
});
