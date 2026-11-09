import { defineConfig } from 'vite';
import { viteSingleFile } from 'vite-plugin-singlefile';
export default defineConfig({
  base: './',
  publicDir: false,
  plugins: [viteSingleFile()],
  build: { outDir: 'dist_single', target: 'es2022', chunkSizeWarningLimit: 8000 },
  worker: { format: 'es' },
});
