import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// base './' so the built app works from any folder (GitHub Pages, a zip, file server).
// /api goes to the FastAPI backend (server.py) during development.
export default defineConfig({
  plugins: [react()],
  base: './',
  server: {
    proxy: { '/api': 'http://localhost:8000' },
  },
})
