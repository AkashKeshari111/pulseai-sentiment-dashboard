import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')

  // 8000 is a popular default and is frequently already taken by another
  // project on the same machine, so this defaults to 8020. Override with
  // VITE_API_PROXY (or API_PORT) in dashboard/.env.local.
  const apiTarget =
    env.VITE_API_PROXY || `http://127.0.0.1:${env.API_PORT || '8020'}`

  return {
    plugins: [react(), tailwindcss()],
    server: {
      port: Number(env.VITE_PORT || 5173),
      // The dashboard calls the API through /api and /health on its own origin.
      // Proxying in dev means no CORS handling and no hard-coded localhost URL
      // in the client, so the same build works unchanged in production behind a
      // reverse proxy.
      proxy: {
        '/api': { target: apiTarget, changeOrigin: true },
        '/health': { target: apiTarget, changeOrigin: true },
      },
    },
    build: {
      outDir: 'dist',
      sourcemap: false,
      rollupOptions: {
        output: {
          // Recharts + d3 is the heaviest dependency; splitting it keeps the
          // initial bundle small so the first paint is not blocked by charts.
          manualChunks: {
            charts: ['recharts'],
            vendor: ['react', 'react-dom', 'react-router-dom'],
          },
        },
      },
    },
  }
})
