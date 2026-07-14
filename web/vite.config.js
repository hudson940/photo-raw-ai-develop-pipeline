import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Built assets are served by the Python backend from web/dist. In dev, proxy the API
// (and image/share/auth routes) to a locally running `python -m pipeline.webui`.
const backend = process.env.BACKEND || "http://127.0.0.1:8765";
const proxy = Object.fromEntries(
  ["/api", "/thumb", "/img", "/auth", "/share"].map((p) => [p, { target: backend, changeOrigin: true }])
);

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy },
  build: { outDir: "dist", emptyOutDir: true, chunkSizeWarningLimit: 900 },
});
