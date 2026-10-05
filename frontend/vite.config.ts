import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// In development the page is served by Vite and `/api` is proxied to the backend, so the code
// uses the same relative paths as in production, where FastAPI serves both (ADR-0002).
const backend = process.env.OFFSCREEN_API ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { proxy: { "/api": backend } },
  test: {
    environment: "jsdom",
    setupFiles: ["src/test/setup.ts"],
    css: false,
  },
});
