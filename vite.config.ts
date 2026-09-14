import { svelte } from "@sveltejs/vite-plugin-svelte";
import { defineConfig } from "vite";

export default defineConfig({
  base: "/reve-atlas/",
  plugins: [svelte()],
  worker: { format: "es" },
  build: {
    target: "esnext",
    chunkSizeWarningLimit: 4096,
  },
});
