import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "tests",
  testMatch: "camera-stream.e2e.spec.ts",
  workers: 1,
  timeout: 30_000,
  use: { baseURL: "http://127.0.0.1:5173", channel: "chrome", headless: true },
  webServer: {
    command: "npm run dev",
    url: "http://127.0.0.1:5173",
    reuseExistingServer: true,
  },
  reporter: "list",
});
