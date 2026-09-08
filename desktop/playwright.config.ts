import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "tests",
  testMatch: "ui.spec.ts",
  workers: 1,
  timeout: 30000,
  use: {
    baseURL: "http://127.0.0.1:5173",
    channel: "chrome",
    headless: true,
    trace: "retain-on-failure",
  },
  reporter: "list",
});
