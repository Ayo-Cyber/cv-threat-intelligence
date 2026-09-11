import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "tests",
  testMatch: "bridge-electron.e2e.spec.ts",
  workers: 1,
  timeout: 30_000,
  reporter: "list",
});
