import { readFileSync } from "node:fs";
import { expect, it } from "vitest";

it("keeps the verifier model identifier out of the customer-facing settings label", () => {
  const source = readFileSync(
    new URL("../src/components/SettingsPanel.tsx", import.meta.url), "utf8",
  );
  expect(source).toContain('"Argus AI verification"');
  expect(source).not.toContain("gate.model");
  expect(source).toContain('"No verifier connected"');
  expect(source).toContain('api.invoke(method, args)');
});
