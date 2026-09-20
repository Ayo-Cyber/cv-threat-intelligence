import { describe, expect, it } from "vitest";
import { MODEL_SIZE, verifierMessage } from "../src/components/VerifierDownload";

describe("what the operator is told about the on-device AI", () => {
  it("says nothing before the status is known", () => {
    expect(verifierMessage(null, null)).toBeNull();
  });

  it("confirms readiness once the model is installed", () => {
    const message = verifierMessage({ mode: "live" }, { state: "done" });
    expect(message?.tone).toBe("ready");
  });

  it("names the size when nothing has been downloaded", () => {
    const message = verifierMessage({ mode: "no-model" }, { state: "idle" });
    expect(message?.tone).toBe("action");
    expect(message?.text).toContain(MODEL_SIZE);
    expect(message?.text).toContain("unverified");
  });

  it("shows progress and says the operator need not wait", () => {
    const message = verifierMessage(
      { mode: "no-model" },
      { state: "pulling", percent: 42.6 },
    );
    expect(message?.tone).toBe("working");
    expect(message?.text).toContain("43%");
    expect(message?.text).toContain("carry on setting up");
  });

  it("clamps a percentage that arrives out of range", () => {
    expect(
      verifierMessage({ mode: "no-model" }, { state: "pulling", percent: 140 })
        ?.text,
    ).toContain("100%");
    expect(
      verifierMessage({ mode: "no-model" }, { state: "pulling", percent: -5 })
        ?.text,
    ).toContain("0%");
  });

  it("treats a missing percentage as zero rather than NaN", () => {
    expect(
      verifierMessage({ mode: "no-model" }, { state: "pulling" })?.text,
    ).toContain("0%");
  });

  it("explains a failed download and that it resumes", () => {
    const message = verifierMessage(
      { mode: "no-model" },
      { state: "error", detail: "connection reset" },
    );
    expect(message?.tone).toBe("action");
    expect(message?.text).toContain("connection reset");
    expect(message?.text).toContain("resumes");
  });

  it("reports an absent runtime separately from an absent model", () => {
    const message = verifierMessage({ mode: "offline" }, { state: "idle" });
    expect(message?.tone).toBe("action");
    expect(message?.text).toContain("runtime");
  });

  it("prefers live progress over the stale no-model status", () => {
    // gate_status is polled before pull_progress, so mid-download the mode is
    // still "no-model"; the operator must see progress, not the prompt again.
    const message = verifierMessage(
      { mode: "no-model" },
      { state: "pulling", percent: 10 },
    );
    expect(message?.tone).toBe("working");
  });
});
