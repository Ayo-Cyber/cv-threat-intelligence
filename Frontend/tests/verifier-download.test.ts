import { describe, expect, it } from "vitest";
import {
  MODEL_SIZE,
  RECOGNITION_SIZE,
  recognitionMessage,
  verifierMessage,
} from "../src/components/VerifierDownload";

describe("what the operator is told about the on-device AI", () => {
  it("says nothing before the status is known", () => {
    expect(verifierMessage(null, null)).toBeNull();
  });

  it("confirms readiness once the model is installed", () => {
    const message = verifierMessage({ mode: "live" }, { state: "done" });
    expect(message?.tone).toBe("ready");
  });

  it("a cloud verifier is described by name, never as a download", () => {
    const live = verifierMessage(
      { mode: "live", cloud: true, label: "OpenRouter", model: "google/gemini-2.5-flash-lite" },
      null,
    );
    expect(live?.tone).toBe("ready");
    expect(live?.text).toContain("OpenRouter");
    expect(live?.text).toContain("gemini-2.5-flash-lite");
    expect(live?.text).not.toContain(MODEL_SIZE);
    const noKey = verifierMessage({ mode: "no-key", cloud: true, label: "Groq" }, null);
    expect(noKey?.tone).toBe("action");
    expect(noKey?.text).toContain("Groq");
    expect(noKey?.text).toContain("API key");
    expect(noKey?.text).not.toContain(MODEL_SIZE);
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

describe("what the operator is told about the object-recognition model", () => {
  it("says nothing before the status is known (older engines have no route)", () => {
    expect(recognitionMessage(null)).toBeNull();
  });
  it("names the size when nothing is installed, and that recognition stays off", () => {
    const m = recognitionMessage({ state: "idle" });
    expect(m?.tone).toBe("action");
    expect(m?.text).toContain(RECOGNITION_SIZE);
    expect(m?.text).toMatch(/recognition stays off/);
  });
  it("shows download progress", () => {
    expect(recognitionMessage({ state: "downloading", percent: 57.4 })?.text).toContain("57%");
  });
  it("explains the load check", () => {
    expect(recognitionMessage({ state: "verifying" })?.text).toMatch(/embedding/);
  });
  it("a failure names the reason and says the rest of Argus still works", () => {
    const m = recognitionMessage({ state: "error", detail: "checksum mismatch" });
    expect(m?.tone).toBe("action");
    expect(m?.text).toContain("checksum mismatch");
    expect(m?.text).toMatch(/everything else works/);
  });
  it("confirms readiness", () => {
    expect(recognitionMessage({ state: "ready", percent: 100 })?.tone).toBe("ready");
  });
});
