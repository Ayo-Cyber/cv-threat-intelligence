import React from "react";
import { createRoot } from "react-dom/client";
import CameraDetails from "../../src/components/CameraDetails";
import SupportDiagnostics from "../../src/components/SupportDiagnostics";
import "../../src/styles.css";

const calls: unknown[] = [];
(window as any).supportCalls = calls;
const api = { invoke: async (method: string, args?: unknown[]) => {
  calls.push({ method, args });
  if (method === "download_diagnostics") return { ok: true, path: "C:\\Argus\\argus-diagnostics-test.zip" };
  if (method === "scene_context") return null;
  if (["presets", "english_rules_status"].includes(method)) return {};
  return [];
} } as any;
createRoot(document.getElementById("root")!).render(<main style={{ padding: 24 }}>
  <SupportDiagnostics api={api} />
  <CameraDetails camera={{ id: "Bay", source: "test", zone_count: 1 }} api={api}
    initialTab="detectors" onChange={async () => {}} notify={() => {}} />
</main>);
