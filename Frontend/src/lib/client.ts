import { createDemo } from "./demo";
import type { Mode, Transport } from "./types";
const demo = createDemo(localStorage);
export function client(mode: Mode): Transport {
  if (mode === "demo") return demo;
  return {
    invoke: async (method, args = []) => {
      if (!window.argusDesktop)
        throw new Error(
          "Open the Electron desktop app to connect to the local engine.",
        );
      return window.argusDesktop.invoke(method, args);
    },
    subscribe: (listener) => {
      if (!window.argusDesktop) return () => {};
      return window.argusDesktop.subscribe(listener);
    },
  };
}
