import { describe, expect, it } from "vitest";
import { initialMode, rememberMode } from "../src/lib/mode";

function store(initial: Record<string, string> = {}) {
  const data = { ...initial };
  return {
    data,
    getItem: (k: string) => (k in data ? data[k] : null),
    setItem: (k: string, v: string) => {
      data[k] = v;
    },
  };
}

describe("initialMode", () => {
  it("opens the desktop app on the real workspace, not the demo fixture", () => {
    expect(initialMode(store(), true)).toBe("engine");
  });

  it("keeps the browser preview on demo: there is no engine to reach", () => {
    expect(initialMode(store(), false)).toBe("demo");
  });

  it("remembers a deliberate switch to demo", () => {
    const s = store();
    rememberMode(s, "demo");
    expect(initialMode(s, true)).toBe("demo");
  });

  it("remembers a deliberate switch back to the engine", () => {
    const s = store({ "argus.workspace.mode": "demo" });
    rememberMode(s, "engine");
    expect(initialMode(s, true)).toBe("engine");
  });

  it("ignores a junk stored value", () => {
    expect(initialMode(store({ "argus.workspace.mode": "wat" }), true)).toBe(
      "engine",
    );
  });

  it("survives storage that throws, as private browsing does", () => {
    const throwing = {
      getItem: () => {
        throw new Error("denied");
      },
      setItem: () => {
        throw new Error("denied");
      },
    };
    expect(initialMode(throwing, true)).toBe("engine");
    expect(() => rememberMode(throwing, "demo")).not.toThrow();
  });

  it("survives no storage at all", () => {
    expect(initialMode(null, true)).toBe("engine");
    expect(() => rememberMode(null, "demo")).not.toThrow();
  });

  it("never lets a stored preference put the browser preview on the engine", () => {
    expect(initialMode(store({ "argus.workspace.mode": "engine" }), false)).toBe(
      "demo",
    );
  });
});
