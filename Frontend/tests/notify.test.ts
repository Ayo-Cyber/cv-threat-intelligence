import { describe, expect, it } from "vitest";
import {
  EMPTY,
  buildNotify,
  maskToken,
  notifyProblem,
  parseNotify,
} from "../src/lib/notify";

const TOKEN = "8691681982:AAH216L_pYnTxTESTTESTTESTTESTTESTTES";

describe("building the notify spec", () => {
  it("keeps console on its own", () => {
    expect(buildNotify(EMPTY)).toBe("console");
  });

  it("assembles telegram the way the engine parses it", () => {
    expect(
      buildNotify({ ...EMPTY, telegram: true, botToken: TOKEN, chatId: "1883642843" }),
    ).toBe(`console,telegram:${TOKEN}:1883642843`);
  });

  it("refuses to write a half-configured telegram channel", () => {
    // `telegram` with no token silently falls back to console in the engine,
    // which looks identical to working. Never emit it.
    expect(buildNotify({ ...EMPTY, telegram: true, botToken: "", chatId: "1" }))
      .toBe("console");
    expect(buildNotify({ ...EMPTY, telegram: true, botToken: TOKEN, chatId: "" }))
      .toBe("console");
  });

  it("never returns an empty spec", () => {
    expect(buildNotify({ ...EMPTY, console: false })).toBe("console");
  });

  it("carries a webhook too", () => {
    expect(buildNotify({ ...EMPTY, webhook: "https://example.com/hook" }))
      .toBe("console,webhook:https://example.com/hook");
  });
});

describe("reading an existing spec back", () => {
  it("splits the chat id off the LAST colon, not the first", () => {
    const parsed = parseNotify(`console,telegram:${TOKEN}:1883642843`);
    expect(parsed.botToken).toBe(TOKEN);
    expect(parsed.chatId).toBe("1883642843");
    expect(parsed.telegram).toBe(true);
    expect(parsed.console).toBe(true);
  });

  it("round-trips", () => {
    const spec = `console,telegram:${TOKEN}:-1001234567890`;
    expect(buildNotify(parseNotify(spec))).toBe(spec);
  });

  it("treats a bare 'telegram' as not configured", () => {
    // The retired console's checkbox wrote exactly this, and it never worked.
    expect(parseNotify("console,telegram").telegram).toBe(false);
  });

  it("defaults to console when the spec is empty or unknown", () => {
    expect(parseNotify("").console).toBe(true);
    expect(parseNotify("carrier-pigeon").console).toBe(true);
  });
});

describe("telling the operator what is missing", () => {
  it("is silent when nothing is wrong", () => {
    expect(notifyProblem(EMPTY)).toBe("");
    expect(
      notifyProblem({ ...EMPTY, telegram: true, botToken: TOKEN, chatId: "123" }),
    ).toBe("");
  });

  it("asks for the token first", () => {
    expect(notifyProblem({ ...EMPTY, telegram: true })).toContain("BotFather");
  });

  it("catches a token pasted without its colon", () => {
    expect(
      notifyProblem({ ...EMPTY, telegram: true, botToken: "AAH216L", chatId: "1" }),
    ).toContain("colon");
  });

  it("catches a chat id that is not a number", () => {
    expect(
      notifyProblem({ ...EMPTY, telegram: true, botToken: TOKEN, chatId: "@mychannel" }),
    ).toContain("number");
  });

  it("accepts a negative group chat id", () => {
    expect(
      notifyProblem({ ...EMPTY, telegram: true, botToken: TOKEN, chatId: "-1001234567890" }),
    ).toBe("");
  });

  it("insists a webhook is a URL", () => {
    expect(notifyProblem({ ...EMPTY, webhook: "example.com" })).toContain("http");
  });
});

describe("masking", () => {
  it("shows the bot id but never the secret", () => {
    const masked = maskToken(TOKEN);
    expect(masked).toContain("8691681982");
    expect(masked).not.toContain("AAH216L_pYnTx");
  });

  it("masks nothing when there is nothing", () => {
    expect(maskToken("")).toBe("");
  });
});
