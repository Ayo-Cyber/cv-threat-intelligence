import { describe, expect, it } from "vitest";
import {
  EMPTY,
  buildNotify,
  emailRecipients,
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

describe("whatsapp", () => {
  const WA = {
    ...EMPTY,
    whatsapp: true,
    twilioSid: "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    twilioToken: "sometoken",
    whatsappFrom: "+14155238886",
    whatsappTo: "+2348012345678",
  };

  it("carries its own credentials so an installed machine can use it", () => {
    // Before, WhatsApp read Twilio creds from environment variables only --
    // unsettable on a machine that runs an installer, so it silently did nothing.
    expect(buildNotify(WA)).toBe(
      "console,whatsapp:ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx:sometoken:" +
        "+14155238886:+2348012345678",
    );
  });

  it("round-trips", () => {
    expect(buildNotify(parseNotify(buildNotify(WA)))).toBe(buildNotify(WA));
  });

  it("refuses to write a half-configured channel", () => {
    expect(buildNotify({ ...WA, twilioToken: "" })).toBe("console");
    expect(buildNotify({ ...WA, whatsappTo: "" })).toBe("console");
  });

  it("checks the SID looks like a Twilio SID", () => {
    expect(notifyProblem({ ...WA, twilioSid: "12345" })).toContain("AC");
  });

  it("asks for an international number", () => {
    expect(notifyProblem({ ...WA, whatsappTo: "0801234" })).toBe("");
    expect(notifyProblem({ ...WA, whatsappTo: "not-a-number" }))
      .toContain("international");
  });

  it("is happy when fully filled in", () => {
    expect(notifyProblem(WA)).toBe("");
  });

  it("a bare 'whatsapp' spec is not configured", () => {
    // That is what the old UI would have written, and it fell back to console.
    expect(parseNotify("console,whatsapp").whatsapp).toBe(false);
  });
});

describe("email over SMTP", () => {
  const email = {
    ...EMPTY,
    email: true,
    smtpHost: "smtp.gmail.com",
    smtpPort: "587",
    smtpSecurity: "starttls" as const,
    smtpUser: "ops@site.com",
    smtpPassword: "p:ss,w@rd&x",
    emailFrom: "argus@site.com",
    emailTo: "a@x.com, b@y.com",
  };

  it("writes one url-encoded channel the engine can split safely", () => {
    const spec = buildNotify(email);
    const [, part] = spec.split(",");
    expect(spec.startsWith("console,email:")).toBe(true);
    expect(spec.split(",").length).toBe(2);          // the password's comma did not split it
    const q = new URLSearchParams(part.slice("email:".length));
    expect(q.get("password")).toBe("p:ss,w@rd&x");
    expect(q.get("to")).toBe("a@x.com;b@y.com");
    expect(q.get("from")).toBe("argus@site.com");
  });

  it("round-trips", () => {
    const back = parseNotify(buildNotify(email));
    expect(back.email).toBe(true);
    expect(back.smtpHost).toBe("smtp.gmail.com");
    expect(back.smtpPassword).toBe("p:ss,w@rd&x");
    expect(back.emailTo).toBe("a@x.com, b@y.com");
    expect(buildNotify(back)).toBe(buildNotify(email));
  });

  it("refuses to write a half-configured email channel", () => {
    expect(buildNotify({ ...email, smtpHost: "" })).toBe("console");
    expect(buildNotify({ ...email, emailTo: "" })).toBe("console");
  });

  it("from falls back to the username", () => {
    const q = new URLSearchParams(buildNotify({ ...email, emailFrom: "" }).split("email:")[1]);
    expect(q.get("from")).toBe("ops@site.com");
  });

  it("names the problem in words the operator can act on", () => {
    expect(notifyProblem({ ...email, smtpHost: "" })).toMatch(/mail server/);
    expect(notifyProblem({ ...email, emailTo: "nobody" })).toMatch(/does not look like an email/);
    expect(notifyProblem({ ...email, smtpPassword: "" })).toMatch(/password/i);
    expect(notifyProblem({ ...email, smtpPort: "abc" })).toMatch(/port/);
    expect(notifyProblem(email)).toBe("");
  });

  it("recipients accept commas, semicolons and newlines", () => {
    expect(emailRecipients("a@x.com; b@y.com\nc@z.com,")).toEqual(["a@x.com", "b@y.com", "c@z.com"]);
  });
});
