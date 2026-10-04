/**
 * The notification destination, as a thing you can fill in rather than a string
 * you must already know how to write.
 *
 * Settings offered one free-text box placeholdered "console". To get alerts on
 * a phone you had to type `console,telegram:<bot_token>:<chat_id>` with nothing
 * anywhere saying so -- and typing just `telegram` silently falls back to the
 * console, because _build_one only matches the "telegram:" prefix. The pilot
 * could not set his own up (20 Sep).
 *
 * The bot token itself contains a colon (123456789:AAH...), so the chat id is
 * the part after the LAST colon. Splitting on the first one handed half the
 * token to the chat id and no message ever left the building (11 Sep, #?).
 */

export type Channels = {
  console: boolean;
  telegram: boolean;
  botToken: string;
  chatId: string;
  whatsapp: boolean;
  twilioSid: string;
  twilioToken: string;
  whatsappFrom: string;
  whatsappTo: string;
  webhook: string;
  email: boolean;
  smtpHost: string;
  smtpPort: string;
  smtpSecurity: SmtpSecurity;
  smtpUser: string;
  smtpPassword: string;
  emailFrom: string;
  /** Recipients, separated by commas, semicolons or newlines. */
  emailTo: string;
};

export type SmtpSecurity = "starttls" | "ssl" | "none";
export const SMTP_DEFAULT_PORT: Record<SmtpSecurity, string> = {
  starttls: "587",
  ssl: "465",
  none: "25",
};

/** Twilio's shared sandbox number, which is what a trial account sends from. */
export const TWILIO_SANDBOX = "+14155238886";

export const EMPTY: Channels = {
  console: true,
  telegram: false,
  botToken: "",
  chatId: "",
  whatsapp: false,
  twilioSid: "",
  twilioToken: "",
  whatsappFrom: TWILIO_SANDBOX,
  whatsappTo: "",
  webhook: "",
  email: false,
  smtpHost: "",
  smtpPort: "587",
  smtpSecurity: "starttls",
  smtpUser: "",
  smtpPassword: "",
  emailFrom: "",
  emailTo: "",
};

/** Recipients as a clean list, whatever separator the operator typed. */
export function emailRecipients(text: string): string[] {
  return text
    .split(/[,;\s]+/)
    .map((r) => r.trim())
    .filter(Boolean);
}

/** Parse the stored spec back into fields the form can show. */
export function parseNotify(spec: string): Channels {
  const out: Channels = { ...EMPTY, console: false };
  for (const raw of (spec || "").split(",")) {
    const part = raw.trim();
    if (!part) continue;
    if (part === "console") out.console = true;
    else if (part.startsWith("telegram:")) {
      const rest = part.slice("telegram:".length);
      const cut = rest.lastIndexOf(":");
      if (cut > 0) {
        out.telegram = true;
        out.botToken = rest.slice(0, cut);
        out.chatId = rest.slice(cut + 1);
      }
    } else if (part.startsWith("whatsapp:")) {
      const fields = part.slice("whatsapp:".length).split(":");
      if (fields.length === 4 && fields.every((f) => f.trim())) {
        out.whatsapp = true;
        [out.twilioSid, out.twilioToken, out.whatsappFrom, out.whatsappTo] =
          fields.map((f) => f.trim());
      }
    } else if (part.startsWith("webhook:")) {
      out.webhook = part.slice("webhook:".length);
    } else if (part.startsWith("email:")) {
      // URL-encoded key=value pairs: a password's colon or a recipient's
      // comma never collides with this colon- and comma-delimited string.
      const q = new URLSearchParams(part.slice("email:".length));
      const host = (q.get("host") || "").trim();
      const to = emailRecipients(q.get("to") || "");
      if (host && to.length) {
        const security = (q.get("security") || "starttls") as SmtpSecurity;
        out.email = true;
        out.smtpHost = host;
        out.smtpSecurity = security in SMTP_DEFAULT_PORT ? security : "starttls";
        out.smtpPort = (q.get("port") || SMTP_DEFAULT_PORT[out.smtpSecurity]).trim();
        out.smtpUser = (q.get("user") || "").trim();
        out.smtpPassword = q.get("password") || "";
        out.emailFrom = (q.get("from") || "").trim();
        out.emailTo = to.join(", ");
      }
    }
  }
  if (!out.console && !out.telegram && !out.whatsapp && !out.webhook && !out.email)
    out.console = true;
  return out;
}

/** Build the spec the engine expects. Console always stays on as a floor. */
export function buildNotify(c: Channels): string {
  const parts: string[] = [];
  if (c.console) parts.push("console");
  if (c.telegram && c.botToken.trim() && c.chatId.trim())
    parts.push(`telegram:${c.botToken.trim()}:${c.chatId.trim()}`);
  if (
    c.whatsapp &&
    c.twilioSid.trim() &&
    c.twilioToken.trim() &&
    c.whatsappFrom.trim() &&
    c.whatsappTo.trim()
  )
    parts.push(
      `whatsapp:${c.twilioSid.trim()}:${c.twilioToken.trim()}:` +
        `${c.whatsappFrom.trim()}:${c.whatsappTo.trim()}`,
    );
  if (c.webhook.trim()) parts.push(`webhook:${c.webhook.trim()}`);
  if (c.email && c.smtpHost.trim() && emailRecipients(c.emailTo).length) {
    const q = new URLSearchParams({
      host: c.smtpHost.trim(),
      port: c.smtpPort.trim() || SMTP_DEFAULT_PORT[c.smtpSecurity],
      security: c.smtpSecurity,
      user: c.smtpUser.trim(),
      password: c.smtpPassword,
      from: c.emailFrom.trim() || c.smtpUser.trim(),
      to: emailRecipients(c.emailTo).join(";"),
    });
    // URLSearchParams encodes "," ";" ":" and "&" — nothing here can split
    // the outer spec. (It writes spaces as "+", which the engine decodes.)
    parts.push(`email:${q.toString()}`);
  }
  return parts.length ? parts.join(",") : "console";
}

/** What is stopping this from working, in words the operator can act on. */
export function notifyProblem(c: Channels): string {
  if (c.telegram && !c.botToken.trim())
    return "Paste the bot token BotFather gave you.";
  if (c.telegram && !c.botToken.includes(":"))
    return "That does not look like a bot token — it should contain a colon, like 123456789:AAH...";
  if (c.telegram && !c.chatId.trim())
    return "Enter the chat ID the alerts should go to.";
  if (c.telegram && !/^-?\d+$/.test(c.chatId.trim()))
    return "A chat ID is a number, and starts with a minus sign for a group.";
  if (c.whatsapp && !c.twilioSid.trim())
    return "Paste your Twilio Account SID (it starts with AC).";
  if (c.whatsapp && !/^AC[0-9a-zA-Z]+$/.test(c.twilioSid.trim()))
    return "A Twilio Account SID starts with AC.";
  if (c.whatsapp && !c.twilioToken.trim())
    return "Paste your Twilio Auth Token.";
  if (c.whatsapp && !/^\+?\d[\d\s-]+$/.test(c.whatsappTo.trim()))
    return "Enter the WhatsApp number to alert, in full international form like +234...";
  if (c.webhook.trim() && !/^https?:\/\//.test(c.webhook.trim()))
    return "A webhook must start with http:// or https://";
  if (c.email) {
    const addr = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!c.smtpHost.trim())
      return "Enter your mail server's address, like smtp.gmail.com.";
    if (c.smtpPort.trim() && !/^\d{1,5}$/.test(c.smtpPort.trim()))
      return "The SMTP port is a number — 587 for STARTTLS, 465 for SSL.";
    const to = emailRecipients(c.emailTo);
    if (!to.length) return "Enter at least one address to send alerts to.";
    const bad = to.find((r) => !addr.test(r));
    if (bad) return `"${bad}" does not look like an email address.`;
    const from = c.emailFrom.trim() || c.smtpUser.trim();
    if (!from) return "Enter the address the alerts should come from.";
    if (!addr.test(from)) return "The from address does not look like an email address.";
    if (c.smtpUser.trim() && !c.smtpPassword)
      return "Enter the password for that mail account (for Gmail, an App Password).";
  }
  return "";
}

/** Never show a live token back on screen. */
export function maskToken(token: string): string {
  const t = token.trim();
  if (!t) return "";
  const cut = t.indexOf(":");
  const head = cut > 0 ? t.slice(0, cut) : t.slice(0, 4);
  return `${head}:${"•".repeat(8)}`;
}
