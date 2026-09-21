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
};

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
    }
  }
  if (!out.console && !out.telegram && !out.whatsapp && !out.webhook)
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
