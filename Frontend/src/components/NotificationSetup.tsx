import { useEffect, useState } from "react";
import { Bell, Eye, EyeOff, Save, Send } from "lucide-react";
import type { Json, Mode, Transport } from "../lib/types";
import {
  EMPTY,
  buildNotify,
  emailRecipients,
  maskToken,
  notifyProblem,
  parseNotify,
  SMTP_DEFAULT_PORT,
  TWILIO_SANDBOX,
  type Channels,
  type SmtpSecurity,
} from "../lib/notify";
import { Notice, Spinner } from "./common";

/**
 * Where alerts go, as fields rather than a string you must already know.
 *
 * Settings used to offer one free-text box placeholdered "console". Telegram
 * meant typing `console,telegram:<bot_token>:<chat_id>`, undocumented anywhere
 * in the app, and typing just `telegram` silently did nothing.
 */
export default function NotificationSetup({
  api,
  mode,
  site,
  onChange,
  notify,
}: {
  api: Transport;
  mode: Mode;
  site: Json;
  onChange: () => Promise<void>;
  notify: (message: string) => void;
}) {
  const [channels, setChannels] = useState<Channels>(() =>
    parseNotify(String(site.notify || "console")),
  );
  const [reveal, setReveal] = useState(false);
  const [revealTwilio, setRevealTwilio] = useState(false);
  const [revealSmtp, setRevealSmtp] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    setChannels(parseNotify(String(site.notify || "console")));
  }, [site.notify]);

  const set = (patch: Partial<Channels>) => {
    setChannels((current) => ({ ...current, ...patch }));
    setSaved(false);
  };
  const problem = notifyProblem(channels);
  const spec = buildNotify(channels);
  const live = mode === "engine";

  async function run(method: string, args: unknown[], message: string) {
    setBusy(true);
    setError("");
    try {
      const result = await api.invoke(method, args);
      if (result && result.ok === false)
        throw new Error(result.message || "That did not work");
      await onChange();
      notify(message);
      if (method === "set_site") setSaved(true);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="settings-section">
      <div>
        <h2>Alerts</h2>
        <p>Where Argus sends an alert when it sees something.</p>
      </div>
      <div className="notify-setup">
        {error && <Notice error>{error}</Notice>}

        <label className="notify-channel">
          <input
            type="checkbox"
            checked={channels.console}
            onChange={(e) => set({ console: e.target.checked })}
          />
          <div>
            <strong>This computer</strong>
            <small>Always recorded here, and shown on the Incidents screen.</small>
          </div>
        </label>

        <label className="notify-channel">
          <input
            type="checkbox"
            checked={channels.telegram}
            onChange={(e) => set({ telegram: e.target.checked })}
          />
          <div>
            <strong>Telegram</strong>
            <small>Alerts on your phone, with the evidence photos and video.</small>
          </div>
        </label>

        {channels.telegram && (
          <div className="notify-telegram">
            <ol className="notify-steps">
              <li>
                In Telegram, message <strong>@BotFather</strong> and send{" "}
                <code>/newbot</code>. It replies with a token that looks like{" "}
                <code>123456789:AAH...</code>
              </li>
              <li>
                Send your new bot a message, then open{" "}
                <strong>@userinfobot</strong> to get your chat ID. For a group,
                add the bot to it — a group ID starts with a minus sign.
              </li>
            </ol>
            <label>
              Bot token
              <div className="reveal-field">
                <input
                  type={reveal ? "text" : "password"}
                  value={channels.botToken}
                  onChange={(e) => set({ botToken: e.target.value })}
                  placeholder="123456789:AAH..."
                  autoComplete="off"
                  spellCheck={false}
                />
                <button
                  type="button"
                  className="icon-button"
                  aria-label={reveal ? "Hide the token" : "Show the token"}
                  onClick={() => setReveal(!reveal)}
                >
                  {reveal ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
            </label>
            <label>
              Chat ID
              <input
                value={channels.chatId}
                onChange={(e) => set({ chatId: e.target.value })}
                placeholder="1883642843, or -1001234567890 for a group"
                autoComplete="off"
                spellCheck={false}
              />
            </label>
            {channels.botToken && (
              <p className="field-note">
                Sending as bot <code>{maskToken(channels.botToken)}</code>
              </p>
            )}
          </div>
        )}

        <label className="notify-channel">
          <input
            type="checkbox"
            checked={channels.whatsapp}
            onChange={(e) => set({ whatsapp: e.target.checked })}
          />
          <div>
            <strong>WhatsApp</strong>
            <small>Alerts to a WhatsApp number, sent through Twilio.</small>
          </div>
        </label>

        {channels.whatsapp && (
          <div className="notify-telegram">
            <ol className="notify-steps">
              <li>
                Create a free account at <strong>twilio.com</strong>. Your
                Account SID and Auth Token are on the console dashboard.
              </li>
              <li>
                Open <strong>Messaging → Try it out → WhatsApp</strong> and
                follow the sandbox join step from the phone that should receive
                alerts. Until you have your own sender, leave the From number as
                Twilio's sandbox.
              </li>
            </ol>
            <label>
              Account SID
              <input
                value={channels.twilioSid}
                onChange={(e) => set({ twilioSid: e.target.value })}
                placeholder="AC..."
                autoComplete="off"
                spellCheck={false}
              />
            </label>
            <label>
              Auth token
              <div className="reveal-field">
                <input
                  type={revealTwilio ? "text" : "password"}
                  value={channels.twilioToken}
                  onChange={(e) => set({ twilioToken: e.target.value })}
                  autoComplete="off"
                  spellCheck={false}
                />
                <button
                  type="button"
                  className="icon-button"
                  aria-label={revealTwilio ? "Hide the token" : "Show the token"}
                  onClick={() => setRevealTwilio(!revealTwilio)}
                >
                  {revealTwilio ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
            </label>
            <label>
              Send alerts to
              <input
                value={channels.whatsappTo}
                onChange={(e) => set({ whatsappTo: e.target.value })}
                placeholder="+2348012345678"
                autoComplete="off"
              />
            </label>
            <label>
              Send from
              <input
                value={channels.whatsappFrom}
                onChange={(e) => set({ whatsappFrom: e.target.value })}
                placeholder={TWILIO_SANDBOX}
                autoComplete="off"
              />
            </label>
          </div>
        )}

        <label className="notify-channel">
          <input
            type="checkbox"
            checked={channels.email}
            onChange={(e) => set({ email: e.target.checked })}
          />
          <div>
            <strong>Email</strong>
            <small>
              One message per alert with the evidence pictures inline, through
              your own mail server (SMTP).
            </small>
          </div>
        </label>

        {channels.email && (
          <div className="notify-telegram">
            <ol className="notify-steps">
              <li>
                <strong>Gmail / Google Workspace:</strong> server{" "}
                <code>smtp.gmail.com</code>, STARTTLS on 587. Turn on 2-step
                verification, then create an <strong>App Password</strong> and
                use it here instead of your normal password.
              </li>
              <li>
                <strong>Outlook / Microsoft 365:</strong>{" "}
                <code>smtp.office365.com</code>, STARTTLS on 587, your full
                email address as the username.
              </li>
              <li>
                An internal relay with no login: leave username and password
                empty and pick <em>None</em> for security.
              </li>
            </ol>
            <div className="notify-grid">
              <label>
                Mail server (SMTP host)
                <input
                  value={channels.smtpHost}
                  onChange={(e) => set({ smtpHost: e.target.value })}
                  placeholder="smtp.gmail.com"
                  autoComplete="off"
                  spellCheck={false}
                />
              </label>
              <label>
                Security
                <select
                  value={channels.smtpSecurity}
                  onChange={(e) => {
                    const security = e.target.value as SmtpSecurity;
                    const wasDefault =
                      channels.smtpPort === SMTP_DEFAULT_PORT[channels.smtpSecurity];
                    set({
                      smtpSecurity: security,
                      ...(wasDefault || !channels.smtpPort
                        ? { smtpPort: SMTP_DEFAULT_PORT[security] }
                        : {}),
                    });
                  }}
                >
                  <option value="starttls">STARTTLS (port 587)</option>
                  <option value="ssl">SSL/TLS (port 465)</option>
                  <option value="none">None (port 25)</option>
                </select>
              </label>
              <label>
                Port
                <input
                  value={channels.smtpPort}
                  onChange={(e) => set({ smtpPort: e.target.value })}
                  placeholder={SMTP_DEFAULT_PORT[channels.smtpSecurity]}
                  inputMode="numeric"
                  autoComplete="off"
                />
              </label>
            </div>
            <label>
              Username
              <input
                value={channels.smtpUser}
                onChange={(e) => set({ smtpUser: e.target.value })}
                placeholder="ops@yourcompany.com"
                autoComplete="off"
                spellCheck={false}
              />
            </label>
            <label>
              Password
              <div className="reveal-field">
                <input
                  type={revealSmtp ? "text" : "password"}
                  value={channels.smtpPassword}
                  onChange={(e) => set({ smtpPassword: e.target.value })}
                  autoComplete="off"
                  spellCheck={false}
                />
                <button
                  type="button"
                  className="icon-button"
                  aria-label={revealSmtp ? "Hide the password" : "Show the password"}
                  onClick={() => setRevealSmtp(!revealSmtp)}
                >
                  {revealSmtp ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
            </label>
            <label>
              Send alerts to
              <textarea
                rows={2}
                value={channels.emailTo}
                onChange={(e) => set({ emailTo: e.target.value })}
                placeholder="security@yourcompany.com, manager@yourcompany.com"
                autoComplete="off"
                spellCheck={false}
              />
            </label>
            <label>
              From address (optional — defaults to the username)
              <input
                value={channels.emailFrom}
                onChange={(e) => set({ emailFrom: e.target.value })}
                placeholder="argus@yourcompany.com"
                autoComplete="off"
                spellCheck={false}
              />
            </label>
            {emailRecipients(channels.emailTo).length > 1 && (
              <p className="field-note">
                {emailRecipients(channels.emailTo).length} recipients
              </p>
            )}
          </div>
        )}

        <label>
          Webhook (optional)
          <input
            value={channels.webhook}
            onChange={(e) => set({ webhook: e.target.value })}
            placeholder="https://your-system.example.com/alerts"
            autoComplete="off"
          />
        </label>

        {problem && <Notice error>{problem}</Notice>}
        {saved && !problem && <Notice>Saved. Send a test to confirm it arrives.</Notice>}

        <div className="actions">
          <button
            className="button primary"
            disabled={busy || Boolean(problem) || !live}
            onClick={() =>
              void run("set_site", [site.name || "", spec], "Alert settings saved")
            }
          >
            {busy ? <Spinner /> : <Save size={16} />}
            Save
          </button>
          <button
            className="button"
            disabled={busy || !live}
            onClick={() =>
              void run(
                "send_test_notification",
                [],
                "Test alert sent — check the destination",
              )
            }
          >
            <Send size={16} />
            Send test alert
          </button>
        </div>
        {!live && (
          <p className="field-note">
            <Bell size={13} /> Switch to your own workspace to change where
            alerts go.
          </p>
        )}
      </div>
    </section>
  );
}
