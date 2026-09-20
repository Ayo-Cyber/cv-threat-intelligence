import { useEffect, useState } from "react";
import { Bell, Eye, EyeOff, Save, Send } from "lucide-react";
import type { Json, Mode, Transport } from "../lib/types";
import {
  EMPTY,
  buildNotify,
  maskToken,
  notifyProblem,
  parseNotify,
  type Channels,
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
