import { useState } from "react";
import { Link2 } from "lucide-react";

/**
 * The guided camera connection form, restored from the PyQt console (v1.8.14).
 *
 * The React shell replaced a form that asked for IP, vendor, username and
 * password with a single "Camera source" box. That box is fine if you already
 * know that a Hikvision substream lives at /Streaming/Channels/102 and a Tapo
 * at /stream1 — and useless if you don't, which is everyone installing their
 * first camera. It also took a camera password as plain visible text.
 *
 * Paths and the URL assembly (including URL-encoding the credentials, so a
 * password containing @ or : cannot corrupt the URL) come from the old
 * console's wizBuildUrl, unchanged in behaviour.
 */

export const VENDORS: { label: string; path: string; note?: string }[] = [
  { label: "Hikvision", path: "/Streaming/Channels/102", note: "substream" },
  { label: "Dahua / Amcrest", path: "/cam/realmonitor?channel=1&subtype=1", note: "substream" },
  { label: "Reolink", path: "/h264Preview_01_sub", note: "substream" },
  { label: "Tapo / TP-Link", path: "/stream2", note: "stream1 = full quality" },
  { label: "Tapo / TP-Link (full quality)", path: "/stream1" },
  { label: "Generic", path: "/stream1" },
];

/** rtsp://user:pass@host:port/path — credentials percent-encoded. */
export function buildRtspUrl(
  host: string,
  path: string,
  user = "",
  password = "",
  port = "554",
): string {
  const h = host.trim();
  if (!h) return "";
  const u = user.trim();
  const auth = u
    ? encodeURIComponent(u) +
      (password ? ":" + encodeURIComponent(password) : "") +
      "@"
    : "";
  const p = (port || "554").trim();
  return `rtsp://${auth}${h}:${p}${path}`;
}

export default function CameraConnection({
  onBuild,
  disabled,
}: {
  onBuild: (url: string) => void;
  disabled?: boolean;
}) {
  const [host, setHost] = useState("");
  const [port, setPort] = useState("554");
  const [vendor, setVendor] = useState(VENDORS[0].path);
  const [user, setUser] = useState("");
  const [password, setPassword] = useState("");
  const [reveal, setReveal] = useState(false);

  const url = buildRtspUrl(host, vendor, user, password, port);
  // Never render the password back to the room; a camera password is often
  // reused, and this screen gets projected during installs.
  const shown = password
    ? url.replace(
        `:${encodeURIComponent(password)}@`,
        reveal ? `:${encodeURIComponent(password)}@` : ":••••@",
      )
    : url;

  return (
    <fieldset className="camera-connect" disabled={disabled}>
      <legend>
        <Link2 size={15} /> Build the connection
      </legend>
      <p className="camera-connect-hint">
        For a network camera, fill these in and Argus assembles the address.
        Already have an RTSP URL? Put it straight in Camera source below.
      </p>
      <div className="camera-connect-grid">
        <label>
          Camera IP
          <input
            value={host}
            onChange={(e) => setHost(e.target.value)}
            placeholder="192.168.1.64"
            autoComplete="off"
          />
        </label>
        <label>
          Port
          <input
            value={port}
            onChange={(e) => setPort(e.target.value)}
            placeholder="554"
            autoComplete="off"
          />
        </label>
        <label>
          Make
          <select value={vendor} onChange={(e) => setVendor(e.target.value)}>
            {VENDORS.map((v) => (
              <option key={v.label} value={v.path}>
                {v.label}
                {v.note ? ` — ${v.note}` : ""}
              </option>
            ))}
          </select>
        </label>
        <label>
          Username
          <input
            value={user}
            onChange={(e) => setUser(e.target.value)}
            placeholder="camera account, not the phone app login"
            autoComplete="off"
          />
        </label>
        <label>
          Password
          <input
            type={reveal ? "text" : "password"}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete="new-password"
          />
        </label>
        <label className="camera-connect-reveal">
          <input
            type="checkbox"
            checked={reveal}
            onChange={(e) => setReveal(e.target.checked)}
          />
          Show password
        </label>
      </div>
      {url && (
        <div className="camera-connect-preview">
          <code>{shown}</code>
          <button type="button" onClick={() => onBuild(url)}>
            Use this address
          </button>
        </div>
      )}
    </fieldset>
  );
}
