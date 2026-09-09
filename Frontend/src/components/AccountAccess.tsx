import { useEffect, useState } from "react";
import { Copy, KeyRound, UserPlus } from "lucide-react";
import { Notice, Spinner } from "./common";
import type { Auth, Json, Transport } from "../lib/types";

export function AccountHelp() {
  const [view, setView] = useState("");
  const [command, setCommand] = useState("");
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);
  useEffect(() => {
    if (view !== "recovery") return;
    window.argusDesktop
      ?.environment()
      .then((env) => setCommand(env.recovery_command || ""))
      .catch((e) => setError(e.message));
  }, [view]);
  return (
    <div className="account-help">
      <div className="actions">
        <button
          type="button"
          className="text-button"
          onClick={() => setView("recovery")}
        >
          <KeyRound size={16} />
          Forgot password?
        </button>
        <button
          type="button"
          className="text-button"
          onClick={() => setView("create")}
        >
          <UserPlus size={16} />
          Create account
        </button>
      </div>
      {error && <Notice error>{error}</Notice>}
      {view === "create" && (
        <Notice>
          An existing owner must create your account. Ask them to open Settings,
          then Users. Existing accounts will not be replaced.
        </Notice>
      )}
      {view === "recovery" && (
        <div>
          <h3>Recover a local account</h3>
          <p>
            On the computer that owns this installation, run this command in
            Terminal. Select your account and choose a new password. Existing
            evidence and other accounts are retained.
          </p>
          {command ? (
            <>
              <label>
                Recovery command
                <textarea readOnly value={command} rows={4} />
              </label>
              <button
                type="button"
                className="button"
                onClick={() => {
                  void navigator.clipboard
                    .writeText(command)
                    .then(() => setCopied(true))
                    .catch(() =>
                      setError("Select the command and copy it manually."),
                    );
                }}
              >
                <Copy size={16} />
                {copied ? "Copied" : "Copy command"}
              </button>
            </>
          ) : (
            <Spinner />
          )}
        </div>
      )}
    </div>
  );
}

export function UsersPanel({ api, auth }: { api: Transport; auth: Auth }) {
  const [users, setUsers] = useState<Json[]>([]);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const [role, setRole] = useState("operator");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const allowed = auth.permissions.includes("manage_users");
  useEffect(() => {
    if (allowed)
      void api
        .invoke<Json[]>("list_users")
        .then(setUsers)
        .catch((e) => setError(e.message));
  }, [api, allowed]);
  if (!allowed) return null;
  return (
    <section className="settings-section">
      <div>
        <h2>Users</h2>
        <p>Accounts authorized by this workspace's owner.</p>
      </div>
      <div>
        {error && <Notice error>{error}</Notice>}
        {success && <p role="status">{success}</p>}
        <ul className="account-list">
          {users.map((user) => (
            <li key={user.username}>
              <strong>{user.username}</strong>
              <span>{user.role}</span>
            </li>
          ))}
        </ul>
        <form
          onSubmit={async (e) => {
            e.preventDefault();
            setError("");
            setSuccess("");
            if (password !== confirmation) {
              setError("Passwords do not match.");
              return;
            }
            if (
              role === "owner" &&
              !confirm(
                "Give this account full owner access to the site and its users?",
              )
            )
              return;
            setBusy(true);
            try {
              await api.invoke("add_user", [username.trim(), password, role]);
              setPassword("");
              setConfirmation("");
              setUsername("");
              setUsers(await api.invoke("list_users"));
              setSuccess("Account created. The new user can now sign in.");
            } catch (e) {
              setError((e as Error).message);
            } finally {
              setBusy(false);
            }
          }}
        >
          <label>
            New account username
            <input
              required
              autoComplete="off"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
            />
          </label>
          <label>
            Access role
            <select value={role} onChange={(e) => setRole(e.target.value)}>
              <option value="operator">Operator</option>
              <option value="installer">Installer</option>
              <option value="owner">Owner</option>
            </select>
          </label>
          <label>
            New account password
            <input
              required
              type="password"
              minLength={12}
              autoComplete="new-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </label>
          <label>
            Confirm account password
            <input
              required
              type="password"
              minLength={12}
              autoComplete="new-password"
              value={confirmation}
              onChange={(e) => setConfirmation(e.target.value)}
            />
          </label>
          <button
            className="button primary"
            disabled={busy || !username.trim()}
          >
            {busy ? <Spinner /> : <UserPlus size={16} />}Create account
          </button>
        </form>
      </div>
    </section>
  );
}
