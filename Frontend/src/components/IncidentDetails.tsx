import { useEffect, useState } from "react";
import { Check, Flag, ShieldCheck } from "lucide-react";
import type { Incident, Json, Transport } from "../lib/types";
import { Badge, Notice, Spinner } from "./common";
export default function IncidentDetails({
  event,
  api,
  onChange,
}: {
  event: Incident;
  api: Transport;
  onChange: () => Promise<void>;
}) {
  const [clip, setClip] = useState<Json>({});
  const [loading, setLoading] = useState(!event.demo_video);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState(event.note || "");
  const [done, setDone] = useState(event.triage_state === "resolved");
  useEffect(() => {
    let active = true;
    if (!event.demo_video)
      api
        .invoke("event_clip", [event.evidence_dir || null])
        .then((r) => active && setClip(r))
        .catch((e) => active && setError(e.message))
        .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, [event.id, api]);
  async function submit(outcome?: string) {
    setBusy(true);
    setError("");
    try {
      const r = await api.invoke(
        outcome ? "resolve_alert" : "acknowledge_alert",
        outcome ? [event.id, outcome, note] : [event.id],
      );
      if (r.ok === false || r.persisted === false)
        throw new Error("The review could not be saved.");
      if (outcome) setDone(true);
      await onChange();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      {error && <Notice error>{error}</Notice>}
      <div className="context-status">
        <Badge
          tone={event.priority.toLowerCase() === "critical" ? "red" : "amber"}
        >
          {event.priority}
        </Badge>
        <span>{new Date(event.ts * 1000).toLocaleString()}</span>
        {event.demo_video && <Badge>Sample incident</Badge>}
      </div>
      <div className="evidence">
        {loading ? (
          <Spinner />
        ) : event.demo_video || clip.uri ? (
          <video controls muted autoPlay src={event.demo_video || clip.uri} />
        ) : clip.frames?.length ? (
          <div className="evidence-frames">
            {clip.frames.map((uri: string, i: number) => (
              <img src={uri} key={i} alt={`Evidence frame ${i + 1}`} />
            ))}
          </div>
        ) : (
          <p>No recorded evidence is available for this event.</p>
        )}
      </div>
      <h3>Verification assessment</h3>
      <p className="assessment">
        {event.reason || "No assessment supplied by the engine."}
      </p>
      <dl className="facts">
        <dt>Camera</dt>
        <dd>{event.camera_id}</dd>
        <dt>Rule</dt>
        <dd>{event.rule}</dd>
        <dt>Review state</dt>
        <dd>{done ? "Resolved" : event.triage_state || event.review}</dd>
        {typeof event.confidence === "number" && (
          <>
            <dt>Model confidence</dt>
            <dd>{Math.round(event.confidence * 100)}%</dd>
          </>
        )}
      </dl>
      <label>
        Operator note
        <textarea
          rows={3}
          value={note}
          onChange={(e) => setNote(e.target.value)}
          placeholder="Record what happened and any action taken."
          disabled={done}
        />
      </label>
      {done ? (
        <Notice>
          Review saved. This incident remains in the activity record.
        </Notice>
      ) : (
        <>
          <button
            className="button"
            disabled={busy || event.triage_state === "acknowledged"}
            onClick={() => void submit()}
          >
            <Flag size={16} />
            Acknowledge incident
          </button>
          <div className="actions">
            <button
              className="button primary"
              disabled={busy}
              onClick={() => void submit("real")}
            >
              <ShieldCheck size={16} />
              Real incident
            </button>
            <button
              className="button"
              disabled={busy}
              onClick={() => void submit("false_alarm")}
            >
              <Check size={16} />
              False alarm
            </button>
            <button
              className="button"
              disabled={busy}
              onClick={() => void submit("inconclusive")}
            >
              Inconclusive
            </button>
          </div>
        </>
      )}
    </>
  );
}
