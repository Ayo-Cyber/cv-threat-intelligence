import { useEffect, useState } from "react";
import { Check, Flag, ShieldCheck } from "lucide-react";
import type { Incident, Transport } from "../lib/types";
import {
  clipRequestKey,
  evidenceView,
  playbackProblem,
  PENDING_POLL_MS,
  type ClipReply,
} from "../lib/evidence";
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
  const [clip, setClip] = useState<ClipReply>({});
  const [loading, setLoading] = useState(!event.demo_video);
  const [videoFailed, setVideoFailed] = useState(false);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState(event.note || "");
  const [done, setDone] = useState(event.triage_state === "resolved");
  const view = evidenceView(clip, event, { videoFailed });
  // Ask by event id (never by evidence path: a freshly pushed alert has
  // none yet), ask again when the settled row arrives with its evidence,
  // and while the engine is still writing, poll until it lands.
  useEffect(() => {
    if (event.demo_video) return;
    let active = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    setVideoFailed(false);
    const fetchClip = () =>
      api
        .invoke<ClipReply>("event_clip", [clipRequestKey(event)])
        .then((r) => {
          if (!active) return;
          setClip(r || {});
          if (evidenceView(r, event) === "pending")
            timer = setTimeout(fetchClip, PENDING_POLL_MS);
        })
        .catch((e) => active && setError(e.message))
        .finally(() => active && setLoading(false));
    void fetchClip();
    return () => {
      active = false;
      if (timer) clearTimeout(timer);
    };
  }, [event.id, event.evidence_dir, api]);
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
  const problem = playbackProblem(clip, videoFailed);
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
      {problem && <Notice>{problem}</Notice>}
      <div className="evidence">
        {loading ? (
          <Spinner />
        ) : event.demo_video ? (
          <video controls muted autoPlay src={event.demo_video} />
        ) : view === "video" ? (
          <video
            controls
            muted
            autoPlay
            src={clip.uri || undefined}
            onError={() => setVideoFailed(true)}
          />
        ) : view === "frames" ? (
          <div className="evidence-frames">
            {(clip.frames || []).map((uri: string, i: number) => (
              <img src={uri} key={i} alt={`Evidence frame ${i + 1}`} />
            ))}
          </div>
        ) : view === "pending" ? (
          <p>
            Evidence is still being written for this alert. The replay appears
            here as soon as the engine has saved it, usually within a few
            seconds of the verdict.
          </p>
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
