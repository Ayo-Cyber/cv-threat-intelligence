import { useEffect, useState } from "react";
import { Camera as CameraIcon, Plus, Trash2, RotateCw } from "lucide-react";
import type { Camera, Transport } from "../lib/types";
import { Notice } from "./common";
import CameraStream from "./CameraStream";

type Snapshot = { uri: string; w: number; h: number; error?: string };
type Entry = { id: string; name: string; state: string };
export default function RegisteredObjects({ camera, api }: { camera: Camera; api: Transport }) {
  const [items, setItems] = useState<Entry[]>([]);
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [start, setStart] = useState<number[] | null>(null);
  const [region, setRegion] = useState<number[] | null>(null);
  const [name, setName] = useState("");
  const [seconds, setSeconds] = useState(8);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [live, setLive] = useState(false);
  useEffect(() => {
    let active = true;
    const refresh = () => api.invoke<Entry[]>("registered_objects", [camera.id])
      .then(r => { if (active) setItems(r); })
      .catch(e => { if (active) setError(e.message); });
    void refresh();
    const timer = setInterval(refresh, 3000);
    return () => { active = false; clearInterval(timer); };
  }, [api, camera.id]);
  async function perform(action: () => Promise<void>) {
    setBusy(true); setError("");
    try { await action(); setItems(await api.invoke("registered_objects", [camera.id])); }
    catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  }
  const point = (e: React.PointerEvent<HTMLDivElement>) => {
    const box = e.currentTarget.getBoundingClientRect();
    return [Math.round(Math.max(0, Math.min(1, (e.clientX-box.left)/box.width)) * snapshot!.w),
      Math.round(Math.max(0, Math.min(1, (e.clientY-box.top)/box.height)) * snapshot!.h)];
  };
  return <section className="registered-objects-editor">
    <h3>Registered objects</h3>
    {error && <Notice error>{error}</Notice>}
    <div className="setting-row">
      <button className="button" disabled={busy} onClick={() => void perform(async () => {
        const next = await api.invoke<Snapshot>("camera_snapshot", [camera.id]);
        if (next.error || !next.uri) throw new Error(next.error || "Camera image unavailable");
        setSnapshot(next); setRegion(null);
      })}><CameraIcon size={16}/> {snapshot ? "Reset selection" : "Open live view"}</button>
    </div>
    {snapshot && <>
      <div aria-label="Object region" className="registered-region" style={{ aspectRatio: `${snapshot.w}/${snapshot.h}`, width: `min(100%, ${60*snapshot.w/snapshot.h}vh)` }}
        onPointerDown={e => { if (busy || !live) return; e.preventDefault(); e.currentTarget.setPointerCapture(e.pointerId); const p = point(e); setStart(p); setRegion([...p, ...p]); }}
        onPointerMove={e => { if (!start) return; const p = point(e); setRegion([Math.min(start[0],p[0]), Math.min(start[1],p[1]), Math.max(start[0],p[0]), Math.max(start[1],p[1])]); }}
        onPointerUp={() => setStart(null)} onPointerCancel={() => { setStart(null); setRegion(null); }}>
        <CameraStream camera={camera} api={api} active onLiveChange={setLive}/>
        {region && <div className="registered-region-box" style={{ left: `${region[0]/snapshot.w*100}%`, top: `${region[1]/snapshot.h*100}%`, width: `${(region[2]-region[0])/snapshot.w*100}%`, height: `${(region[3]-region[1])/snapshot.h*100}%` }}/>}
      </div>
      <label>Object name<input value={name} maxLength={80} onChange={e => setName(e.target.value)}/></label>
      <label>Confirmation time (seconds)<input type="number" min={2} max={300} value={seconds} onChange={e => setSeconds(Number(e.target.value))}/></label>
      <button className="button primary" disabled={busy || !live || !name.trim() || !region || region[2]-region[0]<12 || region[3]-region[1]<12}
        onClick={() => void perform(async () => {
          const result = await api.invoke("register_object_region", [camera.id, name.trim(), region, [snapshot.h,snapshot.w], seconds]);
          if (result.ok === false) throw new Error(result.error || "Registration failed");
          setName(""); setRegion(null);
        })}><Plus size={16}/> Register object</button>
    </>}
    {items.map(item => <div className="setting-row" key={item.id}>
      <div><strong>{item.name}</strong><small>{item.state.replaceAll("_", " ")}</small></div>
      <button className="icon-button" disabled={busy} title={`Recapture ${item.name}`} aria-label={`Recapture ${item.name}`}
        onClick={() => { if (confirm(`Confirm ${item.name} is visible in its registered position before replacing the reference.`)) void perform(async () => { await api.invoke("recapture_registered_object", [camera.id,item.id]); }); }}><RotateCw size={16}/></button>
      <button className="icon-button" disabled={busy} title={`Remove ${item.name}`} aria-label={`Remove ${item.name}`}
        onClick={() => { if (confirm(`Stop monitoring ${item.name}?`)) void perform(async () => { await api.invoke("remove_registered_object", [camera.id,item.id]); }); }}><Trash2 size={16}/></button>
    </div>)}
  </section>;
}
