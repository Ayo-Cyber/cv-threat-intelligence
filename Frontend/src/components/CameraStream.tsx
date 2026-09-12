import { useEffect, useRef, useState } from "react";
import { Maximize2 } from "lucide-react";
import {
  cameraStreamPresentation,
  type MediaEvidence,
  type StreamTransport,
} from "../lib/stream-state";
import type { Camera, Transport } from "../lib/types";
import { resolveCameraStream, type ResolvedCameraStream } from "../lib/whep";
import { Empty, Spinner } from "./common";

type PlayerState =
  | { kind: "loading" }
  | ResolvedCameraStream
  | { kind: "offline"; message: string };

export default function CameraStream({
  camera,
  api,
  active,
  onOpen,
}: {
  camera: Camera;
  api: Transport;
  active: boolean;
  onOpen?: () => void;
}) {
  const video = useRef<HTMLVideoElement>(null);
  const [state, setState] = useState<PlayerState>({
    kind: active ? "loading" : "inactive",
  });
  const [imageFailed, setImageFailed] = useState(false);
  const [evidence, setEvidence] = useState<MediaEvidence>("none");

  useEffect(() => {
    const controller = new AbortController();
    setImageFailed(false);
    setEvidence("none");
    if (!active) {
      setState({ kind: "inactive" });
      return () => controller.abort();
    }
    setState({ kind: "loading" });
    if (video.current)
      void resolveCameraStream({
        cameraId: camera.id,
        active,
        api,
        video: video.current,
        signal: controller.signal,
        onWebRtcTrack: () => setEvidence("webrtc-track"),
      })
        .then((next) => !controller.signal.aborted && setState(next))
        .catch((error) => {
          if (!controller.signal.aborted)
            setState({
              kind: "offline",
              message: (error as Error).message,
            });
        });
    return () => controller.abort();
  }, [active, api, camera.id]);

  const transport: StreamTransport =
    state.kind === "webrtc"
      ? "webrtc"
      : state.kind === "mjpeg"
        ? state.degraded
          ? "mjpeg-fallback"
          : "mjpeg"
        : "none";
  const presentation = cameraStreamPresentation({
    active,
    cameraState: camera.state,
    transport,
    evidence,
    failed: state.kind === "offline" || imageFailed,
  });

  return (
    <div className="camera-media">
      <video
        ref={video}
        className={
          state.kind === "webrtc" && presentation.phase !== "offline"
            ? ""
            : "stream-hidden"
        }
        autoPlay
        muted
        playsInline
      />
      {state.kind === "mjpeg" && presentation.phase !== "offline" ? (
        <img
          src={state.url}
          alt={`${camera.id} live feed`}
          onLoad={() => setEvidence("mjpeg-frame")}
          onError={() => {
            setEvidence("none");
            setImageFailed(true);
          }}
        />
      ) : presentation.phase === "loading" ? (
        <div className="loading">
          <Spinner />
          Connecting feed...
        </div>
      ) : presentation.phase === "offline" ? (
        <Empty title="Camera offline">
          {state.kind === "offline"
            ? state.message
            : imageFailed
              ? "The fallback stream could not be loaded."
              : "Camera health reports that this feed is offline."}
        </Empty>
      ) : presentation.phase === "inactive" ? (
        <Empty title="Preview inactive">
          This camera preview is not visible.
        </Empty>
      ) : presentation.phase === "degraded" && evidence === "none" ? (
        <Empty title="Camera degraded">
          Camera health reports that this feed is reconnecting.
        </Empty>
      ) : null}
      <div className="media-label">
        <span
          className={`status-dot ${
            presentation.phase === "offline"
              ? "off"
              : presentation.phase === "degraded"
                ? "sample"
                : ""
          }`}
        />
        {presentation.label}
      </div>
      {onOpen && (
        <button
          className="media-open"
          title={`Open ${camera.id}`}
          aria-label={`Open ${camera.id}`}
          onClick={onOpen}
        >
          <Maximize2 size={17} />
        </button>
      )}
    </div>
  );
}
