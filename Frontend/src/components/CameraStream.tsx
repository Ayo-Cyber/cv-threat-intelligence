import { useEffect, useRef, useState } from "react";
import { Maximize2 } from "lucide-react";
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

  useEffect(() => {
    const controller = new AbortController();
    setImageFailed(false);
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

  const offline = state.kind === "offline" || imageFailed;
  return (
    <div className="camera-media">
      <video
        ref={video}
        className={state.kind === "webrtc" ? "" : "stream-hidden"}
        autoPlay
        muted
        playsInline
      />
      {state.kind === "mjpeg" && !imageFailed ? (
        <img
          src={state.url}
          alt={`${camera.id} live feed`}
          onError={() => setImageFailed(true)}
        />
      ) : state.kind === "loading" ? (
        <div className="loading">
          <Spinner />
          Connecting feed...
        </div>
      ) : offline ? (
        <Empty title="Camera offline">
          {state.kind === "offline"
            ? state.message
            : "The fallback stream could not be loaded."}
        </Empty>
      ) : state.kind === "inactive" ? (
        <Empty title="Preview inactive">
          This camera preview is not visible.
        </Empty>
      ) : null}
      <div className="media-label">
        <span className={`status-dot ${offline ? "off" : ""}`} />
        {offline
          ? "OFFLINE"
          : state.kind === "mjpeg" && state.degraded
            ? "MJPEG FALLBACK"
            : state.kind === "mjpeg"
              ? "MJPEG FEED"
              : state.kind === "webrtc"
                ? "WEBRTC FEED"
                : state.kind === "loading"
                  ? "CONNECTING"
                  : "INACTIVE"}
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
