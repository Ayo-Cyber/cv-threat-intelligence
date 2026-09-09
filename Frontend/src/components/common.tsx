import { useEffect, useRef, useState } from "react";
import {
  X,
  LoaderCircle,
  VideoOff,
  Maximize2,
  Check,
  AlertTriangle,
} from "lucide-react";
import type { Camera } from "../lib/types";
export function Badge({
  children,
  tone = "neutral",
}: {
  children: React.ReactNode;
  tone?: string;
}) {
  return <span className={`badge ${tone}`}>{children}</span>;
}
export function Empty({
  title,
  children,
}: {
  title: string;
  children?: React.ReactNode;
}) {
  return (
    <div className="empty">
      <VideoOff size={25} />
      <h3>{title}</h3>
      <p>{children}</p>
    </div>
  );
}
export function Spinner() {
  return <LoaderCircle size={17} className="spin" />;
}
export function Notice({
  children,
  error = false,
}: {
  children: React.ReactNode;
  error?: boolean;
}) {
  return (
    <div className={`notice ${error ? "error" : ""}`}>
      <AlertTriangle size={16} />
      <span>{children}</span>
    </div>
  );
}
export function Drawer({
  expanded = false,
  title,
  subtitle,
  children,
  onClose,
}: {
  expanded?: boolean;
  title: string;
  subtitle?: string;
  children: React.ReactNode;
  onClose: () => void;
}) {
  const panel = useRef<HTMLElement>(null);
  const close = useRef<HTMLButtonElement>(null);
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;
  useEffect(() => {
    const last = document.activeElement as HTMLElement;
    close.current?.focus();
    const old = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape") onCloseRef.current();
      if (e.key === "Tab") {
        const list = Array.from(
          panel.current?.querySelectorAll<HTMLElement>(
            'button:not(:disabled),input:not(:disabled),select:not(:disabled),textarea:not(:disabled),[tabindex="0"]',
          ) || [],
        );
        if (e.shiftKey && document.activeElement === list[0]) {
          e.preventDefault();
          list.at(-1)?.focus();
        } else if (!e.shiftKey && document.activeElement === list.at(-1)) {
          e.preventDefault();
          list[0]?.focus();
        }
      }
    };
    document.addEventListener("keydown", key);
    return () => {
      document.body.style.overflow = old;
      document.removeEventListener("keydown", key);
      last?.focus();
    };
  }, []);
  return (
    <div
      className="backdrop"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <section
        className={`drawer ${expanded ? "drawer-expanded" : ""}`}
        ref={panel}
        role="dialog"
        aria-modal="true"
        aria-label={title}
      >
        <div className="drawer-head">
          <div>
            <small>{subtitle}</small>
            <h2>{title}</h2>
          </div>
          <button
            className="icon-button"
            ref={close}
            onClick={onClose}
            aria-label="Close details"
            title="Close"
          >
            <X size={20} />
          </button>
        </div>
        {children}
      </section>
    </div>
  );
}
export function CameraMedia({
  camera,
  stream,
  paused = false,
  controls = false,
  onOpen,
}: {
  camera: Camera;
  stream?: string;
  paused?: boolean;
  controls?: boolean;
  onOpen?: () => void;
}) {
  const video = useRef<HTMLVideoElement>(null);
  const [failed, setFailed] = useState(false);
  useEffect(
    () => setFailed(false),
    [camera.demo_video, stream, camera.snapshot],
  );
  useEffect(() => {
    if (video.current) {
      if (paused) video.current.pause();
      else void video.current.play().catch(() => {});
    }
  }, [paused, camera.demo_video]);
  return (
    <div className="camera-media">
      {failed ? (
        <Empty title="Media unavailable">The source could not be loaded.</Empty>
      ) : camera.demo_video ? (
        <video
          ref={video}
          src={camera.demo_video}
          poster={camera.snapshot}
          autoPlay={!paused}
          controls={controls}
          muted
          loop
          playsInline
          preload="metadata"
          onError={() => setFailed(true)}
        />
      ) : stream ? (
        <img
          src={stream}
          alt={`${camera.id} live feed`}
          onError={() => setFailed(true)}
        />
      ) : camera.snapshot ? (
        <img
          src={camera.snapshot}
          alt={`${camera.id} snapshot`}
          onError={() => setFailed(true)}
        />
      ) : (
        <Empty title="No camera frame">
          Connect the camera to see its view.
        </Empty>
      )}
      <div className="media-label">
        <span className={`status-dot ${camera.demo_video ? "sample" : ""}`} />
        {failed
          ? "UNAVAILABLE"
          : camera.demo_video
            ? "RECORDED CLIP"
            : stream
              ? "CAMERA FEED"
              : "NO LIVE FEED"}
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
export function Toggle({
  checked,
  onChange,
  label,
  disabled = false,
}: {
  checked: boolean;
  onChange: () => void;
  label: string;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-label={label}
      aria-checked={checked}
      disabled={disabled}
      className={`toggle ${checked ? "on" : ""}`}
      onClick={onChange}
    >
      <span>{checked && <Check size={10} />}</span>
    </button>
  );
}
