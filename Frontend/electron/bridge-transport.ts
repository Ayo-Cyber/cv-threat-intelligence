type LegacyInvoke = (method: string, args: unknown[]) => Promise<any>;
type LoadStream = (url: string) => Promise<Response>;

export function createBridgeTransport(
  legacyInvoke: LegacyInvoke,
  loadStream: LoadStream,
) {
  let live: Promise<any> | undefined;
  const publisher = () => {
    live ??= legacyInvoke("live_start", [6]).catch((error) => {
      live = undefined;
      throw error;
    });
    return live;
  };
  return {
    async invoke(method: string, args: unknown[] = []) {
      if (method === "sign_out") {
        if (live) {
          try {
            await live;
            await legacyInvoke("live_stop", []);
          } catch {
            /* Sign-out still revokes the account session if the publisher failed. */
          } finally {
            live = undefined;
          }
        }
        return legacyInvoke(method, args);
      }
      if (method !== "camera_stream") return legacyInvoke(method, args);
      await publisher();
      return {
        kind: "mjpeg" as const,
        url: `argus-stream://camera/${encodeURIComponent(String(args[0]))}`,
      };
    },
    async load(cameraId: string) {
      if (!live) throw new Error("Bridge stream is not active.");
      const descriptor = await live;
      if (!descriptor?.port)
        throw new Error("The legacy bridge did not start a camera publisher.");
      const upstream =
        `http://127.0.0.1:${descriptor.port}/stream/` +
        `${encodeURIComponent(cameraId)}?token=${encodeURIComponent(descriptor.token ?? "")}`;
      return loadStream(upstream);
    },
  };
}
