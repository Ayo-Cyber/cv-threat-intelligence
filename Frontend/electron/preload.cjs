const { contextBridge, ipcRenderer } = require("electron");
contextBridge.exposeInMainWorld(
  "argusDesktop",
  Object.freeze({
    invoke: (method, args = []) =>
      ipcRenderer.invoke("engine:invoke", method, args),
    subscribe: (listener) => {
      if (typeof listener !== "function")
        throw new TypeError("Listener required");
      const handler = (_event, value) => {
        if (
          !value ||
          typeof value !== "object" ||
          !["health", "triage", "alert.new", "alert.update"].includes(
            value.type,
          ) ||
          !("data" in value)
        )
          return;
        listener(Object.freeze({ type: value.type, data: value.data }));
      };
      ipcRenderer.on("engine:event", handler);
      return () => ipcRenderer.removeListener("engine:event", handler);
    },
    environment: () => ipcRenderer.invoke("engine:environment"),
  }),
);
