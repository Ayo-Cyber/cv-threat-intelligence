const { contextBridge, ipcRenderer } = require("electron");
contextBridge.exposeInMainWorld(
  "argusDesktop",
  Object.freeze({
    invoke: (method, args = []) =>
      ipcRenderer.invoke("engine:invoke", method, args),
    environment: () => ipcRenderer.invoke("engine:environment"),
  }),
);
