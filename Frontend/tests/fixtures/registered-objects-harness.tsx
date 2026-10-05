import { createRoot } from "react-dom/client";
import RegisteredObjects from "../../src/components/RegisteredObjects";
import "../../src/styles.css";
import type { Transport } from "../../src/lib/types";
const canvas = document.createElement("canvas");
canvas.width = 640; canvas.height = 360;
const ctx = canvas.getContext("2d")!;
ctx.fillStyle = "#486257"; ctx.fillRect(0,0,640,360);
ctx.fillStyle = "#d9dbd8"; ctx.fillRect(160,90,240,180);
let entries: any[] = [];
(window as any).calls = [];
const api: Transport = {
  subscribe: () => () => {},
  async invoke<T>(method: string, args: unknown[] = []) {
    (window as any).calls.push({method,args});
    let result: any = {};
    if (method === "camera_snapshot") result = {uri:canvas.toDataURL(),w:640,h:360};
    if (method === "camera_stream") result = {kind:"mjpeg",url:canvas.toDataURL(),preview:true};
    if (method === "registered_objects") result = entries;
    if (method === "register_object_region") { entries = [{id:"one",name:args[1],state:"capturing_reference"}]; result = {ok:true}; }
    if (method === "remove_registered_object") entries = [];
    return result as T;
  },
};
createRoot(document.getElementById("root")!).render(<main className="drawer drawer-expanded"><header className="drawer-head"><h2>Camera intelligence</h2></header><fieldset><RegisteredObjects camera={{id:"test",source:"test"}} api={api}/></fieldset></main>);
