import { _electron as electron } from '@playwright/test';
import fs from 'node:fs/promises';
import path from 'node:path';
import readline from 'node:readline';

const root = path.resolve('..');
const out = path.join(root, 'runs/chi_validation/scenarios/09_object_state/live', new Date().toISOString().replaceAll(':','-'));
await fs.mkdir(out, {recursive:true});
await fs.writeFile(path.join(out,'site.json'), JSON.stringify({
 name:'Registered-object live test',configured:true,notify:'console',
 inference:{imgsz:512,confidence:.4,target_fps:4},
 cameras:[{id:'kpi9_webcam',source:'0',config:'configs/chi_object_state_v1.json',registered_objects:[]}]
},null,2));
const env={...process.env,ARGUS_REPO:root,
 ARGUS_PYTHON:(process.env.ARGUS_PYTHON || 'python3'),
 ARGUS_SITE_CONFIG:path.join(out,'site.json'),ARGUS_DB:'runs/chi_validation/desktop/events.db',
 ARGUS_USER_DATA:path.join(root,'runs/chi_validation/desktop/user-data'),ARGUS_API_PORT:'8799',MPLCONFIGDIR:'/private/tmp'};
delete env.ELECTRON_RUN_AS_NODE;
const app=await electron.launch({args:[process.cwd(),'--production'],env,timeout:60000});
let page;
const invoke=(method,args=[])=>page.evaluate(({method,args})=>window.argusDesktop.invoke(method,args),{method,args});
async function capture(label='view') {
 const snap=await invoke('camera_snapshot',['kpi9_webcam']);
 if(snap.error) throw new Error(snap.error);
 await fs.writeFile(path.join(out,`${label}.jpg`),Buffer.from(snap.uri.split(',')[1],'base64'));
 await page.screenshot({path:path.join(out,`${label}-ui.png`)});
 return {w:snap.w,h:snap.h,image:path.join(out,`${label}.jpg`)};
}
try {
 page=await app.firstWindow();
 page.on('console',message=>{if(message.type()==='error') console.log('renderer console: '+message.text());});
 page.on('requestfailed',request=>console.log('request failed: '+request.failure()?.errorText));
 page.on('pageerror',error=>console.log(JSON.stringify({rendererError:String(error)})));
 await page.waitForTimeout(10000);
 if(await page.getByLabel('Username',{exact:true}).isVisible()) {
  const credentials=JSON.parse(await fs.readFile(path.join(root,'runs/chi_validation/desktop/test-credentials.json')));
  await page.getByLabel('Username',{exact:true}).fill(credentials.username);
  await page.getByLabel('Password',{exact:true}).fill(credentials.password);
  await page.getByRole('button',{name:'Sign in',exact:true}).click();
 }
 await page.getByRole('button',{name:'Cameras',exact:true}).click();
 await page.getByRole('button',{name:'Scene review',exact:true}).first().click();
 await page.getByRole('tab',{name:'Registered objects',exact:true}).click();
 await page.getByRole('button',{name:'Open live view',exact:true}).click();
 await page.getByText('LIVE PREVIEW',{exact:true}).waitFor();
 console.log(JSON.stringify({ready:true,out,snapshot:await capture('initial')}));
 if(process.env.ARGUS_CHECK_LIVE_PREVIEW==='1') {
  const region=page.getByLabel('Object region');
  const box=await region.boundingBox();
  await page.mouse.move(box.x+box.width*.3,box.y+box.height*.3);
  await page.mouse.down();
  await page.mouse.move(box.x+box.width*.7,box.y+box.height*.7);
  await page.mouse.up();
  await region.screenshot({path:path.join(out,'live-selection-before.png')});
  await page.waitForTimeout(2500);
  await region.screenshot({path:path.join(out,'live-selection-after.png')});
  console.log(JSON.stringify({livePreviewEvidence:out}));
 }
 const input=readline.createInterface({input:process.stdin,terminal:false});
 for await(const line of input) {
  try {
   const cmd=JSON.parse(line);
   let result;
   if(cmd.action==='close') break;
   if(cmd.action==='capture') { await page.getByRole('button',{name:'Reset selection',exact:true}).click(); result=await capture(cmd.label||'current'); }
   if(cmd.action==='invoke') result=await invoke(cmd.method,cmd.args||[]);
   if(cmd.action==='screenshot') {await page.screenshot({path:path.join(out,'current-ui.png')});result={image:path.join(out,'current-ui.png'),text:await page.locator('body').innerText()};}
   if(cmd.action==='register') {
    const snap=await invoke('camera_snapshot',['kpi9_webcam']);
    const box=await page.getByLabel('Object region').boundingBox();
    const [x1,y1,x2,y2]=cmd.region;
    await page.mouse.move(box.x+box.width*x1/snap.w,box.y+box.height*y1/snap.h);
    await page.mouse.down();
    await page.mouse.move(box.x+box.width*x2/snap.w,box.y+box.height*y2/snap.h);
    await page.mouse.up();
    await page.getByLabel('Object name',{exact:true}).fill(cmd.name);
    await page.getByRole('button',{name:'Register object',exact:true}).click();
    result=await invoke('registered_objects',['kpi9_webcam']);
   }
   console.log(JSON.stringify({action:cmd.action,result}));
  } catch(error) {console.log(JSON.stringify({error:String(error)}));}
 }
} finally {
 if(page && !page.isClosed()) { await page.screenshot({path:path.join(out,'final-ui.png')}).catch(()=>{}); console.log(await page.locator('body').innerText().catch(()=>'')); }
 if(page) await invoke('stop_monitoring').catch(()=>{});
 await app.close();
 console.log('Live test closed; evidence: '+out);
}
