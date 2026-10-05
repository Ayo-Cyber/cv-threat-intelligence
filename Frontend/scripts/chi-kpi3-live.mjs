import { _electron as electron } from '@playwright/test';
import fs from 'node:fs/promises';
import path from 'node:path';
import readline from 'node:readline';

const root = path.resolve('..');
const out = path.join(root, 'runs/chi_validation/scenarios/03_object_identification/live', new Date().toISOString().replaceAll(':', '-'));
await fs.mkdir(out, {recursive:true});
await fs.writeFile(path.join(out, 'site.json'), JSON.stringify({
  name:'Bottle identification test', configured:true, notify:'console',
  inference:{imgsz:512, confidence:.4, target_fps:4},
  cameras:[{id:'kpi3_webcam', source:'0', config:'configs/chi_object_state_v1.json'}],
}, null, 2));
const env = {...process.env, ARGUS_REPO:root,
  ARGUS_PYTHON:(process.env.ARGUS_PYTHON || 'python3'),
  ARGUS_SITE_CONFIG:path.join(out,'site.json'), ARGUS_DB:'runs/chi_validation/desktop/events.db',
  ARGUS_USER_DATA:path.join(root,'runs/chi_validation/desktop/user-data'),
  ARGUS_API_PORT:'8799', MPLCONFIGDIR:'/private/tmp'};
delete env.ELECTRON_RUN_AS_NODE;
const app = await electron.launch({args:[process.cwd(),'--production'],env,timeout:60000});
let page, input;
const invoke = (method,args=[]) => page.evaluate(({method,args}) => window.argusDesktop.invoke(method,args), {method,args});
try {
  page = await app.firstWindow();
  await page.waitForTimeout(10000);
  if(await page.getByLabel('Username',{exact:true}).isVisible()) {
    const credentials = JSON.parse(await fs.readFile(path.join(root,'runs/chi_validation/desktop/test-credentials.json')));
    await page.getByLabel('Username',{exact:true}).fill(credentials.username);
    await page.getByLabel('Password',{exact:true}).fill(credentials.password);
    await page.getByRole('button',{name:'Sign in',exact:true}).click();
  }
  await page.getByRole('button',{name:'Settings',exact:true}).click();
  await page.getByRole('heading',{name:'Object watchlists',exact:true}).scrollIntoViewIfNeeded();
  console.log(JSON.stringify({ready:true,out,text:await page.locator('body').innerText()}));
  input = readline.createInterface({input:process.stdin,terminal:false});
  for await (const line of input) {
    try {
      const cmd = JSON.parse(line);
      if(cmd.action==='close') break;
      let result;
      if(cmd.action==='invoke') result=await invoke(cmd.method,cmd.args||[]);
      if(cmd.action==='click') await page.getByRole(cmd.role||'button',{name:cmd.name,exact:true}).click();
      if(cmd.action==='fill') await page.getByLabel(cmd.label,{exact:true}).fill(cmd.value);
      if(cmd.action==='upload') await page.locator('input[type=file]').setInputFiles(cmd.path);
      if(cmd.action==='screenshot') {
        const file=path.join(out,'current-ui.png');
        await page.screenshot({path:file});
        result={image:file,text:await page.locator('body').innerText()};
      }
      console.log(JSON.stringify({action:cmd.action,result}));
    } catch(error) { console.log(JSON.stringify({error:String(error)})); }
  }
} finally {
  input?.close();
  process.stdin.pause();
  if(page) await invoke('stop_monitoring').catch(()=>{});
  await app.close();
  console.log('Closed test: '+out);
}
