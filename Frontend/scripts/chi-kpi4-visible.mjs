import {_electron as electron} from '@playwright/test';
import fs from 'node:fs/promises';
import path from 'node:path';
const root = path.resolve('..');
const out = path.join(root, 'runs/chi_validation/scenarios/04_normal_movement/evidence', new Date().toISOString().replaceAll(':','-'));
await fs.mkdir(out, {recursive:true});
const env = {...process.env, ARGUS_REPO:root,
 ARGUS_PYTHON:(process.env.ARGUS_PYTHON || 'python3'),
 ARGUS_SITE_CONFIG:'runs/chi_validation/scenarios/04_normal_movement/config.json',
 ARGUS_DB:'runs/chi_validation/desktop/events.db',
 ARGUS_USER_DATA:path.join(root,'runs/chi_validation/desktop/user-data'),
 ARGUS_API_PORT:'8799', MPLCONFIGDIR:'/private/tmp'};
delete env.ELECTRON_RUN_AS_NODE;
const app=await electron.launch({args:[process.cwd(),'--production'],env,timeout:60000});
try {
 const page=await app.firstWindow();
 const errors=[]; page.on('pageerror',e=>errors.push(String(e)));
 await page.waitForTimeout(10000);
 if(await page.getByLabel('Username',{exact:true}).isVisible()) {
  const auth=JSON.parse(await fs.readFile(path.join(root,'runs/chi_validation/desktop/test-credentials.json'),'utf8'));
  await page.getByLabel('Username',{exact:true}).fill(auth.username);
  await page.getByLabel('Password',{exact:true}).fill(auth.password);
  await page.getByRole('button',{name:'Sign in',exact:true}).click();
  await page.waitForTimeout(5000);
 }
 await page.getByRole('button',{name:'Overview',exact:true}).click();
 await page.waitForTimeout(8000);
 await page.screenshot({path:path.join(out,'01-preview.png')});
 await page.getByRole('button',{name:'Start monitoring',exact:true}).click();
 const overlay=page.getByRole('switch',{name:'Person boxes: kpi4_walk',exact:true});
 await overlay.check();
 for(let i=0;i<2;i++) {
  await page.waitForTimeout(20000);
  await page.screenshot({path:path.join(out,`02-monitoring-${i}.png`)});
  await fs.writeFile(path.join(out,`state-${i}.txt`), await page.locator('body').innerText());
  console.log(await page.locator('body').innerText());
 }
 if(await overlay.count()) {
  await overlay.uncheck(); await page.waitForTimeout(2000);
  await page.screenshot({path:path.join(out,'03-overlay-toggle.png')});
 }
 await fs.writeFile(path.join(out,'renderer-errors.json'),JSON.stringify(errors));
 const stop=page.getByRole('button',{name:'Stop monitoring',exact:true});
 if(await stop.isVisible()) await stop.click();
} finally {await app.close();}
