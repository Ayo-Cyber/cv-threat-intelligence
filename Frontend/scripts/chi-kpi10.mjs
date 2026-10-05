import {_electron as electron} from '@playwright/test';
import fs from 'node:fs/promises';
import path from 'node:path';
const root=path.resolve('..');
const scenario=path.join(root,'runs/chi_validation/scenarios/10_concealment');
const out=path.join(scenario,'evidence',new Date().toISOString().replaceAll(':','-'));
await fs.mkdir(out,{recursive:true});
const env={...process.env,ARGUS_REPO:root,ARGUS_PYTHON:(process.env.ARGUS_PYTHON || 'python3'),ARGUS_SITE_CONFIG:path.join(scenario,'site.json'),ARGUS_DB:'runs/chi_validation/desktop/events.db',ARGUS_USER_DATA:path.join(root,'runs/chi_validation/desktop/user-data'),ARGUS_API_PORT:'8799',MPLCONFIGDIR:'/private/tmp',ARGUS_TRACKING_DIAGNOSTICS:'1'};
delete env.ELECTRON_RUN_AS_NODE;
if(process.argv.includes('--dense')) env.ARGUS_SITE_CONFIG=path.join(scenario,'site-dense.json');
if(process.argv.includes('--timed')) env.ARGUS_SITE_CONFIG=path.join(scenario,'site-timed.json');
if(process.argv.includes('--wig')) env.ARGUS_SITE_CONFIG=path.join(scenario,'site-wig.json');
const three = process.argv.includes('--three');
if(three) env.ARGUS_SITE_CONFIG=path.join(scenario,'site-three.json');
const seconds = three ? 180 : 120;
await fs.copyFile(env.ARGUS_SITE_CONFIG,path.join(out,'site-config.json'));
const app=await electron.launch({args:[process.cwd(),'--production'],env,timeout:60000});
let page;
try {
 page=await app.firstWindow();
 await page.waitForTimeout(10000);
 if(await page.getByLabel('Username',{exact:true}).isVisible()) {
  const a=JSON.parse(await fs.readFile(path.join(root,'runs/chi_validation/desktop/test-credentials.json')));
  await page.getByLabel('Username',{exact:true}).fill(a.username);
  await page.getByLabel('Password',{exact:true}).fill(a.password);
  await page.getByRole('button',{name:'Sign in',exact:true}).click();
  await page.waitForTimeout(5000);
 }
 await page.getByRole('button',{name:'Overview',exact:true}).click();
 await page.getByRole('switch',{name:'Person boxes: all cameras',exact:true}).check();
 await fs.writeFile(path.join(out,'run-start.json'),JSON.stringify({started_at:Date.now()/1000,seconds}));
 await page.getByRole('button',{name:'Start monitoring',exact:true}).click();
 console.log(`KPI 10 real engine test running for ${seconds} seconds.`);
 let lastNotice = '';
 for(let i=0;i<seconds;i++) {
  await page.waitForTimeout(1000);
  const notice = await page.locator('.concealment-notice').allTextContents();
  const text = notice.join(' | ');
  if(text && text !== lastNotice)
   await page.screenshot({path:path.join(out,`notice-${i}.png`),fullPage:true});
  lastNotice = text;
  if(i % 20 === 19)
   await page.screenshot({path:path.join(out,`monitoring-${Math.floor(i/20)}.png`),fullPage:true});
 }
 await fs.writeFile(path.join(out,'screen.txt'),await page.locator('body').innerText());
 await fs.copyFile(path.join(root,'runs/chi_validation/desktop/gate_health.json'),path.join(out,'gate-health.json'));
} finally {
 try {
  // Stop through the authenticated bridge: an operator may have a modal open.
  if(page && !page.isClosed()) {
   await page.evaluate(() => window.argusDesktop.invoke('stop_monitoring'));
   await page.waitForTimeout(3000);
  }
 } finally {
  await fs.copyFile(path.join(root,'runs/chi_validation/desktop/monitor.log'),path.join(out,'monitor.log')).catch(()=>{});
  await fs.cp(path.join(root,'runs/chi_validation/desktop/gate'),path.join(out,'gate'),{recursive:true}).catch(()=>{});
  await app.close();
  console.log(`Evidence: ${out}`);
 }
}
