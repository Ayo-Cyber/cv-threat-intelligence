import {_electron as electron} from '@playwright/test';
import fs from 'node:fs/promises';
import path from 'node:path';
const root=path.resolve('..');
const four=process.argv.includes('--four');
const kpi5=process.argv.includes('--kpi5');
const negative=process.argv.includes('--negative');
const realCrowd=process.argv.includes('--real-crowd');
if(realCrowd && (!kpi5 || negative)) throw new Error('--real-crowd requires --kpi5 and cannot use --negative');
if(negative && !kpi5) throw new Error('--negative requires --kpi5');
if(kpi5 && !four) throw new Error('--kpi5 requires --four');
const demo=process.argv.includes('--demo');
const out=path.join(root,'runs/chi_validation/scenarios',kpi5?'05_multiple_people_moving':'04_normal_movement','evidence',realCrowd?'real-crowd':negative?'negative-controls':four?'four-videos':'three-videos',new Date().toISOString().replaceAll(':','-'));
await fs.mkdir(out,{recursive:true});
const env={...process.env,ARGUS_REPO:root,ARGUS_PYTHON:(process.env.ARGUS_PYTHON || 'python3'),ARGUS_SITE_CONFIG:'runs/chi_validation/scenarios/04_normal_movement/three-videos.json',ARGUS_DB:'runs/chi_validation/desktop/events.db',ARGUS_USER_DATA:path.join(root,'runs/chi_validation/desktop/user-data'),ARGUS_API_PORT:'8799',MPLCONFIGDIR:'/private/tmp'};
delete env.ELECTRON_RUN_AS_NODE;
if(four) env.ARGUS_SITE_CONFIG='runs/chi_validation/scenarios/04_normal_movement/four-videos.json';
if(kpi5) env.ARGUS_SITE_CONFIG='runs/chi_validation/scenarios/05_multiple_people_moving/site.json';
if(kpi5 && negative) env.ARGUS_SITE_CONFIG='runs/chi_validation/scenarios/05_multiple_people_moving/negative-site.json';
if(realCrowd) env.ARGUS_SITE_CONFIG='runs/chi_validation/scenarios/05_multiple_people_moving/real-crowd-site.json';
env.ARGUS_TRACKING_DIAGNOSTICS='1';
await fs.copyFile(path.join(root,env.ARGUS_SITE_CONFIG),path.join(out,'site-config.json'));
const app=await electron.launch({args:[process.cwd(),'--production'],env,timeout:60000});
try {
 const page=await app.firstWindow();
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
 await page.getByRole('switch',{name:'Person boxes: retail_walk',exact:true}).uncheck();
 if(await page.getByRole('switch',{name:'Person boxes: street_two',exact:true}).isChecked()===false) throw Error('Individual toggle affected another camera');
 await page.getByRole('switch',{name:'Person boxes: all cameras',exact:true}).uncheck();
 for(const id of ['street_two','retail_walk','caviar_pair',...(four?['crowd_walk']:[])]) {
  if(await page.getByRole('switch',{name:`Person boxes: ${id}`,exact:true}).isChecked()) throw Error('All-off failed');
 }
 await page.getByRole('switch',{name:'Person boxes: all cameras',exact:true}).check();
 await page.getByRole('button',{name:'Start monitoring',exact:true}).click();
 if(demo) {
  console.log('Four-camera demo running. Close the Argus window to finish.');
  await new Promise(resolve => app.on('close',resolve));
 } else {
 for(let i=0;i<3;i++) {
  await page.waitForTimeout(20000);
  await page.screenshot({path:path.join(out,`monitoring-${i}.png`),fullPage:true});
 }
 await page.getByRole('switch',{name:'Person boxes: retail_walk',exact:true}).uncheck();
 await page.waitForTimeout(3000);
 await page.screenshot({path:path.join(out,'single-camera-off.png'),fullPage:true});
 await page.getByRole('switch',{name:'Person boxes: all cameras',exact:true}).uncheck();
 await page.waitForTimeout(3000);
 await page.screenshot({path:path.join(out,'all-off.png'),fullPage:true});
 await fs.writeFile(path.join(out,'result.json'),JSON.stringify({controlsPassed:true,screen:await page.locator('body').innerText()},null,2));
 await fs.copyFile(path.join(root,'runs/chi_validation/desktop/gate_health.json'),path.join(out,'gate-health.json'));
 await page.setViewportSize({width:800,height:900});
 await page.screenshot({path:path.join(out,'narrow-layout.png'),fullPage:true});
 await page.getByRole('button',{name:'Stop monitoring',exact:true}).click();
 await fs.copyFile(path.join(root,'runs/chi_validation/desktop/monitor.log'),path.join(out,'monitor.log'));
 console.log(`${four?'Four':'Three'}-video desktop run completed; evidence: ${out}`);
 }
} finally {await app.close().catch(()=>{});}
