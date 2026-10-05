import { _electron as electron, expect } from '@playwright/test';
import fs from 'node:fs/promises';
import path from 'node:path';
const root = path.resolve('..');
const output = path.join(root,'runs/chi_validation/scenarios/09_object_state/electron',new Date().toISOString().replaceAll(':','-'));
await fs.mkdir(output,{recursive:true});
const site = JSON.parse(await fs.readFile(path.join(root,'runs/chi_validation/scenarios/10_concealment/site-three.json')));
site.cameras = [site.cameras[0]];
site.cameras[0].registered_objects = [];
await fs.writeFile(path.join(output,'site.json'),JSON.stringify(site,null,2));
const env = {...process.env, ARGUS_REPO:root,
 ARGUS_PYTHON:(process.env.ARGUS_PYTHON || 'python3'),
 ARGUS_SITE_CONFIG:path.join(output,'site.json'), ARGUS_DB:'runs/chi_validation/desktop/events.db',
 ARGUS_USER_DATA:path.join(root,'runs/chi_validation/desktop/user-data'), ARGUS_API_PORT:'8799', MPLCONFIGDIR:'/private/tmp'};
delete env.ELECTRON_RUN_AS_NODE;
const app = await electron.launch({args:[process.cwd(),'--production'],env,timeout:60000});
try {
 const page = await app.firstWindow();
 const errors = [];
 page.on('pageerror',e=>errors.push(String(e)));
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
 await page.getByRole('button',{name:'Capture view',exact:true}).click();
 const region=page.getByLabel('Object region');
 await expect(region).toBeVisible({timeout:20000});
 const box=await region.boundingBox();
 await page.mouse.move(box.x+box.width*.2,box.y+box.height*.2);
 await page.mouse.down();
 await page.mouse.move(box.x+box.width*.65,box.y+box.height*.65);
 await page.mouse.up();
 await page.getByLabel('Object name',{exact:true}).fill('UI plumbing test');
 await page.getByRole('button',{name:'Register object',exact:true}).click();
 await expect(page.getByText('UI plumbing test',{exact:true})).toBeVisible();
 await page.screenshot({path:path.join(output,'registered-object.png'),fullPage:true});
 page.on('dialog',dialog=>dialog.accept());
 await page.getByRole('button',{name:'Remove UI plumbing test',exact:true}).click();
 await expect(page.getByText('UI plumbing test',{exact:true})).toHaveCount(0);
 if(errors.length) throw new Error(errors.join('\n'));
 await fs.writeFile(path.join(output,'result.json'),JSON.stringify({passed:true,errors,scope:'Electron/API registration and removal only; no model accuracy test'},null,2));
 console.log(`Electron registration check passed. Evidence: ${output}`);
} finally { await app.close(); }
