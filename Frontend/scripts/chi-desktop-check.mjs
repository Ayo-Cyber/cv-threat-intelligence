import { _electron as electron } from '@playwright/test';
import fs from 'node:fs/promises';
import path from 'node:path';
import {randomBytes} from 'node:crypto';

const root = path.resolve('..');
const output = path.join(root, 'runs/chi_validation/desktop');
await fs.mkdir(output, {recursive: true});
const env = {...process.env,
  ARGUS_REPO: root,
  ARGUS_PYTHON: (process.env.ARGUS_PYTHON || 'python3'),
  ARGUS_SITE_CONFIG: 'runs/chi_validation/site.json',
  ARGUS_DB: 'runs/chi_validation/desktop/events.db',
  ARGUS_USER_DATA: path.join(output, 'user-data'),
  ARGUS_API_PORT: '8799', MPLCONFIGDIR: '/private/tmp'};
delete env.ELECTRON_RUN_AS_NODE;
const app = await electron.launch({args: [process.cwd(), '--production'], env, timeout: 60000});
try {
  const page = await app.firstWindow();
  const errors = [];
  page.on('pageerror', error => errors.push(String(error)));
  await page.waitForTimeout(20000);
  const owner = page.getByRole('button', {name: 'Create owner account', exact: true});
  if (await owner.isVisible()) {
    const password = randomBytes(24).toString('hex');
    await fs.writeFile(path.join(output, 'test-credentials.json'), JSON.stringify({username: 'chi_test', password}), {mode: 0o600});
    await page.getByLabel('Username', {exact: true}).fill('chi_test');
    await page.getByLabel('Password', {exact: true}).fill(password);
    await owner.click();
    await page.waitForTimeout(12000);
  }
  await page.screenshot({path: path.join(output, 'startup.png')});
  const result = {title: await page.title(), text: await page.locator('body').innerText(), errors};
  await fs.writeFile(path.join(output, 'startup.json'), JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result));
} finally {
  await app.close();
}
