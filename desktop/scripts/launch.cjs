const { spawn } = require('node:child_process');
const path = require('node:path');
const electron = require('electron');

// Some Electron-based editors export this for their own child tools. Argus is
// a desktop process, not an Electron-as-Node subprocess.
const env = { ...process.env };
delete env.ELECTRON_RUN_AS_NODE;
const child = spawn(electron, [path.resolve(__dirname, '..'), ...process.argv.slice(2)], {
  env,
  stdio: 'inherit',
});
child.on('error', (error) => { console.error(error.message); process.exitCode = 1; });
child.on('exit', (code) => { process.exitCode = code ?? 1; });
for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => child.kill(signal));
