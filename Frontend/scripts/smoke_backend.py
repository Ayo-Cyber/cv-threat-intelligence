"""Verify the real Python adapter using an isolated site and account database."""
import argparse
import json
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--repo', type=Path, required=True)
args = parser.parse_args()
bridge = Path(__file__).resolve().parents[1] / 'bridge.py'
with tempfile.TemporaryDirectory(prefix='argus-desktop-smoke-') as folder:
    root = Path(folder)
    site = root / 'site.json'
    site.write_text(json.dumps({'name': 'Desktop integration test', 'notify': 'console', 'configured': False, 'cameras': []}))
    commands = [
        ('auth_state', []),
        ('create_first_owner', ['desktop_smoke', secrets.token_urlsafe(24)]),
        ('get_site', []),
        ('add_camera', [{'id': 'smoke_camera', 'source': str(args.repo / 'data/test_clips/empty_warehouse.mp4')}]),
        ('list_cameras', []),
        ('scene_context', ['smoke_camera']),
        ('list_areas', []),
        ('set_camera_rules', ['smoke_camera', {'running': False}]),
        ('monitoring_status', []),
        ('camera_snapshot', ['smoke_camera']),
        ('shutdown', []),
    ]
    data = ''.join(json.dumps({'id': i, 'method': method, 'args': values})+'\n' for i, (method, values) in enumerate(commands))
    p = subprocess.run([sys.executable, str(bridge), '--repo', str(args.repo), '--site', str(site), '--db', str(root/'events.db')], input=data, capture_output=True, text=True, timeout=90)
    replies = []
    for line in p.stdout.splitlines():
        replies.append(json.loads(line))
    errors = [r for r in replies if r.get('error')]
    if p.returncode or errors or len(replies) != len(commands):
        print(p.stderr[-4000:])
        print('Protocol errors:', errors, 'Replies:', len(replies), 'Exit:', p.returncode)
        raise SystemExit(1)
    assert replies[1]['result']['signed_in']
    assert replies[4]['result'][0]['id'] == 'smoke_camera'
    assert replies[9]['result']['uri'].startswith('data:image/jpeg;base64,')
    print('PASS: real backend auth, camera configuration, scene lookup, status and image snapshot.')
    print('All test accounts and site data were isolated; no monitoring started.')
