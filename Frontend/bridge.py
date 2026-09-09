"""Source-run Electron adapter over the existing, permission-enforced backend.

JSON lines over private stdin/stdout, never a public HTTP service. Replace this
transport with the v0.2 Engine API when Ayo's server is available.
"""
import argparse
import contextlib
import json
import os
import signal
import sys

PUBLIC = {'auth_state', 'create_first_owner', 'sign_in'}
METHODS = set('auth_state create_first_owner sign_in sign_out get_site set_site list_cameras add_camera remove_camera test discover_cameras detect_subnet scan list_areas create_area assign_camera_area scene_context scene_review_summary area_context approve_area_context approve_site_context update_scene_context approve_scene_context request_scene_remap enqueue_scene_mapping scene_mapping_progress accept_suggested_zone camera_snapshot list_zones add_zone remove_zone set_camera_rules presets use_case_templates apply_template add_custom_rule remove_custom_rule english_rules_status list_events event_clip acknowledge_alert resolve_alert search_events live_start camera_links start_monitoring stop_monitoring monitoring_status feed_sources switch_feed feed_switch_status setup_state setup_check mark_configured send_test_notification gate_status pull_model pull_progress retention_status set_retention list_users add_user remove_user audit_entries backup_now download_diagnostics value_summary role_table disk_encryption'.split())

METHODS.add('live_stop')

def dispatch(backend, method, args):
    if method not in METHODS or not isinstance(args, list):
        raise ValueError('Unsupported backend command')
    if method not in PUBLIC and not backend.current_user:
        raise PermissionError('Sign in to access the local engine')
    extra_permissions = {
        'mark_configured': 'configure_cameras',
        'send_test_notification': 'configure_site',
        'test': 'configure_cameras', 'scan': 'configure_cameras',
        'detect_subnet': 'configure_cameras',
        'camera_snapshot': 'view_live', 'scene_context': 'view_live',
        'list_cameras': 'view_live', 'list_zones': 'view_live',
        'pull_model': 'configure_site', 'pull_progress': 'configure_site',
        'live_stop': 'view_live',
    }
    if method in extra_permissions:
        backend._require(extra_permissions[method])
    if method == 'event_clip' and args and args[0]:
        permitted = {item.get('evidence_dir') for item in backend.list_events(10000)}
        if args[0] not in permitted:
            raise PermissionError('Evidence must belong to an accessible incident')
    result = getattr(backend, method)(*args)
    if isinstance(result, dict) and result.get('error'):
        raise ValueError(str(result['error']))
    if isinstance(result, dict) and result.get('ok') is False:
        raise ValueError(str(result.get('message') or 'Operation could not be completed'))
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--site', required=True)
    parser.add_argument('--db', required=True)
    args = parser.parse_args()
    os.chdir(args.repo)
    sys.path.insert(0, args.repo)
    protocol = sys.stdout
    # Model imports and older helpers print to stdout. Keep the protocol clean.
    with contextlib.redirect_stdout(sys.stderr):
        from cvti.app.console_backend import ConsoleBackend
        backend = ConsoleBackend(args.site, args.db, enable_demo=False)
    def shutdown_signal(signum, frame):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, shutdown_signal)
    request = {}
    try:
        for line in sys.stdin:
            request = {}
            try:
                if len(line) > 1000000:
                    raise ValueError('Request too large')
                request = json.loads(line)
                if request.get('method') == 'shutdown':
                    break
                with contextlib.redirect_stdout(sys.stderr):
                    result = dispatch(backend, request.get('method'), request.get('args', []))
                reply = {'id': request.get('id'), 'result': result}
            except Exception as error:
                reply = {'id': request.get('id'), 'error': str(error)[:400]}
            protocol.write(json.dumps(reply) + '\n')
            protocol.flush()
    finally:
        with contextlib.redirect_stdout(sys.stderr):
            if backend._monitor is not None:
                backend._monitor_should_run = False
                backend._monitor.terminate()
                try:
                    backend._monitor.wait(timeout=8)
                except Exception:
                    backend._monitor.kill()
            backend.live_stop()
        protocol.write(json.dumps({'id': request.get('id'), 'result': {'ok': True}})+'\n')
        protocol.flush()

if __name__ == '__main__':
    main()
