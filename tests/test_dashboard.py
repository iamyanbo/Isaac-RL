import json
import os
import threading
import urllib.error
import urllib.request

import psutil
import pytest

from isaac_rl.dashboard import Dashboard, History, make_server


def test_full_history_partial_and_replacement(tmp_path):
    path = tmp_path / 'updates.jsonl'
    path.write_bytes(b'{"steps":999999}\n{"steps":1000001}\n{"steps":1234567890}\n{"steps":')
    history = History(path)
    assert [r['steps'] for r in history.read()] == [999999, 1000001, 1234567890]
    with path.open('ab') as stream:
        stream.write(b'1234567891}\nmalformed\n')
    assert len(history.read()) == 4
    assert len(history.read()) == 4
    replacement = tmp_path / 'replacement'
    replacement.write_text('{"steps":42}\n')
    replacement.replace(path)
    assert history.read() == [{'steps': 42}]
    path.write_text('')
    assert history.read() == []


def test_snapshot_liveness_and_nonfinite(tmp_path):
    (tmp_path / 'status.json').write_text(json.dumps(dict(pid=os.getpid(),
        process_started=psutil.Process().create_time(), status='idle', steps=1234567890)))
    (tmp_path / 'updates.jsonl').write_text('{"loss":NaN,"steps":1000001}\n')
    snapshot = json.loads(Dashboard(tmp_path).snapshot())
    assert snapshot['lifecycle'] == 'IDLE'
    assert snapshot['state']['steps'] == 1234567890
    assert snapshot['updates'][0]['loss'] is None
    assert snapshot['run_path'] == str(tmp_path.resolve())
    (tmp_path / 'status.json').write_text(json.dumps(dict(pid=os.getpid(),
        process_started=1, status='training')))
    assert json.loads(Dashboard(tmp_path).snapshot())['lifecycle'] == 'DEAD'


def test_http_read_only_and_host_validation(tmp_path):
    server = make_server(tmp_path, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}'
    try:
        for route in ['/', '/charts.js', '/api/history']:
            with urllib.request.urlopen(url + route) as response:
                assert response.status == 200
                assert response.headers['Cache-Control'] == 'no-store'
        for route in ['/../status.json', '/latest.pt', '/api/stop']:
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(url + route)
            assert error.value.code == 404
        request = urllib.request.Request(url + '/api/history', headers={'Host': 'untrusted.example'})
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        assert error.value.code == 403
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(urllib.request.Request(url + '/api/stop', data=b''))
        assert error.value.code == 501
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
