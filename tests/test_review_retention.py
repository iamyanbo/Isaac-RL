import json

from isaac_rl import storage


def test_retains_first_eligible_bytes_and_never_replaces_existing_snapshot(tmp_path):
    assert storage.retain_review_checkpoints(tmp_path,100)=={}
    storage.atomic_json(tmp_path/'snapshot_steps.json',[103])
    (tmp_path/'latest.pt').write_bytes(b'first optimized checkpoint')
    assert 'milestone_last_saved' not in storage.retain_review_checkpoints(tmp_path,100)
    result=storage.retain_review_checkpoints(tmp_path,106)
    assert result['milestone_last_saved']['target_steps']==103
    assert result['milestone_last_saved']['actual_steps']==106
    assert (tmp_path/'milestone-103.pt').read_bytes()==(tmp_path/'latest.pt').read_bytes()
    (tmp_path/'latest.pt').write_bytes(b'later optimized checkpoint')
    storage.retain_review_checkpoints(tmp_path,112)
    assert (tmp_path/'milestone-103.pt').read_bytes()==b'first optimized checkpoint'
    metadata=json.loads((tmp_path/'milestone-103.json').read_text())
    assert metadata['sha256']==storage.sha256(tmp_path/'milestone-103.pt')


def test_bad_plan_or_archive_failure_is_reported_without_raising(tmp_path,monkeypatch):
    storage.atomic_json(tmp_path/'snapshot_steps.json',['../escape'])
    assert 'positive integer' in storage.retain_review_checkpoints(tmp_path,100)['milestone_archive_error']
    storage.atomic_json(tmp_path/'snapshot_steps.json',[100])
    (tmp_path/'latest.pt').write_bytes(b'checkpoint')
    def fail(*args): raise OSError('archive unavailable')
    monkeypatch.setattr(storage.os,'link',fail)
    result=storage.retain_review_checkpoints(tmp_path,100)
    assert 'archive unavailable' in result['milestone_archive_error']
    assert not (tmp_path/'milestone-100.pt').exists()
    assert not list(tmp_path.glob('milestone-*.tmp'))
