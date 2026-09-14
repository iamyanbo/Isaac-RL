import hashlib
import io
import json
from pathlib import Path

import pytest
import torch

from isaac_rl.evaluate import canonical_seed, park_after_evaluation, success_gate
from isaac_rl.storage import load_snapshot
from isaac_rl.analyze import distribution_summary


def test_snapshot_hash_load_and_archive_share_bytes_even_when_source_changes(tmp_path, monkeypatch):
    original = io.BytesIO()
    torch.save({"steps":123,"weights":torch.tensor([1.,2.])},original)
    replacement = io.BytesIO()
    torch.save({"steps":999,"weights":torch.tensor([9.,9.])},replacement)
    path = tmp_path/"latest.pt"
    path.write_bytes(original.getvalue())
    read_bytes = Path.read_bytes
    def changing_read(self):
        data = read_bytes(self)
        if self == path:
            self.write_bytes(replacement.getvalue())
        return data
    monkeypatch.setattr(Path,"read_bytes",changing_read)
    saved,digest,payload = load_snapshot(path)
    assert saved["steps"] == 123
    assert payload == original.getvalue()
    assert digest == hashlib.sha256(payload).hexdigest()
    assert read_bytes(path) == replacement.getvalue()


def test_seed_normalization_cannot_turn_duplicate_runs_into_unique_evidence():
    assert canonical_seed("aBcD efGh") == "ABCDEFGH"
    records = [dict(seed=str(i),success=True,boss_seen=True,boss_defeated=True,reason="boss_clear") for i in range(100)]
    records[0]["seed"],records[1]["seed"] = "ABCD EFGH","abcdefgh"
    assert not success_gate(records)["unique_seeds"]
    assert not success_gate(records)["passed"]


@pytest.mark.parametrize("dead",[True,False])
def test_completed_evaluation_parks_dead_or_truncated_game_without_changing_scores(tmp_path,dead):
    class Game:
        state = {"seed":"OLD1 SEED","episode":7,"player":{"dead":dead}}
        resets = 0
        def reset(self):
            self.resets += 1
            self.state = {"seed":"NEW1 SEED","episode":8,"stage":1,"character":0,
                "difficulty":0,"player":{"dead":False},"room":{"clear":True,"enemies":0}}
    scored = {"episodes":3,"wins":0,"passed":False,"complete":True}
    result_path = tmp_path/"result.json"
    result_path.write_text(json.dumps(scored))
    game = Game()
    record = park_after_evaluation(game,tmp_path)
    assert game.resets == 1 and record["parked"] and not record["scored"]
    assert record["previous_episode"] == 7 and record["state"]["episode"] == 8
    assert json.loads(result_path.read_text()) == scored
    assert json.loads((tmp_path/"cleanup.json").read_text()) == record


def test_cleanup_timeout_is_reported_separately_and_never_retried(tmp_path):
    class Game:
        state = {"seed":"OLD1 SEED","episode":7}
        resets = 0
        def reset(self):
            self.resets += 1
            raise TimeoutError("pending reset")
    game = Game()
    record = park_after_evaluation(game,tmp_path)
    assert game.resets == 1 and not record["parked"] and not record["scored"]
    assert record["error"] == "TimeoutError: pending reset"
    assert not (tmp_path/"result.json").exists()


def test_distributions_preserve_samples_quantiles_outcomes_and_valid_probability_bounds():
    records = [dict(seed=str(i),success=False,boss_seen=False,boss_defeated=False,
        reason="death" if i<5 else "idle",r=float(i),l=i,rooms=i,combat_clears=i,
        damage_dealt=i,damage_taken=i,kills=i) for i in range(20)]
    report = distribution_summary(records)
    assert report["metrics"]["r"]["samples"] == list(range(20))
    assert report["metrics"]["r"]["median"] == 9.5
    assert report["metrics"]["r"]["p25"] == 4.75
    assert report["metrics"]["r"]["max"] == 19
    assert report["end_reasons"] == {"death":5,"idle":15}
    assert success_gate(records)["wilson_95_lower"] == 0
    records[0]["r"] = float("nan")
    with pytest.raises(ValueError,match="nonfinite"):
        distribution_summary(records)
