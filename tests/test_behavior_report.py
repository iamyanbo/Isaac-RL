from copy import deepcopy
import json

import pytest

from isaac_rl.behavior_eval import BehaviorSummary
from isaac_rl.behavior_report import aggregate, analyze_transitions, complete_snapshot, ratio


def episode(seed="ABCDEFGH", arm="stochastic", repetition=0, combat=2, stationary=1, reward=3):
    behavior = BehaviorSummary().result()
    behavior.update(steps=combat, combat_steps=combat, same_room_combat_steps=combat,
                    combat_stationary=stationary)
    return dict(seed=seed, arm=arm, repetition=repetition, prefix_hash="same", prefix_steps=3,
        combat_reached=bool(combat), success=False, boss_seen=False, reason="death", behavior=behavior,
        r=reward, l=combat, rooms=2, combat_clears=0, damage_dealt=3, damage_taken=6, kills=0)


def test_transition_fraction_uses_count_weighting_and_zero_denominators():
    rows = [episode(combat=2, stationary=1), episode(seed="12345678", combat=8, stationary=1)]
    assert aggregate(rows)["arms"]["stochastic"]["combat_stationary_fraction"] == .2
    assert ratio(0, 0) is None
    assert aggregate([episode(combat=0, stationary=0)])["arms"]["stochastic"]["combat_stationary_fraction"] is None


def test_rejects_duplicate_cases_and_keeps_repetitions_separate():
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate([episode(), episode()])
    result = aggregate([episode(), episode(repetition=1)])
    assert result["arms"]["stochastic"]["episodes"] == 2
    assert result["arms"]["stochastic"]["seeds"] == 1


def test_movement_effects_exclude_mismatches_and_weight_seeds_not_repeats():
    rows = []
    for seed, repeat, difference, matched in [("one", 0, 2, True), ("one", 1, 4, True),
                                             ("two", 0, 9, True), ("bad", 0, 100, False)]:
        normal = episode(seed=seed, repetition=repeat, reward=difference)
        other = deepcopy(normal)
        other.update(arm="stochastic_no_move_combat", r=0, prefix_hash="same" if matched else "different")
        rows.extend([normal, other])
    result = aggregate(rows)["movement_comparison"]
    assert result["eligible_pairs"] == 3 and result["excluded_pairs"] == 1
    assert result["eligible_seeds"] == 2
    assert result["equal_seed_weighted_effects"]["r"]["mean"] == 6
    assert not aggregate(rows)["completion_proven"]


def test_replayed_summary_handles_json_action_keys_and_catches_tampering(tmp_path):
    player = dict(x=100, y=100, hearts=6, soul=0, bombs=0)
    row = dict(seed='ABCDEFGH', arm='stochastic', step=1, before_sequence=1,
        player_before=player, player_after=player, intervention_applied=False,
        room_before=1, room_after=1, combat_before=True, entropy=[1, 1, 1],
        proposed=[0, 2, 1], executed=[0, 2, 1], reward_components={'time':-.01},
        shot_geometrically_aligned=False, damage_dealt=0, damage_taken=0, displacement=0)
    stats = BehaviorSummary()
    stats.add(row)
    record = dict(seed='ABCDEFGH', arm='stochastic', behavior=stats.result())
    record = json.loads(json.dumps(record))
    (tmp_path/'transitions.jsonl').write_text(json.dumps(row)+'\n')
    (tmp_path/'trace.jsonl').write_text(json.dumps(dict(direction='game', message=dict(sequence=1)))+'\n')
    report = analyze_transitions(tmp_path, [record])
    assert report['transition_summaries_verified'] == 1
    assert report['health_and_bomb_commands']['stochastic']['bomb_commands_without_bombs'] == 1
    record['behavior']['combat_steps'] = 0
    with pytest.raises(ValueError, match='summary mismatch'):
        analyze_transitions(tmp_path, [record])


def test_training_snapshot_records_exact_complete_prefix(tmp_path):
    import hashlib
    path = tmp_path/'episodes.jsonl'
    prefix = b'{"steps":12}\n'
    path.write_bytes(prefix+b'{"ste')
    rows, source = complete_snapshot(path)
    assert rows == [{'steps':12}]
    assert source['bytes'] == len(prefix)
    assert source['sha256'] == hashlib.sha256(prefix).hexdigest()
    assert source['omitted_trailing_bytes'] == 5
