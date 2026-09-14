from copy import deepcopy
import io
import json

import numpy as np
import torch

from isaac_rl.behavior_eval import (BehaviorSummary, intervention, native_seed, paired_results,
                                   run_case, transition_metrics)


def scene(room=1, enemies=1, x=100, y=200):
    return dict(room=dict(id=room, enemies=enemies),
        player=dict(x=x, y=y), entities=[[150, 200, 0, 0, 1, 10, 0, 0, 10, 10, 12, 1]],
        events=dict(damage_dealt=0, damage_attempted=0, damage_taken=0, kills=0), episode_frame=1)


def test_ablation_preserves_navigation_and_other_action_heads():
    proposed = np.array([5, 2, 3])
    np.testing.assert_equal(intervention(proposed, scene(enemies=0), 'stochastic_no_move_combat'), proposed)
    np.testing.assert_equal(intervention(proposed, scene(), 'stochastic_no_move_combat'), [0, 2, 3])
    np.testing.assert_equal(intervention(proposed, scene(), 'stochastic'), proposed)
    np.testing.assert_equal(proposed, [5, 2, 3])


def test_metrics_separate_commands_displacement_events_and_teleports():
    before, after = scene(), scene()
    after['events']['damage_dealt'] = 7
    after['episode_frame'] = 9
    record = transition_metrics(before, after, [0, 2, 0], [0, 2, 0], [], [1, 1, 1], 1.39, {'damage':1.4, 'time':-.01})
    stats = BehaviorSummary()
    stats.add(record)
    assert record['shot_geometrically_aligned'] and record['frame_delta'] == 8
    assert stats.result()['damage_during_stationary_transitions'] == 7
    before, after = scene(), scene(room=2, x=900)
    teleport = transition_metrics(before, after, [2, 2, 0], [2, 2, 0], [], [1, 1, 1], 0, {})
    stats.add(teleport)
    assert teleport['displacement'] is None
    assert stats.result()['same_room_combat_steps'] == 1
    assert stats.result()['room_changes'] == 1
    assert abs(stats.result()['reward_components']['time']+.01) < 1e-8


def test_counterfactual_requires_matching_prefix_and_combat():
    baseline = dict(seed='ABCDEFGH', arm='stochastic', prefix_hash='same', prefix_steps=3,
        combat_reached=True, r=5, l=90, damage_dealt=20, damage_taken=3,
        kills=2, combat_clears=1, boss_seen=False, success=False)
    ablated = dict(baseline, arm='stochastic_no_move_combat', damage_taken=6)
    result = paired_results([baseline, ablated])[0]
    assert result['movement_comparison_eligible']
    assert result['normal_minus_ablated']['damage_taken'] == -3
    ablated['prefix_hash'] = 'mismatch'
    assert not paired_results([baseline, ablated])[0]['movement_comparison_eligible']
    ablated.update(prefix_hash='same', combat_reached=False)
    assert not paired_results([baseline, ablated])[0]['movement_comparison_eligible']


def test_paired_runner_preserves_stochastic_prefix_and_logs_executed_actions():
    class Model:
        def __call__(self, obs):
            return torch.zeros((1, 18)), torch.zeros(1)

    class FakeEnv:
        def reset(self, options):
            assert options['game_seed'] == 'ABCD EFGH'
            self.steps = 0
            self.state = scene(enemies=0)
            self.state.update(sequence=1, stage=1)
            return {'grid':np.zeros((1, 1, 1), dtype=np.float32), 'vector':np.zeros(3, dtype=np.float32)}, {}

        def step(self, action):
            self.steps += 1
            self.state = deepcopy(self.state)
            self.state['sequence'] += 1
            self.state['episode_frame'] += 8
            self.state['room']['enemies'] = 1 if self.steps >= 2 else 0
            self.state['player']['x'] += int(action[0])
            info = {'reward_components': {'time':-.01}}
            if self.steps == 4:
                info['episode'] = dict(seed='ABCDEFGH', success=False, r=-.04, l=4,
                    damage_dealt=0, damage_taken=0, kills=0, combat_clears=0,
                    boss_seen=False, reason='idle')
            return {'grid':np.zeros((1, 1, 1), dtype=np.float32), 'vector':np.array([self.state['player']['x']],dtype=np.float32)}, -.01, False, self.steps == 4, info

    baseline_log, ablated_log = io.StringIO(), io.StringIO()
    baseline = run_case(FakeEnv(), Model(), 'stochastic', 'ABCDEFGH', 444, 0, baseline_log, lambda **kw:None)
    ablated = run_case(FakeEnv(), Model(), 'stochastic_no_move_combat', 'ABCDEFGH', 444, 0, ablated_log, lambda **kw:None)
    b = [json.loads(line) for line in baseline_log.getvalue().splitlines()]
    a = [json.loads(line) for line in ablated_log.getvalue().splitlines()]
    assert [x['executed'] for x in b[:2]] == [x['executed'] for x in a[:2]]
    assert all(x['executed'][0] == 0 for x in a[2:])
    assert baseline['prefix_hash'] == ablated['prefix_hash']
    assert paired_results([baseline, ablated])[0]['movement_comparison_eligible']
    assert sum(x['reward'] for x in a) == ablated['r']


def test_native_seed_formatting_preserves_canonical_identity():
    from isaac_rl.evaluate import canonical_seed
    import pytest
    for seed in ['3SLKTM8D', '3slk tm8d', '3SLK TM8D']:
        assert native_seed(seed) == '3SLK TM8D'
        assert canonical_seed(native_seed(seed)) == canonical_seed(seed)
    with pytest.raises(ValueError):
        native_seed('INVALID')
