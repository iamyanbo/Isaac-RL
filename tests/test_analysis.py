from isaac_rl.analyze import summarize


def episode(**overrides):
    return dict(dict(success=False,boss_seen=False,r=-10,l=100,rooms=2,clears=1,
        damage_dealt=5,damage_taken=6,reason="death"),**overrides)


def test_analysis_does_not_average_incomparable_reward_versions():
    report = summarize([episode(),episode(reward_profile="balanced_v2",r=10)])
    assert report["mean_reward"] is None
    assert report["reward_profiles"] == ["balanced_v2","legacy_v1"]
    assert summarize([episode(),episode(r=-20)])["mean_reward"] == -15


def test_missing_legacy_kill_and_combat_data_are_not_fabricated_as_zeros():
    report = summarize([episode(),episode(kills=4,combat_clears=1)])
    assert report["mean_kills"] == 4
    assert report["mean_observed_combat_clears"] == 1
    assert report["metric_coverage"] == dict(kill_episodes=1,combat_clear_episodes=1)
