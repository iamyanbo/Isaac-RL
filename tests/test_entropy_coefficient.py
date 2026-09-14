from copy import deepcopy

import numpy as np
import pytest
import torch

from isaac_rl.ppo import ActorCritic, optimize, resolve_entropy_coef


def test_entropy_inherits_legacy_and_new_checkpoint_values():
    assert resolve_entropy_coef() == .02
    assert resolve_entropy_coef(checkpoint={}) == .02
    assert resolve_entropy_coef(checkpoint={'entropy_coef':.002}) == .002
    assert resolve_entropy_coef(.002, {}, False) == .002
    assert resolve_entropy_coef(None, {'entropy_coef':.002}, True) == .002
    with pytest.raises(ValueError, match='new run'):
        resolve_entropy_coef(.002, {}, True)


@pytest.mark.parametrize('value',[-.1, float('nan'), float('inf'), -float('inf')])
def test_entropy_rejects_invalid_arguments_and_checkpoint_values(value):
    with pytest.raises(ValueError, match='finite and nonnegative'):
        resolve_entropy_coef(value)
    with pytest.raises(ValueError, match='finite and nonnegative'):
        resolve_entropy_coef(checkpoint={'entropy_coef':value})


def test_only_entropy_term_changes_on_identical_frozen_rollout():
    torch.set_num_threads(1)
    torch.manual_seed(41)
    model = ActorCritic()
    with torch.no_grad():
        model.actor.bias.copy_(torch.linspace(-1, 1, 18))
    obs = {'grid':np.zeros((2,10,16,28),np.float32), 'vector':np.zeros((2,392),np.float32)}
    with torch.no_grad():
        actions,log_probs,_,values = model.act({k:torch.from_numpy(v) for k,v in obs.items()})
    rollout = dict(obs=[obs], actions=[actions.numpy()], log_probs=[log_probs.numpy()],
        values=[values.numpy()], rewards=[np.array([1.,-1.],np.float32)],
        next_values=[np.zeros(2,np.float32)], terminated=[np.ones(2,bool)], ended=[np.ones(2,bool)])
    reports = []
    for coefficient in [.02,.002]:
        copy = deepcopy(model)
        # lr=0 isolates the loss calculation; one minibatch avoids update drift.
        optimizer = torch.optim.SGD(copy.parameters(),lr=0)
        reports.append(optimize(copy,optimizer,rollout,epochs=1,batch_size=2,entropy_coef=coefficient))
        for key,value in model.state_dict().items():
            torch.testing.assert_close(copy.state_dict()[key],value,rtol=0,atol=0)
    first,second = reports
    for key in ['policy_loss','value_loss','entropy','kl','explained_variance']:
        assert first[key] == pytest.approx(second[key],abs=1e-7)
    assert second['loss']-first['loss'] == pytest.approx(.018*first['entropy'],abs=1e-7)
