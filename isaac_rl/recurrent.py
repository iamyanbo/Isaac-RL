"""GRU PPO: one current snapshot, explicit states and ordered trajectory SGD."""
from copy import deepcopy

import numpy as np
import torch
from torch import nn

from .history import FRAME_VECTOR_SIZE, HISTORY
from .observation import compatible_observation_layout, observation_manifest
from .ppo import ActorCritic, action_statistics, compute_gae, resolve_entropy_coef


class RecurrentActorCritic(ActorCritic):
    recurrent = True

    def __init__(self):
        super().__init__('combat_gru_v4')
        self.gru = nn.GRU(256,256)
        self.memory_readout = nn.Linear(256,256,bias=False)
        # Preserve the current-frame-only warm start, not the removed stack.
        nn.init.zeros_(self.memory_readout.weight)

    def initial_state(self, batch_size):
        return self.actor.weight.new_zeros(1,batch_size,256)

    def forward(self, obs):
        raise ValueError('GRU requires explicit hidden state; use PolicyMemory or forward_sequence')

    def forward_sequence(self, obs, hidden, episode_starts):
        """Time-major [T,B,...]. Reset only affected games BEFORE their observation.

        Each segment between any episode boundary uses the fused GRU; no padded
        steps exist. Resets also sever gradients to the preceding episode.
        """
        t,b = obs['vector'].shape[:2]
        if hidden.shape != (1,b,256) or episode_starts.shape != (t,b):
            raise ValueError('Recurrent sequence/state shape mismatch')
        flat = {k:v.reshape(t*b,*v.shape[2:]) for k,v in obs.items()}
        features = self.shared(torch.cat([self.spatial(flat['grid']),self.features(flat['vector'])],dim=-1)).reshape(t,b,256)
        cuts = [0]+(torch.nonzero(episode_starts[1:].any(dim=1),as_tuple=False).flatten()+1).tolist()+[t]
        outputs = []
        for start,end in zip(cuts,cuts[1:]):
            hidden = hidden*(~episode_starts[start]).to(hidden.dtype)[None,:,None]
            sequence,hidden = self.gru(features[start:end],hidden)
            outputs.append(sequence)
        combined = features+self.memory_readout(torch.cat(outputs,dim=0))
        return self.actor(combined),self.critic(combined).squeeze(-1),hidden


def project_history_input(tensor, name, target):
    """Explicit removal of old snapshots; same mapping for weights and Adam."""
    if name == 'spatial.0.weight': return tensor[:,:10].clone()
    if name == 'features.0.weight':
        out = tensor.new_zeros(target.shape)
        out[:,:FRAME_VECTOR_SIZE] = tensor[:,:FRAME_VECTOR_SIZE]
        action_offset = HISTORY*FRAME_VECTOR_SIZE+HISTORY*2
        out[:,FRAME_VECTOR_SIZE:FRAME_VECTOR_SIZE+18] = tensor[:,action_offset:action_offset+18]
        return out
    if tensor.shape != target.shape: raise ValueError(f'Unsupported recurrent migration: {name}')
    return tensor.clone()


def load_recurrent_policy(saved, device, with_optimizer):
    parent = saved.get('observation_profile') if saved is not None else None
    if parent not in (None,'combat_history_v3','combat_gru_v4'):
        raise ValueError('GRU fork requires enriched history parent or exact GRU checkpoint')
    if saved is not None:
        if saved['architecture'] != observation_manifest(parent)['architecture'] or not compatible_observation_layout(saved.get('observation_layout'),parent):
            raise ValueError('Checkpoint architecture/observation layout mismatch')
    model = RecurrentActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=3e-4,eps=1e-5) if with_optimizer else None
    if saved is None: return model,optimizer
    if parent == 'combat_gru_v4':
        model.load_state_dict(saved['model'])
        if optimizer is not None: optimizer.load_state_dict(saved['optimizer'])
        return model,optimizer
    state = model.state_dict()
    for name,value in saved['model'].items():
        state[name] = project_history_input(value,name,state[name])
    model.load_state_dict(state)
    if optimizer is not None:
        old = saved['optimizer']
        if len(old['param_groups']) != 1: raise ValueError('Expected one inherited Adam parameter group')
        old_names = list(saved['model'])
        old_ids = old['param_groups'][0]['params']
        if len(old_names) != len(old_ids): raise ValueError('Parent model/Adam parameter mismatch')
        new = optimizer.state_dict()
        new_names = list(dict(model.named_parameters()))
        new_ids = dict(zip(new_names,new['param_groups'][0]['params']))
        for name,old_id in zip(old_names,old_ids):
            slots = deepcopy(old['state'].get(old_id,{}))
            for key,value in slots.items():
                if torch.is_tensor(value) and value.ndim:
                    slots[key] = project_history_input(value,name,state[name])
            if slots: new['state'][new_ids[name]] = slots
        group = deepcopy(old['param_groups'][0])
        group['params'] = new['param_groups'][0]['params']
        new['param_groups'] = [group]
        optimizer.load_state_dict(new)
    return model,optimizer


class PolicyMemory:
    """Collector/evaluator-owned memory; value probes must NOT advance it twice."""
    def __init__(self, model, batch_size):
        self.model = model
        self.recurrent = getattr(model,'recurrent',False)
        self.hidden = model.initial_state(batch_size) if self.recurrent else None
        self.starts = np.ones(batch_size,dtype=bool)

    def rollout_state(self):
        if not self.recurrent: return {}
        return dict(hidden_states=self.hidden.detach().cpu().numpy()[0].copy(),
                    episode_starts=self.starts.copy())

    def forward(self, obs):
        if not self.recurrent: return self.model(obs)
        logits,values,hidden = self.model.forward_sequence({k:v.unsqueeze(0) for k,v in obs.items()},
            self.hidden,torch.as_tensor(self.starts,device=self.hidden.device).unsqueeze(0))
        self.hidden = hidden.detach()
        self.starts.fill(False)
        return logits[0],values[0]

    def act(self, obs, deterministic=False):
        if not self.recurrent: return self.model.act(obs,deterministic=deterministic)
        logits,values = self.forward(obs)
        return action_statistics(logits,values,deterministic=deterministic)

    def value(self, obs):
        if not self.recurrent: return self.model(obs)[1]
        _,values,_ = self.model.forward_sequence({k:v.unsqueeze(0) for k,v in obs.items()},self.hidden,
            torch.zeros((1,len(self.starts)),dtype=torch.bool,device=self.hidden.device))
        return values[0]

    def reset_done(self, ended):
        self.starts = np.asarray(ended,dtype=bool).reshape(-1).copy()
        if self.recurrent:
            self.hidden = self.hidden*(~torch.as_tensor(self.starts,device=self.hidden.device))[None,:,None]


def sequence_batches(time_steps, environments, batch_size, permutation=None):
    """Shuffle complete contiguous sequences, never transitions inside them.

    Normal six-worker settings: six sequences of256, one per minibatch. Smaller
    explicit batches split trajectories and use their recorded starting states.
    Tail sequences are unpadded and grouped only with equal lengths.
    """
    if min(time_steps,environments,batch_size) < 1: raise ValueError('Positive sequence dimensions required')
    length = min(time_steps,batch_size)
    sequences = [(e,t,min(t+length,time_steps)) for e in range(environments) for t in range(0,time_steps,length)]
    order = np.random.permutation(len(sequences)) if permutation is None else permutation
    groups = {}
    for index in order:
        item = sequences[index];size = item[2]-item[1]
        group = groups.setdefault(size,[]);group.append(item)
        if len(group)*size+size > batch_size:
            yield group[:];group.clear()
    for group in groups.values():
        if group: yield group


def optimize_recurrent(model, optimizer, rollout, device, epochs, batch_size, clip, entropy_coef,target_kl,gamma,lam):
    entropy_coef = resolve_entropy_coef(entropy_coef)
    advantages,returns = compute_gae(rollout['rewards'],rollout['values'],rollout['next_values'],rollout['terminated'],rollout['ended'],gamma,lam)
    if advantages.ndim == 1: advantages,returns = advantages[:,None],returns[:,None]
    t,b = advantages.shape
    def shaped(name,tail=()):
        return torch.as_tensor(np.asarray(rollout[name]).reshape(t,b,*tail),device=device)
    starts = shaped('episode_starts').bool()
    ended = np.asarray(rollout['ended']).reshape(t,b)
    if t > 1 and not np.array_equal(starts[1:].cpu().numpy(),ended[:-1]):
        raise ValueError('Episode-start masks do not match rollout endings')
    hidden = shaped('hidden_states',(256,))
    observations = {}
    for key in rollout['obs'][0]:
        v = np.stack([o[key] for o in rollout['obs']])
        tail = v.shape[-3:] if key == 'grid' else v.shape[-1:]
        observations[key] = torch.as_tensor(v.reshape(t,b,*tail),device=device)
    actions,old_logs,old_values = shaped('actions',(3,)),shaped('log_probs'),shaped('values')
    adv = torch.as_tensor(advantages,device=device)
    adv = (adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    targets = torch.as_tensor(returns,device=device)
    reports,used_tokens = [],0
    for _ in range(epochs):
        epoch_kls,epoch_sizes = [],[]
        for batch in sequence_batches(t,b,batch_size):
            def take(tensor): return torch.stack([tensor[start:end,e] for e,start,end in batch],dim=1)
            initial = torch.stack([hidden[start,e] for e,start,end in batch],dim=0).unsqueeze(0).detach()
            logits,values,_ = model.forward_sequence({k:take(v) for k,v in observations.items()},initial,take(starts))
            _,logs,entropy,_ = action_statistics(logits,values,take(actions))
            log_ratio = logs-take(old_logs);ratio = log_ratio.exp()
            advantage = take(adv)
            policy_loss = torch.maximum(-advantage*ratio,-advantage*ratio.clamp(1-clip,1+clip)).mean()
            previous,target = take(old_values),take(targets)
            clipped = previous+(values-previous).clamp(-clip,clip)
            value_loss = .5*torch.maximum((values-target).square(),(clipped-target).square()).mean()
            entropy_loss = entropy.mean()
            loss = policy_loss+.5*value_loss-entropy_coef*entropy_loss
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite recurrent PPO loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(),.5)
            optimizer.step()
            with torch.no_grad(): kl = ((ratio-1)-log_ratio).mean().item()
            size = logs.numel();used_tokens += size
            epoch_kls.append(kl);epoch_sizes.append(size)
            reports.append(dict(loss=loss.item(),policy_loss=policy_loss.item(),value_loss=value_loss.item(),
                entropy=entropy_loss.item(),kl=kl,grad_norm=float(grad_norm)))
        if np.average(epoch_kls,weights=epoch_sizes) > target_kl: break
    result = {k:float(np.mean([r[k] for r in reports])) for k in reports[0]}
    result['explained_variance'] = float(1-np.var(returns-old_values.cpu().numpy())/max(np.var(returns),1e-8))
    result.update(recurrent_sequence_length=min(t,batch_size),recurrent_tokens=used_tokens)
    return result
