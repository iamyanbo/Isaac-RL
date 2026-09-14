"""Clipped PPO with GAE, time-limit bootstrapping, and factored actions."""
import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .observation import CHANNELS, VECTOR_SIZE
from .rewards import GAMMA


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(CHANNELS,16,3,padding=1,stride=2),nn.ReLU(),
            nn.Conv2d(16,32,3,padding=1,stride=2),nn.ReLU(),nn.Flatten(),
            nn.Linear(32*4*7,128),nn.Tanh(),
        )
        self.features = nn.Sequential(nn.Linear(VECTOR_SIZE,256),nn.Tanh())
        self.shared = nn.Sequential(nn.Linear(384,256),nn.Tanh(),nn.Linear(256,256),nn.Tanh())
        self.actor = nn.Linear(256,18)
        self.critic = nn.Linear(256,1)
        for layer in self.modules():
            if isinstance(layer,(nn.Linear,nn.Conv2d)):
                nn.init.orthogonal_(layer.weight,math.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor.weight,0.01)
        nn.init.orthogonal_(self.critic.weight,1)

    def forward(self, obs):
        features = self.shared(torch.cat([self.spatial(obs["grid"]),self.features(obs["vector"])],dim=-1))
        return self.actor(features),self.critic(features).squeeze(-1)

    def act(self, obs, actions=None, deterministic=False):
        logits,values = self(obs)
        distributions = [Categorical(logits=part) for part in logits.split([9,5,4],dim=-1)]
        if actions is None:
            actions = torch.stack([d.probs.argmax(-1) if deterministic else d.sample() for d in distributions],dim=-1)
        log_prob = torch.stack([d.log_prob(actions[:,i]) for i,d in enumerate(distributions)],dim=-1).sum(-1)
        entropy = torch.stack([d.entropy() for d in distributions],dim=-1).sum(-1)
        return actions,log_prob,entropy,values


def as_tensor(obs, device="cpu"):
    return {key:torch.as_tensor(value,device=device,dtype=torch.float32).unsqueeze(0) for key,value in obs.items()}


def compute_gae(rewards, values, next_values, terminated, ended, gamma=GAMMA, lam=0.95):
    """Bootstrap truncations, but never propagate GAE across a reset boundary."""
    rewards,values,next_values = map(lambda x:np.asarray(x,np.float32),(rewards,values,next_values))
    terminated,ended = np.asarray(terminated,dtype=np.float32),np.asarray(ended,dtype=np.float32)
    advantages = np.zeros_like(rewards)
    carry = np.zeros_like(rewards[0])
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma*next_values[i]*(1-terminated[i]) - values[i]
        carry = delta + gamma*lam*(1-ended[i])*carry
        advantages[i] = carry
    return advantages,advantages+values


def optimize(model, optimizer, rollout, device="cpu", epochs=4, batch_size=64,
             clip=0.2, entropy_coef=0.02, target_kl=0.025):
    advantages,returns = compute_gae(rollout["rewards"],rollout["values"],rollout["next_values"],
                                    rollout["terminated"],rollout["ended"])
    # Rollouts may be [time] or [time, independent environments]. Compute GAE
    # along time first, then flatten for SGD; never join different games' returns.
    obs = {}
    for key in rollout["obs"][0]:
        values = np.stack([o[key] for o in rollout["obs"]])
        shape = values.shape[-3:] if key == "grid" else values.shape[-1:]
        obs[key] = torch.as_tensor(values.reshape(-1,*shape),device=device)
    actions = torch.as_tensor(np.stack(rollout["actions"]).reshape(-1,3),device=device)
    old_log_probs = torch.as_tensor(np.asarray(rollout["log_probs"]).reshape(-1),device=device)
    old_values_array = np.asarray(rollout["values"]).reshape(-1)
    old_values = torch.as_tensor(old_values_array,device=device)
    adv = torch.as_tensor(advantages.reshape(-1),device=device)
    adv = (adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    returns = returns.reshape(-1)
    targets = torch.as_tensor(returns,device=device)
    reports = []
    count = len(returns)
    for _ in range(epochs):
        indices = np.random.permutation(count)
        epoch_kls = []
        for start in range(0,count,batch_size):
            idx = torch.as_tensor(indices[start:start+batch_size],device=device)
            _,log_probs,entropy,values = model.act({key:val[idx] for key,val in obs.items()},actions[idx])
            log_ratio = log_probs-old_log_probs[idx]
            ratio = log_ratio.exp()
            policy_loss = torch.maximum(-adv[idx]*ratio,-adv[idx]*ratio.clamp(1-clip,1+clip)).mean()
            clipped_values = old_values[idx]+(values-old_values[idx]).clamp(-clip,clip)
            value_loss = 0.5*torch.maximum((values-targets[idx]).square(),(clipped_values-targets[idx]).square()).mean()
            entropy_loss = entropy.mean()
            loss = policy_loss+0.5*value_loss-entropy_coef*entropy_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(),0.5)
            optimizer.step()
            with torch.no_grad():
                kl = ((ratio-1)-log_ratio).mean().item()
            epoch_kls.append(kl)
            reports.append({"loss":loss.item(),"policy_loss":policy_loss.item(),"value_loss":value_loss.item(),
                "entropy":entropy_loss.item(),"kl":kl,"grad_norm":float(grad_norm)})
        if np.mean(epoch_kls) > target_kl:
            break
    result = {key:float(np.mean([r[key] for r in reports])) for key in reports[0]}
    result["explained_variance"] = float(1-np.var(returns-old_values_array)/max(np.var(returns),1e-8))
    return result
