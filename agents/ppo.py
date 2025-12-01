# agents/ppo.py
"""
Proximal Policy Optimization (PPO) agent implementation (PyTorch).

Features:
- Works with discrete (Categorical) and continuous (Gaussian) action spaces.
- Uses Generalized Advantage Estimation (GAE) to compute advantages.
- Implements PPO clipped surrogate objective with value loss and entropy bonus.
- Includes helper methods for collecting rollouts and updating the networks.
- Save / load utilities.
- Designed to be called by an external training loop.

Notes:
- This implementation assumes Gymnasium-style env.step returns:
    next_obs, reward, terminated, truncated, info
  and env.reset returns (obs, info).
- For continuous actions we sample from a Normal distribution and clamp to env bounds.
"""

import time
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal


# -------------------------
# Simple MLP builder
# -------------------------
def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


# -------------------------
# Actor for discrete action spaces
# -------------------------
class DiscreteActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(64, 64)):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation=nn.ReLU)
    def forward(self, obs):
        return self.logits_net(obs)


# -------------------------
# Actor for continuous action spaces (Gaussian)
# returns mean; log_std is a learnable parameter (per action dim)
# -------------------------
class GaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(64, 64), init_log_std=-0.5):
        super().__init__()
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation=nn.ReLU)
        # learnable log std parameter (one per action dim)
        self.log_std = nn.Parameter(init_log_std * torch.ones(act_dim))

    def forward(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return mu, std


# -------------------------
# Critic: state-value function
# -------------------------
class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes=(64, 64)):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation=nn.ReLU)
    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)  # output shape: (batch,)


# -------------------------
# PPO Agent
# -------------------------
class PPOAgent:
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: str = "cpu",
        hidden_sizes=(64, 64),
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        clip_ratio: float = 0.2,
        target_kl: float = None,
        train_epochs: int = 10,
        batch_size: int = 64,
        gamma: float = 0.99,
        lam: float = 0.95,
        entropy_coef: float = 0.0,
    ):
        """
        observation_space, action_space: Gym spaces
        device: "cpu" or "cuda"
        lr_actor, lr_critic: learning rates
        clip_ratio: PPO clip epsilon
        target_kl: optional early stop if KL > target_kl during update
        train_epochs: how many epochs to run per update
        batch_size: minibatch size for updates
        gamma: discount factor
        lam: GAE lambda
        entropy_coef: coefficient for entropy bonus
        """
        self.device = torch.device(device)
        self.obs_space = observation_space
        self.act_space = action_space

        obs_dim = observation_space.shape[0]
        self.discrete = isinstance(action_space, gym.spaces.Discrete)
        if self.discrete:
            act_dim = action_space.n
            self.actor = DiscreteActor(obs_dim, act_dim, hidden_sizes).to(self.device)
        else:
            act_dim = action_space.shape[0]
            self.actor = GaussianActor(obs_dim, act_dim, hidden_sizes).to(self.device)

        self.critic = Critic(obs_dim, hidden_sizes).to(self.device)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # PPO hyperparams
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.train_epochs = train_epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.lam = lam
        self.entropy_coef = entropy_coef

    # -------------------------
    # Policy action + log_prob + value
    # -------------------------
    def step(self, obs: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Given a single observation, choose an action (sampled),
        return (action_np, logp_t, value_t)
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)  # shape (1, obs_dim)
        with torch.no_grad():
            value = self.critic(obs_t).cpu().numpy().squeeze(0)  # scalar
            if self.discrete:
                logits = self.actor(obs_t)
                dist = Categorical(logits=logits)
                action_t = dist.sample()
                logp = dist.log_prob(action_t)
                action = action_t.cpu().numpy().squeeze()
            else:
                mu, std = self.actor(obs_t)
                dist = Normal(mu, std)
                action_t = dist.sample()
                logp = dist.log_prob(action_t).sum(axis=-1)
                action = action_t.cpu().numpy().squeeze()
                # Clamp to env action bounds
                action = np.clip(action, self.act_space.low, self.act_space.high)
        return action, logp.cpu().numpy().squeeze(), value

    # -------------------------
    # Deterministic action for evaluation (mean for continuous, argmax for discrete)
    # -------------------------
    def act_deterministic(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if self.discrete:
                logits = self.actor(obs_t)
                action = torch.argmax(logits, dim=-1).cpu().numpy().squeeze()
            else:
                mu, _ = self.actor(obs_t)
                action = mu.cpu().numpy().squeeze()
                action = np.clip(action, self.act_space.low, self.act_space.high)
        return action

    # -------------------------
    # Compute GAE advantages + returns
    # -------------------------
    @staticmethod
    def compute_gae(rewards, values, dones, last_value, gamma, lam):
        """
        rewards: array length T
        values: array length T (V(s_0)...V(s_{T-1}))
        dones: array length T (bools)
        last_value: V(s_T) (for bootstrapping)
        returns: length T (discounted sum targets)
        advantages: length T
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae_lam = 0
        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            last_gae_lam = delta + gamma * lam * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam
        returns = advantages + values
        return returns, advantages

    # -------------------------
    # Convert rollout buffers to torch tensors and do PPO update
    # -------------------------
    def update(self, obs_buf, act_buf, logp_buf, ret_buf, adv_buf):
        """
        Performs training (multiple epochs over the collected rollout).
        Inputs are numpy arrays with matching first dimension N (timesteps collected).
        """
        obs = torch.as_tensor(obs_buf, dtype=torch.float32, device=self.device)
        act = torch.as_tensor(act_buf, dtype=torch.float32 if not self.discrete else torch.int64, device=self.device)
        old_logp = torch.as_tensor(logp_buf, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(ret_buf, dtype=torch.float32, device=self.device)
        adv = torch.as_tensor(adv_buf, dtype=torch.float32, device=self.device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        N = obs.shape[0]
        batch_size = self.batch_size
        for epoch in range(self.train_epochs):
            # shuffle indices and create minibatches
            idxs = np.random.permutation(N)
            for start in range(0, N, batch_size):
                end = start + batch_size
                mb_idx = idxs[start:end]

                mb_obs = obs[mb_idx]
                mb_act = act[mb_idx]
                mb_old_logp = old_logp[mb_idx]
                mb_returns = returns[mb_idx]
                mb_adv = adv[mb_idx]

                # current policy
                if self.discrete:
                    logits = self.actor(mb_obs)
                    dist = Categorical(logits=logits)
                    mb_logp = dist.log_prob(mb_act)
                    entropy = dist.entropy().mean()
                    # action values for critic input: critic uses states only
                else:
                    mu, std = self.actor(mb_obs)
                    dist = Normal(mu, std)
                    mb_logp = dist.log_prob(mb_act).sum(axis=-1)
                    entropy = dist.entropy().sum(axis=-1).mean()

                # ratio for clipped surrogate
                ratio = torch.exp(mb_logp - mb_old_logp)

                # surrogate loss
                surrogate1 = ratio * mb_adv
                surrogate2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * mb_adv
                policy_loss = -torch.min(surrogate1, surrogate2).mean()

                # value function loss (MSE)
                value = self.critic(mb_obs)
                value_loss = ((mb_returns - value) ** 2).mean()

                # total loss
                loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy

                # update actor & critic jointly
                self.optimizer_actor.zero_grad()
                self.optimizer_critic.zero_grad()
                loss.backward()
                # gradient clipping (optional safety)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.optimizer_actor.step()
                self.optimizer_critic.step()

            # optional KL early stopping check (compute approximate KL between old and new)
            if self.target_kl is not None:
                with torch.no_grad():
                    if self.discrete:
                        logits = self.actor(obs)
                        new_dist = Categorical(logits=logits)
                        old_dist = Categorical(logits=torch.exp(old_logp))  # not exact: we don't have old logits stored
                        # We cannot compute exact KL without old logits; skip robust KL check in this simple impl.
                        pass

    # -------------------------
    # Save / Load
    # -------------------------
    def save(self, path_prefix: str):
        torch.save(self.actor.state_dict(), f"{path_prefix}_actor.pt")
        torch.save(self.critic.state_dict(), f"{path_prefix}_critic.pt")

    def load(self, path_prefix: str, map_location: str = None):
        device = self.device if map_location is None else torch.device(map_location)
        self.actor.load_state_dict(torch.load(f"{path_prefix}_actor.pt", map_location=device))
        self.critic.load_state_dict(torch.load(f"{path_prefix}_critic.pt", map_location=device))

    # -------------------------
    # Helper: collect rollout from environment for exactly `steps` steps
    # -------------------------
    def collect_rollout(self, env: gym.Env, rollout_steps: int, render: bool = False):
        """
        Collect `rollout_steps` transitions by interacting with env.
        Returns buffers: obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf
        """
        obs_buf = []
        act_buf = []
        logp_buf = []
        rew_buf = []
        val_buf = []
        done_buf = []

        obs, _ = env.reset()
        for _ in range(rollout_steps):
            action, logp, value = self.step(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            obs_buf.append(obs.copy())
            act_buf.append(action.copy() if not self.discrete else int(action))
            logp_buf.append(float(logp))
            rew_buf.append(float(reward))
            val_buf.append(float(value))
            done_buf.append(done)

            if render:
                env.render()

            obs = next_obs
            if done:
                # reset environment immediately after terminal state to continue collecting fixed number of steps
                obs, _ = env.reset()

        # compute last value for bootstrap (V(s_T))
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            last_value = float(self.critic(obs_t).cpu().numpy().squeeze(0))

        return (
            np.array(obs_buf),
            np.array(act_buf),
            np.array(logp_buf, dtype=np.float32),
            np.array(rew_buf, dtype=np.float32),
            np.array(val_buf, dtype=np.float32),
            np.array(done_buf, dtype=np.uint8),
            last_value,
        )

    # -------------------------
    # High-level convenience: run one update cycle (collect + compute GAE + update)
    # -------------------------
    def rollout_and_update(self, env: gym.Env, rollout_steps: int, render: bool = False):
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf, last_value = self.collect_rollout(env, rollout_steps, render)
        returns, adv = self.compute_gae(rew_buf, val_buf, done_buf, last_value, self.gamma, self.lam)
        self.update(obs_buf, act_buf, logp_buf, returns, adv)
        metrics = {
            "rollout_step_count": len(rew_buf),
            "rollout_return_mean": float(np.sum(rew_buf) / max(1, len(rew_buf))),
        }
        return metrics
