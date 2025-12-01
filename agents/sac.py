# agents/sac.py
"""
Soft Actor-Critic (SAC) agent implementation (PyTorch).

Features:
- Supports continuous action spaces only (Gaussian policy).
- Twin Q networks (Soft Q-function) for Clipped Double-Q.
- Automatic entropy tuning (alpha-learning).
- Target networks updated using Polyak averaging.
- Actor outputs mean + log_std (clamped) to a squashed (tanh) policy.
- Replay-buffer-based training (external buffer required).
- Designed for clean use with a training loop.

This file intentionally does NOT implement a ReplayBuffer inside the agent,
to keep separation of concerns. The training loop should supply batches.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
import gymnasium as gym


# ----------------------------------------------------------------------
# Helper: MLP builder
# ----------------------------------------------------------------------
def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i+1]), act()]
    return nn.Sequential(*layers)


# ----------------------------------------------------------------------
# Gaussian Policy with Tanh Squash
# ----------------------------------------------------------------------
class SquashedGaussianActor(nn.Module):
    """
    Outputs a Gaussian distribution (mu, std), samples an action,
    then applies a tanh squash + re-scales to environment action bounds.
    """

    def __init__(self, obs_dim, act_dim, act_limit, hidden_sizes=(256, 256)):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes) + [act_dim*2])
        self.act_limit = act_limit
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2

    def forward(self, obs):
        out = self.net(obs)
        mu, log_std = torch.chunk(out, 2, dim=-1)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, obs):
        """
        Sample action + compute log_prob with tanh correction.
        Returns:
          action: scaled to env bounds
          log_prob
          mu_tanh: deterministic action
        """
        mu, std = self.forward(obs)
        dist = Normal(mu, std)

        # Reparameterization trick: sample epsilon, compute a = mu + std * eps
        raw_action = dist.rsample()  # rsample uses reparameterization
        action = torch.tanh(raw_action)

        # log probability correction for tanh squash
        # log_prob_raw - log(1 - tanh(a)^2)
        log_prob = dist.log_prob(raw_action).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        # Scale to env bounds
        action_scaled = action * self.act_limit
        mu_tanh = torch.tanh(mu) * self.act_limit
        return action_scaled, log_prob, mu_tanh


# ----------------------------------------------------------------------
# Twin Q Networks
# ----------------------------------------------------------------------
class QNetwork(nn.Module):
    """Q(s,a) approximator."""
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256)):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1])

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q(x)


# ----------------------------------------------------------------------
# SAC Agent
# ----------------------------------------------------------------------
class SACAgent:
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        device="cpu",
        hidden_sizes=(256, 256),
        gamma=0.99,
        polyak=0.995,
        lr=3e-4,
        alpha_lr=3e-4,
        auto_alpha=True,
        target_entropy=None,
    ):
        """
        observation_space, action_space: gymnasium spaces
        device: "cpu" or "cuda"
        hidden_sizes: sizes for all MLPs
        gamma: discount factor
        polyak: target network update coefficient
        auto_alpha: whether to learn entropy coefficient α
        target_entropy: optional; defaults to -action_dim
        """

        assert isinstance(action_space, gym.spaces.Box), \
            "SAC only supports continuous action spaces."

        self.device = torch.device(device)
        self.obs_dim = observation_space.shape[0]
        self.act_dim = action_space.shape[0]
        self.act_limit = float(action_space.high[0])

        # Core networks ----------------------------------------------------
        self.actor = SquashedGaussianActor(
            self.obs_dim, self.act_dim, self.act_limit, hidden_sizes
        ).to(self.device)

        self.q1 = QNetwork(self.obs_dim, self.act_dim, hidden_sizes).to(self.device)
        self.q2 = QNetwork(self.obs_dim, self.act_dim, hidden_sizes).to(self.device)

        # Target networks (initialized equal to main networks)
        self.q1_target = QNetwork(self.obs_dim, self.act_dim, hidden_sizes).to(self.device)
        self.q2_target = QNetwork(self.obs_dim, self.act_dim, hidden_sizes).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Optimizers -------------------------------------------------------
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)

        # Entropy coefficient α -------------------------------------------
        self.auto_alpha = auto_alpha
        if target_entropy is None:
            target_entropy = -self.act_dim

        if auto_alpha:
            # alpha is learned as exp(log_alpha)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
            self.target_entropy = target_entropy
        else:
            self.alpha = 0.2  # fixed entropy coefficient

        # Hyperparameters --------------------------------------------------
        self.gamma = gamma
        self.polyak = polyak

    # ------------------------------------------------------------------
    # Select action (deterministic OR stochastic)
    # ------------------------------------------------------------------
    def act(self, obs, deterministic=False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                mu, _ = self.actor.forward(obs_t)
                action = torch.tanh(mu) * self.act_limit
            else:
                action, _, _ = self.actor.sample(obs_t)
        return action.cpu().numpy().squeeze()

    # ------------------------------------------------------------------
    # One gradient update using a batch from replay buffer
    # ------------------------------------------------------------------
    def update(self, batch):
        """
        batch: dict with keys:
          - obs
          - act
          - rew
          - next_obs
          - done
        All should be numpy arrays.
        """
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        act = torch.as_tensor(batch["act"], dtype=torch.float32, device=self.device)
        rew = torch.as_tensor(batch["rew"], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        done = torch.as_tensor(batch["done"], dtype=torch.float32, device=self.device).unsqueeze(1)

        # -----------------------------
        # 1) Compute target Q values
        # -----------------------------
        with torch.no_grad():
            next_action, next_logp, _ = self.actor.sample(next_obs)

            q1_target = self.q1_target(next_obs, next_action)
            q2_target = self.q2_target(next_obs, next_action)
            q_target = torch.min(q1_target, q2_target)

            if self.auto_alpha:
                alpha = self.log_alpha.exp()
            else:
                alpha = self.alpha

            backup = rew + self.gamma * (1 - done) * (q_target - alpha * next_logp)

        # -----------------------------
        # 2) Update Q-networks
        # -----------------------------
        q1_loss = ((self.q1(obs, act) - backup) ** 2).mean()
        q2_loss = ((self.q2(obs, act) - backup) ** 2).mean()
        q_loss = q1_loss + q2_loss

        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        q_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        # -----------------------------
        # 3) Update policy (actor)
        # -----------------------------
        action_pi, logp_pi, _ = self.actor.sample(obs)

        q1_pi = self.q1(obs, action_pi)
        q2_pi = self.q2(obs, action_pi)
        q_pi = torch.min(q1_pi, q2_pi)

        if self.auto_alpha:
            alpha = self.log_alpha.exp()

        actor_loss = (alpha * logp_pi - q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # -----------------------------
        # 4) Update entropy coefficient α
        # -----------------------------
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        # -----------------------------
        # 5) Polyak Average target networks
        # -----------------------------
        with torch.no_grad():
            for p, p_t in zip(self.q1.parameters(), self.q1_target.parameters()):
                p_t.data.mul_(self.polyak)
                p_t.data.add_((1 - self.polyak) * p.data)

            for p, p_t in zip(self.q2.parameters(), self.q2_target.parameters()):
                p_t.data.mul_(self.polyak)
                p_t.data.add_((1 - self.polyak) * p.data)

        return {
            "q_loss": q_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": alpha.item() if self.auto_alpha else self.alpha,
        }

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def save(self, prefix):
        torch.save(self.actor.state_dict(), f"{prefix}_actor.pt")
        torch.save(self.q1.state_dict(), f"{prefix}_q1.pt")
        torch.save(self.q2.state_dict(), f"{prefix}_q2.pt")

    def load(self, prefix):
        self.actor.load_state_dict(torch.load(f"{prefix}_actor.pt", map_location=self.device))
        self.q1.load_state_dict(torch.load(f"{prefix}_q1.pt", map_location=self.device))
        self.q2.load_state_dict(torch.load(f"{prefix}_q2.pt", map_location=self.device))
