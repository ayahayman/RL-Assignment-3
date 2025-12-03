# models/A2C.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


# -------------------------------------------------
# Actor network
# -------------------------------------------------
class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, act_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# -------------------------------------------------
# Critic network
# -------------------------------------------------
class CriticNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_sizes):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze(-1)


# -------------------------------------------------
# A2C Agent
# -------------------------------------------------
class A2CAgent:
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes=(128, 128),
        actor_lr=3e-4,
        critic_lr=3e-4,
        gamma=0.99,
        entropy_coef=0.001,
        device="cpu",
    ):
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.device = device

        self.act_dim = act_dim
        
        self.actor = ActorNetwork(obs_dim, act_dim, hidden_sizes).to(device)
        self.critic = CriticNetwork(obs_dim, hidden_sizes).to(device)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Replay buffers for episode-based update
        self.obs_buf = []
        self.act_buf = []
        self.rew_buf = []
        self.done_buf = []
        self.logp_buf = []

    # -------------------------------------------------------------
    # ACTION SELECTION
    # -------------------------------------------------------------
    def select_action(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        logits = self.actor(obs_t)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        action = dist.sample()
        logp = dist.log_prob(action).item()

        return action.item(), logp  # EXACTLY 2 VALUES

    # -------------------------------------------------------------
    def store(self, obs, act, rew, done, logp):
        self.obs_buf.append(obs)
        self.act_buf.append(act)
        self.rew_buf.append(rew)
        self.done_buf.append(done)
        self.logp_buf.append(logp)

    # -------------------------------------------------------------
    # EPISODIC RETURNS
    # -------------------------------------------------------------
    def compute_returns(self):
        R = 0
        returns = []
        for r, d in zip(reversed(self.rew_buf), reversed(self.done_buf)):
            if d:
                R = 0
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32, device=self.device)

    # -------------------------------------------------------------
    # A2C UPDATE
    # -------------------------------------------------------------
    def update(self):
        obs = torch.tensor(np.array(self.obs_buf), dtype=torch.float32, device=self.device)
        acts = torch.tensor(self.act_buf, dtype=torch.long, device=self.device)
        logps_old = torch.tensor(self.logp_buf, dtype=torch.float32, device=self.device)

        returns = self.compute_returns()

        # Critic forward
        values = self.critic(obs)
        advantage = returns - values.detach()

        # Actor forward
        logits = self.actor(obs)
        probs = F.softmax(logits, dim=-1)
        d = torch.distributions.Categorical(probs)
        logps = d.log_prob(acts)

        entropy = d.entropy().mean()

        actor_loss = -(logps * advantage).mean() - self.entropy_coef * entropy
        critic_loss = F.mse_loss(values, returns)

        # Update actor
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Update critic
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Clear buffers
        self.obs_buf.clear()
        self.act_buf.clear()
        self.rew_buf.clear()
        self.done_buf.clear()
        self.logp_buf.clear()

        return actor_loss.item(), critic_loss.item()
