# models/SAC_Discrete.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.defs import Transition


# ============================================================
# Soft Actor (Categorical policy for discrete SAC)
# ============================================================
class SACDiscreteActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(128, 128)):
        super().__init__()

        layers = []
        last = state_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, action_dim))  # logits
        self.model = nn.Sequential(*layers)

    def forward(self, state):
        logits = self.model(state)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return probs, log_probs

    def sample_action(self, state):
        probs, log_probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        action_log_prob = log_probs.gather(1, action.unsqueeze(1))
        return action, action_log_prob


# ============================================================
# Soft Q-Network (Q(s,a))
# ============================================================
class SACDiscreteCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(128, 128)):
        super().__init__()

        layers = []
        last = state_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, action_dim))  # Q(s,a) for each action
        self.model = nn.Sequential(*layers)

    def forward(self, state):
        return self.model(state)  # returns Q-values for all actions


# ============================================================
# SAC-Discrete Agent (Single-file version)
# ============================================================
class SACDiscreteAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_sizes=(128, 128),
        gamma=0.99,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        automatic_entropy_tuning=True,
        device="cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        # networks
        self.actor = SACDiscreteActor(state_dim, action_dim, hidden_sizes).to(device)
        self.critic1 = SACDiscreteCritic(state_dim, action_dim, hidden_sizes).to(device)
        self.critic2 = SACDiscreteCritic(state_dim, action_dim, hidden_sizes).to(device)

        self.target_critic1 = SACDiscreteCritic(state_dim, action_dim, hidden_sizes).to(device)
        self.target_critic2 = SACDiscreteCritic(state_dim, action_dim, hidden_sizes).to(device)

        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

        # entropy tuning
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.target_entropy = -np.log(action_dim)

        if automatic_entropy_tuning:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        else:
            self.log_alpha = torch.tensor(np.log(0.1), device=device)

        self.gamma = gamma
        self.tau = 0.005
        self.memory = []

    @property
    def alpha(self):
        return self.log_alpha.exp()

    # ---------------------------------------------------------
    # ACTION SELECTION
    # ---------------------------------------------------------
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs, _ = self.actor.forward(state)
        action = probs.argmax(dim=-1).item()
        return action

    # ---------------------------------------------------------
    def store_transition(self, transition):
        self.memory.append(transition)

    # ---------------------------------------------------------
    def update(self, batch_size=256):
        if len(self.memory) < batch_size:
            return None

        batch = Transition(*zip(*self.memory))

        state = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action = torch.LongTensor(np.array(batch.action)).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(np.array(batch.reward)).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done = torch.FloatTensor(np.array(batch.done)).unsqueeze(1).to(self.device)

        # ----------------------------------------------------
        # 1. Target critic values
        # ----------------------------------------------------
        with torch.no_grad():
            next_probs, next_log_probs = self.actor.forward(next_state)

            q1_next = self.target_critic1(next_state)
            q2_next = self.target_critic2(next_state)
            q_next = torch.min(q1_next, q2_next)

            v_next = (next_probs * (q_next - self.alpha * next_log_probs)).sum(dim=1, keepdim=True)

            target_q = reward + self.gamma * (1 - done) * v_next

        # ----------------------------------------------------
        # 2. Update critic 1
        # ----------------------------------------------------
        q1 = self.critic1(state).gather(1, action)
        critic1_loss = F.mse_loss(q1, target_q)

        self.critic1_opt.zero_grad()
        critic1_loss.backward()
        self.critic1_opt.step()

        # ----------------------------------------------------
        # 3. Update critic 2
        # ----------------------------------------------------
        q2 = self.critic2(state).gather(1, action)
        critic2_loss = F.mse_loss(q2, target_q)

        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        self.critic2_opt.step()

        # ----------------------------------------------------
        # 4. Update actor
        # ----------------------------------------------------
        probs, log_probs = self.actor.forward(state)

        q1_all = self.critic1(state)
        q2_all = self.critic2(state)
        q_min = torch.min(q1_all, q2_all)

        actor_loss = (probs * (self.alpha * log_probs - q_min)).sum(dim=1).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ----------------------------------------------------
        # 5. Automatic entropy tuning
        # ----------------------------------------------------
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

        # ----------------------------------------------------
        # 6. Soft update target networks
        # ----------------------------------------------------
        self._soft_update(self.critic1, self.target_critic1)
        self._soft_update(self.critic2, self.target_critic2)

        return actor_loss.item(), critic1_loss.item(), critic2_loss.item()

    # ---------------------------------------------------------
    def _soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    # ---------------------------------------------------------
    def reset_memory(self):
        self.memory = []

    # ---------------------------------------------------------
    def save(self, folder):
        torch.save(self.actor.state_dict(), f"{folder}/actor.pth")
        torch.save(self.critic1.state_dict(), f"{folder}/critic1.pth")
        torch.save(self.critic2.state_dict(), f"{folder}/critic2.pth")

    def load(self, folder):
        self.actor.load_state_dict(torch.load(f"{folder}/actor.pth"))
        self.critic1.load_state_dict(torch.load(f"{folder}/critic1.pth"))
        self.critic2.load_state_dict(torch.load(f"{folder}/critic2.pth"))
