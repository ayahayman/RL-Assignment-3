# agents/a2c.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class A2CAgent:
    def __init__(
        self,
        actor_network,
        critic_network,
        actor_lr=1e-4,
        critic_lr=5e-4,
        gamma=0.99,
        entropy_weight=0.01,
        device="cpu"
    ):
        """
        A2C Agent:
        - actor_network: nn.Module that outputs action probabilities
        - critic_network: nn.Module that outputs state value V(s)
        - actor_lr: learning rate for policy
        - critic_lr: learning rate for value function
        - gamma: discount factor
        - entropy_weight: encourages exploration
        """
        self.actor = actor_network.to(device)
        self.critic = critic_network.to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.entropy_weight = entropy_weight
        self.device = device

    # --------------------------------------------------------------------
    # Pick action from policy π(a|s)
    # --------------------------------------------------------------------
    def choose_action(self, state):
        """
        Input:
            state: numpy array from environment
        Returns:
            action (int)
            log_prob (tensor)
            value (tensor)
        """
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_probs = self.actor(state_tensor)
        distribution = Categorical(action_probs)
        action = distribution.sample()
        value = self.critic(state_tensor)

        return action.item(), distribution.log_prob(action), value

    # --------------------------------------------------------------------
    # Compute Advantage
    # --------------------------------------------------------------------
    def compute_advantage(self, reward, value, next_value, done):
        target_value = reward if done else reward + self.gamma * next_value
        advantage = target_value - value
        return advantage, target_value

    # --------------------------------------------------------------------
    # Update Actor & Critic
    # --------------------------------------------------------------------
    def update(self, log_prob, value, reward, next_value, done, state):
        """
        Performs one A2C update step.
        """
        advantage, target_value = self.compute_advantage(reward, value, next_value, done)

        # -------------------------
        # Critic update
        # -------------------------
        critic_loss = (target_value.detach() - value) ** 2
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # -------------------------
        # Actor update
        # -------------------------
        actor_loss = -log_prob * advantage.detach()

        # Entropy bonus for exploration
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        probs = self.actor(state_tensor)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        actor_loss -= self.entropy_weight * entropy

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    # --------------------------------------------------------------------
    # Save / Load models
    # --------------------------------------------------------------------
    def save(self, path):
        checkpoint = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict()
        }
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
