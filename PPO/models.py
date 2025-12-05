# models.py
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

class ActorCritic(nn.Module):
    """Actor-Critic Network for PPO"""
    def __init__(self, state_dim, action_dim, hidden_dim=64, continuous=False):
        super(ActorCritic, self).__init__()
        self.continuous = continuous
        
        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head
        if continuous:
            self.actor_mean = nn.Linear(hidden_dim, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)
        
        # Initialize weights for stability
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with smaller values for stability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        features = self.shared(state)
        value = self.critic(features)
        
        if self.continuous:
            mean = self.actor_mean(features)
            std = torch.exp(self.actor_log_std.clamp(-20, 2))
            return mean, std, value
        else:
            action_logits = self.actor(features)
            # Clamp logits to prevent overflow
            action_logits = torch.clamp(action_logits, -20, 20)
            action_probs = torch.softmax(action_logits, dim=-1)
            # Add small epsilon and renormalize for numerical stability
            action_probs = action_probs + 1e-8
            action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
            return action_probs, value
    
    def get_action(self, state):
        if self.continuous:
            mean, std, value = self.forward(state)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            return action, log_prob, value
        else:
            action_probs, value = self.forward(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action, log_prob, value
    
    def evaluate(self, states, actions):
        if self.continuous:
            mean, std, values = self.forward(states)
            dist = Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        else:
            action_probs, values = self.forward(states)
            dist = Categorical(action_probs)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
        
        return log_probs, values.squeeze(), entropy