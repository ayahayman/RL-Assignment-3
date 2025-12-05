import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    """Q-network that takes state-action pairs for SAC."""

    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        # Q1 architecture
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_value = nn.Linear(hidden_dim, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.q_value(x)
        return q