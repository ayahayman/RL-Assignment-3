import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
import random

from models.actor_critic.actor import Actor
from models.actor_critic.critic import Critic

# Automatically select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"SAC using device: {device}")

# SAC Agent Implementation
class SACAgent:

    def __init__(self, state_dim, action_dim, hidden_dim, gamma,
                 actor_lr, critic_lr, tau=0.005, alpha=0.2,
                 batch_size=256, buffer_size=1000000,
                 action_low=-1.0, action_high=1.0):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.device = device
        
        # Action scaling for environments with action bounds != [-1, 1]
        self.action_low = torch.FloatTensor([action_low]).to(device)
        self.action_high = torch.FloatTensor([action_high]).to(device)
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0

        # Actor network
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Critic networks (Q1 and Q2)
        self.critic1 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

        # Target critic networks
        self.target_critic1 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.target_critic2 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # Replay buffer
        self.memory = deque(maxlen=buffer_size)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if evaluate:
            # Use mean action for evaluation
            mean, _ = self.actor(state)
            action = torch.tanh(mean)
        else:
            # Sample action during training
            action, _ = self.actor.sample(state)
        
        # Scale action from [-1, 1] to [action_low, action_high]
        action = action * self.action_scale + self.action_bias
        return action.detach().cpu().numpy()[0]
    
    def store_transition(self, transition):
        self.memory.append(transition)

    def update(self):
        if len(self.memory) < self.batch_size:
            return 0.0, 0.0, 0.0

        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        state = torch.FloatTensor(np.array(state)).to(self.device)
        action = torch.FloatTensor(np.array(action)).to(self.device)
        reward = torch.FloatTensor(np.array(reward)).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        done = torch.FloatTensor(np.array(done)).unsqueeze(1).to(self.device)

        # ========== Update Critics ==========
        with torch.no_grad():
            # Sample actions from current policy for next states
            next_action, next_log_prob = self.actor.sample(next_state)
            
            # Compute target Q-values
            target_q1 = self.target_critic1(next_state, next_action)
            target_q2 = self.target_critic2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q

        # Update Critic 1
        current_q1 = self.critic1(state, action)
        critic1_loss = F.mse_loss(current_q1, target_q)
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        # Update Critic 2
        current_q2 = self.critic2(state, action)
        critic2_loss = F.mse_loss(current_q2, target_q)
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # ========== Update Actor ==========
        new_action, log_prob = self.actor.sample(state)
        q1_new = self.critic1(state, new_action)
        q2_new = self.critic2(state, new_action)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_prob - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ========== Soft update target networks ==========
        self._soft_update(self.critic1, self.target_critic1)
        self._soft_update(self.critic2, self.target_critic2)

        return actor_loss.item(), critic1_loss.item(), critic2_loss.item()

    def _soft_update(self, source, target):
        """Soft update target network parameters."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def reset_memory(self):
        self.memory.clear()

    def save_models(self, actor_path, critic1_path, critic2_path):
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic1.state_dict(), critic1_path)
        torch.save(self.critic2.state_dict(), critic2_path)
    
    def load_models(self, actor_path, critic1_path, critic2_path):
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic1.load_state_dict(torch.load(critic1_path))
        self.critic2.load_state_dict(torch.load(critic2_path))
