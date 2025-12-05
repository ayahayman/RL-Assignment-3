# agent.py
import torch
import torch.optim as optim
import numpy as np
from models import ActorCritic
from memory import PPOMemory

class PPOAgent:
    """Proximal Policy Optimization Agent"""
    def __init__(self, state_dim, action_dim, continuous=False, 
                 lr=3e-4, gamma=0.99, gae_lambda=0.95, 
                 clip_epsilon=0.2, entropy_coef=0.01, epochs=10, batch_size=64):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.epochs = epochs
        self.batch_size = batch_size
        self.continuous = continuous
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = ActorCritic(state_dim, action_dim, continuous=continuous).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.memory = PPOMemory()
    
    def select_action(self, state):
        """Select action using current policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, value = self.policy.get_action(state)
        
        if self.continuous:
            return action.cpu().numpy()[0], log_prob.item(), value.item()
        else:
            return action.item(), log_prob.item(), value.item()
    
    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        return advantages
    
    def update(self, next_state):
        """Update policy using PPO algorithm"""
        states, actions, old_log_probs, rewards, dones, values = self.memory.get()
        
        # Compute advantages
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.continuous:
                _, _, next_value = self.policy.forward(next_state_tensor)
            else:
                _, next_value = self.policy.forward(next_state_tensor)
            next_value = next_value.item()
        
        advantages = self.compute_gae(rewards, values, dones, next_value)
        
        # Compute returns
        returns = [adv + val for adv, val in zip(advantages, values)]
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        if self.continuous:
            actions = torch.FloatTensor(np.array(actions)).to(self.device)
        else:
            actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages (only if we have more than 1 sample)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(self.epochs):
            # Random mini-batches
            indices = np.random.permutation(len(states))
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Evaluate actions
                log_probs, state_values, entropy = self.policy.evaluate(batch_states, batch_actions)
                
                # Compute ratio
                ratios = torch.exp(log_probs - batch_old_log_probs)
                
                # Compute surrogate losses
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                
                # Compute losses
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * (batch_returns - state_values).pow(2).mean()
                entropy_loss = -self.entropy_coef * entropy.mean()
                
                loss = actor_loss + critic_loss + entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                
                # Check for NaN and stop if found
                if torch.isnan(loss):
                    print("Warning: NaN detected in loss, skipping update")
                    break
        
        self.memory.clear()
    
    def save(self, filepath):
        """Save model weights"""
        torch.save(self.policy.state_dict(), filepath)
    
    def load(self, filepath):
        """Load model weights"""
        self.policy.load_state_dict(torch.load(filepath, map_location=self.device))
        self.policy.eval()