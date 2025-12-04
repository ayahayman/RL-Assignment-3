# memory.py

class PPOMemory:
    """Memory buffer for storing PPO trajectories"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
    
    def store(self, state, action, log_prob, reward, done, value):
        """Store a transition"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
    
    def clear(self):
        """Clear all stored transitions"""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
    
    def get(self):
        """Get all stored transitions"""
        return (self.states, self.actions, self.log_probs, 
                self.rewards, self.dones, self.values)
    
    def __len__(self):
        return len(self.states)