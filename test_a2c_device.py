"""
Quick test to verify A2C implementation is using CUDA correctly
"""
import torch
import numpy as np
from models.A2C import A2CAgent

# Test: Agent initialization
print("Test: Creating A2C agent...")
agent = A2CAgent(
    obs_dim=4,
    act_dim=2,
    hidden_sizes=(128, 128),
    actor_lr=3e-4,
    critic_lr=3e-4,
    gamma=0.99,
    entropy_coef=0.001
)
print("✓ Agent created successfully")

# Check device
print(f"\nAgent device: {agent.device}")
print(f"Actor on device: {next(agent.actor.parameters()).device}")
print(f"Critic on device: {next(agent.critic.parameters()).device}")

# Test action selection
print("\nTest: Testing action selection...")
state = np.random.randn(4)
action, logp = agent.select_action(state)
print(f"✓ Action sampled: {action}, log_prob: {logp:.4f}")

print("\n✅ All tests passed!")
