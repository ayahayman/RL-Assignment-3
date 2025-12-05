"""
Quick test to verify SAC implementation is working correctly
"""
import torch
import numpy as np
from models.SAC import SACAgent

# Test 1: Agent initialization
print("Test 1: Creating SAC agent...")
agent = SACAgent(
    state_dim=4,
    action_dim=1,
    hidden_dim=256,
    gamma=0.99,
    actor_lr=0.0003,
    critic_lr=0.0003,
    tau=0.005,
    alpha=0.2,
    batch_size=32,
    buffer_size=10000
)
print("✓ Agent created successfully")

# Test 2: Action selection
print("\nTest 2: Testing action selection...")
state = np.random.randn(4)
action = agent.select_action(state)
print(f"✓ Action sampled: {action}, shape: {action.shape}")
assert action.shape == (1,), "Action should be 1-dimensional"

# Test 3: Store transitions
print("\nTest 3: Storing transitions...")
for i in range(50):
    state = np.random.randn(4)
    action = np.random.randn(1)
    reward = np.random.randn()
    next_state = np.random.randn(4)
    done = False
    agent.store_transition((state, action, reward, next_state, done))
print(f"✓ Stored {len(agent.memory)} transitions")

# Test 4: Update (should work now)
print("\nTest 4: Testing update...")
actor_loss, critic1_loss, critic2_loss = agent.update()
print(f"✓ Update successful!")
print(f"  Actor loss: {actor_loss:.4f}")
print(f"  Critic1 loss: {critic1_loss:.4f}")
print(f"  Critic2 loss: {critic2_loss:.4f}")

print("\n✅ All tests passed!")
