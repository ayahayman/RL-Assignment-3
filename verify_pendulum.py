import gymnasium as gym
import numpy as np

def verify_pendulum():
    env = gym.make('Pendulum-v1')
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    torque_bins = np.linspace(-2.0, 2.0, 11)
    print(f"Torque bins: {torque_bins}")
    
    state, _ = env.reset()
    total_reward = 0
    
    for i in range(10):
        # Pick a random discrete action
        action_idx = np.random.randint(0, 11)
        continuous_action = np.array([torque_bins[action_idx]], dtype=np.float32)
        
        print(f"Step {i}: Discrete {action_idx} -> Continuous {continuous_action}")
        
        next_state, reward, terminated, truncated, _ = env.step(continuous_action)
        total_reward += reward
        
        if terminated or truncated:
            break
            
    print("Pendulum verification successful!")
    env.close()

if __name__ == "__main__":
    verify_pendulum()