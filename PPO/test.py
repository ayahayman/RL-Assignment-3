# test.py
import gymnasium as gym
import numpy as np
import wandb

def test_agent(agent, env_name, num_tests=100, render=False, verbose=True, use_wandb=False):
    """
    Test trained agent on environment
    
    Args:
        agent: Trained PPO agent
        env_name: Name of the Gymnasium environment
        num_tests: Number of test episodes
        render: Whether to render the environment
        verbose: Print test progress
    
    Returns:
        test_rewards: List of test episode rewards
        test_lengths: List of test episode lengths
    """
    if render:
        env = gym.make(env_name, render_mode='human')
    else:
        env = gym.make(env_name)
    
    # Setup discretization for Pendulum
    if env_name == 'Pendulum-v1':
        torque_bins = np.linspace(-2.0, 2.0, 11)
    
    test_rewards = []
    test_lengths = []
    
    for test in range(num_tests):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action, _, _ = agent.select_action(state)
            
            # Convert discrete action to continuous for Pendulum
            if env_name == 'Pendulum-v1':
                continuous_action = np.array([torque_bins[action]], dtype=np.float32)
                next_state, reward, terminated, truncated, _ = env.step(continuous_action)
            else:
                next_state, reward, terminated, truncated, _ = env.step(action)
            
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        test_rewards.append(episode_reward)
        test_lengths.append(episode_length)
        
        if verbose and (test + 1) % 20 == 0:
            print(f"Test {test + 1}/{num_tests} completed")
    
    env.close()
    
    # Log test statistics to wandb
    if use_wandb:
        wandb.log({
            f"{env_name}/test_mean_reward_final": np.mean(test_rewards),
            f"{env_name}/test_std_reward_final": np.std(test_rewards),
            f"{env_name}/test_min_reward": np.min(test_rewards),
            f"{env_name}/test_max_reward": np.max(test_rewards),
            f"{env_name}/test_mean_length_final": np.mean(test_lengths)
        })
    
    if verbose:
        print(f"\nTest Results:")
        print(f"Mean Reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
        print(f"Mean Length: {np.mean(test_lengths):.2f} ± {np.std(test_lengths):.2f}")
        print(f"Min/Max Reward: {np.min(test_rewards):.2f} / {np.max(test_rewards):.2f}")
    
    return test_rewards, test_lengths