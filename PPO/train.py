# train.py
import gymnasium as gym
import numpy as np
import wandb
from agent import PPOAgent

def train_ppo(env_name, hyperparams, episodes=500, update_freq=2048, verbose=True, use_wandb=False):
    """
    Train PPO agent on a given environment
    
    Args:
        env_name: Name of the Gymnasium environment
        hyperparams: Dictionary of hyperparameters
        episodes: Number of training episodes
        update_freq: Update policy every N steps
        verbose: Print training progress
        use_wandb: Whether to log to Weights & Biases
    
    Returns:
        agent: Trained PPO agent
        rewards_history: List of episode rewards
        episode_lengths: List of episode lengths
    """
    # Initialize wandb
    if use_wandb:
        wandb.init(
            project="RL-Assignment-3-PPO",
            name=f"PPO_{env_name}",
            config={
                "environment": env_name,
                "episodes": episodes,
                "update_freq": update_freq,
                **hyperparams
            },
            reinit=True
        )
    
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    
    # Check if continuous or discrete action space
    # Force Pendulum to use discrete actions
    if env_name == 'Pendulum-v1':
        continuous = False
        action_dim = 11  # Discretize torque into 11 bins: -2.0 to 2.0
        torque_bins = np.linspace(-2.0, 2.0, action_dim)
    elif isinstance(env.action_space, gym.spaces.Box):
        continuous = True
        action_dim = env.action_space.shape[0]
    else:
        continuous = False
        action_dim = env.action_space.n
    
    # Environment-specific adjustments
    # Use n_steps from hyperparams if provided
    if 'n_steps' in hyperparams:
        update_freq = hyperparams['n_steps']
    elif env_name == 'MountainCar-v0':
        update_freq = 1024  # More frequent updates for sparse rewards
    elif env_name == 'Pendulum-v1':
        update_freq = 2048  # Keep default
    
    # Create agent
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        continuous=continuous,
        lr=hyperparams.get('learning_rate', 0.0003),
        gamma=hyperparams.get('gamma', hyperparams.get('discount_factor', 0.99)),
        gae_lambda=hyperparams.get('gae_lambda', 0.95),
        clip_epsilon=hyperparams.get('clip_epsilon', hyperparams.get('clip_range', 0.2)),
        entropy_coef=hyperparams.get('entropy_coef', 0.01),
        epochs=hyperparams.get('epochs', hyperparams.get('n_epochs', 10)),
        batch_size=hyperparams.get('batch_size', 64)
    )
    
    rewards_history = []
    episode_lengths = []
    steps = 0
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action, log_prob, value = agent.select_action(state)
            
            # Convert discrete action to continuous for Pendulum
            if env_name == 'Pendulum-v1':
                continuous_action = np.array([torque_bins[action]], dtype=np.float32)
                next_state, reward, terminated, truncated, _ = env.step(continuous_action)
            else:
                next_state, reward, terminated, truncated, _ = env.step(action)
            
            done = terminated or truncated
            
            agent.memory.store(state, action, log_prob, reward, done, value)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            steps += 1
            
            # Update policy
            if steps % update_freq == 0 and len(agent.memory) > 0:
                agent.update(next_state)
        
        # Update at end of episode if we have enough data
        if len(agent.memory) >= agent.batch_size:
            agent.update(next_state)
        
        rewards_history.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Log to wandb
        if use_wandb:
            wandb.log({
                f"{env_name}/train_episode": episode + 1,
                f"{env_name}/train_reward": episode_reward,
                f"{env_name}/train_length": episode_length,
                f"{env_name}/train_avg_reward_50": np.mean(rewards_history[-50:]) if len(rewards_history) >= 50 else np.mean(rewards_history),
                f"{env_name}/train_avg_length_50": np.mean(episode_lengths[-50:]) if len(episode_lengths) >= 50 else np.mean(episode_lengths),
                f"{env_name}/train_total_steps": steps
            })
        
        if verbose and (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            avg_length = np.mean(episode_lengths[-50:])
            print(f"Episode {episode + 1}/{episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Length: {avg_length:.2f}")
    
    env.close()
    
    return agent, rewards_history, episode_lengths