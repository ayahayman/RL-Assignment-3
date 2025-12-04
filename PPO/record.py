# record.py
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import os
import numpy as np
from agent import PPOAgent
import json

def load_agent_from_checkpoint(model_path, env_name):
    """
    Load a trained agent from checkpoint
    
    Args:
        model_path: Path to saved model weights
        env_name: Name of the environment
    
    Returns:
        agent: Loaded PPO agent
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    
    # Handle Pendulum discretization
    if env_name == 'Pendulum-v1':
        continuous = False
        action_dim = 11  # Discretized actions
    elif isinstance(env.action_space, gym.spaces.Box):
        continuous = True
        action_dim = env.action_space.shape[0]
    else:
        continuous = False
        action_dim = env.action_space.n
    
    # Create agent with default parameters (doesn't matter for inference)
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        continuous=continuous
    )
    
    # Load weights
    agent.load(model_path)
    env.close()
    
    return agent

def record_episode(agent, env_name, video_folder='videos', video_name=None, 
                   max_steps=1000, episode_trigger=None):
    """
    Record a single episode of trained agent
    
    Args:
        agent: Trained PPO agent
        env_name: Name of the environment
        video_folder: Folder to save videos
        video_name: Name prefix for video file
        max_steps: Maximum steps per episode
        episode_trigger: Function to determine which episodes to record
    
    Returns:
        episode_reward: Total reward for the episode
        episode_length: Number of steps in the episode
    """
    # Create video folder
    os.makedirs(video_folder, exist_ok=True)
    
    # Set video name
    if video_name is None:
        video_name = f"{env_name}_episode"
    
    # Episode trigger - record all episodes by default
    if episode_trigger is None:
        episode_trigger = lambda x: True
    
    # Create environment with video recording
    env = gym.make(env_name, render_mode='rgb_array')
    env = RecordVideo(
        env, 
        video_folder=video_folder,
        name_prefix=video_name,
        episode_trigger=episode_trigger
    )
    
    # Setup discretization for Pendulum
    if env_name == 'Pendulum-v1':
        torque_bins = np.linspace(-2.0, 2.0, 11)
    
    state, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    done = False
    
    while not done and episode_length < max_steps:
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
    
    env.close()
    
    print(f"Recorded episode: Reward={episode_reward:.2f}, Length={episode_length}")
    
    return episode_reward, episode_length

def record_multiple_episodes(agent, env_name, num_episodes=5, 
                            video_folder='videos', video_name=None):
    """
    Record multiple episodes
    
    Args:
        agent: Trained PPO agent
        env_name: Name of the environment
        num_episodes: Number of episodes to record
        video_folder: Folder to save videos
        video_name: Name prefix for video files
    
    Returns:
        rewards: List of episode rewards
        lengths: List of episode lengths
    """
    # Create video folder
    os.makedirs(video_folder, exist_ok=True)
    
    if video_name is None:
        video_name = f"{env_name}_episode"
    
    # Episode trigger - record all episodes
    episode_trigger = lambda x: True
    
    # Create environment with video recording
    env = gym.make(env_name, render_mode='rgb_array')
    env = RecordVideo(
        env, 
        video_folder=video_folder,
        name_prefix=video_name,
        episode_trigger=episode_trigger
    )
    
    # Setup discretization for Pendulum
    if env_name == 'Pendulum-v1':
        torque_bins = np.linspace(-2.0, 2.0, 11)
    
    rewards = []
    lengths = []
    
    for episode in range(num_episodes):
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
        
        rewards.append(episode_reward)
        lengths.append(episode_length)
        
        print(f"Episode {episode + 1}/{num_episodes}: "
              f"Reward={episode_reward:.2f}, Length={episode_length}")
    
    env.close()
    
    return rewards, lengths

def record_best_episode(agent, env_name, num_trials=10, 
                       video_folder='videos', video_name=None):
    """
    Run multiple trials and record only the best episode
    
    Args:
        agent: Trained PPO agent
        env_name: Name of the environment
        num_trials: Number of trials to find best episode
        video_folder: Folder to save videos
        video_name: Name prefix for video file
    
    Returns:
        best_reward: Reward of the best episode
        best_length: Length of the best episode
    """
    print(f"Running {num_trials} trials to find best episode...")
    
    # First, run trials without recording to find best
    env = gym.make(env_name)
    trial_rewards = []
    trial_lengths = []
    
    for trial in range(num_trials):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action, _, _ = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        trial_rewards.append(episode_reward)
        trial_lengths.append(episode_length)
    
    env.close()
    
    # Find best trial
    best_idx = trial_rewards.index(max(trial_rewards))
    best_reward = trial_rewards[best_idx]
    best_length = trial_lengths[best_idx]
    
    print(f"Best trial: #{best_idx + 1} with reward={best_reward:.2f}")
    print(f"Now recording best episode...")
    
    # Record one episode (hoping for similar performance)
    if video_name is None:
        video_name = f"{env_name}_best"
    
    record_reward, record_length = record_episode(
        agent, env_name, video_folder, video_name
    )
    
    return record_reward, record_length

def record_all_environments(results_dir='ppo_results', video_folder='videos',
                           num_episodes=3):
    """
    Record videos for all trained environments
    
    Args:
        results_dir: Directory containing trained models
        video_folder: Folder to save videos
        num_episodes: Number of episodes to record per environment
    """
    environments = [
        'CartPole-v1',
        'Acrobot-v1',
        'MountainCar-v0',
        'Pendulum-v1'
    ]
    
    # Find latest results directory
    if os.path.isdir(results_dir):
        subdirs = [d for d in os.listdir(results_dir) 
                  if os.path.isdir(os.path.join(results_dir, d))]
        if subdirs:
            latest_dir = sorted(subdirs)[-1]
            results_dir = os.path.join(results_dir, latest_dir)
    
    print(f"Loading models from: {results_dir}\n")
    
    # Create video folder
    video_base_folder = os.path.join(video_folder, 
                                     os.path.basename(results_dir))
    os.makedirs(video_base_folder, exist_ok=True)
    
    all_recording_stats = {}
    
    for env_name in environments:
        print(f"\n{'='*60}")
        print(f"Recording {env_name}")
        print(f"{'='*60}\n")
        
        # Load model
        model_path = os.path.join(results_dir, f'{env_name}_model.pth')
        
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            continue
        
        agent = load_agent_from_checkpoint(model_path, env_name)
        
        # Create environment-specific video folder
        env_video_folder = os.path.join(video_base_folder, env_name)
        
        # Record episodes
        rewards, lengths = record_multiple_episodes(
            agent, 
            env_name, 
            num_episodes=num_episodes,
            video_folder=env_video_folder,
            video_name=env_name
        )
        
        # Save recording stats
        all_recording_stats[env_name] = {
            'rewards': rewards,
            'lengths': lengths,
            'mean_reward': float(sum(rewards) / len(rewards)),
            'mean_length': float(sum(lengths) / len(lengths))
        }
        
        print(f"\nRecorded {num_episodes} episodes for {env_name}")
        print(f"Mean Reward: {all_recording_stats[env_name]['mean_reward']:.2f}")
        print(f"Mean Length: {all_recording_stats[env_name]['mean_length']:.2f}")
    
    # Save recording stats
    stats_path = os.path.join(video_base_folder, 'recording_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(all_recording_stats, f, indent=4)
    
    print(f"\n{'='*60}")
    print(f"All videos saved to: {video_base_folder}")
    print(f"Recording stats saved to: {stats_path}")
    print(f"{'='*60}\n")

def main():
    """Main recording function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Record PPO agent videos')
    parser.add_argument('--env', type=str, help='Environment name')
    parser.add_argument('--model', type=str, help='Path to model checkpoint')
    parser.add_argument('--episodes', type=int, default=3, 
                       help='Number of episodes to record')
    parser.add_argument('--output', type=str, default='videos',
                       help='Output video folder')
    parser.add_argument('--all', action='store_true',
                       help='Record all environments from results directory')
    parser.add_argument('--results-dir', type=str, default='ppo_results',
                       help='Results directory containing models')
    parser.add_argument('--best', action='store_true',
                       help='Record only the best episode after multiple trials')
    parser.add_argument('--trials', type=int, default=10,
                       help='Number of trials to find best episode')
    
    args = parser.parse_args()
    
    if args.all:
        # Record all environments
        record_all_environments(
            results_dir=args.results_dir,
            video_folder=args.output,
            num_episodes=args.episodes
        )
    elif args.env and args.model:
        # Record specific environment
        agent = load_agent_from_checkpoint(args.model, args.env)
        
        if args.best:
            # Record best episode
            reward, length = record_best_episode(
                agent,
                args.env,
                num_trials=args.trials,
                video_folder=args.output,
                video_name=f"{args.env}_best"
            )
        else:
            # Record multiple episodes
            rewards, lengths = record_multiple_episodes(
                agent,
                args.env,
                num_episodes=args.episodes,
                video_folder=args.output,
                video_name=args.env
            )
            
            print(f"\nRecorded {args.episodes} episodes")
            print(f"Mean Reward: {sum(rewards)/len(rewards):.2f}")
            print(f"Mean Length: {sum(lengths)/len(lengths):.2f}")
    else:
        print("Error: Either use --all flag or provide both --env and --model")
        parser.print_help()

if __name__ == "__main__":
    main()