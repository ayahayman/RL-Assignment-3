# record_videos.py
"""
Simple script to record videos of trained PPO agents
Usage: python record_videos.py
"""

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import os
import glob
import numpy as np
from agent import PPOAgent

def load_trained_agent(model_path, env_name):
    """Load a trained agent"""
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
    
    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, continuous=continuous)
    agent.load(model_path)
    env.close()
    
    return agent

def record_videos(env_name, model_path, output_folder='videos', num_episodes=5):
    """Record videos of agent performance"""
    
    print(f"\n{'='*60}")
    print(f"Recording {env_name}")
    print(f"{'='*60}\n")
    
    # Load agent
    try:
        agent = load_trained_agent(model_path, env_name)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Create output folder
    env_video_folder = os.path.join(output_folder, env_name)
    os.makedirs(env_video_folder, exist_ok=True)
    
    # Create environment with video recording
    env = gym.make(env_name, render_mode='rgb_array')
    env = RecordVideo(
        env,
        video_folder=env_video_folder,
        name_prefix=f"{env_name}",
        episode_trigger=lambda x: True  # Record all episodes
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
        
        print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, Length={episode_length}")
    
    env.close()
    
    abs_path = os.path.abspath(env_video_folder)
    print(f"\nVideos saved to: {abs_path}")
    
    # List generated video files
    video_files = glob.glob(os.path.join(env_video_folder, "*.mp4"))
    if video_files:
        print("Generated files:")
        for vf in video_files:
            print(f"  - {os.path.basename(vf)}")
            
    print(f"Mean Reward: {sum(rewards)/len(rewards):.2f}")
    print(f"Mean Length: {sum(lengths)/len(lengths):.2f}\n")

def main():
    """Record videos for selected trained models"""
    
    # Find the latest results directory
    # Check both current directory and PPO subdirectory
    results_base = 'ppo_results'
    if not os.path.exists(results_base):
        results_base = os.path.join('PPO', 'ppo_results')
    
    if not os.path.exists(results_base):
        print(f"Error: ppo_results directory not found!")
        print("Please run main.py first to train the models.")
        return
    
    # Get latest results folder
    subdirs = [d for d in os.listdir(results_base) 
               if os.path.isdir(os.path.join(results_base, d))]
    
    if not subdirs:
        print(f"No results found in {results_base}")
        return
    
    latest_dir = sorted(subdirs)[-1]
    results_dir = os.path.join(results_base, latest_dir)
    
    print(f"\nLoading models from: {results_dir}")
    
    # Environment list
    environments = {
        '1': ('CartPole-v1', 'CartPole-v1_model.pth'),
        '2': ('Acrobot-v1', 'Acrobot-v1_model.pth'),
        '3': ('MountainCar-v0', 'MountainCar-v0_model.pth'),
        '4': ('Pendulum-v1', 'Pendulum-v1_model.pth')
    }
    
    # Show menu
    print("\n" + "="*60)
    print("Select environments to record (comma-separated):")
    print("="*60)
    for key, (env_name, _) in environments.items():
        model_path = os.path.join(results_dir, environments[key][1])
        status = "✓" if os.path.exists(model_path) else "✗"
        print(f"  {key}. {env_name} {status}")
    print("  5. Record ALL environments")
    print("="*60)
    
    choice = input("\nEnter your choice (e.g., 1,3 or 5 for all): ").strip()
    
    # Parse selection
    selected_envs = []
    if choice == '5':
        selected_envs = list(environments.keys())[:4]  # Exclude '5' itself
    else:
        selected_envs = [c.strip() for c in choice.split(',') if c.strip() in environments]
    
    if not selected_envs:
        print("No valid selection made.")
        return
    
    # Get number of episodes
    try:
        num_episodes = int(input("\nNumber of episodes to record per environment (default=5): ").strip() or "5")
    except ValueError:
        num_episodes = 5
    
    # Create videos folder
    video_folder = os.path.join('videos', latest_dir)
    
    # Record videos for selected environments
    for key in selected_envs:
        env_name, model_file = environments[key]
        model_path = os.path.join(results_dir, model_file)
        
        if os.path.exists(model_path):
            record_videos(env_name, model_path, video_folder, num_episodes=num_episodes)
        else:
            print(f"Model not found: {model_path}\n")
    
    print(f"{'='*60}")
    print(f"All videos saved to: {video_folder}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()