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
    """Record videos for trained models"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Record videos of trained PPO agents')
    parser.add_argument('--env', type=str, required=True, 
                        choices=['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0', 'Pendulum-v1'],
                        help='Environment name')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the trained model file (.pth)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to record (default: 5)')
    parser.add_argument('--output', type=str, default='videos',
                        help='Output folder for videos (default: videos)')
    
    args = parser.parse_args()
    
    # Validate model path
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    # Record videos
    record_videos(args.env, args.model, args.output, num_episodes=args.episodes)

if __name__ == "__main__":
    main()