# utils.py
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def save_results(results, filepath):
    """Save results to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert numpy types to native Python types
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    results = convert_numpy(results)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {filepath}")

def load_results(filepath):
    """Load results from JSON file"""
    with open(filepath, 'r') as f:
        results = json.load(f)
    return results

def plot_training_curves(rewards, lengths, env_name, save_path=None):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Raw rewards
    axes[0, 0].plot(rewards, alpha=0.3, label='Raw')
    if len(rewards) >= 50:
        moving_avg = np.convolve(rewards, np.ones(50)/50, mode='valid')
        axes[0, 0].plot(range(49, len(rewards)), moving_avg, label='MA(50)', linewidth=2)
    axes[0, 0].set_title(f'{env_name} - Training Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Episode lengths
    axes[0, 1].plot(lengths, alpha=0.3, label='Raw')
    if len(lengths) >= 50:
        moving_avg = np.convolve(lengths, np.ones(50)/50, mode='valid')
        axes[0, 1].plot(range(49, len(lengths)), moving_avg, label='MA(50)', linewidth=2)
    axes[0, 1].set_title(f'{env_name} - Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Reward distribution (last 100 episodes)
    axes[1, 0].hist(rewards[-100:], bins=20, edgecolor='black', alpha=0.7)
    axes[1, 0].set_title('Reward Distribution (Last 100 Episodes)')
    axes[1, 0].set_xlabel('Reward')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Cumulative reward
    cumulative_rewards = np.cumsum(rewards)
    axes[1, 1].plot(cumulative_rewards)
    axes[1, 1].set_title('Cumulative Reward')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Cumulative Reward')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.close()

def plot_test_results(test_rewards, test_lengths, env_name, save_path=None):
    """Plot test results"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Reward distribution
    axes[0].hist(test_rewards, bins=20, edgecolor='black', alpha=0.7, color='green')
    axes[0].axvline(np.mean(test_rewards), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[0].set_title(f'{env_name} - Test Reward Distribution')
    axes[0].set_xlabel('Reward')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Length distribution
    axes[1].hist(test_lengths, bins=20, edgecolor='black', alpha=0.7, color='blue')
    axes[1].axvline(np.mean(test_lengths), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[1].set_title(f'{env_name} - Test Episode Length Distribution')
    axes[1].set_xlabel('Episode Length')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Scatter plot
    axes[2].scatter(test_lengths, test_rewards, alpha=0.6)
    axes[2].set_title('Reward vs Episode Length')
    axes[2].set_xlabel('Episode Length')
    axes[2].set_ylabel('Reward')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Test results saved to {save_path}")
    
    plt.close()

def plot_hyperparameter_comparison(tuning_results, env_name, save_path=None):
    """Plot hyperparameter tuning results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Extract data with backward compatibility
    learning_rates = [r['params'].get('learning_rate') for r in tuning_results]
    discount_factors = [r['params'].get('gamma', r['params'].get('discount_factor')) for r in tuning_results]
    batch_sizes = [r['params'].get('batch_size') for r in tuning_results]
    clip_epsilons = [r['params'].get('clip_range', r['params'].get('clip_epsilon')) for r in tuning_results]
    avg_rewards = [r['avg_reward'] for r in tuning_results]
    
    # Learning rate vs reward
    axes[0, 0].scatter(learning_rates, avg_rewards, s=100, alpha=0.6)
    axes[0, 0].set_xlabel('Learning Rate')
    axes[0, 0].set_ylabel('Average Reward')
    axes[0, 0].set_title('Learning Rate vs Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Discount factor vs reward
    axes[0, 1].scatter(discount_factors, avg_rewards, s=100, alpha=0.6, color='green')
    axes[0, 1].set_xlabel('Gamma (Discount Factor)')
    axes[0, 1].set_ylabel('Average Reward')
    axes[0, 1].set_title('Gamma vs Reward')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Batch size vs reward
    axes[1, 0].scatter(batch_sizes, avg_rewards, s=100, alpha=0.6, color='red')
    axes[1, 0].set_xlabel('Batch Size')
    axes[1, 0].set_ylabel('Average Reward')
    axes[1, 0].set_title('Batch Size vs Reward')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Clip epsilon vs reward
    axes[1, 1].scatter(clip_epsilons, avg_rewards, s=100, alpha=0.6, color='purple')
    axes[1, 1].set_xlabel('Clip Range')
    axes[1, 1].set_ylabel('Average Reward')
    axes[1, 1].set_title('Clip Range vs Reward')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'{env_name} - Hyperparameter Analysis', fontsize=16, y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Hyperparameter comparison saved to {save_path}")
    
    plt.close()

def print_summary(results):
    """Print summary of all results"""
    print(f"\n{'='*80}")
    print(f"{'FINAL SUMMARY':^80}")
    print(f"{'='*80}\n")
    
    for env_name, env_results in results.items():
        print(f"\n{env_name}:")
        print(f"{'-'*60}")
        print(f"  Best Hyperparameters:")
        for param, value in env_results['best_hyperparameters'].items():
            print(f"    {param}: {value}")
        print(f"\n  Test Statistics:")
        print(f"    Mean Reward: {env_results['test_stats']['mean_reward']:.2f} ± "
              f"{env_results['test_stats']['std_reward']:.2f}")
        print(f"    Mean Episode Length: {env_results['test_stats']['mean_length']:.2f} ± "
              f"{env_results['test_stats']['std_length']:.2f}")
        print(f"    Min/Max Reward: {env_results['test_stats']['min_reward']:.2f} / "
              f"{env_results['test_stats']['max_reward']:.2f}")
    
    print(f"\n{'='*80}\n")

def create_results_directory(base_dir='ppo_results'):
    """Create directory for results with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(base_dir, timestamp)
    os.makedirs(results_dir, exist_ok=True)
    return results_dir