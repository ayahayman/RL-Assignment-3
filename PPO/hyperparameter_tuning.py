# hyperparameter_tuning.py
import numpy as np
from train import train_ppo

def hyperparameter_search(env_name, num_trials=5, episodes_per_trial=200):
    """
    Perform random hyperparameter search
    
    Args:
        env_name: Name of the Gymnasium environment
        num_trials: Number of hyperparameter combinations to try
        episodes_per_trial: Number of episodes to train each trial
    
    Returns:
        best_params: Best hyperparameters found
        results: List of all trial results
    """
    # Define hyperparameter ranges based on configuration
    # Using ranges from the provided configuration
    param_ranges = {
        'learning_rate': [0.00015, 0.0003, 0.0006],  # 0.5x, 1x, 2x multipliers
        'gamma': [0.95, 0.99, 0.999],  # low, default, high
        'gae_lambda': [0.9, 0.95, 0.98],  # low, default, high
        'clip_range': [0.1, 0.2, 0.3],  # low, default, high
        'entropy_coef': [0.005, 0.01, 0.02],  # 0.5x, 1x, 2x multipliers
        'batch_size': [32, 64, 128],  # low, default, high
        'n_epochs': [3, 10, 20],  # low, default, high
        'n_steps': [1024, 2048, 4096]  # low, default, high
    }
    
    # Environment-specific overrides if needed
    if env_name == 'MountainCar-v0':
        # MountainCar needs higher learning rate for sparse rewards
        param_ranges['learning_rate'] = [5e-4, 1e-3, 3e-3]
        param_ranges['n_steps'] = [1024, 2048]
    elif env_name == 'Pendulum-v1':
        # Pendulum tuning
        param_ranges['gamma'] = [0.9, 0.95, 0.99]
        param_ranges['learning_rate'] = [1e-4, 3e-4, 1e-3]
    
    best_reward = -float('inf')
    best_params = None
    results = []
    
    print(f"\n{'='*60}")
    print(f"Hyperparameter Tuning for {env_name}")
    print(f"{'='*60}\n")
    
    for trial in range(num_trials):
        # Random sample hyperparameters
        hyperparams = {
            'learning_rate': float(np.random.choice(param_ranges['learning_rate'])),
            'gamma': float(np.random.choice(param_ranges['gamma'])),
            'gae_lambda': float(np.random.choice(param_ranges['gae_lambda'])),
            'clip_range': float(np.random.choice(param_ranges['clip_range'])),
            'entropy_coef': float(np.random.choice(param_ranges['entropy_coef'])),
            'batch_size': int(np.random.choice(param_ranges['batch_size'])),
            'n_epochs': int(np.random.choice(param_ranges['n_epochs'])),
            'n_steps': int(np.random.choice(param_ranges['n_steps']))
        }
        
        print(f"Trial {trial + 1}/{num_trials}")
        print(f"Parameters: {hyperparams}")
        
        # Train with these hyperparameters
        agent, rewards, lengths = train_ppo(
            env_name, 
            hyperparams, 
            episodes=episodes_per_trial,
            verbose=False
        )
        
        # Evaluate on last 50 episodes
        avg_reward = np.mean(rewards[-50:])
        results.append({
            'params': hyperparams,
            'avg_reward': avg_reward,
            'all_rewards': rewards
        })
        
        print(f"Average Reward (last 50 episodes): {avg_reward:.2f}\n")
        
        if avg_reward > best_reward:
            best_reward = avg_reward
            best_params = hyperparams
    
    print(f"{'='*60}")
    print(f"Best Parameters: {best_params}")
    print(f"Best Average Reward: {best_reward:.2f}")
    print(f"{'='*60}\n")
    
    return best_params, results


def grid_search(env_name, episodes_per_trial=200):
    """
    Perform grid search over hyperparameters (more exhaustive)
    
    Args:
        env_name: Name of the Gymnasium environment
        episodes_per_trial: Number of episodes to train each trial
    
    Returns:
        best_params: Best hyperparameters found
        results: List of all trial results
    """
    # Define hyperparameter grid
    param_grid = {
        'discount_factor': [0.95, 0.99],
        'learning_rate': [1e-4, 3e-4],
        'batch_size': [64, 128],
        'clip_epsilon': [0.2]
    }
    
    best_reward = -float('inf')
    best_params = None
    results = []
    
    print(f"\n{'='*60}")
    print(f"Grid Search for {env_name}")
    print(f"{'='*60}\n")
    
    trial = 0
    total_trials = (len(param_grid['discount_factor']) * 
                   len(param_grid['learning_rate']) * 
                   len(param_grid['batch_size']) * 
                   len(param_grid['clip_epsilon']))
    
    for gamma in param_grid['discount_factor']:
        for lr in param_grid['learning_rate']:
            for batch_size in param_grid['batch_size']:
                for clip_eps in param_grid['clip_epsilon']:
                    trial += 1
                    
                    hyperparams = {
                        'discount_factor': gamma,
                        'learning_rate': lr,
                        'batch_size': batch_size,
                        'clip_epsilon': clip_eps
                    }
                    
                    print(f"Trial {trial}/{total_trials}")
                    print(f"Parameters: {hyperparams}")
                    
                    # Train with these hyperparameters
                    agent, rewards, lengths = train_ppo(
                        env_name, 
                        hyperparams, 
                        episodes=episodes_per_trial,
                        verbose=False
                    )
                    
                    # Evaluate on last 50 episodes
                    avg_reward = np.mean(rewards[-50:])
                    results.append({
                        'params': hyperparams,
                        'avg_reward': avg_reward,
                        'all_rewards': rewards
                    })
                    
                    print(f"Average Reward (last 50 episodes): {avg_reward:.2f}\n")
                    
                    if avg_reward > best_reward:
                        best_reward = avg_reward
                        best_params = hyperparams
    
    print(f"{'='*60}")
    print(f"Best Parameters: {best_params}")
    print(f"Best Average Reward: {best_reward:.2f}")
    print(f"{'='*60}\n")
    
    return best_params, results