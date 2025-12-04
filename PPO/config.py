# config.py
"""
Configuration file for PPO training
Contains default hyperparameters and environment settings
"""

# Default PPO hyperparameters
DEFAULT_PPO_PARAMS = {
    'learning_rate': 3e-4,
    'discount_factor': 0.99,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.2,
    'epochs': 10,
    'batch_size': 64,
    'update_freq': 2048
}

# Hyperparameter search space
HYPERPARAM_SEARCH_SPACE = {
    'discount_factor': [0.95, 0.98, 0.99],
    'learning_rate': [1e-4, 3e-4, 5e-4],
    'batch_size': [32, 64, 128],
    'clip_epsilon': [0.1, 0.2, 0.3]
}

# Environment-specific settings
ENV_CONFIGS = {
    'CartPole-v1': {
        'max_episodes': 500,
        'update_freq': 2048,
        'solved_threshold': 475.0
    },
    'Acrobot-v1': {
        'max_episodes': 500,
        'update_freq': 2048,
        'solved_threshold': -100.0
    },
    'MountainCar-v0': {
        'max_episodes': 500,
        'update_freq': 2048,
        'solved_threshold': -110.0
    },
    'Pendulum-v1': {
        'max_episodes': 500,
        'update_freq': 2048,
        'solved_threshold': -200.0
    }
}

# Training settings
TRAINING_CONFIG = {
    'num_tuning_trials': 5,
    'episodes_per_trial': 200,
    'final_training_episodes': 500,
    'test_episodes': 100,
    'random_seed': 42
}

# Neural network architecture
NN_CONFIG = {
    'hidden_dim': 64,
    'activation': 'relu'
}

# Results directory
RESULTS_DIR = 'ppo_results'