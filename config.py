"""
Central configuration file for A2C training.
Modify hyperparameters for each environment here.
"""
ENVIRONMENTS = {
    "CartPole-v1": {
        "target_reward": 475
    },
    "Acrobot-v1": {
        "target_reward": -100
    },
    "MountainCar-v0": {
        "target_reward": -110
    },
    "Pendulum-v1": {
        "target_reward": -200
    }
}
A2C_CONFIG = {

    # =====================================================
    # DEFAULT SETTINGS (used when env does not override)
    # =====================================================
    "default": {
        "actor_lr": 3e-4,
        "critic_lr": 3e-4,

        "gamma": 0.99,
        "entropy_coef": 0.001,

        "hidden_sizes": (128, 128),

        "episodes": 500,
        "seed": None,

        "action_bins": 9,  # for discretized continuous environments
        "save_dir": "trained_models/A2C",
        "wandb": True,
    },

    # =====================================================
    # ENV-SPECIFIC OVERRIDES
    # =====================================================

    "CartPole-v1": {
        "actor_lr": 7e-4,
        "critic_lr": 7e-4,

        "gamma": 0.99,
        "entropy_coef": 0.001,

        "hidden_sizes": (128, 128),

        "episodes": 500,
        "seed": 42,
    },

    "MountainCar-v0": {
        "actor_lr": 5e-4,
        "critic_lr": 5e-4,

        "gamma": 0.995,
        "entropy_coef": 0.005,

        "hidden_sizes": (128, 128),

        "episodes": 2000,
        "seed": 123,
    },

    "Acrobot-v1": {
        "actor_lr": 3e-4,
        "critic_lr": 3e-4,

        "gamma": 0.99,
        "entropy_coef": 0.001,

        "hidden_sizes": (128, 128),

        "episodes": 1500,
        "seed": None,
    },

    "Pendulum-v1": {
        "actor_lr": 0.0003,
        "critic_lr": 0.001,      # critic needs larger LR to stabilize value estimates

        "gamma": 0.95,
        "entropy_coef": 0.01,

        "hidden_sizes": (128, 128),    # better for continuous control

        "episodes": 500,
        "action_bins": 9,              # discrete torque levels

        
    },
}


SAC_CONFIG = {
    "default": {
        "Gamma": 0.999,
        "Actor LR": 0.0004,
        "Critic 1 LR": 0.001,
        "Critic 2 LR": 0.001,
        "Entropy Coef": 0.01,
        "Hidden Dim": 128,
        "Training Episodes": 500
    },
    "CartPole-v1": {
        "Gamma": 0.99,
        "Actor LR": 0.0004,
        "Critic 1 LR": 0.0005,
        "Critic 2 LR": 0.0005,
        "Entropy Coef": 0.01,
        "Hidden Dim": 128,
        "Training Episodes": 500,
        "n_steps": 5

    },
    "Acrobot-v1": {
        "Gamma": 0.99,
        "Actor LR": 0.0005,
        "Critic 1 LR": 0.001,
        "Critic 2 LR": 0.001,
        "Entropy Coef": 0,
        "Hidden Dim": 128,
        "Training Episodes": 500
    },
    "MountainCar-v0": {
        "Gamma": 0.99,
        "Actor LR": 0.0007,
        "Critic 1 LR": 0.001,
        "Critic 2 LR": 0.001,
        "Entropy Coef": 0,
        "Hidden Dim": 128,
        "Training Episodes": 1000
    },
   "Pendulum-v1": {
        "Gamma": 0.97,
        "Actor LR": 0.00025,
        "Critic 1 LR": 0.0005,
        "Critic 2 LR": 0.0005,
        "Entropy Coef": 0.005,
        "Hidden Dim": 256,
        "Training Episodes": 1000
    
    }
}
# Add this new function for SAC
def get_sac_config(env_name: str) -> dict:
    """
    Returns merged SAC configuration:
    default values + environment-specific overrides.
    """
    cfg = dict(SAC_CONFIG["default"])
    
    if env_name in SAC_CONFIG:
        cfg.update(SAC_CONFIG[env_name])
    
    return cfg

# Keep your existing get_config for A2C
def get_config(env_name: str) -> dict:
    """
    Returns merged A2C configuration:
    default values + environment-specific overrides.
    """
    cfg = dict(A2C_CONFIG["default"])
    
    if env_name in A2C_CONFIG:
        cfg.update(A2C_CONFIG[env_name])
    
    return cfg