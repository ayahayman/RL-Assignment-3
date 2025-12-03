"""
Central configuration file for A2C training.
Modify hyperparameters for each environment here.
"""

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
        "entropy_coef": 0.001,

        "hidden_sizes": (128, 128),    # better for continuous control

        "episodes": 500,
        "action_bins": 9,              # discrete torque levels

        
    },
}


def get_config(env_name: str) -> dict:
    """
    Returns merged configuration:
    default values + environment-specific overrides.
    """
    cfg = dict(A2C_CONFIG["default"])

    if env_name in A2C_CONFIG:
        cfg.update(A2C_CONFIG[env_name])

    return cfg
