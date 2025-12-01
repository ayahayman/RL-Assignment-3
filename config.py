"""
config.py
---------
This file stores ALL hyperparameter definitions used in the project.
It also defines the hyperparameter search space that train.py will loop over.

The goal:
- Provide clean organization
- Allow easy tuning
- Avoid hard-coding values inside the training loop
"""

# ============================================
# DEFAULT HYPERPARAMETERS FOR EACH AGENT
# ============================================

DEFAULT_HYPERPARAMS = {
    "episodes": 500,
    "discount_factor": 0.99,     # (Gamma γ)
    "learning_rate": 0.0003,     # NN learning rate
    "batch_size": 64,
    "replay_memory_size": 50000,
    "epsilon_decay": 0.995,      # For exploration (A2C/PPO ignore this)
}


# ============================================
# HYPERPARAMETER TUNING SEARCH SPACE
# ============================================

# The assignment requires tuning these:
DISCOUNT_FACTORS = [0.95, 0.99]
LEARNING_RATES = [1e-2, 1e-3, 3e-4]
MEMORY_SIZES = [20000, 50000, 100000]
BATCH_SIZES = [32, 64, 128]
EPSILON_DECAYS = [0.99, 0.995, 0.999]


# ============================================
# BUILD THE HYPERPARAMETER COMBINATIONS
# ============================================

HYPERPARAM_CONFIG = []

for gamma in DISCOUNT_FACTORS:
    for lr in LEARNING_RATES:
        for mem in MEMORY_SIZES:
            for batch in BATCH_SIZES:
                for eps in EPSILON_DECAYS:
                    combo = {
                        "episodes": DEFAULT_HYPERPARAMS["episodes"],
                        "discount_factor": gamma,
                        "learning_rate": lr,
                        "replay_memory_size": mem,
                        "batch_size": batch,
                        "epsilon_decay": eps
                    }
                    HYPERPARAM_CONFIG.append(combo)
