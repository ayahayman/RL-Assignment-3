# utils/env_utils.py

import gymnasium as gym
import numpy as np
import os


def make_env(env_id: str, seed: int | None = None, render_mode: str | None = None):
    """
    Create a Gymnasium environment with optional seed and render_mode.

    Args:
        env_id: e.g. "CartPole-v1", "Pendulum-v1"
        seed: integer or None
        render_mode: None, or "rgb_array" / "human"; for recording we usually want "rgb_array"
    
    Returns:
        env: a Gymnasium env ready to use with env.reset() -> obs, info
    """
    # Many Gym environment constructors accept render_mode (some don't)
    kwargs = {}
    if render_mode is not None:
        # pass render_mode if the env supports it
        kwargs["render_mode"] = render_mode

    env = gym.make(env_id, **kwargs)

    # seeding - Gymnasium recommends seeding when you reset:
    if seed is not None:
        # set numpy + env seed for reproducibility
        np.random.seed(seed)
        # resetting with seed ensures both env & action_space seeds propagate
        env.reset(seed=seed)
        try:
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
        except Exception:
            pass

    return env


def safe_close_env(env):
    """Safely close an environment if it has close()."""
    try:
        env.close()
    except Exception:
        pass
