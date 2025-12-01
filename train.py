import gymnasium as gym
import torch
import numpy as np
from agents.a2c import A2CAgent
from agents.sac import SACAgent
from agents.ppo import PPOAgent
from utils.env_utils import make_env
from utils.video_recorder import *
from config import HYPERPARAM_CONFIG
import wandb

"""
train.py
--------
This is the MAIN SCRIPT that controls the entire project. It does:

1. Selecting environment (CartPole, Acrobot…etc.)
2. Selecting RL algorithm (A2C / SAC / PPO)
3. Running hyperparameter tuning loops
4. Training the agent
5. Logging results to Weights & Biases
6. Saving trained model
7. Running evaluation test episodes
8. Recording video of the trained agent

You run this file using:
    python train.py
"""

def train_one_model(env_name, agent_name, hyperparams):
    """
    Train ONE model with ONE hyperparameter set.

    This function:
    - Creates env
    - Instantiates agent
    - Runs training loop
    - Logs results to wandb
    """
    # Create the environment
    env = make_env(env_name)

    # Select agent class
    if agent_name == "A2C":
        agent = A2CAgent(env, hyperparams)
    elif agent_name == "SAC":
        agent = SACAgent(env, hyperparams)
    elif agent_name == "PPO":
        agent = PPOAgent(env, hyperparams)
    else:
        raise ValueError("Invalid agent name!")

    # Initialize wandb experiment
    wandb.init(
        project="RL-PolicyGradients",
        config=hyperparams,
        name=f"{agent_name}-{env_name}"
    )

    total_rewards = []

    # Training loop for one model
    for episode in range(hyperparams["episodes"]):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            episode_reward += reward

        total_rewards.append(episode_reward)

        wandb.log({"episode_reward": episode_reward})

        print(f"[{agent_name}] Ep {episode} — Reward: {episode_reward}")

    # Save model
    agent.save(f"{agent_name}-{env_name}.pth")

    env.close()

    return total_rewards


def hyperparameter_search(env_name, agent_name):
    """
    Iterates over many hyperparameter combinations.

    Your assignment requires tuning:
    - Discount factor
    - Learning rate
    - Replay memory size
    - Batch size
    - Epsilon decay (for exploration)
    """
    all_results = {}

    for hp_set in HYPERPARAM_CONFIG:
        print("\n==============================")
        print("Running new hyperparameter set:")
        print(hp_set)
        print("==============================\n")

        results = train_one_model(env_name, agent_name, hp_set)
        all_results[str(hp_set)] = results

    return all_results


if __name__ == "__main__":
    # Choose environment and agent manually for now
    env_name = "CartPole-v1"
    agent_name = "A2C"

    # Start hyperparameter tuning
    results = hyperparameter_search(env_name, agent_name)
