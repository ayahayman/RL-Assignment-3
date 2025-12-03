# a2c_main.py (Interactive version)

import os
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import wandb

from models.A2C import A2CAgent


# ============================================================
# ENVIRONMENTS MENU
# ============================================================

ENV_MENU = {
    1: ("CartPole-v1", True),           # discrete
    2: ("MountainCar-v0", True),        # discrete
    3: ("Acrobot-v1", True),            # discrete
    4: ("Pendulum-v1", False)           # continuous → discretized
}

# torque discretization for Pendulum
PENDULUM_BINS = 9
PENDULUM_TORQUES = np.linspace(-2.0, 2.0, PENDULUM_BINS).astype(np.float32)


# ============================================================
# MAKE ENV + ACTION SPACE WRAPPER
# ============================================================

def make_env_and_actions(env_name: str, is_discrete: bool):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]

    if is_discrete:
        act_dim = env.action_space.n
        action_values = None
    else:
        # Pendulum: discretize continuous torque
        act_dim = PENDULUM_BINS
        action_values = PENDULUM_TORQUES

    return env, obs_dim, act_dim, action_values


# ============================================================
# TRAINING LOOP
# ============================================================

def train_a2c(env, agent, episodes, action_values, use_wandb, save_dir):

    os.makedirs(save_dir, exist_ok=True)
    env_name = env.spec.id

    print(f"\n🚀 Training A2C on {env_name} for {episodes} episodes...\n")

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action_idx, logp = agent.select_action(obs)

            # map discrete → real continuous torque (Pendulum)
            if action_values is not None:
                action = np.array([action_values[action_idx]], dtype=np.float32)
                next_obs, reward, terminated, truncated, _ = env.step(action)
            else:
                next_obs, reward, terminated, truncated, _ = env.step(action_idx)

            done = terminated or truncated
            agent.store(obs, action_idx, reward, done, logp)

            obs = next_obs
            total_reward += reward
            steps += 1

        # update
        actor_loss, critic_loss = agent.update()

        if use_wandb:
            wandb.log({
                "episode": ep,
                "reward": total_reward,
                "steps": steps,
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
            })

        print(f"Episode {ep+1}/{episodes} | Reward={total_reward:.2f} "
              f"| Steps={steps} | A_Loss={actor_loss:.4f} | C_Loss={critic_loss:.4f}")

    # save final model
    save_path = os.path.join(save_dir, f"a2c_{env_name}.pth")
    torch.save({
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "env": env_name,
        "act_dim": agent.act_dim
    }, save_path)

    print(f"\n💾 Model saved to {save_path}")

    return save_path


# ============================================================
# MAIN MENU (NO ARGUMENTS)
# ============================================================

if __name__ == "__main__":

    print("\n======= A2C TRAINER =======")
    print("Select environment to train:")
    print("1 → CartPole-v1")
    print("2 → MountainCar-v0")
    print("3 → Acrobot-v1")
    print("4 → Pendulum-v1 (discretized)")
    print("===========================\n")

    choice = int(input("Enter choice (1–4): ").strip())

    if choice not in ENV_MENU:
        print("❌ Invalid choice.")
        exit()

    env_name, is_discrete = ENV_MENU[choice]

    # episodes
    episodes = int(input("Number of training episodes (default=500): ") or "500")

    # wandb logging?
    use_wandb = input("Enable Weights & Biases logging? (y/n): ").lower().startswith("y")

    # learning settings
    lr = 3e-4
    gamma = 0.99
    entropy_coef = 0.001
    hidden_sizes = (128, 128)

    # create environment
    env, obs_dim, act_dim, action_values = make_env_and_actions(env_name, is_discrete)

    # initialize wandb
    if use_wandb:
        wandb.init(
            project="RL-Assignment3",
            name=f"A2C_{env_name}",
            config={
                "env": env_name,
                "episodes": episodes,
                "lr": lr,
                "gamma": gamma,
                "entropy_coef": entropy_coef,
                "hidden_sizes": hidden_sizes,
                "discrete": is_discrete,
                "action_bins": PENDULUM_BINS if not is_discrete else None,
            }
        )

    # Create agent
    agent = A2CAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=hidden_sizes,
        lr=lr,
        gamma=gamma,
        entropy_coef=entropy_coef,
        device="cpu"
    )

    # Train
    checkpoint = train_a2c(
        env=env,
        agent=agent,
        episodes=episodes,
        action_values=action_values,
        use_wandb=use_wandb,
        save_dir="trained_models/A2C"
    )

    if use_wandb:
        wandb.save(checkpoint)
        wandb.finish()

    env.close()

    print("\n🎉 Training Complete!\n")
