# a2c_main.py (Interactive version, now uses config.py)

import os
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import wandb

from models.A2C import A2CAgent
from config import get_config   # <-- NEW: import config loader


# ============================================================
# ENVIRONMENTS MENU
# ============================================================

ENV_MENU = {
    1: ("CartPole-v1", True),
    2: ("MountainCar-v0", True),
    3: ("Acrobot-v1", True),
    4: ("Pendulum-v1", False)   # continuous → discretized
}


# ============================================================
# MAKE ENV + ACTION SPACE WRAPPER
# ============================================================

def make_env_and_actions(env_name: str, is_discrete: bool, action_bins: int):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]

    if is_discrete:
        act_dim = env.action_space.n
        action_values = None
    else:
        # discretize continuous torque
        low = float(env.action_space.low.min())
        high = float(env.action_space.high.max())
        action_values = np.linspace(low, high, action_bins).astype(np.float32)
        act_dim = len(action_values)

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

        actor_loss, critic_loss = agent.update()

        if use_wandb:
            wandb.log({
                "reward": total_reward,
                "steps": steps,
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
            }, step=ep)

        print(
            f"Episode {ep+1}/{episodes} | Reward={total_reward:.2f} "
            f"| Steps={steps} | A_Loss={actor_loss:.4f} | C_Loss={critic_loss:.4f}"
        )

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

    # Load full config for this environment
    cfg = get_config(env_name)

    # Unpack settings
    actor_lr              = cfg["actor_lr"]
    critic_lr           =cfg["critic_lr"]
    gamma           = cfg["gamma"]
    entropy_coef    = cfg["entropy_coef"]
    hidden_sizes    = cfg["hidden_sizes"]
    episodes        = cfg["episodes"]
    seed            = cfg["seed"]
    action_bins     = cfg["action_bins"]
    save_dir        = cfg["save_dir"]
    use_wandb       = cfg["wandb"]

    print(f"\n📌 Loaded hyperparameters from config.py:\n{cfg}\n")

    # Set seed
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Create environment
    env, obs_dim, act_dim, action_values = make_env_and_actions(env_name, is_discrete, action_bins)

    # Initialize wandb
    if use_wandb:
        wandb.init(
            project="RL-Assignment3",
            name=f"A2C_{env_name}",
            config=cfg
        )

    # Create agent (will use CUDA if available)
    agent = A2CAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=hidden_sizes,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        gamma=gamma,
        entropy_coef=entropy_coef
    )

    # Train
    checkpoint = train_a2c(
        env=env,
        agent=agent,
        episodes=episodes,
        action_values=action_values,
        use_wandb=use_wandb,
        save_dir=save_dir
    )

    if use_wandb:
        wandb.save(checkpoint)
        wandb.finish()

    env.close()

    print("\n🎉 Training Complete!\n")
