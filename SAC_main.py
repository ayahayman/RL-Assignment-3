# sac_discrete_main.py (Interactive version, same style as a2c_main.py)

import os
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import wandb

from models.SAC import SACDiscreteAgent
from config import get_config


# ============================================================
# ENVIRONMENTS MENU (DISCRETE ONLY)
# ============================================================

ENV_MENU = {
    1: ("CartPole-v1", True),
    2: ("MountainCar-v0", True),
    3: ("Acrobot-v1", True),
}

print("\n⚡ SAC-Discrete Trainer")
print("Supports only discrete action spaces.\n")


# ============================================================
# MAKE ENV
# ============================================================

def make_env(env_name: str):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    return env, obs_dim, act_dim


# ============================================================
# TRAINING LOOP
# ============================================================

def train_sac(env, agent, episodes, batch_size, use_wandb, save_dir):

    os.makedirs(save_dir, exist_ok=True)
    env_name = env.spec.id

    print(f"\n🚀 Training SAC-Discrete on {env_name} for {episodes} episodes...\n")

    for ep in range(episodes):

        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = agent.select_action(obs)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated   # <-- FIX HERE

            agent.store_transition((obs, action, reward, next_obs, done))

            obs = next_obs
            total_reward += reward
            steps += 1

            losses = agent.update(batch_size)


        if losses is None:
            actor_loss = critic1_loss = critic2_loss = 0.0
        else:
            actor_loss, critic1_loss, critic2_loss = losses

        # wandb logging
        if use_wandb:
            wandb.log({
                "episode": ep,
                "reward": total_reward,
                "steps": steps,
                "actor_loss": actor_loss,
                "critic1_loss": critic1_loss,
                "critic2_loss": critic2_loss
            })

        print(
            f"Episode {ep+1}/{episodes} | Reward={total_reward:.2f} "
            f"| Steps={steps} | A_Loss={actor_loss:.3f} "
            f"| C1_Loss={critic1_loss:.3f} | C2_Loss={critic2_loss:.3f}"
        )

    # save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"sac_discrete_{env_name}_{timestamp}")
    os.makedirs(save_path, exist_ok=True)
    agent.save(save_path)

    print(f"\n💾 SAC-Discrete model saved to folder: {save_path}\n")

    return save_path


# ============================================================
# MAIN MENU (INTERACTIVE)
# ============================================================

if __name__ == "__main__":

    print("\n======= SAC-Discrete TRAINER =======")
    print("Select environment to train:")
    print("1 → CartPole-v1")
    print("2 → MountainCar-v0")
    print("3 → Acrobot-v1")
    print("====================================\n")

    choice = int(input("Enter choice (1–3): ").strip())

    if choice not in ENV_MENU:
        print("❌ Invalid choice.")
        exit()

    env_name, _ = ENV_MENU[choice]

    # load config
    cfg = get_config(env_name)

    actor_lr     = cfg["actor_lr"]
    critic_lr    = cfg["critic_lr"]
    gamma        = cfg["gamma"]
    hidden_sizes = cfg["hidden_sizes"]
    episodes     = cfg["episodes"]
    seed         = cfg["seed"]
    save_dir     = cfg["save_dir"].replace("A2C", "SAC_Discrete")
    use_wandb    = cfg["wandb"]

    print(f"\n📌 Loaded hyperparameters:\n{cfg}\n")

    # custom episodes?
    episodes = int(input(f"Number of episodes (default={episodes}): ") or episodes)

    # wandb enable?
    use_wandb = (
        input("Enable Weights & Biases logging? (y/n): ").lower().startswith("y")
        if use_wandb else False
    )

    # seed
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # create environment
    env, obs_dim, act_dim = make_env(env_name)

    # wandb init
    if use_wandb:
        wandb.init(
            project="RL-SAC-Discrete",
            name=f"SAC_Discrete_{env_name}",
            config=cfg
        )

    # create agent
    agent = SACDiscreteAgent(
        state_dim=obs_dim,
        action_dim=act_dim,
        hidden_sizes=hidden_sizes,
        gamma=gamma,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        alpha_lr=3e-4,
        automatic_entropy_tuning=True,
        device="cpu"
    )

    # train
    save_path = train_sac(
        env=env,
        agent=agent,
        episodes=episodes,
        batch_size=256,
        use_wandb=use_wandb,
        save_dir=save_dir
    )

    if use_wandb:
        wandb.finish()

    env.close()
    print("\n🎉 SAC-Discrete Training Complete!\n")
