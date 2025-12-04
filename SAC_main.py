import os
import numpy as np
import torch
import gymnasium as gym
import wandb  # Add W&B import
from utils.discrete_pendulum import DiscretePendulum
from models.SAC import SACAgent
from config import get_sac_config

# ============================================================
# ENVIRONMENTS MENU
# ============================================================

ENV_MENU = {
    1: "CartPole-v1",
    2: "MountainCar-v0",
    3: "Acrobot-v1",
    4: "Pendulum-v1"
}

# ============================================================
# MAKE ENV + ACTION SPACE WRAPPER
# ============================================================

def make_env_and_actions(env_name: str, action_bins: int):
    if env_name == "Pendulum-v1":
        env = DiscretePendulum(gym.make(env_name), num_actions=action_bins)
    else:
        env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) else env.action_space.shape[0]

    return env, state_dim, action_dim

# ============================================================
# TRAINING LOOP
# ============================================================
def train_sac(env, agent, episodes, save_dir, use_wandb):
    os.makedirs(save_dir, exist_ok=True)
    env_name = env.spec.id

    print(f"\n🚀 Training SAC on {env_name} for {episodes} episodes...\n")

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        # Reset episode-level tracking
        episode_actor_loss = 0.0
        episode_critic1_loss = 0.0
        episode_critic2_loss = 0.0
        update_count = 0

        while not done:
            # --- Select action ---
            action = agent.select_action(obs)

            # --- Environment step ---
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # --- Store transition ---
            agent.store_transition((obs, action, reward, next_obs, done))

            obs = next_obs
            total_reward += reward
            steps += 1

            # --- UPDATE LOGIC ---
            should_update = False

            if env_name == "MountainCar-v0":
                # For MountainCar, update ONLY once at episode end
                should_update = done
            else:
                # Normal n-step update
                if len(agent.memory) >= agent.n_steps or done:
                    should_update = True

            if should_update:
                actor_l, c1_l, c2_l = agent.update()
                agent.reset_memory()

                # Accumulate losses for reporting
                episode_actor_loss += actor_l
                episode_critic1_loss += c1_l
                episode_critic2_loss += c2_l
                update_count += 1

        # Average losses for logging
        if update_count > 0:
            episode_actor_loss /= update_count
            episode_critic1_loss /= update_count
            episode_critic2_loss /= update_count

        # Log metrics to W&B
        if use_wandb:
            wandb.log({
                "episode": ep,
                "reward": total_reward,
                "steps": steps,
                "actor_loss": episode_actor_loss,
                "critic1_loss": episode_critic1_loss,
                "critic2_loss": episode_critic2_loss,
            })

        print(
            f"Episode {ep+1}/{episodes} | Reward={total_reward:.2f} | Steps={steps} "
            f"| A_Loss={episode_actor_loss:.4f} | C1_Loss={episode_critic1_loss:.4f} "
            f"| C2_Loss={episode_critic2_loss:.4f}"
        )

    # ============================================================
    # Save ALL SAC components in one file
    # ============================================================

    save_path = os.path.join(save_dir, f"sac_{env_name}.pth")
    torch.save({
        "actor": agent.actor.state_dict(),
        "critic1": agent.critic1.state_dict(),
        "critic2": agent.critic2.state_dict(),
        "env": env_name,
        "state_dim": agent.state_dim,
        "action_dim": agent.action_dim,
        "hidden_dim": agent.hidden_dim,
        "gamma": agent.gamma,
        "actor_lr": agent.actor_lr,
        "critic1_lr": agent.critic1_lr,
        "critic2_lr": agent.critic2_lr,
        "entropy_coef": agent.entropy_coef,
        "n_steps": agent.n_steps
    }, save_path)

    print(f"\n💾 Model saved to {save_path}")
    return save_path


# ============================================================
# MAIN MENU (NO ARGUMENTS)
# ============================================================

if __name__ == "__main__":
    print("\n======= SAC TRAINER =======")
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

    env_name = ENV_MENU[choice]
    print("Selected environment:", env_name)
    # Load full config for this environment
    cfg = get_sac_config(env_name)

    # Unpack settings
    gamma = cfg["Gamma"]
    actor_lr = cfg["Actor LR"]
    critic1_lr = cfg["Critic 1 LR"]
    critic2_lr = cfg["Critic 2 LR"]
    entropy_coef = cfg["Entropy Coef"]
    hidden_dim = cfg["Hidden Dim"]
    episodes = cfg["Training Episodes"]
    action_bins = cfg.get("Action Bins", 9)  # Default to 9 for Pendulum
    save_dir = "trained_models/SAC"

    # W&B logging
    use_wandb = cfg.get("wandb", True)
    if use_wandb:
        wandb.init(
            project="RL-Assignment3",
            name=f"SAC_{env_name}",
            config=cfg
        )

    print(f"\n📌 Loaded hyperparameters from config.py:\n{cfg}\n")

    # Create environment
    env, state_dim, action_dim = make_env_and_actions(env_name, action_bins)

    # Create agent
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        gamma=gamma,
        actor_lr=actor_lr,
        critic1_lr=critic1_lr,
        critic2_lr=critic2_lr,
        entropy_coef=entropy_coef
    )

    # Train
    train_sac(env=env, agent=agent, episodes=episodes, save_dir=save_dir, use_wandb=use_wandb)

    if use_wandb:
        wandb.finish()

    env.close()

    print("\n🎉 Training Complete!\n")