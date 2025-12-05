import os
import numpy as np
import torch
import gymnasium as gym
import wandb
from utils.continuous_cartpole import ContinuousCartPole, ContinuousMountainCar, ContinuousAcrobot
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
    """
    Create environment with appropriate wrapper to make it continuous.
    All environments now use continuous action spaces for SAC.
    """
    base_env = gym.make(env_name)
    
    if env_name == "CartPole-v1":
        # Wrap CartPole to make it continuous
        env = ContinuousCartPole(base_env)
    elif env_name == "MountainCar-v0":
        # Wrap MountainCar to make it continuous
        env = ContinuousMountainCar(base_env)
    elif env_name == "Acrobot-v1":
        # Acrobot already works well with discrete actions for SAC
        env = base_env
    elif env_name == "Pendulum-v1":
        # Pendulum is ALREADY continuous - SAC's native environment!
        # No wrapper needed
        env = base_env
    else:
        env = base_env

    state_dim = env.observation_space.shape[0]
    
    # Now all environments have continuous action spaces
    if isinstance(env.action_space, gym.spaces.Box):
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

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
            # Update every step if we have enough samples
            if len(agent.memory) >= agent.batch_size:
                actor_l, c1_l, c2_l = agent.update()

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
                "reward": total_reward,
                "steps": steps,
                "actor_loss": episode_actor_loss,
                "critic1_loss": episode_critic1_loss,
                "critic2_loss": episode_critic2_loss,
            }, step=ep)

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
        "target_critic1": agent.target_critic1.state_dict(),
        "target_critic2": agent.target_critic2.state_dict(),
        "env": env_name,
        "state_dim": agent.state_dim,
        "action_dim": agent.action_dim,
        "hidden_dim": agent.hidden_dim,
        "gamma": agent.gamma,
        "tau": agent.tau,
        "actor_lr": agent.actor_lr,
        "critic_lr": agent.critic_lr,
        "alpha": agent.alpha,
        "batch_size": agent.batch_size,
        "buffer_size": len(agent.memory),
        "action_low": agent.action_low.cpu().item(),
        "action_high": agent.action_high.cpu().item()
    }, save_path)

    print(f"\n💾 Model saved to {save_path}")
    return save_path


# ============================================================
# MAIN MENU
# ============================================================

if __name__ == "__main__":
    print("\n======= SAC TRAINER (CONTINUOUS) =======")
    print("Select environment to train:")
    print("1 → CartPole-v1 (Continuous Wrapper)")
    print("2 → MountainCar-v0 (Continuous Wrapper)")
    print("3 → Acrobot-v1 (Continuous Wrapper)")
    print("4 → Pendulum-v1 (Native Continuous)")
    print("========================================\n")

    choice = int(input("Enter choice (1–4): ").strip())

    if choice not in ENV_MENU:
        print("❌ Invalid choice.")
        exit()

    env_name = ENV_MENU[choice]
    print(f"✅ Selected environment: {env_name}")
    
    # Load full config for this environment
    cfg = get_sac_config(env_name)

    # Unpack settings
    gamma = cfg["Gamma"]
    actor_lr = cfg["Actor LR"]
    critic_lr = cfg["Critic LR"]
    tau = cfg.get("Tau", 0.005)
    alpha = cfg.get("Alpha", 0.2)
    hidden_dim = cfg["Hidden Dim"]
    episodes = cfg["Training Episodes"]
    batch_size = cfg.get("Batch Size", 256)
    buffer_size = cfg.get("Buffer Size", 1000000)
    action_bins = cfg.get("Action Bins", 9)
    save_dir = "trained_models/SAC"

    # W&B logging
    use_wandb = cfg.get("wandb", True)
    if use_wandb:
        wandb.init(
            project="RL-Assignment3",
            name=f"SAC_{env_name}_Continuous",
            config=cfg
        )

    print(f"\n📌 Loaded hyperparameters from config.py:\n{cfg}\n")

    # Create environment with continuous wrapper
    env, state_dim, action_dim = make_env_and_actions(env_name, action_bins)
    
    print(f"📊 State dim: {state_dim}, Action dim: {action_dim}")
    print(f"📊 Action space: {env.action_space}\n")

    # Get action bounds from environment
    if isinstance(env.action_space, gym.spaces.Box):
        action_low = float(env.action_space.low[0])
        action_high = float(env.action_space.high[0])
    else:
        action_low = -1.0
        action_high = 1.0
    
    print(f"📊 Action bounds: [{action_low}, {action_high}]\n")

    # Create agent
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        gamma=gamma,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        tau=tau,
        alpha=alpha,
        batch_size=batch_size,
        buffer_size=buffer_size,
        action_low=action_low,
        action_high=action_high
    )

    # Train
    train_sac(env=env, agent=agent, episodes=episodes, save_dir=save_dir, use_wandb=use_wandb)

    if use_wandb:
        wandb.finish()

    env.close()

    print("\n🎉 Training Complete!\n")