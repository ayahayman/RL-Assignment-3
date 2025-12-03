import os
import numpy as np
import torch
import gymnasium as gym
from datetime import datetime

# Import A2C & SAC
from models.A2C import A2CAgent
from models.SAC import SACAgent
from config import get_config

# ===============================
# ENV MENU
# ===============================
ENV_MENU = {
    1: ("CartPole-v1", True),
    2: ("MountainCar-v0", True),
    3: ("Acrobot-v1", True),
    4: ("Pendulum-v1", False),
}

# Pendulum discrete torques (must match training)
PENDULUM_TORQUES = np.linspace(-2, 2, 9).astype(np.float32)


# ============================================================
# RUN EPISODE WITH A2C
# ============================================================
def run_episode_a2c(env, agent, discrete):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits = agent.actor(obs_t)
        probs = torch.softmax(logits, dim=-1)
        action_idx = torch.argmax(probs).item()

        if discrete:
            action = action_idx
        else:
            action = np.array([PENDULUM_TORQUES[action_idx]], dtype=np.float32)

        obs, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        total_reward += reward
        steps += 1

    return steps, total_reward


# ============================================================
# RUN EPISODE WITH SAC
# ============================================================
def run_episode_sac(env, agent, discrete):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_idx = agent.select_action(obs)

        if discrete:
            action = action_idx
        else:
            action = np.array([PENDULUM_TORQUES[action_idx]], dtype=np.float32)

        obs, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        total_reward += reward
        steps += 1

    return steps, total_reward


# ============================================================
# MAIN RECORD FUNCTION
# ============================================================
def record_video(algorithm, env_name, is_discrete, episodes):

    # Load config
    cfg = get_config(env_name)

    # Load environment
    env = gym.make(env_name, render_mode="rgb_array")
    video_folder = f"videos/{algorithm}_{env_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(video_folder, exist_ok=True)

    env = gym.wrappers.RecordVideo(env, video_folder, episode_trigger=lambda ep: True)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)

    print(f"\n🎥 Videos will be saved to: {video_folder}")

    # Get dims
    obs_dim = env.observation_space.shape[0]
    act_dim = cfg["action_bins"] if not is_discrete else env.action_space.n

    # ============================================================
    # LOAD AGENT
    # ============================================================
    if algorithm == "A2C":
        model_path = f"trained_models/A2C/a2c_{env_name}.pth"

        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return

        print(f"📦 Loading A2C model: {model_path}")

        agent = A2CAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=cfg["hidden_sizes"],
            actor_lr=cfg["actor_lr"],
            critic_lr=cfg["critic_lr"],
            gamma=cfg["gamma"],
            entropy_coef=cfg["entropy_coef"],
        )

        checkpoint = torch.load(model_path, map_location="cpu")
        agent.actor.load_state_dict(checkpoint["actor"])
        agent.critic.load_state_dict(checkpoint["critic"])

        run_episode_fn = run_episode_a2c

    else:  # SAC
        base = f"trained_models/SAC/sac_{env_name}"
        actor_path = base + "_actor.pth"
        critic1_path = base + "_critic1.pth"
        critic2_path = base + "_critic2.pth"

        if not (os.path.exists(actor_path) and os.path.exists(critic1_path) and os.path.exists(critic2_path)):
            print("❌ SAC model files not found!")
            return

        print(f"📦 Loading SAC model: {actor_path}")

        agent = SACAgent(
            state_dim=obs_dim,
            action_dim=act_dim,
            hidden_dim=128,
            gamma=cfg["gamma"],
            actor_lr=cfg["actor_lr"],
            critic1_lr=cfg["critic_lr"],
            critic2_lr=cfg["critic_lr"],
            entropy_coef=cfg["entropy_coef"],
        )

        agent.load_models(actor_path, critic1_path, critic2_path)

        run_episode_fn = run_episode_sac

    # ============================================================
    # RECORD EPISODES
    # ============================================================
    print(f"\n🎬 Recording {episodes} episodes...\n")

    for ep in range(episodes):
        steps, reward = run_episode_fn(env, agent, is_discrete)
        print(f"Episode {ep+1}: Steps={steps}, Reward={reward:.1f}")

    env.close()
    print(f"\n🎉 DONE! Videos saved to: {video_folder}\n")


# ============================================================
# INTERACTIVE MENU
# ============================================================
if __name__ == "__main__":

    print("\n=========== VIDEO RECORDER ===========")
    print("Choose Algorithm:")
    print("1 → A2C")
    print("2 → SAC")
    print("======================================\n")

    algo_choice = int(input("Enter choice (1–2): ").strip())
    algorithm = "A2C" if algo_choice == 1 else "SAC"

    print("\nChoose Environment:")
    for k, (env_name, _) in ENV_MENU.items():
        print(f"{k} → {env_name}")
    print("======================================\n")

    env_choice = int(input("Enter choice (1–4): ").strip())

    if env_choice not in ENV_MENU:
        print("❌ Invalid environment choice.")
        exit()

    env_name, is_discrete = ENV_MENU[env_choice]

    episodes = int(input("\nNumber of episodes to record (default=3): ") or "3")

    record_video(algorithm, env_name, is_discrete, episodes)
