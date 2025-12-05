import os
import numpy as np
import torch
import gymnasium as gym
from datetime import datetime

# Import A2C & SAC
from models.A2C import A2CAgent
from models.SAC import SACAgent
from utils.continuous_cartpole import ContinuousCartPole, ContinuousMountainCar, ContinuousAcrobot
from config import get_config, get_sac_config

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
# RUN EPISODE WITH SAC (Continuous)
# ============================================================
def run_episode_sac(env, agent):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        with torch.no_grad():
            action = agent.select_action(obs, evaluate=True)

        obs, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        total_reward += reward
        steps += 1

    return steps, total_reward


# ============================================================
# MAIN RECORD FUNCTION
# ============================================================
def record_video(algorithm, env_name, is_discrete, episodes):
    # Load environment
    base_env = gym.make(env_name, render_mode="rgb_array")
    
    # Apply continuous wrappers for SAC
    if algorithm == "SAC":
        if env_name == "CartPole-v1":
            base_env = ContinuousCartPole(base_env)
        elif env_name == "MountainCar-v0":
            base_env = ContinuousMountainCar(base_env)
        # Acrobot uses discrete actions, no wrapper needed
    
    video_folder = f"videos/{algorithm}_{env_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(video_folder, exist_ok=True)

    env = gym.wrappers.RecordVideo(base_env, video_folder, episode_trigger=lambda ep: True)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)

    print(f"\n🎥 Videos will be saved to: {video_folder}")

    # Get dims
    obs_dim = env.observation_space.shape[0]
    
    if algorithm == "A2C":
        cfg = get_config(env_name)
        act_dim = cfg["action_bins"] if not is_discrete else env.action_space.n
    else:  # SAC
        cfg = get_sac_config(env_name)
        act_dim = env.action_space.shape[0]

    # ============================================================
    # LOAD AGENT
    # ============================================================
    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Recording on device: {device}")
    
    if algorithm == "A2C":
        model_path = f"trained_models/A2C/a2c_{env_name}.pth"

        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return

        print(f"📦 Loading A2C model: {model_path}")

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        agent = A2CAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=cfg.get("hidden_sizes", (128, 128))
        )

        agent.actor.load_state_dict(checkpoint["actor"])
        agent.critic.load_state_dict(checkpoint["critic"])

        run_episode_fn = run_episode_a2c

    else:  # SAC
        model_path = f"trained_models/SAC/sac_{env_name}.pth"

        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return

        print(f"📦 Loading SAC model: {model_path}")

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        agent = SACAgent(
            state_dim=obs_dim,
            action_dim=act_dim,
            hidden_dim=checkpoint["hidden_dim"],
            gamma=checkpoint["gamma"],
            actor_lr=checkpoint["actor_lr"],
            critic_lr=checkpoint["critic_lr"],
            tau=checkpoint["tau"],
            alpha=checkpoint["alpha"],
            batch_size=checkpoint.get("batch_size", 256),
            buffer_size=checkpoint.get("buffer_size", 1000000),
            action_low=checkpoint.get("action_low", -1.0),
            action_high=checkpoint.get("action_high", 1.0)
        )

        agent.actor.load_state_dict(checkpoint["actor"])
        agent.critic1.load_state_dict(checkpoint["critic1"])
        agent.critic2.load_state_dict(checkpoint["critic2"])

        run_episode_fn = run_episode_sac

    # ============================================================
    # RECORD EPISODES
    # ============================================================
    print(f"\n🎬 Recording {episodes} episodes...\n")

    for ep in range(episodes):
        if algorithm == "A2C":
            steps, reward = run_episode_fn(env, agent, is_discrete)
        else:  # SAC
            steps, reward = run_episode_fn(env, agent)
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
