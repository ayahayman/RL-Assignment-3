import os
from datetime import datetime
import gymnasium as gym
import torch
import numpy as np
from models.A2C import A2CAgent

# ===================================
# AVAILABLE ENVIRONMENTS
# ===================================

ENV_MENU = {
    1: {
        "key": "cartpole",
        "env_name": "CartPole-v1",
        "state_dim": 4,
        "action_dim": 2,
        "discrete": True
    },
    2: {
        "key": "mountaincar",
        "env_name": "MountainCar-v0",
        "state_dim": 2,
        "action_dim": 3,
        "discrete": True
    },
    3: {
        "key": "acrobot",
        "env_name": "Acrobot-v1",
        "state_dim": 6,
        "action_dim": 3,
        "discrete": True
    },
    4: {
        "key": "pendulum",
        "env_name": "Pendulum-v1",
        "state_dim": 3,
        "action_dim": 9,       # discretized torque bins
        "discrete": False
    }
}

# Torques for discretized Pendulum
PENDULUM_TORQUES = np.linspace(-2.0, 2.0, 9)


# ===================================
# RECORD VIDEO FUNCTION
# ===================================

def record_video(env_config, num_episodes=3, max_steps=1000):

    env_name = env_config["env_name"]
    state_dim = env_config["state_dim"]
    action_dim = env_config["action_dim"]
    discrete = env_config["discrete"]

    # ------------------------------------------------------
    # AUTO LOAD MODEL PATH
    # ------------------------------------------------------
    model_path = f"trained_models/A2C/a2c_{env_name}.pth"

    if not os.path.exists(model_path):
        print(f"\n❌ ERROR: Model not found: {model_path}")
        return

    print(f"\n📦 Loading model: {model_path}")

    # ------------------------------------------------------
    # Environment setup
    # ------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_folder = f"videos/A2C_{env_name}_{timestamp}"
    os.makedirs(video_folder, exist_ok=True)

    print(f"🎥 Saving videos in: {video_folder}")

    env = gym.make(env_name, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, video_folder,
                                   episode_trigger=lambda ep: True)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)

    # ------------------------------------------------------
    # Load model
    # ------------------------------------------------------
    agent = A2CAgent(
        obs_dim=state_dim,
        act_dim=action_dim,
        hidden_sizes=(128, 128),
        device="cpu"
    )

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    agent.actor.load_state_dict(checkpoint["actor"])
    agent.critic.load_state_dict(checkpoint["critic"])

    print("✅ Model loaded successfully\n")

    # ------------------------------------------------------
    # RECORD EPISODES
    # ------------------------------------------------------
    print(f"🎬 Recording {num_episodes} episodes...\n")

    for ep in range(num_episodes):

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
                # Pendulum torque
                torque = PENDULUM_TORQUES[action_idx]
                action = np.array([torque], dtype=np.float32)

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_reward += reward
            steps += 1

        print(f"Episode {ep+1}: Steps={steps}, Reward={total_reward:.1f}")

    env.close()
    print(f"\n🎉 DONE! Videos saved to: {video_folder}\n")


# ===================================
# INTERACTIVE MENU (NO CLI ARGUMENTS)
# ===================================

if __name__ == "__main__":

    print("\n========= VIDEO RECORDER =========")
    print("Choose environment:")
    print("1 → CartPole-v1")
    print("2 → MountainCar-v0")
    print("3 → Acrobot-v1")
    print("4 → Pendulum-v1 (discrete)")
    print("=================================\n")

    choice = int(input("Enter choice (1–4): ").strip())

    if choice not in ENV_MENU:
        print("❌ Invalid choice.")
        exit()

    env_config = ENV_MENU[choice]

    episodes = int(input("\nNumber of episodes to record (default=3): ") or "3")

    record_video(env_config, num_episodes=episodes, max_steps=1000)
