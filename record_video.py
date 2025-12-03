import os
import sys
from datetime import datetime
import warnings
import gymnasium as gym
import torch
import numpy as np

from models.A2C import A2CAgent

warnings.filterwarnings("ignore", category=UserWarning)

# ===================================
# ENVIRONMENT CONFIGURATIONS
# ===================================

ENV_CONFIGS = {
    "cartpole": {
        "env_name": "CartPole-v1",
        "state_dim": 4,
        "action_dim": 2,
        "discrete": True
    },
    "acrobot": {
        "env_name": "Acrobot-v1",
        "state_dim": 6,
        "action_dim": 3,
        "discrete": True
    },
    "mountaincar": {
        "env_name": "MountainCar-v0",
        "state_dim": 2,
        "action_dim": 3,
        "discrete": True
    },
    "pendulum": {
        "env_name": "Pendulum-v1",
        "state_dim": 3,
        "action_dim": 9,     # DISCRETIZED TORQUES
        "discrete": False    # env itself uses continuous, but we discretize
    }
}

# Predefined torque values for discretized pendulum (MUST match training)
PENDULUM_TORQUES = np.linspace(-2.0, 2.0, 9)


# ===================================
# RECORD FUNCTION
# ===================================

def record_video(agent_type, environment_name, model_path, num_episodes=3, max_steps=1000):

    print("📌 DEBUG: record_video() called")
    print(f"  agent_type={agent_type}")
    print(f"  environment_name={environment_name}")
    print(f"  model_path={model_path}")
    print(f"  num_episodes={num_episodes}")
    print(f"  max_steps={max_steps}\n")

    if environment_name not in ENV_CONFIGS:
        print(f"❌ Environment '{environment_name}' not found in config.")
        print(f"Available: {list(ENV_CONFIGS.keys())}")
        return

    config = ENV_CONFIGS[environment_name]

    # ------------------------------------------------------
    # 1. Create output folder
    # ------------------------------------------------------
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    video_folder = f"videos/{agent_type}_{environment_name}_{timestamp}"
    os.makedirs(video_folder, exist_ok=True)

    print(f"🎥 Saving videos in: {video_folder}")

    # ------------------------------------------------------
    # 2. Setup environment
    # ------------------------------------------------------
    print("📌 DEBUG: Creating gym environment...")
    env = gym.make(config["env_name"], render_mode="rgb_array")

    # Wrap for video recording
    env = gym.wrappers.RecordVideo(env, video_folder, episode_trigger=lambda ep: True)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)

    # ------------------------------------------------------
    # 3. Load model
    # ------------------------------------------------------
    print("📌 DEBUG: Loading model...")

    agent = A2CAgent(
        obs_dim=config["state_dim"],
        act_dim=config["action_dim"],
        hidden_sizes=(128, 128),
        device="cpu"
    )

    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        agent.actor.load_state_dict(checkpoint["actor"])
        agent.critic.load_state_dict(checkpoint["critic"])
    except Exception as e:
        print(f"❌ Failed to load model file: {model_path}")
        print("Error:", e)
        env.close()
        return

    print("✅ Model loaded successfully\n")

    # ------------------------------------------------------
    # 4. Record episodes
    # ------------------------------------------------------
    print(f"🎬 Starting recording for {num_episodes} episodes...\n")

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

            # --------------------------------------------------
            # Convert discrete action → continuous torque (Pendulum)
            # --------------------------------------------------
            if environment_name == "pendulum":
                torque = PENDULUM_TORQUES[action_idx]
                action = np.array([torque], dtype=np.float32)
            else:
                action = action_idx  # normal discrete envs

            # Step environment
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_reward += reward
            steps += 1

        print(f"Episode {ep+1}: Steps={steps}, Reward={total_reward:.1f}")

    env.close()
    print(f"\n🎉 Recording complete! Videos saved at: {video_folder}")


# ===================================
# CLI MODE
# ===================================

if __name__ == "__main__":
    print("📌 DEBUG: Script started")
    print("sys.argv =", sys.argv)

    if len(sys.argv) >= 4:
        agent_type = sys.argv[1]
        environment_name = sys.argv[2]
        model_path = sys.argv[3]
        num_episodes = int(sys.argv[4]) if len(sys.argv) > 4 else 3
        max_steps = int(sys.argv[5]) if len(sys.argv) > 5 else 1000

        print("\n📌 DEBUG: Arguments parsed, calling record_video()...\n")
        record_video(agent_type, environment_name, model_path, num_episodes, max_steps)

    else:
        print("❌ Not enough arguments provided.")
        print("Usage:")
        print("python record_video.py <agent_type> <env> <model_path> [episodes] [max_steps]")
