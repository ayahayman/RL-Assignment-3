import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.A2C import A2CAgent
import os

# ============================================================
# ENV MENU
# ============================================================
ENV_MENU = {
    1: ("CartPole-v1", True, 2),
    2: ("MountainCar-v0", True, 3),
    3: ("Acrobot-v1", True, 3),
    4: ("Pendulum-v1", False, 9)      # continuous but discretized
}

# Discretized torque values for Pendulum
PENDULUM_TORQUES = np.linspace(-2.0, 2.0, 9).astype(np.float32)


# ============================================================
# EVALUATE FUNCTION
# ============================================================
def evaluate(env_name, act_dim, discrete, episodes):

    model_path = f"trained_models/A2C/a2c_{env_name}.pth"

    if not os.path.exists(model_path):
        print(f"\n❌ ERROR: Model not found: {model_path}")
        return

    print(f"\n📦 Loading model from: {model_path}")

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    agent = A2CAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=(128, 128),
        device="cpu"
    )

    agent.actor.load_state_dict(checkpoint["actor"])
    agent.critic.load_state_dict(checkpoint["critic"])

    print(f"Environment: {env_name}")
    print(f"Discrete: {discrete}   |   Action dim: {act_dim}\n")

    durations = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
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

            next_obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            obs = next_obs
            steps += 1

        durations.append(steps)
        print(f"Episode {ep+1}/{episodes}: {steps} steps")

    env.close()

    durations = np.array(durations)

    print("\n========== TEST SUMMARY ============")
    print(f"Mean duration:  {durations.mean():.2f}")
    print(f"Std deviation: {durations.std():.2f}")
    print(f"Min duration:  {durations.min()}")
    print(f"Max duration:  {durations.max()}")
    print("====================================\n")

    filename = f"{env_name}_a2c_test_hist.png"
    plt.figure(figsize=(8, 5))
    plt.hist(durations, bins=20, edgecolor='black')
    plt.title(f"A2C Test Durations ({env_name})")
    plt.xlabel("Episode Length")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)

    print(f"Saved histogram: {filename}")


# ============================================================
# INTERACTIVE MENU (NO ARGUMENTS)
# ============================================================
if __name__ == "__main__":

    print("\n=== Choose environment to test ===")
    print("1 → CartPole-v1")
    print("2 → MountainCar-v0")
    print("3 → Acrobot-v1")
    print("4 → Pendulum-v1 (discrete)")
    print("=================================\n")

    choice = int(input("Enter choice (1–4): ").strip())

    if choice not in ENV_MENU:
        print("❌ Invalid choice.")
        exit()

    env_name, discrete, act_dim = ENV_MENU[choice]

    episodes = int(input("How many episodes to test? (default 100): ") or "100")

    evaluate(
        env_name=env_name,
        act_dim=act_dim,
        discrete=discrete,
        episodes=episodes
    )
