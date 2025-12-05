import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from models.A2C import A2CAgent

# ============================================================
# ENV MENU
# ============================================================
ENV_MENU = {
    1: ("CartPole-v1", True, 2),
    2: ("MountainCar-v0", True, 3),
    3: ("Acrobot-v1", True, 3),
    4: ("Pendulum-v1", False, 9)  # continuous but discretized
}

# Discretized torque values for Pendulum
PENDULUM_TORQUES = np.linspace(-2.0, 2.0, 9).astype(np.float32)


# ============================================================
# EVALUATE FUNCTION
# ============================================================
def evaluate(model_path, env_name, act_dim, discrete, episodes):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")
    
    # Load checkpoint with weights_only=False
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    agent = A2CAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=(128, 128)
    )

    agent.actor.load_state_dict(checkpoint["actor"])
    agent.critic.load_state_dict(checkpoint["critic"])

    print(f"\nLoaded model: {model_path}")
    print(f"Environment: {env_name}")
    print(f"Discrete: {discrete}   |   Action dim: {act_dim}\n")

    rewards = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = agent.actor(obs_t)
            probs = torch.softmax(logits, dim=-1)

            action_idx = torch.argmax(probs).item()

            if discrete:
                action = action_idx
            else:
                action = np.array([PENDULUM_TORQUES[action_idx]], dtype=np.float32)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            obs = next_obs
            total_reward += reward

        rewards.append(total_reward)
        print(f"Episode {ep+1}/{episodes}: Reward={total_reward:.2f}")

    env.close()

    rewards = np.array(rewards)

    print("\n========== TEST SUMMARY ============")
    print(f"Mean reward:  {rewards.mean():.2f}")
    print(f"Std deviation: {rewards.std():.2f}")
    print(f"Min reward:  {rewards.min()}")
    print(f"Max reward:  {rewards.max()}")
    print("====================================\n")

    # Save rewards graph in graphs/A2C folder
    os.makedirs("graphs/A2C", exist_ok=True)
    graph_filename = os.path.join("graphs", "A2C", f"{env_name}_a2c_test_rewards.png")
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, episodes + 1), rewards, marker='o', linestyle='-', color='b')
    plt.title(f"A2C Test Rewards ({env_name})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    plt.savefig(graph_filename)

    print(f"Saved rewards graph: {graph_filename}")


# ============================================================
# INTERACTIVE MODE (NO ARGUMENTS)
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

    # Automatically construct the model path
    model_path = os.path.join("trained_models", "A2C", f"a2c_{env_name}.pth")

    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        exit()

    episodes = int(input("How many episodes to test? (default 100): ") or "100")

    evaluate(
        model_path=model_path,
        env_name=env_name,
        act_dim=act_dim,
        discrete=discrete,
        episodes=episodes
    )
