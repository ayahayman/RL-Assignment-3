import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.SAC import SACAgent
from utils.continuous_cartpole import ContinuousCartPole, ContinuousMountainCar, ContinuousAcrobot
import os

# ============================================================
# ENV MENU
# ============================================================
ENV_MENU = {
    1: "CartPole-v1",
    2: "MountainCar-v0",
    3: "Acrobot-v1",
    4: "Pendulum-v1"
}

# ============================================================
# EVALUATE FUNCTION
# ============================================================
def evaluate(model_path, env_name, episodes):
    # Create environment with wrapper
    base_env = gym.make(env_name)
    
    if env_name == "CartPole-v1":
        env = ContinuousCartPole(base_env)
    elif env_name == "MountainCar-v0":
        env = ContinuousMountainCar(base_env)
    elif env_name == "Acrobot-v1":
        env = base_env  # Acrobot uses discrete actions
    elif env_name == "Pendulum-v1":
        env = base_env
    else:
        env = base_env

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")
    
    # Load checkpoint
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

    print(f"\nLoaded model: {model_path}")
    print(f"Environment: {env_name}")
    print(f"State dim: {obs_dim}   |   Action dim: {act_dim}\n")

    rewards = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(obs, evaluate=True)
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

    # Save rewards graph in graphs/SAC folder
    os.makedirs("graphs/SAC", exist_ok=True)
    graph_filename = os.path.join("graphs", "SAC", f"{env_name}_sac_test_rewards.png")
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, episodes + 1), rewards, marker='o', linestyle='-', color='b')
    plt.title(f"SAC Test Rewards ({env_name})")
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
    print("4 → Pendulum-v1")
    print("=================================\n")

    choice = int(input("Enter choice (1–4): ").strip())

    if choice not in ENV_MENU:
        print("❌ Invalid choice.")
        exit()

    env_name = ENV_MENU[choice]

    # Automatically construct the model path
    model_path = os.path.join("trained_models", "SAC", f"sac_{env_name}.pth")

    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        exit()

    episodes = int(input("How many episodes to test? (default 100): ") or "100")

    evaluate(
        model_path=model_path,
        env_name=env_name,
        episodes=episodes
    )