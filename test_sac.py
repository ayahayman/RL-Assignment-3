import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt

from models.SAC import SACAgent   # your SAC implementation


# ============================================================
# ENV MENU
# ============================================================
ENV_MENU = {
    1: ("CartPole-v1", True, 2),
    2: ("MountainCar-v0", True, 3),
    3: ("Acrobot-v1", True, 3),
    4: ("Pendulum-v1", False, 9),   # continuous but discretized
}

# Discretized Pendulum torques
PENDULUM_TORQUES = np.linspace(-2.0, 2.0, 9).astype(np.float32)


# ============================================================
# EVALUATE FUNCTION
# ============================================================
def evaluate(env_name, discrete, act_dim, episodes):

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]

    # -------------------------------
    # Load SAC model files
    # -------------------------------
    print("\n📦 Loading SAC model...")

    base = f"trained_models/SAC/sac_{env_name}"
    actor_path = base + "_actor.pth"
    critic1_path = base + "_critic1.pth"
    critic2_path = base + "_critic2.pth"

    try:
        agent = SACAgent(
            state_dim=obs_dim,
            action_dim=act_dim,
            hidden_dim=128,
            gamma=0.99,
            actor_lr=0.0003,
            critic1_lr=0.0003,
            critic2_lr=0.0003,
            entropy_coef=0.01,
        )

        agent.load_models(actor_path, critic1_path, critic2_path)
        print("✅ SAC model loaded successfully.\n")

    except FileNotFoundError:
        print("❌ ERROR: SAC model files not found!")
        print("Expected:")
        print(" ", actor_path)
        print(" ", critic1_path)
        print(" ", critic2_path)
        return

    # -------------------------------
    # RUN TEST EPISODES
    # -------------------------------

    print(f"Testing SAC on {env_name} for {episodes} episodes...\n")

    durations = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        steps = 0

        while not done:

            action_idx = agent.select_action(obs)

            # map discrete index → real action
            if discrete:
                action = action_idx
            else:
                action = np.array([PENDULUM_TORQUES[action_idx]], dtype=np.float32)

            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

        durations.append(steps)
        print(f"Episode {ep+1}/{episodes}: {steps} steps")

    env.close()

    durations = np.array(durations)

    # -------------------------------
    # PRINT RESULT SUMMARY
    # -------------------------------
    print("\n========== TEST SUMMARY ============")
    print(f"Mean duration:  {durations.mean():.2f}")
    print(f"Std deviation: {durations.std():.2f}")
    print(f"Min duration:  {durations.min()}")
    print(f"Max duration:  {durations.max()}")
    print("====================================\n")

    # -------------------------------
    # SAVE HISTOGRAM
    # -------------------------------
    filename = f"{env_name}_sac_test_hist.png"
    plt.figure(figsize=(8, 5))
    plt.hist(durations, bins=20, edgecolor="black")
    plt.title(f"SAC Test Episode Durations ({env_name})")
    plt.xlabel("Episode Length")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)

    print(f"📊 Saved histogram: {filename}")


# ============================================================
# INTERACTIVE MODE
# ============================================================
if __name__ == "__main__":

    print("\n=== Choose environment to test (SAC) ===")
    print("1 → CartPole-v1")
    print("2 → MountainCar-v0")
    print("3 → Acrobot-v1")
    print("4 → Pendulum-v1 (discrete torques)")
    print("=======================================\n")

    choice = int(input("Enter choice (1–4): ").strip())

    if choice not in ENV_MENU:
        print("❌ Invalid option.")
        exit()

    env_name, discrete, act_dim = ENV_MENU[choice]

    episodes = int(input("\nHow many episodes to test? (default 100): ") or "100")

    evaluate(
        env_name=env_name,
        discrete=discrete,
        act_dim=act_dim,
        episodes=episodes
    )
