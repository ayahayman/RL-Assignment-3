import argparse
import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.A2C import A2CAgent


def evaluate(model_path, env_name, episodes=100, render=False):
    # ----------------------------------------------
    # Load environment
    # ----------------------------------------------
    env = gym.make(env_name, render_mode="human" if render else None)

    obs_dim = env.observation_space.shape[0]

    # ----------------------------------------------
    # Load checkpoint
    # ----------------------------------------------
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    actor_state = checkpoint["actor"]
    critic_state = checkpoint["critic"]

    # If continuous env → read action bins from checkpoint
    if "act_dim" in checkpoint:
        act_dim = checkpoint["act_dim"]
    else:
        # fallback (old models)
        if hasattr(env.action_space, "n"):
            act_dim = env.action_space.n
        else:
            act_dim = 9  # default
            print("⚠ WARNING: act_dim not found in checkpoint. Using default=9 for Pendulum.")

    # ----------------------------------------------
    # Handle continuous vs discrete environments
    # ----------------------------------------------
    if hasattr(env.action_space, "n"):
        discrete = True
        action_values = None
    else:
        discrete = False
        low = float(env.action_space.low.min())
        high = float(env.action_space.high.max())
        action_values = np.linspace(low, high, act_dim).astype(np.float32)

    # ----------------------------------------------
    # Re-create agent architecture
    # ----------------------------------------------
    agent = A2CAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=(128, 128),
        device="cpu"
    )

    agent.actor.load_state_dict(actor_state)
    agent.critic.load_state_dict(critic_state)

    print(f"\nLoaded trained A2C model from {model_path}")
    print(f"Environment: {env_name}")
    print(f"Discrete: {discrete} | Action dim: {act_dim}\n")

    durations = []

    # ----------------------------------------------
    # Run test episodes
    # ----------------------------------------------
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        steps = 0

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = agent.actor(obs_t)
            probs = torch.softmax(logits, dim=-1)
            action_idx = torch.argmax(probs).item()

            # convert discrete → continuous for Pendulum
            if not discrete:
                action = np.array([action_values[action_idx]], dtype=np.float32)
            else:
                action = action_idx

            next_obs, _, terminated, truncated, _ = env.step(action)
            obs = next_obs
            done = terminated or truncated
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

    # Save histogram
    plt.figure(figsize=(8, 5))
    plt.hist(durations, bins=20, edgecolor='black')
    plt.title(f"A2C Test Episode Durations ({env_name})")
    plt.xlabel("Episode Length (steps)")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    filename = f"{env_name}_a2c_test_hist.png"
    plt.savefig(filename)
    print(f"Saved histogram: {filename}")

    return durations


# ----------------------------------------------
# CLI
# ----------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--render", action="store_true")

    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        env_name=args.env,
        episodes=args.episodes,
        render=args.render
    )
