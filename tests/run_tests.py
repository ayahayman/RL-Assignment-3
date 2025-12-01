# tests/run_tests.py

import os
import csv
import argparse
import importlib
import time
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym

from utils.env_utils import make_env
from utils.env_utils import safe_close_env


def default_action_fn_from_agent(agent, obs, env):
    """
    Generic deterministic action for evaluation if agent lacks `act_deterministic`.
    - If agent has act_deterministic(obs) use it.
    - Else if discrete: use actor logits -> argmax
    - Else continuous: use actor mu (mean) or require agent.act(obs)
    """
    # priority: act_deterministic, act, actor network inference
    if hasattr(agent, "act_deterministic"):
        return agent.act_deterministic(obs)
    if hasattr(agent, "act"):
        # many agents implement act(obs, deterministic=True)
        try:
            return agent.act(obs, deterministic=True)
        except TypeError:
            # fallback to non-deterministic
            return agent.act(obs)
    # fallback: attempt to use actor/net directly (discrete argmax or continuous mu)
    if hasattr(agent, "actor"):
        # assume actor is a torch module
        import torch
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = agent.actor(obs_t.to(next(agent.actor.parameters()).device))
        # discrete actor might return logits
        if isinstance(out, tuple):
            # continuous: (mu, std)
            mu, _ = out
            action = mu.cpu().numpy().squeeze(0)
            # clip to env bounds if possible
            if hasattr(env.action_space, "low"):
                action = np.clip(action, env.action_space.low, env.action_space.high)
            return action
        else:
            # treat out as logits -> argmax
            logits = out
            action = int(np.argmax(logits.cpu().numpy(), axis=-1).squeeze(0))
            return action
    raise RuntimeError("Agent does not supply a deterministic evaluation method.")


def load_agent_from_checkpoint(agent_type: str, env, checkpoint_prefix: str, device: str = "cpu"):
    """
    Load an agent object from a checkpoint.
    This function includes expected loading behaviors for the three agents we implemented:
    - A2C: expects a saved state dict with keys 'actor','critic' saved by save(path)
    - PPO: expects two files: prefix_actor.pt & prefix_critic.pt
    - SAC: expects prefix_actor.pt, prefix_q1.pt, prefix_q2.pt
    You may need to adjust file names if your save scheme differs.
    """
    # Import agents module (assumes package structure root has 'agents' package)
    import torch

    if agent_type.upper() == "A2C":
        from agents.a2c import A2CAgent
        # For A2C we need the model architectures to instantiate the agent before loading
        # You must instantiate actor/critic networks matching the architecture used at training
        # For convenience, we assume the actor/critic classes were simple MLPs of the same shape.
        # Adjust sizes to match your trained networks.
        from torch import nn

        obs_dim = env.observation_space.shape[0]
        # detection for discrete vs continuous action spaces
        if isinstance(env.action_space, gym.spaces.Discrete):
            act_dim = env.action_space.n
            # discrete actor -> return probabilities via a small MLP
            actor_net = nn.Sequential(
                nn.Linear(obs_dim, 128),
                nn.ReLU(),
                nn.Linear(128, act_dim),
                nn.Softmax(dim=-1)
            )
        else:
            act_dim = env.action_space.shape[0]
            actor_net = nn.Sequential(
                nn.Linear(obs_dim, 128),
                nn.ReLU(),
                nn.Linear(128, act_dim),
            )
        # critic
        critic_net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        agent = A2CAgent(actor_net, critic_net, device=device)
        # load saved checkpoint
        checkpoint_path = f"{checkpoint_prefix}.pth"
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"A2C checkpoint not found: {checkpoint_path}")
        agent.load(checkpoint_path)
        return agent

    elif agent_type.upper() == "PPO":
        from agents.ppo import PPOAgent
        agent = PPOAgent(env.observation_space, env.action_space, device=device)
        # PPO save convention: prefix_actor.pt, prefix_critic.pt
        actor_path = f"{checkpoint_prefix}_actor.pt"
        critic_path = f"{checkpoint_prefix}_critic.pt"
        if not os.path.exists(actor_path) or not os.path.exists(critic_path):
            raise FileNotFoundError(f"PPO checkpoint files not found at {actor_path} / {critic_path}")
        agent.load(checkpoint_prefix)
        return agent

    elif agent_type.upper() == "SAC":
        from agents.sac import SACAgent
        agent = SACAgent(env.observation_space, env.action_space, device=device)
        actor_path = f"{checkpoint_prefix}_actor.pt"
        q1_path = f"{checkpoint_prefix}_q1.pt"
        q2_path = f"{checkpoint_prefix}_q2.pt"
        if not (os.path.exists(actor_path) and os.path.exists(q1_path) and os.path.exists(q2_path)):
            raise FileNotFoundError(f"SAC checkpoint files not found at {actor_path} / {q1_path} / {q2_path}")
        agent.load(checkpoint_prefix)
        return agent
    else:
        raise ValueError("Unsupported agent type. Choose A2C / PPO / SAC.")


def evaluate_agent(
    agent,
    env_id: str,
    n_episodes: int = 100,
    max_episode_steps: int | None = None,
    checkpoint_prefix: str | None = None,
    device: str = "cpu",
    save_csv: str | None = "results.csv",
    verbose: bool = True,
):
    """
    Evaluate `agent` (already loaded) on env_id for n_episodes and record:
    - episode durations (steps)
    - total rewards per episode

    Returns a dict with arrays and also saves CSV + plots.
    """
    env = make_env(env_id, render_mode=None)
    durations = []
    rewards = []

    # build deterministic action function wrapper
    def act_fn(obs):
        # try agent.act_deterministic, then agent.act(obs, deterministic=True), then fallback
        if hasattr(agent, "act_deterministic"):
            return agent.act_deterministic(obs)
        if hasattr(agent, "act"):
            try:
                return agent.act(obs, deterministic=True)
            except TypeError:
                return agent.act(obs)
        # fallback generic
        return default_action_fn_from_agent(agent, obs, env)

    for i in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0
        while True:
            action = act_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            ep_reward += reward
            steps += 1
            if max_episode_steps is not None and steps >= max_episode_steps:
                break
            if done:
                break

        durations.append(steps)
        rewards.append(ep_reward)
        if verbose and (i % 10 == 0):
            print(f"Episode {i:03d} / {n_episodes} — reward={ep_reward:.2f} steps={steps}")

    safe_close_env(env)

    durations = np.array(durations)
    rewards = np.array(rewards)

    # Save CSV
    if save_csv is not None:
        csv_path = save_csv
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "steps", "reward"])
            for i, (s, r) in enumerate(zip(durations.tolist(), rewards.tolist())):
                writer.writerow([i, s, r])
        if verbose:
            print(f"Saved results CSV to {csv_path}")

    # Plots: durations histogram + rewards line
    base = os.path.splitext(save_csv)[0] if save_csv is not None else f"eval_{env_id}"
    hist_path = f"{base}_durations_hist.png"
    plt.figure(figsize=(8, 4))
    plt.hist(durations, bins=20)
    plt.title(f"{env_id} episode durations (n={n_episodes})")
    plt.xlabel("episode steps")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(hist_path)
    if verbose:
        print(f"Saved durations histogram to {hist_path}")
    plt.close()

    reward_path = f"{base}_rewards.png"
    plt.figure(figsize=(10, 4))
    plt.plot(rewards, marker=".", linestyle="-")
    plt.title(f"{env_id} episode rewards (n={n_episodes})")
    plt.xlabel("episode")
    plt.ylabel("total reward")
    plt.tight_layout()
    plt.savefig(reward_path)
    if verbose:
        print(f"Saved rewards plot to {reward_path}")
    plt.close()

    stats = {
        "durations": durations,
        "rewards": rewards,
        "mean_duration": float(durations.mean()),
        "std_duration": float(durations.std()),
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
        "csv": csv_path,
        "hist": hist_path,
        "reward_plot": reward_path,
    }
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_type", type=str, required=True, help="A2C / PPO / SAC")
    parser.add_argument("--env", type=str, required=True, help="Gym env id")
    parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint prefix (no suffix)")
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=str, default="results.csv", help="CSV path to save results")
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()

    # Prepare env to create a dummy env for architecture detection when loading agent
    dummy_env = make_env(args.env)
    agent = load_agent_from_checkpoint(args.agent_type, dummy_env, args.checkpoint, device=args.device)
    # close dummy env
    safe_close_env(dummy_env)

    t0 = time.time()
    stats = evaluate_agent(
        agent,
        args.env,
        n_episodes=args.n_episodes,
        max_episode_steps=args.max_steps,
        device=args.device,
        save_csv=args.out,
        verbose=True,
    )
    dt = time.time() - t0
    print("Evaluation finished in {:.2f}s".format(dt))
    print("Mean reward:", stats["mean_reward"], "Std reward:", stats["std_reward"])
    print("Mean duration:", stats["mean_duration"], "Std duration:", stats["std_duration"])


if __name__ == "__main__":
    main()
