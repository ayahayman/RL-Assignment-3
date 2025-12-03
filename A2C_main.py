# a2c_main.py
import argparse
import os
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import wandb

from models.A2C import A2CAgent  # expects the final A2C implementation (no value_buf)


def parse_args():
    p = argparse.ArgumentParser("A2C Trainer")
    p.add_argument("--env", type=str, required=True, help="Gym environment name (e.g. CartPole-v1)")
    p.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    p.add_argument("--entropy", type=float, default=0.001, help="Entropy coefficient")
    p.add_argument("--hidden-sizes", type=int, nargs="+", default=[128, 128], help="Hidden sizes for networks")
    p.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    p.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    p.add_argument("--action-bins", type=int, default=9, help="Number of discrete bins for continuous envs (Pendulum)")
    p.add_argument("--save-dir", type=str, default="checkpoints", help="Where to save model checkpoints")
    return p.parse_args()


def make_env_and_actions(env_name: str, action_bins: int):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]

    # determine action handling
    if hasattr(env.action_space, "n"):
        act_dim = env.action_space.n
        action_values = None
        discrete = True
    else:
        # continuous -> discretize to `action_bins` values in action space bounds
        # assume 1D Box (Pendulum)
        low = float(env.action_space.low.min())
        high = float(env.action_space.high.max())
        action_values = np.linspace(low, high, action_bins).astype(np.float32)
        act_dim = len(action_values)
        discrete = False

    return env, obs_dim, act_dim, action_values, discrete


def rollout_and_train(
    env,
    agent: A2CAgent,
    episodes: int,
    action_values,
    discrete: bool,
    use_wandb: bool,
    save_dir: str,
):
    os.makedirs(save_dir, exist_ok=True)
    print(f"Training A2C on {env.spec.id} for {episodes} episodes")
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0

        while not done:
            action_idx, logp = agent.select_action(obs)  # action_idx is int, logp is float
            # map discrete action index to real action for continuous envs
            if action_values is not None:
                real_action = np.array([action_values[action_idx]], dtype=np.float32)
                next_obs, reward, terminated, truncated, _ = env.step(real_action)
            else:
                next_obs, reward, terminated, truncated, _ = env.step(action_idx)

            done = terminated or truncated

            agent.store(obs, action_idx, reward, done, logp)
            obs = next_obs
            ep_reward += reward
            steps += 1

        # update after full episode
        actor_loss, critic_loss = agent.update()

        # logging
        if use_wandb:
            wandb.log({
                "episode": ep,
                "reward": ep_reward,
                "steps": steps,
                "actor_loss": actor_loss,
                "critic_loss": critic_loss
            })

        print(f"Episode {ep+1}/{episodes}  Reward={ep_reward:.2f}  Steps={steps}  A_Loss={actor_loss:.4f}  C_Loss={critic_loss:.4f}")

    # save checkpoint
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"a2c_{env.spec.id}_{timestamp}.pth")
    torch.save({
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "env": env.spec.id,
        "act_dim": agent.actor.model[-1].out_features if hasattr(agent.actor, "model") else None
    }, save_path)
    print(f"Model saved to {save_path}")
    return save_path


def main():
    args = parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    env, obs_dim, act_dim, action_values, discrete = make_env_and_actions(args.env, args.action_bins)

    # initialize wandb if enabled
    if not args.no_wandb:
        wandb.init(
            project="RL-Assignment3",
            name=f"A2C_{args.env}",
            config={
                "algorithm": "A2C",
                "environment": args.env,
                "episodes": args.episodes,
                "lr": args.lr,
                "gamma": args.gamma,
                "entropy_coef": args.entropy,
                "hidden_sizes": args.hidden_sizes,
                "discrete": discrete,
                "action_bins": args.action_bins if not discrete else None
            }
        )
        use_wandb = True
    else:
        use_wandb = False

    # Create agent
    agent = A2CAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=tuple(args.hidden_sizes),
        lr=args.lr,
        gamma=args.gamma,
        entropy_coef=args.entropy,
        device="cpu"
    )

    # Train
    checkpoint = rollout_and_train(env, agent, args.episodes, action_values, discrete, use_wandb, args.save_dir)

    if use_wandb:
        # upload checkpoint as artifact (optional)
        wandb.save(checkpoint)
        wandb.finish()

    env.close()


if __name__ == "__main__":
    main()
