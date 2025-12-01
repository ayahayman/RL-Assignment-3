# utils/video_recorder.py

import os
from gymnasium.wrappers import RecordVideo
from typing import Callable
from utils.env_utils import make_env, safe_close_env


def record_single_episode(
    env_id: str,
    agent_act_fn: Callable[[object], object],
    destination_folder: str = "videos",
    filename_prefix: str = "run",
    max_steps: int | None = None,
    seed: int | None = None,
):
    """
    Record a single episode and save the MP4/webm file under destination_folder.

    Args:
        env_id: gym id
        agent_act_fn: a callable f(obs) -> action. Should be deterministic (evaluation)
        destination_folder: where to store the video(s)
        filename_prefix: string prefix for the filename
        max_steps: optional limit on steps for safety
        seed: optional seed for the environment

    Returns:
        path to the recorded video file (str)
    """
    os.makedirs(destination_folder, exist_ok=True)

    # Gymnasium RecordVideo requires env to be created with render_mode="rgb_array"
    env = make_env(env_id, seed=seed, render_mode="rgb_array")
    # wrap with RecordVideo to automatically write files
    video_wrapper = RecordVideo(env, video_folder=destination_folder, name_prefix=filename_prefix)

    obs, _ = video_wrapper.reset(seed=seed)
    done = False
    step = 0
    while True:
        action = agent_act_fn(obs)
        obs, reward, terminated, truncated, info = video_wrapper.step(action)
        done = bool(terminated or truncated)
        step += 1
        if done or (max_steps is not None and step >= max_steps):
            break

    # We must close the wrapper to flush and write the file
    safe_close_env(video_wrapper)

    # The RecordVideo wrapper will save the video file inside destination_folder.
    # The file name is not returned by the wrapper; we can return the last file created in the folder.
    recorded_files = sorted(
        [os.path.join(destination_folder, f) for f in os.listdir(destination_folder)],
        key=os.path.getmtime,
    )
    if not recorded_files:
        raise RuntimeError("No video files were produced by RecordVideo.")
    return recorded_files[-1]


def record_multiple_episodes(
    env_id: str,
    agent_act_fn: Callable[[object], object],
    n_episodes: int = 3,
    destination_folder: str = "videos",
    filename_prefix: str = "run",
    max_steps: int | None = None,
    seed: int | None = None,
):
    """
    Record multiple episodes. Returns list of file paths.
    """
    paths = []
    for i in range(n_episodes):
        prefix = f"{filename_prefix}_{i}"
        s = seed + i if seed is not None else None
        p = record_single_episode(
            env_id,
            agent_act_fn,
            destination_folder=destination_folder,
            filename_prefix=prefix,
            max_steps=max_steps,
            seed=s,
        )
        paths.append(p)
    return paths
