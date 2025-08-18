import argparse
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def find_latest_pair(trajectories_dir: Path) -> Tuple[Path, Path]:
    action_files = sorted(
        [p for p in trajectories_dir.iterdir() if p.is_file() and p.name.startswith("action_") and p.suffix == ".pkl"]
    )
    if not action_files:
        raise FileNotFoundError(f"No action_###.pkl files found in {trajectories_dir}")
    latest_action = action_files[-1]
    idx = latest_action.stem.split("_")[-1]
    obs_path = trajectories_dir / f"obs_{idx}.pkl"
    if not obs_path.exists():
        raise FileNotFoundError(f"Matching obs file not found: {obs_path}")
    return latest_action, obs_path


def human_shape(x: Any) -> str:
    try:
        if isinstance(x, np.ndarray):
            return f"ndarray{tuple(x.shape)} {x.dtype}"
        if isinstance(x, (list, tuple)):
            return f"list[{len(x)}]"
        if isinstance(x, (int, float, str, bool)):
            return f"scalar<{type(x).__name__}>"
        return type(x).__name__
    except Exception:
        return type(x).__name__


def summarize_episode_dict(ep_dict: Dict[str, Any]) -> Dict[str, str]:
    summary: Dict[str, str] = {}
    for key, val in ep_dict.items():
        if isinstance(val, np.ndarray):
            summary[key] = f"ndarray shape={val.shape}, dtype={val.dtype}"
        elif isinstance(val, list):
            if len(val) == 0:
                summary[key] = "list[0]"
            else:
                sample = val[0]
                summary[key] = f"list[{len(val)}] of {human_shape(sample)}"
        else:
            summary[key] = human_shape(val)
    return summary


def print_header(title: str) -> None:
    bar = "=" * len(title)
    print(bar)
    print(title)
    print(bar)


def analyze_pair(action_path: Path, obs_path: Path) -> None:
    print_header("Loading pickles")
    print(f"action: {action_path}")
    print(f"obs   : {obs_path}")

    with open(action_path, "rb") as f:
        action_obj = pickle.load(f)
    with open(obs_path, "rb") as f:
        obs_obj = pickle.load(f)

    if not isinstance(action_obj, dict) or not isinstance(obs_obj, dict):
        raise ValueError("Both action and obs files must contain dict objects")

    action_eps = sorted(action_obj.keys())
    obs_eps = sorted(obs_obj.keys())

    print()
    print_header("Episodes")
    print(f"#action_episodes = {len(action_eps)}")
    print(f"#obs_episodes    = {len(obs_eps)}")
    common_eps = sorted(set(action_eps).intersection(obs_eps))
    print(f"#common_episodes = {len(common_eps)}")
    if len(common_eps) == 0:
        print("WARNING: No common episode keys between action and obs")

    # Inspect one representative episode (prefer the first common)
    sample_ep = common_eps[0] if common_eps else (action_eps[0] if action_eps else None)
    if sample_ep is None:
        print("No episodes to inspect.")
        return

    print()
    print_header(f"Episode detail: {sample_ep}")
    aep = action_obj.get(sample_ep, {})
    oep = obs_obj.get(sample_ep, {})
    if not isinstance(aep, dict) or not isinstance(oep, dict):
        print("Selected episode content is not a dict; skipping detailed summary.")
        return

    # Timesteps: try infer from a standard action key if available
    candidate_action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
        "action.waist",
    ]
    timesteps = None
    for k in candidate_action_keys:
        if k in aep and isinstance(aep[k], np.ndarray) and aep[k].ndim >= 1:
            timesteps = aep[k].shape[0]
            break
    print(f"Estimated timesteps (from actions): {timesteps}")

    print()
    print("Action keys and shapes:")
    a_summary = summarize_episode_dict(aep)
    for k in sorted(a_summary.keys()):
        print(f"  - {k}: {a_summary[k]}")

    print()
    print("Observation keys and shapes:")
    o_summary = summarize_episode_dict(oep)
    for k in sorted(o_summary.keys()):
        print(f"  - {k}: {o_summary[k]}")

    # Cross-check alignment for common numeric keys present in both action and obs
    print()
    print_header("Cross-check (length sanity)")
    numeric_obs_candidates = [
        "video.ego_view",
        "video.ego_view_pad_res256_freq20",
        "video.world_view",
        "video.rs_view",
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.waist",
    ]
    def head_shape(x: Any) -> Tuple[int, Tuple[int, ...]]:
        if isinstance(x, np.ndarray):
            return x.shape[0], x.shape
        if isinstance(x, list):
            return len(x), (len(x),)
        return -1, ()

    # report a few common keys
    for key in candidate_action_keys:
        if key in aep and isinstance(aep[key], np.ndarray):
            alen, ashape = head_shape(aep[key])
            print(f"  action {key}: len={alen}, shape={ashape}")
            break

    for key in numeric_obs_candidates:
        if key in oep:
            olen, oshape = head_shape(oep[key])
            print(f"  obs    {key}: len={olen}, shape={oshape}")
            # only print first present one for brevity
            break


def main() -> None:
    default_dir = (Path(__file__).resolve().parent / "trajectories").resolve()
    parser = argparse.ArgumentParser(description="Inspect saved action / obs trajectory pickles")
    parser.add_argument("--dir", type=str, default=str(default_dir), help="Directory containing action_###.pkl and obs_###.pkl")
    parser.add_argument("--action", type=str, default=None, help="Explicit path to action_###.pkl (optional)")
    parser.add_argument("--obs", type=str, default=None, help="Explicit path to obs_###.pkl (optional)")
    args = parser.parse_args()

    trajectories_dir = Path(args.dir)
    if args.action and args.obs:
        action_path = Path(args.action)
        obs_path = Path(args.obs)
    else:
        action_path, obs_path = find_latest_pair(trajectories_dir)

    analyze_pair(action_path, obs_path)


if __name__ == "__main__":
    main()

import argparse
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def find_latest_pair(trajectories_dir: Path) -> Tuple[Path, Path]:
    action_files = sorted(
        [p for p in trajectories_dir.iterdir() if p.is_file() and p.name.startswith("action_") and p.suffix == ".pkl"]
    )
    if not action_files:
        raise FileNotFoundError(f"No action_###.pkl files found in {trajectories_dir}")
    latest_action = action_files[-1]
    idx = latest_action.stem.split("_")[-1]
    obs_path = trajectories_dir / f"obs_{idx}.pkl"
    if not obs_path.exists():
        raise FileNotFoundError(f"Matching obs file not found: {obs_path}")
    return latest_action, obs_path


def human_shape(x: Any) -> str:
    try:
        if isinstance(x, np.ndarray):
            return f"ndarray{tuple(x.shape)} {x.dtype}"
        if isinstance(x, (list, tuple)):
            return f"list[{len(x)}]"
        if isinstance(x, (int, float, str, bool)):
            return f"scalar<{type(x).__name__}>"
        return type(x).__name__
    except Exception:
        return type(x).__name__


def summarize_episode_dict(ep_dict: Dict[str, Any]) -> Dict[str, str]:
    summary: Dict[str, str] = {}
    for key, val in ep_dict.items():
        if isinstance(val, np.ndarray):
            summary[key] = f"ndarray shape={val.shape}, dtype={val.dtype}"
        elif isinstance(val, list):
            if len(val) == 0:
                summary[key] = "list[0]"
            else:
                sample = val[0]
                summary[key] = f"list[{len(val)}] of {human_shape(sample)}"
        else:
            summary[key] = human_shape(val)
    return summary


def print_header(title: str) -> None:
    bar = "=" * len(title)
    print(bar)
    print(title)
    print(bar)


def analyze_pair(action_path: Path, obs_path: Path) -> None:
    print_header("Loading pickles")
    print(f"action: {action_path}")
    print(f"obs   : {obs_path}")

    with open(action_path, "rb") as f:
        action_obj = pickle.load(f)
    with open(obs_path, "rb") as f:
        obs_obj = pickle.load(f)

    if not isinstance(action_obj, dict) or not isinstance(obs_obj, dict):
        raise ValueError("Both action and obs files must contain dict objects")

    action_eps = sorted(action_obj.keys())
    obs_eps = sorted(obs_obj.keys())

    print()
    print_header("Episodes")
    print(f"#action_episodes = {len(action_eps)}")
    print(f"#obs_episodes    = {len(obs_eps)}")
    common_eps = sorted(set(action_eps).intersection(obs_eps))
    print(f"#common_episodes = {len(common_eps)}")
    if len(common_eps) == 0:
        print("WARNING: No common episode keys between action and obs")

    # Inspect one representative episode (prefer the first common)
    sample_ep = common_eps[0] if common_eps else (action_eps[0] if action_eps else None)
    if sample_ep is None:
        print("No episodes to inspect.")
        return

    print()
    print_header(f"Episode detail: {sample_ep}")
    aep = action_obj.get(sample_ep, {})
    oep = obs_obj.get(sample_ep, {})
    if not isinstance(aep, dict) or not isinstance(oep, dict):
        print("Selected episode content is not a dict; skipping detailed summary.")
        return

    # Timesteps: try infer from a standard action key if available
    candidate_action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
        "action.waist",
    ]
    timesteps = None
    for k in candidate_action_keys:
        if k in aep and isinstance(aep[k], np.ndarray) and aep[k].ndim >= 1:
            timesteps = aep[k].shape[0]
            break
    print(f"Estimated timesteps (from actions): {timesteps}")

    print()
    print("Action keys and shapes:")
    a_summary = summarize_episode_dict(aep)
    for k in sorted(a_summary.keys()):
        print(f"  - {k}: {a_summary[k]}")

    print()
    print("Observation keys and shapes:")
    o_summary = summarize_episode_dict(oep)
    for k in sorted(o_summary.keys()):
        print(f"  - {k}: {o_summary[k]}")

    # Cross-check alignment for common numeric keys present in both action and obs
    print()
    print_header("Cross-check (length sanity)")
    numeric_obs_candidates = [
        "video.ego_view",
        "video.ego_view_pad_res256_freq20",
        "video.world_view",
        "video.rs_view",
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.waist",
    ]
    def head_shape(x: Any) -> Tuple[int, Tuple[int, ...]]:
        if isinstance(x, np.ndarray):
            return x.shape[0], x.shape
        if isinstance(x, list):
            return len(x), (len(x),)
        return -1, ()

    # report a few common keys
    for key in candidate_action_keys:
        if key in aep and isinstance(aep[key], np.ndarray):
            alen, ashape = head_shape(aep[key])
            print(f"  action {key}: len={alen}, shape={ashape}")
            break

    for key in numeric_obs_candidates:
        if key in oep:
            olen, oshape = head_shape(oep[key])
            print(f"  obs    {key}: len={olen}, shape={oshape}")
            # only print first present one for brevity
            break


def main() -> None:
    default_dir = (Path(__file__).resolve().parent / "trajectories").resolve()
    parser = argparse.ArgumentParser(description="Inspect saved action / obs trajectory pickles")
    parser.add_argument("--dir", type=str, default=str(default_dir), help="Directory containing action_###.pkl and obs_###.pkl")
    parser.add_argument("--action", type=str, default=None, help="Explicit path to action_###.pkl (optional)")
    parser.add_argument("--obs", type=str, default=None, help="Explicit path to obs_###.pkl (optional)")
    args = parser.parse_args()

    trajectories_dir = Path(args.dir)
    if args.action and args.obs:
        action_path = Path(args.action)
        obs_path = Path(args.obs)
    else:
        action_path, obs_path = find_latest_pair(trajectories_dir)

    analyze_pair(action_path, obs_path)


if __name__ == "__main__":
    main()


