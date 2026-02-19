"""
Visualize model predictions on real robot data (inference recordings).

Real data structure:
    <inference_data_path>/<scenario_name>/
        data.npz      — robot states, action_chunks, language instructions, etc.
        video.mp4      — observation video (1 frame per timestep)
        visualization.mp4

Training data is used only for metadata (normalization stats, modality config).

Usage (local model):
    python scripts/visualize_real_data.py \
        --inference-data-path /home/hogunkee/inference_data \
        --dataset-path /data1/hogun/dataset/RealG1_walk_pnp_can_0203 \
        --model-path /data2/hogun/ckpts/0203_groot/checkpoint-20000 \
        --scenario inference_20260212_135731 \
        --data-config unitree_g1_locophase_upper \
        --embodiment-tag g1 \
        --output-dir /tmp/real_data_vis

Usage (with inference server):
    python scripts/visualize_real_data.py \
        --inference-data-path /home/hogunkee/inference_data \
        --dataset-path /data1/hogun/dataset/RealG1_walk_pnp_can_0203 \
        --scenario inference_20260212_135731 \
        --data-config unitree_g1_locophase_upper \
        --embodiment-tag g1 \
        --use-server --host localhost --port 5555 \
        --output-dir /tmp/real_data_vis

Visualize all scenarios in a directory:
    python scripts/visualize_real_data.py \
        --inference-data-path /home/hogunkee/inference_data \
        --dataset-path /data1/hogun/dataset/RealG1_walk_pnp_can_0203 \
        --model-path /data2/hogun/ckpts/0203_groot/checkpoint-20000 \
        --scenario all \
        --data-config unitree_g1_locophase_upper \
        --embodiment-tag g1 \
        --output-dir /tmp/real_data_vis
"""

import argparse
import os
import sys
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gr00t.data.dataset import LeRobotSingleDataset, ModalityConfig
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP


# ─────────────────────────── argument parsing ───────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize model predictions on real robot data"
    )
    # Real data
    parser.add_argument("--inference-data-path", type=str, required=True,
                        help="Root path containing scenario folders (e.g. /home/hogunkee/inference_data)")
    parser.add_argument("--scenario", type=str, required=True,
                        help="Scenario folder name (e.g. inference_20260212_135731) or 'all' for all scenarios")

    # Training data (for metadata / normalization)
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to training dataset (for metadata/normalization stats)")

    # Model
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to model checkpoint (for local inference)")
    parser.add_argument("--use-server", action="store_true",
                        help="Use running inference server instead of loading model locally")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=5555)

    # Config
    parser.add_argument("--data-config", type=str, default="unitree_g1_locophase_upper")
    parser.add_argument("--embodiment-tag", type=str, default="g1")
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--video-backend", type=str, default="decord")

    # Visualization options
    parser.add_argument("--output-dir", type=str, default="/tmp/real_data_vis")
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Number of timesteps to sample for per-step visualization")
    parser.add_argument("--step-indices", type=int, nargs="+", default=None,
                        help="Specific timestep indices to visualize (overrides --num-samples)")
    parser.add_argument("--trajectory-mode", action="store_true",
                        help="Visualize full trajectory with overlaid model predictions")
    parser.add_argument("--chunk-stride", type=int, default=None,
                        help="Stride for sliding window in trajectory mode (default: action_horizon)")
    parser.add_argument("--max-chunks", type=int, default=50,
                        help="Maximum number of inference chunks in trajectory mode")

    return parser.parse_args()


# ─────────────────────── load training dataset (metadata only) ──────────────

def load_training_dataset(args):
    """Load the training dataset to obtain metadata and normalization stats."""
    data_config_cls = DATA_CONFIG_MAP[args.data_config]
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()
    embodiment_tag = EmbodimentTag(args.embodiment_tag)

    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=modality_configs,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=args.video_backend,
    )
    return dataset, data_config_cls, modality_configs


# ──────────────────────── load real robot data ──────────────────────────────

def load_real_data(scenario_dir):
    """Load real robot data from a scenario directory.

    Returns:
        npz_data: dict-like from np.load (keys: state, action_chunk, language_instruction, ...)
        num_steps: total number of timesteps
    """
    npz_path = os.path.join(scenario_dir, "data.npz")
    video_path = os.path.join(scenario_dir, "video.mp4")

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"data.npz not found in {scenario_dir}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"video.mp4 not found in {scenario_dir}")

    npz_data = np.load(npz_path, allow_pickle=True)
    num_steps = npz_data["state"].shape[0]

    # Verify video frame count matches
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if frame_count != num_steps:
        print(f"  ⚠ Warning: video has {frame_count} frames but npz has {num_steps} steps")

    return npz_data, num_steps


def get_video_frame(video_path, step_idx):
    """Extract a single frame from the video at the given timestep index.

    Returns:
        frame: (H, W, 3) uint8 numpy array in RGB order
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, step_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Failed to read frame {step_idx} from {video_path}")
    # OpenCV reads BGR, convert to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def get_video_frames_batch(video_path, step_indices):
    """Extract multiple frames from the video efficiently.

    Returns:
        frames: dict mapping step_idx -> (H, W, 3) uint8 numpy array (RGB)
    """
    frames = {}
    cap = cv2.VideoCapture(video_path)
    for idx in sorted(step_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            print(f"  ⚠ Failed to read frame {idx}")
    cap.release()
    return frames


# ────────────────── build observation dict for the model ────────────────────

def build_observation(npz_data, video_path, step_idx, data_config_cls, modality_json):
    """Build the observation dict that the model/policy expects for a single timestep.

    The model expects:
        "video.<key>":  (1, H, W, C) uint8
        "state.<key>":  (1, D) float64
        "annotation.<key>": list[str]
    """
    obs = {}

    # --- Video ---
    frame = get_video_frame(video_path, step_idx)  # (H, W, 3) uint8
    for vkey in data_config_cls.video_keys:
        obs[vkey] = frame[np.newaxis, ...]  # (1, H, W, C)

    # --- State (split concatenated state vector into per-key arrays) ---
    full_state = npz_data["state"][step_idx]  # (state_dim,)
    state_mapping = modality_json["state"]
    for skey in data_config_cls.state_keys:
        subkey = skey.replace("state.", "")
        if subkey in state_mapping:
            start = state_mapping[subkey]["start"]
            end = state_mapping[subkey]["end"]
            obs[skey] = full_state[start:end][np.newaxis, :].astype(np.float64)  # (1, D)
        else:
            print(f"  ⚠ State key '{subkey}' not found in modality.json, skipping")

    # --- Language ---
    for lkey in data_config_cls.language_keys:
        instruction = str(npz_data["language_instruction"][step_idx])
        obs[lkey] = [instruction]

    return obs


def split_action_chunk(action_chunk, data_config_cls, modality_json):
    """Split a flat action chunk array into per-key dict.

    Args:
        action_chunk: (T, action_dim) or (action_dim,)
        data_config_cls: data config with action_keys
        modality_json: modality mapping with start/end indices

    Returns:
        dict mapping action key -> (T, D) array
    """
    if action_chunk.ndim == 1:
        action_chunk = action_chunk[np.newaxis, :]

    action_mapping = modality_json["action"]
    actions = {}
    for akey in data_config_cls.action_keys:
        subkey = akey.replace("action.", "")
        if subkey in action_mapping:
            start = action_mapping[subkey]["start"]
            end = action_mapping[subkey]["end"]
            actions[akey] = action_chunk[:, start:end]
    return actions


# ─────────────────────── policy loading ─────────────────────────────────────

def load_policy_local(args, modality_configs):
    """Load model locally."""
    from gr00t.model.policy import Gr00tPolicy

    data_config_cls = DATA_CONFIG_MAP[args.data_config]
    modality_config = data_config_cls.modality_config()
    modality_transform = data_config_cls.transform()

    policy = Gr00tPolicy(
        model_path=args.model_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
    )
    return policy


def load_policy_remote(args):
    """Connect to running inference server."""
    from gr00t.eval.robot import RobotInferenceClient
    client = RobotInferenceClient(host=args.host, port=args.port)
    return client


# ─────────────────────── visualization functions ────────────────────────────

def visualize_sample(
    sample_idx, step_idx, frame, gt_actions, pred_actions,
    data_config_cls, output_dir, scenario_name
):
    """Visualize a single timestep: input image + GT vs predicted action chunk.

    Each action key gets a per-dimension subplot figure.
    """
    os.makedirs(output_dir, exist_ok=True)
    action_keys = data_config_cls.action_keys

    # --- Plot 1: Input image ---
    fig_img, ax_img = plt.subplots(1, 1, figsize=(6, 6))
    ax_img.imshow(frame)
    ax_img.set_title(f"Input Image (scenario={scenario_name}, step={step_idx})")
    ax_img.axis("off")
    fig_img.tight_layout()
    fig_img.savefig(os.path.join(output_dir, f"sample_{sample_idx:03d}_image.png"), dpi=150)
    plt.close(fig_img)

    # --- Plot 2: Per action key, per dimension comparison ---
    for key in action_keys:
        gt = gt_actions.get(key)
        has_gt = gt is not None
        has_pred = pred_actions is not None and key in pred_actions

        if not has_gt and not has_pred:
            continue

        # Determine dimensions
        ref = gt if has_gt else pred_actions[key]
        if ref.ndim == 1:
            ref = ref.reshape(1, -1)
        D = ref.shape[1]

        ncols = min(D, 4)
        nrows = (D + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows), squeeze=False)

        gt_arr = None
        time_gt = None
        T_gt = 0
        if has_gt:
            gt_arr = gt if gt.ndim > 1 else gt.reshape(1, -1)
            T_gt = gt_arr.shape[0]
            time_gt = np.arange(T_gt)

        pred_arr = None
        time_pred = None
        T_pred = 0
        if has_pred:
            pred_arr = pred_actions[key]
            if pred_arr.ndim == 1:
                pred_arr = pred_arr.reshape(-1, D)
            T_pred = pred_arr.shape[0]
            time_pred = np.arange(T_pred)

        for d in range(D):
            r, c = divmod(d, ncols)
            ax = axes[r, c]
            if has_gt and gt_arr is not None:
                ax.plot(time_gt, gt_arr[:, d], 'b-', linewidth=1.5, label="GT (recorded)")
            if has_pred and pred_arr is not None:
                ax.plot(time_pred, pred_arr[:, d], 'r--', linewidth=1.5, label="Model Pred")
            ax.set_title(f"dim {d}", fontsize=10)
            ax.set_xlabel("time step")
            ax.grid(True, alpha=0.3)
            if d == 0:
                ax.legend(fontsize=8)

        # Hide unused axes
        for d in range(D, nrows * ncols):
            r, c = divmod(d, ncols)
            axes[r, c].set_visible(False)

        safe_key = key.replace(".", "_")
        T_display = T_gt if has_gt else T_pred
        fig.suptitle(
            f"{key} — Action Chunk (scenario={scenario_name}, step={step_idx}, horizon={T_display})",
            fontsize=13, fontweight='bold'
        )
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"sample_{sample_idx:03d}_{safe_key}.png"), dpi=150)
        plt.close(fig)

    # --- Plot 3: All keys overview ---
    num_keys = len(action_keys)
    valid_keys = [k for k in action_keys if k in gt_actions or (pred_actions and k in pred_actions)]
    if not valid_keys:
        return

    fig_overview, axes_ov = plt.subplots(len(valid_keys), 1,
                                          figsize=(12, 3 * len(valid_keys)), squeeze=False)
    for i, key in enumerate(valid_keys):
        ax = axes_ov[i, 0]
        has_gt = key in gt_actions
        has_pred = pred_actions is not None and key in pred_actions

        ref = gt_actions[key] if has_gt else pred_actions[key]
        if ref.ndim == 1:
            ref = ref.reshape(1, -1)
        D = ref.shape[1]

        for d in range(D):
            color = plt.colormaps['tab10'](d % 10)
            if has_gt:
                gt = gt_actions[key]
                if gt.ndim == 1:
                    gt = gt.reshape(1, -1)
                ax.plot(np.arange(gt.shape[0]), gt[:, d], '-', color=color,
                        alpha=0.8, linewidth=1.2, label=f"GT d{d}")
            if has_pred:
                pred = pred_actions[key]
                if pred.ndim == 1:
                    pred = pred.reshape(-1, D)
                ax.plot(np.arange(pred.shape[0]), pred[:, d], '--', color=color,
                        alpha=0.6, linewidth=1.2)

        ax.set_title(f"{key} (D={D})")
        ax.set_xlabel("time step")
        ax.grid(True, alpha=0.3)

    legend_elements = [
        Line2D([0], [0], color='gray', linestyle='-', linewidth=2, label='GT (solid)'),
        Line2D([0], [0], color='gray', linestyle='--', linewidth=2, label='Pred (dashed)'),
    ]
    axes_ov[0, 0].legend(handles=legend_elements, loc='upper right', fontsize=9)
    fig_overview.suptitle(
        f"Action Chunk Overview (scenario={scenario_name}, step={step_idx})",
        fontsize=14, fontweight='bold'
    )
    fig_overview.tight_layout()
    fig_overview.savefig(os.path.join(output_dir, f"sample_{sample_idx:03d}_overview.png"), dpi=150)
    plt.close(fig_overview)


def visualize_trajectory(
    npz_data, video_path, data_config_cls, modality_json,
    output_dir, scenario_name, policy=None,
    chunk_stride=None, max_chunks=50
):
    """Visualize the full trajectory: plot recorded action chunks + model predictions.

    Overlays model predictions at regular intervals on top of the recorded trajectory.
    """
    os.makedirs(output_dir, exist_ok=True)
    action_keys = data_config_cls.action_keys
    action_horizon = len(data_config_cls.action_indices)
    action_mapping = modality_json["action"]

    num_steps = npz_data["state"].shape[0]

    if chunk_stride is None:
        chunk_stride = action_horizon

    # Build the full recorded action trajectory from action_chunk data
    # action_chunk[t] is the chunk predicted/recorded at timestep t
    # We use action_chunk[t, 0, :] as the "executed" action at step t
    recorded_actions_flat = npz_data["action_chunk"][:, 0, :]  # (num_steps, action_dim)

    # Split into per-key
    recorded_actions = {}
    for akey in action_keys:
        subkey = akey.replace("action.", "")
        if subkey in action_mapping:
            start = action_mapping[subkey]["start"]
            end = action_mapping[subkey]["end"]
            recorded_actions[akey] = recorded_actions_flat[:, start:end]  # (num_steps, D)

    # Compute model prediction chunks at regular intervals
    pred_chunks = []  # list of (start_idx, pred_actions_dict)
    if policy is not None:
        chunk_starts = list(range(0, max(1, num_steps - action_horizon + 1), chunk_stride))
        if len(chunk_starts) > max_chunks:
            chunk_starts = chunk_starts[:max_chunks]
        print(f"  Running inference on {len(chunk_starts)} chunks...")
        for start_idx in chunk_starts:
            obs = build_observation(npz_data, video_path, start_idx, data_config_cls, modality_json)
            try:
                pred = policy.get_action(obs)
                pred_chunks.append((start_idx, pred))
            except Exception as e:
                print(f"    ⚠ Inference failed at step {start_idx}: {e}")

    # Also collect recorded action chunks at the same positions for comparison
    recorded_chunks = []
    for start_idx in range(0, max(1, num_steps - action_horizon + 1), chunk_stride):
        if start_idx + action_horizon <= npz_data["action_chunk"].shape[0]:
            chunk = npz_data["action_chunk"][start_idx]  # (action_horizon, action_dim)
            chunk_dict = split_action_chunk(chunk, data_config_cls, modality_json)
            recorded_chunks.append((start_idx, chunk_dict))

    # Plot per action key
    for key in action_keys:
        if key not in recorded_actions:
            continue

        full_recorded = recorded_actions[key]  # (num_steps, D)
        T_total, D = full_recorded.shape

        ncols = min(D, 4)
        nrows = (D + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
        time_all = np.arange(T_total)

        for d in range(D):
            r, c = divmod(d, ncols)
            ax = axes[r, c]

            # Full recorded trajectory (executed action at each step)
            ax.plot(time_all, full_recorded[:, d], 'b-', linewidth=0.8, alpha=0.6,
                    label="Recorded" if d == 0 else None)

            # Overlay recorded action chunks (the full chunk, not just first action)
            for ci, (start_idx, chunk_dict) in enumerate(recorded_chunks):
                if key in chunk_dict:
                    chunk = chunk_dict[key]
                    T_chunk = chunk.shape[0]
                    t_chunk = np.arange(start_idx, start_idx + T_chunk)
                    ax.plot(t_chunk, chunk[:, d], '-', color='cornflowerblue', alpha=0.25, linewidth=0.6)

            # Overlay model predictions
            for ci, (start_idx, pred_dict) in enumerate(pred_chunks):
                if key in pred_dict:
                    pred = pred_dict[key]
                    if pred.ndim == 1:
                        pred = pred.reshape(-1, D)
                    T_pred = pred.shape[0]
                    t_pred = np.arange(start_idx, start_idx + T_pred)
                    ax.plot(t_pred, pred[:, d], '-', color='r', alpha=0.4, linewidth=1.0,
                            label="Model Pred" if (d == 0 and ci == 0) else None)

            ax.set_title(f"dim {d}", fontsize=10)
            ax.set_xlabel("global time step")
            ax.grid(True, alpha=0.3)
            if d == 0:
                ax.legend(fontsize=8)

        for d in range(D, nrows * ncols):
            r, c = divmod(d, ncols)
            axes[r, c].set_visible(False)

        safe_key = key.replace(".", "_")
        fig.suptitle(
            f"{scenario_name} — {key} (T={T_total}, chunk={action_horizon}, stride={chunk_stride})",
            fontsize=13, fontweight='bold'
        )
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"traj_{safe_key}.png"), dpi=150)
        plt.close(fig)

    print(f"  ✓ Trajectory plots saved to {output_dir}/traj_*.png")


# ───────────────────────── info printing ────────────────────────────────────

def print_real_data_info(npz_data, video_path, scenario_name):
    """Print summary of the real data loaded from a scenario."""
    print(f"\n{'='*70}")
    print(f"REAL DATA: {scenario_name}")
    print(f"{'='*70}")
    for k in npz_data.keys():
        v = npz_data[k]
        if isinstance(v, np.ndarray):
            extra = ""
            if v.dtype in [np.float32, np.float64]:
                extra = f", min={v.min():.4f}, max={v.max():.4f}"
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}{extra}")
        else:
            print(f"  {k}: {type(v)}")

    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"  video: {w}x{h}, {fps} fps, {nframes} frames")

    if "language_instruction" in npz_data:
        instructions = npz_data["language_instruction"]
        unique_instructions = list(set(str(x) for x in instructions))
        print(f"  language instructions: {unique_instructions}")
    print(f"{'='*70}\n")


def print_config_info(data_config_cls, dataset):
    """Print data config and metadata info."""
    print(f"\n{'='*70}")
    print("DATA CONFIG & METADATA")
    print(f"{'='*70}")
    print(f"  Config: {data_config_cls.__class__.__name__}")
    print(f"  video_keys:          {data_config_cls.video_keys}")
    print(f"  state_keys:          {data_config_cls.state_keys}")
    print(f"  action_keys:         {data_config_cls.action_keys}")
    print(f"  language_keys:       {data_config_cls.language_keys}")
    print(f"  observation_indices: {data_config_cls.observation_indices}")
    print(f"  action_indices:      {data_config_cls.action_indices}")
    print(f"  action_horizon:      {len(data_config_cls.action_indices)}")

    # Normalization stats from training data
    metadata = dataset.metadata
    print(f"\n--- Normalization Statistics ---")
    for modality in ["state", "action"]:
        stats = getattr(metadata.statistics, modality, {})
        for subkey, vals in stats.items():
            if hasattr(vals, 'min') and hasattr(vals, 'max'):
                print(f"  {modality}.{subkey}:")
                print(f"    min: {[round(v, 4) for v in vals.min]}")
                print(f"    max: {[round(v, 4) for v in vals.max]}")
    print(f"{'='*70}\n")


# ────────────────────── process one scenario ────────────────────────────────

def process_scenario(
    scenario_name, inference_data_path, dataset, data_config_cls,
    modality_json, policy, args
):
    """Process a single scenario: load data, run inference, visualize."""
    scenario_dir = os.path.join(inference_data_path, scenario_name)
    video_path = os.path.join(scenario_dir, "video.mp4")
    output_dir = os.path.join(args.output_dir, scenario_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n▶ Processing scenario: {scenario_name}")

    # Load real data
    npz_data, num_steps = load_real_data(scenario_dir)
    print_real_data_info(npz_data, video_path, scenario_name)

    action_horizon = len(data_config_cls.action_indices)

    if args.trajectory_mode:
        # ── Trajectory mode ──
        visualize_trajectory(
            npz_data, video_path, data_config_cls, modality_json,
            output_dir, scenario_name, policy=policy,
            chunk_stride=args.chunk_stride, max_chunks=args.max_chunks,
        )
    else:
        # ── Per-sample mode ──
        # Select timesteps
        if args.step_indices is not None:
            step_indices = [s for s in args.step_indices if s < num_steps]
        else:
            np.random.seed(42)
            safe_len = max(1, num_steps - action_horizon)
            n = min(args.num_samples, safe_len)
            step_indices = sorted(np.random.choice(safe_len, n, replace=False).tolist())

        print(f"  Visualizing {len(step_indices)} timesteps: {step_indices}")

        # Pre-load all needed frames
        frames = get_video_frames_batch(video_path, step_indices)

        for s_idx, step in enumerate(step_indices):
            print(f"\n  --- Sample {s_idx}: step={step} ---")

            # Get the recorded action chunk at this timestep
            recorded_chunk = npz_data["action_chunk"][step]  # (action_horizon, action_dim)
            gt_actions = split_action_chunk(recorded_chunk, data_config_cls, modality_json)
            for k, v in gt_actions.items():
                print(f"    GT {k}: shape={v.shape}")

            # Model prediction
            pred_actions = None
            if policy is not None:
                obs = build_observation(npz_data, video_path, step, data_config_cls, modality_json)
                print("    Running inference...")
                for k, v in obs.items():
                    if isinstance(v, np.ndarray):
                        print(f"      obs {k}: shape={v.shape}, dtype={v.dtype}")
                    else:
                        print(f"      obs {k}: {v}")
                try:
                    pred_actions = policy.get_action(obs)
                    for k, v in pred_actions.items():
                        if isinstance(v, np.ndarray):
                            print(f"    Pred {k}: shape={v.shape}")
                except Exception as e:
                    print(f"    ⚠ Inference failed: {e}")
                    import traceback
                    traceback.print_exc()
                    pred_actions = None

            # Get the frame
            frame = frames.get(step)
            if frame is None:
                frame = get_video_frame(video_path, step)

            # Visualize
            visualize_sample(
                s_idx, step, frame, gt_actions, pred_actions,
                data_config_cls, output_dir, scenario_name
            )
            print(f"    ✓ Saved to {output_dir}/sample_{s_idx:03d}_*.png")

    print(f"\n✅ Scenario '{scenario_name}' done! Outputs → {output_dir}")


# ──────────────────────────── main ──────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load training dataset for metadata
    print("Loading training dataset (for metadata/normalization)...")
    dataset, data_config_cls, modality_configs = load_training_dataset(args)
    print_config_info(data_config_cls, dataset)

    # 2. Load modality mapping from training data
    modality_json_path = os.path.join(args.dataset_path, "meta", "modality.json")
    with open(modality_json_path, "r") as f:
        modality_json = json.load(f)
    print(f"Loaded modality mapping from {modality_json_path}")

    # 3. Load policy
    policy = None
    if args.use_server:
        print(f"\nConnecting to inference server at {args.host}:{args.port}...")
        policy = load_policy_remote(args)
    elif args.model_path:
        print(f"\nLoading model from {args.model_path}...")
        policy = load_policy_local(args, modality_configs)

    # 4. Determine scenarios
    if args.scenario.lower() == "all":
        scenario_names = sorted([
            d for d in os.listdir(args.inference_data_path)
            if os.path.isdir(os.path.join(args.inference_data_path, d))
        ])
        print(f"\nFound {len(scenario_names)} scenarios: {scenario_names}")
    else:
        scenario_names = [args.scenario]

    # 5. Process each scenario
    for scenario_name in scenario_names:
        scenario_dir = os.path.join(args.inference_data_path, scenario_name)
        if not os.path.isdir(scenario_dir):
            print(f"⚠ Scenario directory not found: {scenario_dir}, skipping")
            continue
        process_scenario(
            scenario_name, args.inference_data_path, dataset, data_config_cls,
            modality_json, policy, args
        )

    print(f"\n🎉 All done! Visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
