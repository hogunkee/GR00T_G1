"""
Visualize action chunks from training data using a trained GR00T model.

Usage (with inference server already running):
    python scripts/visualize_action_chunks.py \
        --dataset-path /data1/hogun/dataset/RealG1_walk_pnp_can_0203 \
        --model-path /data2/hogun/ckpts/0203_groot/checkpoint-20000 \
        --data-config unitree_g1_locophase_upper \
        --embodiment-tag g1 \
        --num-samples 5 \
        --output-dir /tmp/action_chunk_vis

Or use --use-server to query the running inference server instead of loading the model locally:
    python scripts/visualize_action_chunks.py \
        --dataset-path /data1/hogun/dataset/RealG1_walk_pnp_can_0203 \
        --data-config unitree_g1_locophase_upper \
        --embodiment-tag g1 \
        --use-server --host localhost --port 5555 \
        --num-samples 5 \
        --output-dir /tmp/action_chunk_vis

Visualize a continuous trajectory (GT only, no model needed):
    python scripts/visualize_action_chunks.py \
        --dataset-path /data1/hogun/dataset/RealG1_walk_pnp_can_0203 \
        --data-config unitree_g1_locophase_upper \
        --embodiment-tag g1 \
        --episode-idx 0 --trajectory-mode \
        --output-dir /tmp/action_chunk_vis
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gr00t.data.dataset import LeRobotSingleDataset, ModalityConfig
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize action chunks from training data")
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to model checkpoint (for local inference)")
    parser.add_argument("--data-config", type=str, default="unitree_g1_locophase_upper")
    parser.add_argument("--embodiment-tag", type=str, default="g1")
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Number of random samples to visualize")
    parser.add_argument("--output-dir", type=str, default="/tmp/action_chunk_vis")
    parser.add_argument("--episode-idx", type=int, default=None,
                        help="Specific episode index to visualize (if not set, random)")
    parser.add_argument("--step-idx", type=int, default=None,
                        help="Specific step index within the episode")
    parser.add_argument("--use-server", action="store_true",
                        help="Use running inference server instead of loading model locally")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--video-backend", type=str, default="decord")
    parser.add_argument("--trajectory-mode", action="store_true",
                        help="Visualize full trajectory with sliding-window action chunks")
    parser.add_argument("--chunk-stride", type=int, default=None,
                        help="Stride for sliding window in trajectory mode (default: chunk_size)")
    return parser.parse_args()


def load_dataset(args):
    """Load dataset with the same config used in training."""
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


def load_policy_local(args, modality_configs):
    """Load model locally (same as inference_service.py)."""
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


def get_raw_step_data(dataset, trajectory_id, base_index):
    """Get raw (untransformed) data for a single step."""
    data = {}
    dataset.curr_traj_data = dataset.get_trajectory_data(trajectory_id)
    for modality in dataset.modality_keys:
        for key in dataset.modality_keys[modality]:
            data[key] = dataset.get_data_by_modality(trajectory_id, modality, key, base_index)
    return data


def prepare_obs_for_server(raw_data, data_config_cls):
    """Prepare observation dict from raw data for inference server / policy.
    
    The server expects the same format as the client sends:
      - video: (1, H, W, C) uint8
      - state: (1, D) float
      - annotation: list[str]
    """
    obs = {}
    for key in data_config_cls.video_keys:
        # raw_data[key] shape: (T, H, W, C), T=1 for observation_indices=[0]
        obs[key] = raw_data[key].astype(np.uint8)

    for key in data_config_cls.state_keys:
        # raw_data[key] shape: (T, D), T=1
        obs[key] = raw_data[key].astype(np.float64)

    for key in data_config_cls.language_keys:
        obs[key] = raw_data[key]  # list[str]

    return obs


def get_gt_actions(raw_data, data_config_cls):
    """Extract ground truth action chunk from raw data."""
    gt = {}
    for key in data_config_cls.action_keys:
        gt[key] = raw_data[key]  # shape: (action_horizon, D)
    return gt


def get_full_trajectory_actions(dataset, trajectory_id, data_config_cls):
    """Get the full trajectory's action data (all timesteps, no chunking)."""
    dataset.curr_traj_data = dataset.get_trajectory_data(trajectory_id)
    traj_data = dataset.curr_traj_data

    full_actions = {}
    le_modality_meta = dataset.lerobot_modality_meta
    for key in data_config_cls.action_keys:
        subkey = key.replace("action.", "")
        le_cfg = le_modality_meta.action[subkey]
        le_key = le_cfg.original_key if le_cfg.original_key else subkey
        data_array = np.stack(traj_data[le_key])  # (T_total, D_full)
        le_indices = np.arange(le_cfg.start, le_cfg.end)
        full_actions[key] = data_array[:, le_indices]  # (T_total, D)
    return full_actions


def visualize_sample(
    sample_idx, trajectory_id, base_index, raw_data, gt_actions, pred_actions,
    data_config_cls, output_dir
):
    """Visualize a single sample: GT vs predicted action chunk + input image.
    Each action key gets its own figure with per-dimension subplots.
    """
    os.makedirs(output_dir, exist_ok=True)

    action_keys = data_config_cls.action_keys
    action_horizon = len(data_config_cls.action_indices)

    # --- Plot 1: Input image ---
    video_key = data_config_cls.video_keys[0]
    img = raw_data[video_key][0]  # (H, W, C), first frame
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

    fig_img, ax_img = plt.subplots(1, 1, figsize=(6, 6))
    ax_img.imshow(img)
    ax_img.set_title(f"Input Image (traj={trajectory_id}, step={base_index})")
    ax_img.axis("off")
    fig_img.tight_layout()
    fig_img.savefig(os.path.join(output_dir, f"sample_{sample_idx:03d}_image.png"), dpi=150)
    plt.close(fig_img)

    # --- Plot 2: Per action key, per dimension comparison ---
    for key in action_keys:
        gt = gt_actions[key]  # (T_gt, D)
        T_gt, D = gt.shape

        has_pred = (pred_actions is not None and key in pred_actions)
        if has_pred:
            pred = pred_actions[key]
            if pred.ndim == 1:
                pred = pred.reshape(-1, D)
            T_pred = pred.shape[0]
        else:
            T_pred = 0

        ncols = min(D, 4)
        nrows = (D + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows), squeeze=False)

        time_gt = np.arange(T_gt)

        for d in range(D):
            r, c = divmod(d, ncols)
            ax = axes[r, c]
            ax.plot(time_gt, gt[:, d], 'b-', linewidth=1.5, label="GT")
            if has_pred:
                time_pred = np.arange(T_pred)
                ax.plot(time_pred, pred[:, d], 'r--', linewidth=1.5, label="Pred")
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
        fig.suptitle(
            f"{key} — Action Chunk (traj={trajectory_id}, step={base_index}, horizon={T_gt})",
            fontsize=13, fontweight='bold'
        )
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"sample_{sample_idx:03d}_{safe_key}.png"), dpi=150)
        plt.close(fig)

    # --- Plot 3: All keys overview (mean ± std across dims) ---
    num_keys = len(action_keys)
    fig_overview, axes_ov = plt.subplots(num_keys, 1, figsize=(12, 3 * num_keys), squeeze=False)
    for i, key in enumerate(action_keys):
        ax = axes_ov[i, 0]
        gt = gt_actions[key]
        T_gt, D = gt.shape
        time_gt = np.arange(T_gt)

        # Plot each dim with distinct color
        for d in range(D):
            color = plt.colormaps['tab10'](d % 10)
            ax.plot(time_gt, gt[:, d], '-', color=color, alpha=0.8, linewidth=1.2, label=f"GT d{d}")
            if pred_actions is not None and key in pred_actions:
                pred = pred_actions[key]
                if pred.ndim == 1:
                    pred = pred.reshape(-1, D)
                time_pred = np.arange(pred.shape[0])
                ax.plot(time_pred, pred[:, d], '--', color=color, alpha=0.6, linewidth=1.2)

        ax.set_title(f"{key} (D={D})")
        ax.set_xlabel("time step")
        ax.grid(True, alpha=0.3)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', linestyle='-', linewidth=2, label='GT (solid)'),
        Line2D([0], [0], color='gray', linestyle='--', linewidth=2, label='Pred (dashed)'),
    ]
    axes_ov[0, 0].legend(handles=legend_elements, loc='upper right', fontsize=9)
    fig_overview.suptitle(
        f"Action Chunk Overview (traj={trajectory_id}, step={base_index})",
        fontsize=14, fontweight='bold'
    )
    fig_overview.tight_layout()
    fig_overview.savefig(os.path.join(output_dir, f"sample_{sample_idx:03d}_overview.png"), dpi=150)
    plt.close(fig_overview)


def visualize_trajectory(
    dataset, trajectory_id, data_config_cls, output_dir,
    policy=None, chunk_stride=None
):
    """Visualize a full trajectory: plot the entire GT action sequence,
    and overlay predicted action chunks at regular intervals.
    """
    os.makedirs(output_dir, exist_ok=True)
    action_keys = data_config_cls.action_keys
    action_horizon = len(data_config_cls.action_indices)

    if chunk_stride is None:
        chunk_stride = action_horizon

    # Get full trajectory actions
    full_actions = get_full_trajectory_actions(dataset, trajectory_id, data_config_cls)
    traj_idx = np.where(dataset.trajectory_ids == trajectory_id)[0][0]
    traj_len = dataset.trajectory_lengths[traj_idx]

    print(f"  Trajectory {trajectory_id}: {traj_len} steps, action_horizon={action_horizon}, stride={chunk_stride}")

    # Compute prediction chunks at regular intervals
    pred_chunks = []  # list of (start_idx, pred_actions_dict)
    if policy is not None:
        chunk_starts = list(range(0, traj_len - action_horizon + 1, chunk_stride))
        if len(chunk_starts) > 50:
            # Limit to avoid very long inference
            chunk_starts = chunk_starts[:50]
        print(f"  Running inference on {len(chunk_starts)} chunks...")
        for start_idx in chunk_starts:
            raw_data = get_raw_step_data(dataset, trajectory_id, start_idx)
            obs = prepare_obs_for_server(raw_data, data_config_cls)
            try:
                pred = policy.get_action(obs)
                pred_chunks.append((start_idx, pred))
            except Exception as e:
                print(f"    ⚠ Inference failed at step {start_idx}: {e}")

    # Plot per action key
    for key in action_keys:
        full_gt = full_actions[key]  # (T_total, D)
        T_total, D = full_gt.shape

        ncols = min(D, 4)
        nrows = (D + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
        time_all = np.arange(T_total)

        for d in range(D):
            r, c = divmod(d, ncols)
            ax = axes[r, c]
            # Full GT trajectory
            ax.plot(time_all, full_gt[:, d], 'b-', linewidth=1.0, alpha=0.8, label="GT" if d == 0 else None)
            # Overlay predicted chunks
            for ci, (start_idx, pred_dict) in enumerate(pred_chunks):
                if key in pred_dict:
                    pred = pred_dict[key]
                    if pred.ndim == 1:
                        pred = pred.reshape(-1, D)
                    T_pred = pred.shape[0]
                    t_pred = np.arange(start_idx, start_idx + T_pred)
                    color = plt.colormaps['hsv'](ci / max(len(pred_chunks), 1))
                    ax.plot(t_pred, pred[:, d], '-', color='r', alpha=0.4, linewidth=1.0,
                            label="Pred" if (d == 0 and ci == 0) else None)

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
            f"Trajectory {trajectory_id} — {key} (T={T_total}, chunk={action_horizon}, stride={chunk_stride})",
            fontsize=13, fontweight='bold'
        )
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"traj_{trajectory_id:03d}_{safe_key}.png"), dpi=150)
        plt.close(fig)

    print(f"  ✓ Trajectory plots saved to {output_dir}/traj_{trajectory_id:03d}_*.png")


def print_transform_pipeline_info(data_config_cls, dataset):
    """Print detailed info about the transform pipeline."""
    print("\n" + "=" * 70)
    print("TRANSFORM PIPELINE ANALYSIS")
    print("=" * 70)

    transforms = data_config_cls.transform()
    print(f"\nTransform chain ({len(transforms.transforms)} steps):")
    for i, t in enumerate(transforms.transforms):
        cls_name = t.__class__.__name__
        details = ""
        if hasattr(t, 'height') and hasattr(t, 'width'):
            details += f" → resize to ({t.height}, {t.width})"
        if hasattr(t, 'scale'):
            details += f" → crop scale={t.scale}"
        if hasattr(t, 'brightness'):
            details += f" → jitter(b={t.brightness}, c={t.contrast}, s={t.saturation}, h={t.hue})"
        if hasattr(t, 'normalization_modes') and t.normalization_modes:
            details += f" → norm_modes={t.normalization_modes}"
        if hasattr(t, 'max_state_dim'):
            details += f" → max_state={t.max_state_dim}, max_action={t.max_action_dim}"
        if hasattr(t, 'state_horizon'):
            details += f" → state_h={t.state_horizon}, action_h={t.action_horizon}"
        print(f"  [{i}] {cls_name}{details}")

    # Print modality config
    print(f"\n--- Data Config: {data_config_cls.__class__.__name__} ---")
    print(f"  video_keys:       {data_config_cls.video_keys}")
    print(f"  state_keys:       {data_config_cls.state_keys}")
    print(f"  action_keys:      {data_config_cls.action_keys}")
    print(f"  observation_indices: {data_config_cls.observation_indices}")
    print(f"  action_indices:      {data_config_cls.action_indices}")

    # Print dataset metadata (normalization stats)
    metadata = dataset.metadata
    print(f"\n--- Normalization Statistics (min_max) ---")
    for modality in ["state", "action"]:
        stats = getattr(metadata.statistics, modality, {})
        for subkey, vals in stats.items():
            if hasattr(vals, 'min') and hasattr(vals, 'max'):
                print(f"  {modality}.{subkey}:")
                print(f"    min: {[round(v, 4) for v in vals.min]}")
                print(f"    max: {[round(v, 4) for v in vals.max]}")

    # Print video info
    video_meta = metadata.modalities.video
    for vk, vm in video_meta.items():
        print(f"\n--- Video: {vk} ---")
        print(f"  original resolution: {vm.resolution} (W, H)")
        print(f"  channels: {vm.channels}, fps: {vm.fps}")

    print("=" * 70)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load dataset
    print("Loading dataset...")
    dataset, data_config_cls, modality_configs = load_dataset(args)

    # 2. Print transform pipeline info
    print_transform_pipeline_info(data_config_cls, dataset)

    # 3. Load policy (local or remote)
    policy = None
    if args.use_server:
        print(f"\nConnecting to inference server at {args.host}:{args.port}...")
        policy = load_policy_remote(args)
    elif args.model_path:
        print(f"\nLoading model from {args.model_path}...")
        policy = load_policy_local(args, modality_configs)

    # 4. Select samples
    np.random.seed(42)

    if args.trajectory_mode:
        # --- Trajectory mode: visualize full trajectory with overlaid chunks ---
        if args.episode_idx is not None:
            traj_ids = [args.episode_idx]
        else:
            traj_ids = [dataset.trajectory_ids[0]]

        for traj_id in traj_ids:
            visualize_trajectory(
                dataset, traj_id, data_config_cls, args.output_dir,
                policy=policy, chunk_stride=args.chunk_stride
            )
        print(f"\n✅ Done! Trajectory visualizations saved to: {args.output_dir}")
        return

    # --- Sample mode: individual action chunk comparisons ---
    if args.episode_idx is not None:
        # Use specific episode
        traj_idx = np.where(dataset.trajectory_ids == args.episode_idx)[0][0]
        traj_len = dataset.trajectory_lengths[traj_idx]
        if args.step_idx is not None:
            base_indices = [args.step_idx]
        else:
            # Avoid edges where action chunk runs off the end
            action_horizon = len(data_config_cls.action_indices)
            safe_len = max(1, traj_len - action_horizon)
            base_indices = np.random.choice(safe_len, min(args.num_samples, safe_len), replace=False)
        samples = [(args.episode_idx, bi) for bi in base_indices]
    else:
        # Random samples from the entire dataset
        total = len(dataset)
        indices = np.random.choice(total, min(args.num_samples, total), replace=False)
        samples = [dataset.all_steps[i] for i in indices]

    print(f"\nVisualizing {len(samples)} samples...")
    print(f"Output directory: {args.output_dir}")

    # 5. Visualize each sample
    for s_idx, (traj_id, base_idx) in enumerate(samples):
        print(f"\n--- Sample {s_idx}: trajectory={traj_id}, step={base_idx} ---")

        # Get raw data
        raw_data = get_raw_step_data(dataset, traj_id, base_idx)

        # Print raw data shapes
        for k, v in raw_data.items():
            if isinstance(v, np.ndarray):
                print(f"  raw {k}: shape={v.shape}, dtype={v.dtype}, "
                      f"min={v.min():.4f}, max={v.max():.4f}")
            else:
                print(f"  raw {k}: {v}")

        # GT actions
        gt_actions = get_gt_actions(raw_data, data_config_cls)

        # Predicted actions (if policy available)
        pred_actions = None
        if policy is not None:
            obs = prepare_obs_for_server(raw_data, data_config_cls)
            print("  Running inference...")
            try:
                pred_actions = policy.get_action(obs)
                for k, v in pred_actions.items():
                    if isinstance(v, np.ndarray):
                        print(f"  pred {k}: shape={v.shape}")
            except Exception as e:
                print(f"  ⚠ Inference failed: {e}")
                pred_actions = None

        # Visualize
        visualize_sample(
            s_idx, traj_id, base_idx, raw_data, gt_actions, pred_actions,
            data_config_cls, args.output_dir
        )
        print(f"  ✓ Saved to {args.output_dir}/sample_{s_idx:03d}_*.png")

    # 6. Summary: overlay GT action chunks from sampled steps on full trajectory
    if args.episode_idx is not None:
        traj_id = args.episode_idx
        full_actions = get_full_trajectory_actions(dataset, traj_id, data_config_cls)
        action_horizon = len(data_config_cls.action_indices)

        for key in data_config_cls.action_keys:
            full_gt = full_actions[key]
            T_total, D = full_gt.shape
            ncols = min(D, 4)
            nrows = (D + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
            time_all = np.arange(T_total)

            for d in range(D):
                r, c = divmod(d, ncols)
                ax = axes[r, c]
                # Full trajectory GT
                ax.plot(time_all, full_gt[:, d], 'k-', linewidth=0.8, alpha=0.4, label="Full GT")
                # Overlay sampled chunks
                colors_list = plt.colormaps['tab10'](np.linspace(0, 1, len(samples)))
                for si, (_, base_idx) in enumerate(samples):
                    chunk_time = np.arange(base_idx, min(base_idx + action_horizon, T_total))
                    chunk_gt = full_gt[chunk_time, d]
                    ax.plot(chunk_time, chunk_gt, '-', color=colors_list[si], linewidth=2.0,
                            alpha=0.8, label=f"step={base_idx}" if d == 0 else None)
                    ax.axvline(base_idx, color=colors_list[si], linestyle=':', alpha=0.5)

                ax.set_title(f"dim {d}", fontsize=10)
                ax.set_xlabel("global time step")
                ax.grid(True, alpha=0.3)
                if d == 0:
                    ax.legend(fontsize=7)

            for d in range(D, nrows * ncols):
                r, c = divmod(d, ncols)
                axes[r, c].set_visible(False)

            safe_key = key.replace(".", "_")
            fig.suptitle(
                f"Sampled Chunks on Trajectory {traj_id} — {key}",
                fontsize=13, fontweight='bold'
            )
            fig.tight_layout()
            fig.savefig(os.path.join(args.output_dir, f"summary_{safe_key}.png"), dpi=150)
            plt.close(fig)

    print(f"\n✅ Done! All visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
