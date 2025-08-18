# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pickle
import torch

# Required for robocasa environments
import robocasa  # noqa: F401
import robosuite  # noqa: F401
from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401

from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.data.dataset import ModalityConfig
from gr00t.eval.service import BaseInferenceClient
from gr00t.eval.wrappers.multistep_wrapper import MultiStepWrapper
from gr00t.eval.wrappers.video_recording_wrapper import (
    VideoRecorder,
    VideoRecordingWrapper,
)
from gr00t.eval.wrappers.multi_video_recording_wrapper import (
    MultiVideoRecorder,
    MultiVideoRecordingWrapper,
)
from gr00t.model.policy import BasePolicy

# from gymnasium.envs.registration import registry

# print("Available environments:")
# for env_spec in registry.values():
#     print(env_spec.id)


@dataclass
class VideoConfig:
    """Configuration for video recording settings."""

    video_dir: Optional[str] = None
    steps_per_render: int = 2
    fps: int = 10
    codec: str = "h264"
    input_pix_fmt: str = "rgb24"
    crf: int = 22
    thread_type: str = "FRAME"
    thread_count: int = 1


@dataclass
class MultiStepConfig:
    """Configuration for multi-step environment settings."""

    video_delta_indices: np.ndarray = field(default=np.array([0]))
    state_delta_indices: np.ndarray = field(default=np.array([0]))
    n_action_steps: int = 16
    max_episode_steps: int = 1440


@dataclass
class SimulationConfig:
    """Main configuration for simulation environment."""

    env_name: str
    n_episodes: int = 2
    n_envs: int = 1
    video: VideoConfig = field(default_factory=VideoConfig)
    multistep: MultiStepConfig = field(default_factory=MultiStepConfig)
    multi_video: bool = False
    traj_path: str = ''


class SimulationInferenceClient(BaseInferenceClient, BasePolicy):
    """Client for running simulations and communicating with the inference server."""

    def __init__(
        self,
        g1_metadata: Optional[dict] = None,
        gr1_metadata: Optional[dict] = None,
        host: str = "localhost",
        port: int = 5555,
        direct_passthrough: bool = False,
        fixed_hand: float = 0.0,
        fixed_wrist: float = 0.0,
        fixed_waist: float = 0.0,
        debug: bool = False,
    ):
        """Initialize the simulation client with server connection details.

        If direct_passthrough is True (or metadata is missing), actions from the trajectory
        will be passed directly to the environment for arms, while hands and waist are
        set to fixed constants. This bypasses metadata-based transforms.
        """
        super().__init__(host=host, port=port)
        self.env = None

        self.direct_passthrough = bool(direct_passthrough or (g1_metadata is None) or (gr1_metadata is None))
        self.fixed_hand = float(fixed_hand)
        self.fixed_waist = float(fixed_waist)
        self.fixed_wrist = float(fixed_wrist)
        self.debug = bool(debug)

        if not self.direct_passthrough:
            # G1 modality transform
            self.g1_data_config = DATA_CONFIG_MAP["dex31_g1_arms_waist"]
            self.g1_action_transform = self.g1_data_config.action_transform()
            self.g1_action_transform.set_metadata(g1_metadata)

            # GR1 modality transform
            self.gr1_data_config = DATA_CONFIG_MAP["fourier_gr1_arms_waist"]
            self.gr1_action_transform = self.gr1_data_config.action_transform()
            self.gr1_action_transform.set_metadata(gr1_metadata)

    def load_trajectory(self, traj_path):
        # load trajectory data
        with open(traj_path, 'rb') as f:
            self.trajectories = pickle.load(f)
            self.episodes = sorted(list(self.trajectories.keys()))

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        return

    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        return

    def transfer_action(self, gr1_action):
        if self.direct_passthrough:
            # Directly map arms from GR1 trajectory to G1 env, fix hands and waist
            # gr1_action values are shaped (num_env=1, n_action_steps, dim)
            left_arm = gr1_action.get("action.left_arm")
            right_arm = gr1_action.get("action.right_arm")

            if left_arm is None or right_arm is None:
                raise ValueError("Trajectory must contain 'action.left_arm' and 'action.right_arm' for direct passthrough mode")

            # --- 동적으로 env가 기대하는 DoF 읽기 ---
            space = getattr(self.env, "single_action_space", None)
            if space is None:
                raise RuntimeError("VectorEnv not initialized or single_action_space unavailable.")

            expected = {
                "action.left_arm": space["action.left_arm"].shape[-1],
                "action.right_arm": space["action.right_arm"].shape[-1],
                "action.left_hand": space["action.left_hand"].shape[-1],
                "action.right_hand": space["action.right_hand"].shape[-1],
                "action.waist": space["action.waist"].shape[-1],
            }

            # trajectory에서 스텝 수
            n_action_steps = left_arm.shape[1]

            # 타입/복사 정책
            left_arm = left_arm.astype(np.float32, copy=False)
            right_arm = right_arm.astype(np.float32, copy=False)

            # --- 팔 차원 검증 ---
            if left_arm.shape[-1] != expected["action.left_arm"]:
                raise ValueError(f"left_arm dim mismatch: traj={left_arm.shape[-1]} vs env={expected['action.left_arm']}")
            if right_arm.shape[-1] != expected["action.right_arm"]:
                raise ValueError(f"right_arm dim mismatch: traj={right_arm.shape[-1]} vs env={expected['action.right_arm']}")

            # --- GR1→G1 관절 순서 및 범위 보정 ---
            # GR1 순서: [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_pitch, wrist_yaw, wrist_roll, wrist_pitch]
            # G1 순서:  [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_pitch, wrist_roll, wrist_pitch, wrist_yaw]
            
            # 1. 손목 관절 순서 재배치 (wrist_yaw와 wrist_roll 위치 바뀜)
            left_arm_reordered = left_arm.copy()
            right_arm_reordered = right_arm.copy()
            
            # GR1[4]=wrist_yaw → G1[6]=wrist_yaw
            # GR1[5]=wrist_roll → G1[4]=wrist_roll  
            # GR1[6]=wrist_pitch → G1[5]=wrist_pitch
            left_arm_reordered[..., 4] = left_arm[..., 5]  # wrist_roll
            left_arm_reordered[..., 5] = left_arm[..., 6]  # wrist_pitch
            left_arm_reordered[..., 6] = left_arm[..., 4]  # wrist_yaw
            
            right_arm_reordered[..., 4] = right_arm[..., 5]  # wrist_roll
            right_arm_reordered[..., 5] = right_arm[..., 6]  # wrist_pitch
            right_arm_reordered[..., 6] = right_arm[..., 4]  # wrist_yaw
            
            
            left_arm = left_arm_reordered
            right_arm = right_arm_reordered
            left_arm[..., 3] = left_arm[..., 3] + np.pi/2
            right_arm[..., 3] = right_arm[..., 3] + np.pi/2

            # --- 손/허리 배열을 env 기대 DoF로 생성 ---
            left_hand = np.full((1, n_action_steps, expected["action.left_hand"]), self.fixed_hand, dtype=np.float32)
            right_hand = np.full((1, n_action_steps, expected["action.right_hand"]), self.fixed_hand, dtype=np.float32)
            waist = np.full((1, n_action_steps, expected["action.waist"]), self.fixed_waist, dtype=np.float32)

            g1_action = {
                "action.left_arm": left_arm,
                "action.right_arm": right_arm,
                "action.left_hand": left_hand,
                "action.right_hand": right_hand,
                "action.waist": waist,
            }

            # --- 최종 검증(안전망) ---
            for k, v in g1_action.items():
                exp = expected[k]
                if v.shape[1] != n_action_steps or v.shape[-1] != exp:
                    raise ValueError(f"{k} shape mismatch: got {v.shape}, expected (1, {n_action_steps}, {exp})")

            return g1_action
        else:
            data = self.gr1_action_transform.apply(gr1_action)
            normalized_action = torch.cat([data.pop(key) for key in self.gr1_data_config.action_keys], dim=-1)
            # normalized_action = self.gr1_action_transform.unapply({"action": gr1_action})
            g1_action = self.g1_action_transform.unapply({"action": normalized_action})
            return g1_action

    def setup_environment(self, config: SimulationConfig) -> gym.vector.VectorEnv:
        """Set up the simulation environment based on the provided configuration."""
        # Create environment functions for each parallel environment
        env_fns = [partial(_create_single_env, config=config, idx=i) for i in range(config.n_envs)]
        # Create vector environment (sync for single env, async for multiple)
        if config.n_envs == 1:
            return gym.vector.SyncVectorEnv(env_fns)
        else:
            return gym.vector.AsyncVectorEnv(
                env_fns,
                shared_memory=False,
                context="spawn",
            )

    def run_simulation(self, config: SimulationConfig) -> Tuple[str, List[bool]]:
        """Run the simulation for the specified number of episodes."""
        self.load_trajectory(config.traj_path)
        start_time = time.time()
        print(
            f"Running {config.n_episodes} episodes for {config.env_name} with {config.n_envs} environments"
        )
        # Set up the environment
        self.env = self.setup_environment(config)
        # Initialize tracking variables
        episode_lengths = []
        current_rewards = [0] * config.n_envs
        current_lengths = [0] * config.n_envs
        completed_episodes = 0
        current_successes = [False] * config.n_envs
        episode_successes = []

        # Initial environment reset
        obs, _ = self.env.reset()
        ep_idx = 0
        timestep = 0
        # Main simulation loop
        while ep_idx < config.n_episodes:
            # Process observations and get actions from the server
            actions = self._get_actions_from_trajectory(ep_idx, timestep)
            
            # Check if we've reached the end of trajectory data
            if actions is None:
                print(f"[INFO] Reached end of trajectory data at timestep {timestep}, ending episode {ep_idx}")
                # Force episode termination
                terminations = [True] * config.n_envs
                truncations = [False] * config.n_envs
                next_obs = obs
                rewards = [0.0] * config.n_envs
                env_infos = {"success": [[False]] * config.n_envs}
            else:
                # obs[key_obs] : (num_env, ...)
                # actions[key_joint] : (num_env, 16, dim_joint)

                # Step the environment
                if self.debug:
                    print(f"[DEBUG] Stepping environment with actions:")
                    for k, v in actions.items():
                        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                try:
                    next_obs, rewards, terminations, truncations, env_infos = self.env.step(actions)
                    if self.debug:
                        print(f"[DEBUG] Environment step successful")
                except Exception as e:
                    print(f"[ERROR] Environment step failed: {e}")
                    print(f"[DEBUG] Action details:")
                    for k, v in actions.items():
                        print(f"  {k}: shape={v.shape}, dtype={v.dtype}, min={v.min()}, max={v.max()}")
                    raise

            # Update episode tracking
            current_successes[0] |= bool(env_infos["success"][0][0])
            current_rewards[0] += rewards[0]
            current_lengths[0] += 1
            timestep += 1

            # If episode ended, store results
            if terminations[0] or truncations[0]:
                episode_lengths.append(current_lengths[0])
                episode_successes.append(current_successes[0])
                current_successes[0] = False
                completed_episodes += 1
                # Reset trackers for this environment
                current_rewards[0] = 0
                current_lengths[0] = 0
                ep_idx += 1
                timestep = 0

            obs = next_obs

        # Clean up
        self.env.reset()
        self.env.close()
        self.env = None

        print(
            f"Collecting {config.n_episodes} episodes took {time.time() - start_time:.2f} seconds"
        )
        assert (
            len(episode_successes) >= config.n_episodes
        ), f"Expected at least {config.n_episodes} episodes, got {len(episode_successes)}"
        return config.env_name, episode_successes

    def _get_actions_from_trajectory(self, ep_idx, timestep) -> Dict[str, Any]:
        episode = self.episodes[ep_idx]
        trajectory = self.trajectories[episode]
        
        # Check if we've reached the end of trajectory data
        max_timesteps = min(len(v) for v in trajectory.values())
        if timestep >= max_timesteps:
            # Return None to signal episode should end
            return None
            
        gr1_action = {k:a[timestep:timestep+1] for k,a in trajectory.items()}
        if False:
            print()
            print("gr1 action:")
            print(gr1_action)
            for k in gr1_action.keys():
                print(k, gr1_action[k].shape, type(gr1_action[k]))
            exit()
        g1_action = self.transfer_action(gr1_action)
        if self.debug:
            # Print shapes as seen by vector env (batch dim should be 1); multistep wrapper expects (n_action_steps, dim)
            print("[DEBUG] action shapes before env.step:")
            for k,v in g1_action.items():
                try:
                    print(k, v.shape)
                except Exception:
                    print(k, type(v))
        return g1_action


def _create_single_env(config: SimulationConfig, idx: int) -> gym.Env:
    """Create a single environment with appropriate wrappers."""
    # Create base environment
    env = gym.make(config.env_name, enable_render=True)
    # Add video recording wrapper if needed (only for the first environment)
    if config.video.video_dir is not None:
        if config.multi_video:
            video_recorder = MultiVideoRecorder.create_h264(
                fps=config.video.fps,
                codec=config.video.codec,
                input_pix_fmt=config.video.input_pix_fmt,
                crf=config.video.crf,
                thread_type=config.video.thread_type,
                thread_count=config.video.thread_count,
            )
            env = MultiVideoRecordingWrapper(
                env,
                video_recorder,
                video_dir=Path(config.video.video_dir),
                steps_per_render=config.video.steps_per_render,
            )
        else:
            video_recorder = VideoRecorder.create_h264(
                fps=config.video.fps,
                codec=config.video.codec,
                input_pix_fmt=config.video.input_pix_fmt,
                crf=config.video.crf,
                thread_type=config.video.thread_type,
                thread_count=config.video.thread_count,
            )
            env = VideoRecordingWrapper(
                env,
                video_recorder,
                video_dir=Path(config.video.video_dir),
                steps_per_render=config.video.steps_per_render,
            )
    # Add multi-step wrapper
    env = MultiStepWrapper(
        env,
        video_delta_indices=config.multistep.video_delta_indices,
        state_delta_indices=config.multistep.state_delta_indices,
        n_action_steps=config.multistep.n_action_steps,
        max_episode_steps=config.multistep.max_episode_steps,
    )
    return env


def run_evaluation(
    env_name: str,
    host: str = "localhost",
    port: int = 5555,
    video_dir: Optional[str] = None,
    n_episodes: int = 2,
    n_envs: int = 1,
    n_action_steps: int = 2,
    max_episode_steps: int = 100,
) -> Tuple[str, List[bool]]:
    """
    Simple entry point to run a simulation evaluation.
    Args:
        env_name: Name of the environment to run
        host: Hostname of the inference server
        port: Port of the inference server
        video_dir: Directory to save videos (None for no videos)
        n_episodes: Number of episodes to run
        n_envs: Number of parallel environments
        n_action_steps: Number of action steps per environment step
        max_episode_steps: Maximum number of steps per episode
    Returns:
        Tuple of environment name and list of episode success flags
    """
    # Create configuration
    config = SimulationConfig(
        env_name=env_name,
        n_episodes=n_episodes,
        n_envs=n_envs,
        video=VideoConfig(video_dir=video_dir),
        multistep=MultiStepConfig(
            n_action_steps=n_action_steps, max_episode_steps=max_episode_steps
        ),
    )
    # Create client and run simulation
    client = SimulationInferenceClient(host=host, port=port)
    results = client.run_simulation(config)
    # Print results
    print(f"Results for {env_name}:")
    print(f"Success rate: {np.mean(results[1]):.2f}")
    return results


if __name__ == "__main__":
    # Example usage
    run_evaluation(
        env_name="robocasa_gr1_arms_only_fourier_hands/TwoArmPnPCarPartBrakepedal_GR1ArmsOnlyFourierHands_Env",
        host="localhost",
        port=5555,
        video_dir="./videos",
    )