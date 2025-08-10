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

    def __init__(self, g1_metadata, gr1_metadata, host: str = "localhost", port: int = 5555):
        """Initialize the simulation client with server connection details."""
        super().__init__(host=host, port=port)
        self.env = None

        # G1 modality tansform
        g1_data_config = DATA_CONFIG_MAP["dex31_g1_arms_waist"]
        self.g1_action_transform = g1_data_config.action_transform()
        self.g1_action_transform.set_metadata(g1_metadata)

        # GR1 modality transform
        gr1_data_config = DATA_CONFIG_MAP["fourier_gr1_arms_waist"]
        self.gr1_action_transform = gr1_data_config.action_transform()
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
        normalized_action = self.gr1_action_transform.apply(gr1_action)
        #normalized_action = self.gr1_action_transform.unapply({"action": gr1_action})
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
            # obs[key_obs] : (num_env, ...)
            # actions[key_joint] : (num_env, 16, dim_joint)

            # Step the environment
            next_obs, rewards, terminations, truncations, env_infos = self.env.step(actions)

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
        gr1_action = {k:a[timestep:timestep+1] for k,a in trajectory.items()}
        if False:
            print()
            print("gr1 action:")
            print(gr1_action)
            for k in gr1_action.keys():
                print(k, gr1_action[k].shape, type(gr1_action[k]))
            exit()
        g1_action = self.transfer_action(gr1_action)
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
