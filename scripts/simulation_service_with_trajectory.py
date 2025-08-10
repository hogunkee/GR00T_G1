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

import argparse
import os

import json
import pickle
import numpy as np
from pathlib import Path

from gr00t.eval.robot import RobotInferenceServer
from gr00t.eval.simulation_with_trajectory import (
    MultiStepConfig,
    SimulationConfig,
    SimulationInferenceClient,
    VideoConfig,
)
from gr00t.model.policy import Gr00tPolicy
from gr00t.data.schema import DatasetMetadata
from gr00t.data.embodiment_tags import EmbodimentTag

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model checkpoint directory.",
        default="<PATH_TO_YOUR_MODEL>",  # change this to your model path
    )
    parser.add_argument(
        "--embodiment_tag",
        type=str,
        help="The embodiment tag for the model.",
        default="<EMBODIMENT_TAG>",  # change this to your embodiment tag
    )
    parser.add_argument(
        "--env_name",
        type=str,
        help="Name of the environment to run.",
        default="<ENV_NAME>",  # change this to your environment name
    )
    parser.add_argument("--port", type=int, help="Port number for the server.", default=5555)
    parser.add_argument(
        "--host", type=str, help="Host address for the server.", default="localhost"
    )
    parser.add_argument("--video_dir", type=str, help="Directory to save videos.", default=None)
    parser.add_argument("--n_episodes", type=int, help="Number of episodes to run.", default=2)
    parser.add_argument("--n_envs", type=int, help="Number of parallel environments.", default=1)
    parser.add_argument(
        "--n_action_steps",
        type=int,
        help="Number of action steps per environment step.",
        default=16,
    )
    parser.add_argument(
        "--max_episode_steps", type=int, help="Maximum number of steps per episode.", default=1440
    )
    # server mode
    parser.add_argument("--server", action="store_true", help="Run the server.")
    # client mode
    parser.add_argument("--client", action="store_true", help="Run the client")
    # log
    parser.add_argument("--log", action="store_true", help="Save the log.")
    # multi-cameras  
    parser.add_argument("--multi_video", action="store_true", help="Save the multi camera images.")
    # load trajectory data
    parser.add_argument("--traj_path", type=str, default="data/trajectories/action_000.pkl")
    args = parser.parse_args()

    if args.server:
        # Create a policy
        policy = Gr00tPolicy(
            model_path=args.model_path,
            embodiment_tag=args.embodiment_tag,
        )

        # Start the server
        server = RobotInferenceServer(policy, port=args.port)
        server.run()

    elif args.client:
        # Create a simulation client
        metadata_path = os.path.join(args.model_path, "experiment_cfg", "metadata.json")
        with open(metadata_path, "r") as f:
            metadatas = json.load(f)
        # Get metadata for the specific embodiment
        g1_metadata_dict = metadatas.get(EmbodimentTag("g1").value)
        g1_metadata = DatasetMetadata.model_validate(g1_metadata_dict)
        gr1_metadata_dict = metadatas.get(EmbodimentTag("gr1").value)
        gr1_metadata = DatasetMetadata.model_validate(gr1_metadata_dict)
        simulation_client = SimulationInferenceClient(g1_metadata, gr1_metadata, host=args.host, port=args.port)

        # Create simulation configuration
        config = SimulationConfig(
            env_name=args.env_name,
            n_episodes=args.n_episodes,
            n_envs=args.n_envs,
            video=VideoConfig(video_dir=args.video_dir),
            multistep=MultiStepConfig(
                n_action_steps=args.n_action_steps, max_episode_steps=args.max_episode_steps
            ),
            multi_video=args.multi_video,
            traj_path=args.traj_path,
        )

        # Run the simulation
        print(f"Running simulation for {args.env_name}...")
        env_name, episode_successes = simulation_client.run_simulation(config)

        # Print results
        print(f"Results for {env_name}:")
        print(f"Success rate: {np.mean(episode_successes):.2f}")
        if args.log:
            with open(os.path.join(args.video_dir, 'result.txt'), 'w') as f:
                f.write(f"Running simulation for {args.env_name}...\n")
                f.write(f"Results for {env_name}:\n")
                f.write(f"Success rate: {np.mean(episode_successes):.2f}")

    else:
        raise ValueError("Please specify either --server or --client")
