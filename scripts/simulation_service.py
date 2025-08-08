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

import pickle
import numpy as np
from pathlib import Path

from gr00t.eval.robot import RobotInferenceServer
from gr00t.eval.simulation import (
    MultiStepConfig,
    SimulationConfig,
    SimulationInferenceClient,
    VideoConfig,
)
from gr00t.model.policy import Gr00tPolicy

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
    # save trajectory data
    parser.add_argument("--save_data", action="store_true", help="Save trajectory data.")
    parser.add_argument("--save_dir", type=str, default="data/trajectories/")
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
        simulation_client = SimulationInferenceClient(host=args.host, port=args.port)

        print("Available modality configs:")
        modality_config = simulation_client.get_modality_config()
        print(modality_config.keys())

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
            save_data=args.save_data,
        )

        # Run the simulation
        print(f"Running simulation for {args.env_name}...")
        if args.save_data:
            env_name, episode_successes, data_actions, data_obs = simulation_client.run_simulation(config)

            # save trajectory data
            save_dir = Path(args.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            num_data = len([f for f in os.listdir(save_dir) if "action_" in f])
            with open(save_dir / ("action_%03d.pkl"%num_data), 'wb') as f:
                pickle.dump(data_actions, f)
            with open(save_dir / ("obs_%03d.pkl"%num_data), 'wb') as f:
                pickle.dump(data_obs, f)
            #np.savez_compressed(save_dir / ("action_%03d.npz"%num_data), **data_actions)
            #np.savez_compressed(save_dir / ("obs_%03d.npz"%num_data), **data_obs)
        else:
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
