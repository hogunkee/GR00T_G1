#!/usr/bin/env python3
"""
G1 Robot Tabletop Environment Simulation with Multi-view Video Recording

This script is based on simulation_with_trajectory.py and creates a simulation where 
the G1ArmsAndWaistDex31Hands robot stands still in a tabletop environment while 
recording multi-view videos.

Usage:
    python g1_tabletop_simulation_trajectory.py --video_dir ./videos --multiview --steps 500
"""

import argparse
import time
import sys
import os
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
import gymnasium as gym
from functools import partial

# Add the robocasa and robosuite paths to enable imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'robocasa_g1'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'robosuite_g1'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gr00t_g1'))

import robocasa  # Register robocasa environments
import robosuite
from robocasa.utils.gym_utils import GrootRoboCasaEnv

# Import video recording utilities from simulation_with_trajectory
from gr00t.eval.wrappers.video_recording_wrapper import (
    VideoRecorder,
    VideoRecordingWrapper,
)
from gr00t.eval.wrappers.multi_video_recording_wrapper import (
    MultiVideoRecorder,
    MultiVideoRecordingWrapper,
)
from gr00t.eval.wrappers.multistep_wrapper import MultiStepWrapper


@dataclass
class VideoConfig:
    """Configuration for video recording settings."""

    video_dir: str = None
    steps_per_render: int = 2
    fps: int = 30
    codec: str = "h264"
    input_pix_fmt: str = "rgb24"
    crf: int = 22
    thread_type: str = "FRAME"
    thread_count: int = 1


@dataclass
class MultiStepConfig:
    """Configuration for multi-step environment settings."""

    video_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    state_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    n_action_steps: int = 16
    max_episode_steps: int = 1440


@dataclass
class SimulationConfig:
    """Main configuration for simulation environment."""

    env_name: str
    n_episodes: int = 1
    n_envs: int = 1
    video: VideoConfig = field(default_factory=VideoConfig)
    multistep: MultiStepConfig = field(default_factory=MultiStepConfig)
    multi_video: bool = False


class G1TabletopSimulationWithTrajectory:
    """G1 robot tabletop simulation using trajectory-based infrastructure."""
    
    def __init__(self, robot_name="G1ArmsAndWaistDex31Hands"):
        """
        Initialize the G1 tabletop simulation.
        
        Args:
            robot_name (str): Name of the G1 robot variant to use
        """
        self.robot_name = robot_name
        self.env = None
        
        print(f"Initializing G1 Tabletop Simulation with robot: {robot_name}")
        
    def setup_environment(self, config: SimulationConfig) -> gym.vector.VectorEnv:
        """Set up the simulation environment based on the provided configuration."""
        # Create environment functions for each parallel environment
        env_fns = [partial(self._create_single_env, config=config, idx=i) for i in range(config.n_envs)]
        # Create vector environment (sync for single env)
        if config.n_envs == 1:
            return gym.vector.SyncVectorEnv(env_fns)
        else:
            return gym.vector.AsyncVectorEnv(
                env_fns,
                shared_memory=False,
                context="spawn",
            )

    def _create_single_env(self, config: SimulationConfig, idx: int) -> gym.Env:
        """Create a single environment with appropriate wrappers."""
        print(f"Creating environment: {config.env_name}")
        
        # Create base environment - this will use the robocasa gym registration
        env = gym.make(config.env_name, enable_render=True)
        
        print(f"✓ Base environment created successfully")
        
        # Add video recording wrapper if needed
        if config.video.video_dir is not None:
            print(f"Setting up video recording in: {config.video.video_dir}")
            
            # Create video directory
            video_path = Path(config.video.video_dir)
            video_path.mkdir(parents=True, exist_ok=True)
            
            if config.multi_video:
                print("Setting up multi-view video recording...")
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
                    video_dir=video_path,
                    steps_per_render=config.video.steps_per_render,
                )
                print("✓ Multi-view video recording wrapper added")
            else:
                print("Setting up single video recording...")
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
                    video_dir=video_path,
                    steps_per_render=config.video.steps_per_render,
                )
                print("✓ Single video recording wrapper added")
        
        # Add multi-step wrapper with disabled termination
        env = MultiStepWrapper(
            env,
            video_delta_indices=config.multistep.video_delta_indices,
            state_delta_indices=config.multistep.state_delta_indices,
            n_action_steps=config.multistep.n_action_steps,
            max_episode_steps=None,  # Disable episode step limit
        )
        print("✓ Multi-step wrapper added")
        
        return env

    def get_static_action(self, env):
        """
        Generate a static action that keeps the robot in place.
        
        Args:
            env: The vectorized environment
            
        Returns:
            dict: Action dictionary with zero values to maintain current pose
        """
        # Get action space from the vector environment
        action_space = env.single_action_space
        
        # Create zero actions for all action components
        action = {}
        for key, space in action_space.spaces.items():
            # Create array of zeros with the right shape for multistep (n_action_steps, dim)
            if hasattr(space, 'shape'):
                # For vector env, we need (n_envs, n_action_steps, dim)
                action_shape = (1, 16, space.shape[-1])  # 1 env, 16 action steps, action dim
                action[key] = np.zeros(action_shape, dtype=np.float32)
        
        return action

    def run_simulation(self, config: SimulationConfig) -> None:
        """Run the simulation for the specified number of episodes."""
        start_time = time.time()
        print(f"Running {config.n_episodes} episodes for {config.env_name} with {config.n_envs} environments")

        # Set up the environment
        self.env = self.setup_environment(config)

        # Print environment info (이 부분은 이전과 동일합니다)
        if hasattr(self.env, 'envs') and len(self.env.envs) > 0:
            base_env = self.env.envs[0]
            while hasattr(base_env, 'env'):
                base_env = base_env.env
            if hasattr(base_env, 'robots') and len(base_env.robots) > 0:
                robot = base_env.robots[0]
                print(f"✓ Robot loaded: {robot.robot_model.naming_prefix}")
                print(f"  - Total DOF: {robot.dof}")
                if hasattr(robot, 'gripper') and robot.gripper:
                    left_gripper = robot.gripper.get('left', None)
                    right_gripper = robot.gripper.get('right', None)
                    if left_gripper:
                        print(f"  - Left hand DOF: {left_gripper.dof}")
                    if right_gripper:
                        print(f"  - Right hand DOF: {right_gripper.dof}")
                if hasattr(robot, 'joints'):
                    print(f"  - Total joints: {len(robot.joints)}")
                
                # Check robot placement and offset
                if hasattr(robot, 'init_qpos'):
                    print(f"  - Robot init_qpos: {robot.init_qpos}")
                if hasattr(robot, 'base_pos'):
                    print(f"  - Robot base_pos: {robot.base_pos}")
                if hasattr(robot.robot_model, 'base_pos'):
                    print(f"  - Robot model base_pos: {robot.robot_model.base_pos}")
                if hasattr(robot.robot_model, 'base_ori'):
                    print(f"  - Robot model base_ori: {robot.robot_model.base_ori}")

        print(f"Action space: {self.env.single_action_space}")

        # Main simulation loop
        for ep_idx in range(config.n_episodes):
            print(f"\n--- Starting Episode {ep_idx + 1}/{config.n_episodes} ---")

            # Initial environment reset for the episode
            obs, _ = self.env.reset()
            print("✓ Environment reset completed.")
            
            # <<-- 변경점 1: 초기 상태(timestep 0) 출력 -->>
            self._print_states(0)

            # Determine max timesteps for this episode
            # 'args.steps'를 사용하도록 수정합니다.
            max_timesteps = config.multistep.max_episode_steps

            for timestep in range(max_timesteps):
                # Get static action (all zeros to stay in place)
                actions = self.get_static_action(self.env)

                # Step the environment
                try:
                    next_obs, rewards, terminations, truncations, env_infos = self.env.step(actions)
                    obs = next_obs

                    # <<-- 변경점 2: 매 스텝 후의 상태 출력 -->>
                    # timestep은 0부터 시작하므로 +1을 해줘서 1부터 20까지 출력되게 합니다.
                    self._print_states(timestep + 1)
                    
                    # Debug termination information (이전과 동일, 필요시 주석 해제)
                    if terminations[0] or truncations[0]:
                        # print(f"[DEBUG] Episode terminated at step {timestep + 1}")
                        # print(f"  - Termination: {terminations[0]}, Truncation: {truncations[0]}")
                        break # 조기 종료 조건 발생 시 루프 탈출
                
                except Exception as e:
                    print(f"✗ Environment step failed: {e}")
                    for k, v in actions.items():
                        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                    raise
        
        # Clean up
        print("\nFinalizing simulation...")
        self.env.close()
        self.env = None

        print(f"✓ Simulation completed in {time.time() - start_time:.2f} seconds")

        # Video finalization message
        if config.video.video_dir:
            print(f"✓ Videos saved to: {config.video.video_dir}")


    def _print_states(self, timestep):
        """Print cup world pose, joint positions, and robot base position."""
        try:
            # Get the base robosuite environment
            base_env = self.env.envs[0]
            while hasattr(base_env, 'env'):
                base_env = base_env.env
            
            print(f"\n--- States at timestep {timestep} ---")
            
            # 1. Cup world pose T_world_cup: [x, y, z, qx, qy, qz, qw]
            target_obj_name = None
            
            # Find the body name associated with the first object in the environment
            # TabletopObjectShowcase 에서는 보통 'obj_main' 입니다.
            if hasattr(base_env, 'objects') and base_env.objects:
                try:
                    # base_env.objects가 dict인 경우
                    if isinstance(base_env.objects, dict):
                        if len(base_env.objects) > 0:
                            first_obj = list(base_env.objects.values())[0]
                        else:
                            first_obj = None
                    # base_env.objects가 list인 경우
                    elif isinstance(base_env.objects, list):
                        first_obj = base_env.objects[0] if len(base_env.objects) > 0 else None
                    else:
                        first_obj = None
                        
                    if first_obj:
                        # robosuite 에서는 종종 객체의 실제 body 이름이 name + "_main" 입니다.
                        potential_body_name = first_obj.name + "_main"
                    else:
                        potential_body_name = None
                except (KeyError, IndexError, AttributeError) as e:
                    print(f"Error accessing objects: {e}")
                    first_obj = None
                    potential_body_name = None
                
                # 시뮬레이터에 해당 body 이름이 있는지 확인합니다.
                if potential_body_name in base_env.sim.model.body_names:
                    target_obj_name = potential_body_name
                else:
                    # 만약 위 규칙이 맞지 않으면, 그냥 객체 이름 자체를 사용해봅니다.
                    if first_obj.name in base_env.sim.model.body_names:
                         target_obj_name = first_obj.name
            
            if target_obj_name:
                # Get body id and world pose
                cup_body_id = base_env.sim.model.body_name2id(target_obj_name)
                cup_pos = base_env.sim.data.xpos[cup_body_id].copy()  # [x, y, z]
                cup_quat = base_env.sim.data.xquat[cup_body_id].copy() # [w, x, y, z]
                # Convert to [x, y, z, qx, qy, qz, qw] format
                cup_pose = [cup_pos[0], cup_pos[1], cup_pos[2], 
                            cup_quat[1], cup_quat[2], cup_quat[3], cup_quat[0]]
                print(f"T_world_{target_obj_name}: {cup_pose}")
            else:
                print("Target object not found in the simulation bodies.")
                # 디버깅 정보 출력
                if hasattr(base_env, 'objects'):
                    if isinstance(base_env.objects, dict):
                        print(f"Available robosuite objects (dict): {list(base_env.objects.keys())}")
                        if len(base_env.objects) > 0:
                            first_obj = list(base_env.objects.values())[0]
                            print(f"First object name: {first_obj.name if hasattr(first_obj, 'name') else 'no name'}")
                    elif isinstance(base_env.objects, list):
                        print(f"Available robosuite objects (list): {[obj.name for obj in base_env.objects if hasattr(obj, 'name')]}")
                    else:
                        print(f"Objects type: {type(base_env.objects)}, value: {base_env.objects}")
                else:
                    print("No objects attribute found")
                
                # 시뮬레이션 body 이름들 (cup 관련 것들만)
                cup_bodies = [name for name in base_env.sim.model.body_names if 'cup' in name.lower()]
                obj_bodies = [name for name in base_env.sim.model.body_names if 'obj' in name.lower()]
                print(f"Cup-related bodies: {cup_bodies}")
                print(f"Object-related bodies: {obj_bodies}")

        except Exception as e:
            import traceback
            print(f"Error getting object pose: {e}")
            traceback.print_exc()
        
        # 2. G1 Robot 31개 joint positions
        try:
            if hasattr(base_env, 'robots') and len(base_env.robots) > 0:
                robot = base_env.robots[0]
                
                # Get all joint positions (qpos)
                joint_positions = base_env.sim.data.qpos.copy()
                print(f"All joint positions (qpos): {joint_positions}")
                print(f"Total qpos length: {len(joint_positions)}")
                
                # Get robot-specific joint positions
                if hasattr(robot, 'joints'):
                    robot_joint_positions = []
                    robot_joint_names = []
                    for joint_name in robot.joints:
                        try:
                            joint_id = base_env.sim.model.joint_name2id(joint_name)
                            joint_qpos_addr = base_env.sim.model.get_joint_qpos_addr(joint_name)
                            if isinstance(joint_qpos_addr, tuple):
                                # Multiple DOF joint (like freejoint)
                                start, end = joint_qpos_addr
                                joint_pos = joint_positions[start:end]
                            else:
                                # Single DOF joint
                                joint_pos = joint_positions[joint_qpos_addr]
                            robot_joint_positions.extend(joint_pos if hasattr(joint_pos, '__len__') else [joint_pos])
                            robot_joint_names.append(joint_name)
                        except:
                            pass
                    
                    print(f"Robot joints ({len(robot_joint_names)}): {robot_joint_names}")
                    print(f"Robot joint positions ({len(robot_joint_positions)}): {robot_joint_positions}")
                    
                    # Get robot DOF and joint mapping
                    print(f"Robot DOF: {robot.dof}")
                    if hasattr(robot, '_ref_joint_pos_indexes'):
                        ref_joint_positions = joint_positions[robot._ref_joint_pos_indexes]
                        print(f"Robot reference joint positions ({len(ref_joint_positions)}): {ref_joint_positions}")
                
        except Exception as e:
            print(f"Error getting robot joint positions: {e}")
        
        # 3. Robot pelvis/base world coordinates
        try:
            if hasattr(base_env, 'robots') and len(base_env.robots) > 0:
                robot = base_env.robots[0]
                
                # Find robot base body
                base_body_name = None
                if hasattr(robot.robot_model, 'base_body'):
                    base_body_name = robot.robot_model.base_body
                    print(f"Robot base body from model: '{base_body_name}'")
                
                # Try common base body names
                common_base_names = ['base', 'pelvis', 'robot0_base', 'robot0_pelvis']
                for name in common_base_names:
                    if name in base_env.sim.model.body_names:
                        if base_body_name is None:
                            base_body_name = name
                        print(f"Found body: '{name}'")
                
                if base_body_name and base_body_name in base_env.sim.model.body_names:
                    base_body_id = base_env.sim.model.body_name2id(base_body_name)
                    base_position_world = base_env.sim.data.body_xpos[base_body_id].copy()
                    base_quaternion_world = base_env.sim.data.body_xquat[base_body_id].copy()
                    
                    # Convert to [x, y, z, qx, qy, qz, qw] format
                    base_pose_world = [base_position_world[0], base_position_world[1], base_position_world[2],
                                     base_quaternion_world[1], base_quaternion_world[2], base_quaternion_world[3], base_quaternion_world[0]]
                    print(f"T_world_{base_body_name}: {base_pose_world}")
                else:
                    print("Robot base body not found")
                    # Print all body names for debugging
                    robot_bodies = [name for name in base_env.sim.model.body_names if 'robot' in name.lower() or 'pelvis' in name.lower() or 'base' in name.lower()]
                    print(f"Robot-related bodies: {robot_bodies}")
                    
        except Exception as e:
            print(f"Error getting robot base coordinates: {e}")
                


def main():
    """Main function to run the G1 tabletop simulation."""
    parser = argparse.ArgumentParser(description="G1 Robot Tabletop Simulation with Trajectory Infrastructure")
    parser.add_argument("--steps", type=int, default=500,
                       help="Number of simulation steps per episode (default: 500)")
    parser.add_argument("--episodes", type=int, default=1,
                       help="Number of episodes to run (default: 1)")
    parser.add_argument("--robot", type=str, default="G1ArmsAndWaistDex31Hands",
                       help="G1 robot variant to use (default: G1ArmsAndWaistDex31Hands)")
    parser.add_argument("--env", type=str, default="g1_unified/PnPCupToPlate_G1ArmsAndWaistDex31Hands_Env",
                       help="Tabletop environment to use")
    parser.add_argument("--video_dir", type=str, default=None,
                       help="Directory to save video recordings (default: None)")
    parser.add_argument("--multiview", action="store_true",
                       help="Enable multi-view video recording")
    parser.add_argument("--fps", type=int, default=30,
                       help="Video FPS (default: 30)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("G1 Robot Tabletop Simulation (Trajectory-based)")
    print("=" * 60)
    print(f"Robot: {args.robot}")
    print(f"Environment: {args.env}")
    print(f"Episodes: {args.episodes}")
    print(f"Steps per episode: {args.steps}")
    print(f"Video directory: {args.video_dir if args.video_dir else 'Not set'}")
    print(f"Multi-view recording: {'Enabled' if args.multiview else 'Disabled'}")
    print(f"Video FPS: {args.fps}")
    print("=" * 60)
    
    # Create video configuration
    video_config = VideoConfig(
        video_dir=args.video_dir,
        fps=args.fps,
        steps_per_render=2,
    )
    
    # Create simulation configuration
    config = SimulationConfig(
        env_name=args.env,
        n_episodes=args.episodes,
        n_envs=1,
        video=video_config,
        multistep=MultiStepConfig(
            n_action_steps=16,
            max_episode_steps=args.steps
        ),
        multi_video=args.multiview,
    )
    
    # Create simulation instance
    simulation = G1TabletopSimulationWithTrajectory(robot_name=args.robot)
    
    try:
        # Run simulation
        simulation.run_simulation(config)
        
    except KeyboardInterrupt:
        print("\n⚠ Simulation interrupted by user")
        
    except Exception as e:
        print(f"\n✗ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Always clean up
        if simulation.env is not None:
            simulation.env.close()
        print("\nSimulation ended.")


if __name__ == "__main__":
    main()
