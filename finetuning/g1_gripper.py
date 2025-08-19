#!/usr/bin/env python3
"""
G1 Robot Tabletop Environment Simulation with Dex31 Hand Gripper Movement

This script creates a simulation where the G1ArmsAndWaistDex31Hands robot performs
a natural, smoothed left hand gripper movement (close then open) in a tabletop 
environment while recording multi-view videos.

The gripper motion sequence:
- First 1/3 of simulation: smoothly close left hand (ease-in-out)
- Next 1/3 of simulation: smoothly open left hand (ease-in-out)
- Final 1/3 of simulation: keep left hand open

Usage:
    python g1_gripper.py --video_dir ./videos --multiview --steps 500
"""

import argparse
import time
import sys
import os
import math
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
import gymnasium as gym
from functools import partial

# Add the robocasa and robosuite paths to enable imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'robocasa_g1'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'robosuite_g1'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gr00t'))

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
        self.robot_name = robot_name
        self.env = None
        self._total_steps = 500  # Default value
        
        print(f"Initializing G1 Tabletop Simulation with robot: {robot_name}")

    def _ease_in_out_cosine(self, t):
        """Cosine-based easing function for smooth start and end."""
        return 0.5 * (1 - math.cos(t * math.pi))

    def setup_environment(self, config: SimulationConfig) -> gym.vector.VectorEnv:
        """Set up the simulation environment."""
        env_fns = [partial(self._create_single_env, config=config, idx=i) for i in range(config.n_envs)]
        if config.n_envs == 1:
            return gym.vector.SyncVectorEnv(env_fns)
        else:
            return gym.vector.AsyncVectorEnv(
                env_fns,
                shared_memory=False,
                context="spawn",
            )

    def _create_single_env(self, config: SimulationConfig, idx: int) -> gym.Env:
        """Create a single environment with wrappers."""
        print(f"Creating environment: {config.env_name}")
        env = gym.make(
            config.env_name, 
            enable_render=True,
            layout_ids=0,  # 특정 레이아웃 고정 (TABLETOP = 0)
            style_ids=0,   # 특정 스타일 고정
        )
        print(f"✓ Base environment created successfully")
        
        if config.video.video_dir is not None:
            video_path = Path(config.video.video_dir)
            video_path.mkdir(parents=True, exist_ok=True)
            if config.multi_video:
                recorder_class = MultiVideoRecorder
                wrapper_class = MultiVideoRecordingWrapper
                print("Setting up multi-view video recording...")
            else:
                recorder_class = VideoRecorder
                wrapper_class = VideoRecordingWrapper
                print("Setting up single-view video recording...")

            video_recorder = recorder_class.create_h264(
                fps=config.video.fps, codec=config.video.codec,
                input_pix_fmt=config.video.input_pix_fmt, crf=config.video.crf,
                thread_type=config.video.thread_type, thread_count=config.video.thread_count,
            )
            env = wrapper_class(
                env, video_recorder, video_dir=video_path,
                steps_per_render=config.video.steps_per_render,
            )
            print(f"✓ {wrapper_class.__name__} added")
        
        env = MultiStepWrapper(
            env,
            video_delta_indices=config.multistep.video_delta_indices,
            state_delta_indices=config.multistep.state_delta_indices,
            n_action_steps=config.multistep.n_action_steps,
            max_episode_steps=None,
        )
        print("✓ MultiStepWrapper added")
        return env

    def get_gripper_action(self, env, timestep, n_action_steps):
        """
        Generate action with smoothed left hand gripper movement.
        Interpolate actions at sub-step level for smoother motion.
        """
        action_space = env.single_action_space
        action = {k: np.zeros((1, n_action_steps, v.shape[-1]), dtype=np.float32) 
                  for k, v in action_space.spaces.items() if hasattr(v, 'shape')}
        
        if 'action.left_arm' in action:
            total_sim_steps = self._total_steps * n_action_steps
            close_phase_end = total_sim_steps // 3
            
            # Left arm 7DOF 자세 (CUROBO planned values에서 arm 부분만)
            default_pose = np.array([
                # Left arm joints (7개) - CUROBO planned values
                -1.015, 0.781, 0.217, 0.592, -0.046, -0.036, -1.188
            ])
            
            # 들어올린 자세
            lifted_pose = np.array([
                # Left arm joints (7개)
                0.0, 0.1, 0.0, -0.25, 0.0, 0.0, 0.0
            ])
            
            for sub_step in range(n_action_steps):
                current_sim_step = timestep * n_action_steps + sub_step
                
                if current_sim_step < close_phase_end:
                    # 처음 1/3 구간: CUROBO가 계획한 자세 유지
                    action['action.left_arm'][0, sub_step, :] = default_pose
                else:
                    # 나머지 2/3 구간: 컵을 들어올린 자세로 변경
                    action['action.left_arm'][0, sub_step, :] = lifted_pose

        # Waist 관절 제어 (별도로 처리)
        if 'action.waist' in action:
            total_sim_steps = self._total_steps * n_action_steps
            close_phase_end = total_sim_steps // 3
            
            # Waist 3DOF 자세 (CUROBO planned values에서 waist 부분만)
            default_waist = np.array([
                # Waist joints (3개) - CUROBO planned values
                -0.686, -0.519, 0.518  # waist_yaw, waist_roll, waist_pitch
            ])
            
            # 들어올린 자세의 waist
            lifted_waist = np.array([
                # Waist joints (3개)
                0.0, 0.0, 0.0  # waist_yaw, waist_roll, waist_pitch
            ])
            
            for sub_step in range(n_action_steps):
                current_sim_step = timestep * n_action_steps + sub_step
                
                if current_sim_step < close_phase_end:
                    action['action.waist'][0, sub_step, :] = default_waist
                else:
                    action['action.waist'][0, sub_step, :] = lifted_waist

        # 오른팔 joint 값 설정 (기본값 유지)
        if 'action.right_arm' in action:
            right_arm_pose = np.array([0.0, -0.1, 0.0, -0.2, 0.0, 0.0, 0.0])
            for sub_step in range(n_action_steps):
                action['action.right_arm'][0, sub_step, :] = right_arm_pose

        if 'action.left_hand' in action:
            total_sim_steps = self._total_steps * n_action_steps
            close_phase_end = total_sim_steps // 3
            
            # Dex31Hand joint limits 조정 (필요 시 robosuite 문서 확인)
            open_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            close_pose = np.array([-0.8, -0.8, -0.8, -0.8, 0.8, 0.4, 0.4])  # 관절 제한 내에서 조정

            for sub_step in range(n_action_steps):
                current_sim_step = timestep * n_action_steps + sub_step
                delta = sub_step / n_action_steps  # Sub-step 내 보간 비율 (0~1)

                if current_sim_step < close_phase_end:
                    # 닫기 동작 (처음 1/3 구간)
                    linear_progress = current_sim_step / close_phase_end
                    eased_progress = self._ease_in_out_cosine(linear_progress)
                    # Sub-step 내 추가 보간
                    eased_progress = eased_progress + delta * (self._ease_in_out_cosine((current_sim_step + 1) / close_phase_end) - eased_progress) / n_action_steps
                    gripper_action = open_pose + eased_progress * (close_pose - open_pose)
                else:
                    # 닫힌 상태 유지 (나머지 2/3 구간)
                    gripper_action = close_pose
                
                action['action.left_hand'][0, sub_step, :] = gripper_action

        return action

    def _print_robot_base_pose(self, timestep):
        """Print robot base world pose."""
        try:
            base_env = self.env.envs[0]
            while hasattr(base_env, 'env'):
                base_env = base_env.env
            
            # G1 로봇의 base body는 'robot0_base'
            base_body_name = 'robot0_base'
            
            if base_body_name in base_env.sim.model.body_names:
                base_body_id = base_env.sim.model.body_name2id(base_body_name)
                base_position = base_env.sim.data.body_xpos[base_body_id].copy()
                base_quaternion = base_env.sim.data.body_xquat[base_body_id].copy()
                
                # [x, y, z, qx, qy, qz, qw] 형식으로 변환
                base_pose = [base_position[0], base_position[1], base_position[2],
                            base_quaternion[1], base_quaternion[2], base_quaternion[3], base_quaternion[0]]
                
                print(f"Step {timestep:3d} - T_world_{base_body_name}: {base_pose}")
            else:
                print(f"Step {timestep:3d} - Base body '{base_body_name}' not found")
                
        except Exception as e:
            print(f"Error getting robot base pose: {e}")

    def _print_all_joints_dof(self, timestep):
        """Print all 31 DOF joint positions."""
        try:
            base_env = self.env.envs[0]
            while hasattr(base_env, 'env'):
                base_env = base_env.env
            
            # G1ArmsAndWaistDex31Hands의 모든 관절들 (31개)
            all_joints = [
                # Waist joints (3개)
                'robot0_waist_yaw_joint', 'robot0_waist_roll_joint', 'robot0_waist_pitch_joint',
                # Left arm joints (7개)
                'robot0_left_shoulder_pitch_joint', 'robot0_left_shoulder_roll_joint', 'robot0_left_shoulder_yaw_joint',
                'robot0_left_elbow_joint', 'robot0_left_wrist_roll_joint', 'robot0_left_wrist_pitch_joint', 'robot0_left_wrist_yaw_joint',
                # Right arm joints (7개)
                'robot0_right_shoulder_pitch_joint', 'robot0_right_shoulder_roll_joint', 'robot0_right_shoulder_yaw_joint',
                'robot0_right_elbow_joint', 'robot0_right_wrist_roll_joint', 'robot0_right_wrist_pitch_joint', 'robot0_right_wrist_yaw_joint',
                # Left hand joints (7개)
                'robot0_left_hand_thumb_0_joint', 'robot0_left_hand_thumb_1_joint', 'robot0_left_hand_thumb_2_joint',
                'robot0_left_hand_middle_0_joint', 'robot0_left_hand_middle_1_joint', 'robot0_left_hand_index_0_joint', 'robot0_left_hand_index_1_joint',
                # Right hand joints (7개)
                'robot0_right_hand_thumb_0_joint', 'robot0_right_hand_thumb_1_joint', 'robot0_right_hand_thumb_2_joint',
                'robot0_right_hand_middle_0_joint', 'robot0_right_hand_middle_1_joint', 'robot0_right_hand_index_0_joint', 'robot0_right_hand_index_1_joint'
            ]
            
            all_joint_positions = []
            for joint_name in all_joints:
                try:
                    joint_id = base_env.sim.model.joint_name2id(joint_name)
                    joint_pos = base_env.sim.data.qpos[joint_id]
                    all_joint_positions.append(joint_pos)
                except:
                    all_joint_positions.append(0.0)
            
            # 카테고리별로 출력
            print(f"Step {timestep:3d} - All 31 DOF:")
            print(f"  Waist (3): {[f'{pos:.3f}' for pos in all_joint_positions[0:3]]}")
            print(f"  Left Arm (7): {[f'{pos:.3f}' for pos in all_joint_positions[3:10]]}")
            print(f"  Right Arm (7): {[f'{pos:.3f}' for pos in all_joint_positions[10:17]]}")
            print(f"  Left Hand (7): {[f'{pos:.3f}' for pos in all_joint_positions[17:24]]}")
            print(f"  Right Hand (7): {[f'{pos:.3f}' for pos in all_joint_positions[24:31]]}")
            
        except Exception as e:
            print(f"Error getting all joints DOF: {e}")

    def _print_left_arm_dof(self, timestep):
        """Print left arm 7DOF joint positions."""
        try:
            # Get the base robosuite environment
            base_env = self.env.envs[0]
            while hasattr(base_env, 'env'):
                base_env = base_env.env
            
            # Get robot
            robot = base_env.robots[0]
            
            # Get left arm joint positions (robot0_ 접두사 추가)
            left_arm_joints = [
                'robot0_left_shoulder_pitch_joint',
                'robot0_left_shoulder_roll_joint', 
                'robot0_left_shoulder_yaw_joint',
                'robot0_left_elbow_joint',
                'robot0_left_wrist_roll_joint',
                'robot0_left_wrist_pitch_joint',
                'robot0_left_wrist_yaw_joint'
            ]
            
            left_arm_positions = []
            for joint_name in left_arm_joints:
                joint_id = base_env.sim.model.joint_name2id(joint_name)
                joint_pos = base_env.sim.data.qpos[joint_id]
                left_arm_positions.append(joint_pos)
            
            print(f"Step {timestep:3d} - Left Arm 7DOF: {[f'{pos:.3f}' for pos in left_arm_positions]}")
            
        except Exception as e:
            print(f"Error getting left arm DOF: {e}")

    def _print_table_info(self, timestep):
        """Print table world pose for CUROBO obstacle list."""
        try:
            base_env = self.env.envs[0]
            while hasattr(base_env, 'env'):
                base_env = base_env.env
            
            # 테이블 body 찾기 (실제 이름: table_main_group_main)
            table_body_names = ['table_main_group_main', 'table', 'table_main', 'counter', 'counter_main']
            table_body_name = None
            
            for name in table_body_names:
                if name in base_env.sim.model.body_names:
                    table_body_name = name
                    break
            
            if table_body_name:
                table_body_id = base_env.sim.model.body_name2id(table_body_name)
                table_position = base_env.sim.data.body_xpos[table_body_id].copy()
                table_quaternion = base_env.sim.data.body_xquat[table_body_id].copy()
                
                # 테이블의 geom 정보 찾기 (크기 정보)
                table_size = [0.61, 0.375, 0.92]  # 기본값 (테이블 크기)
                
                # body에 연결된 geom들 찾기
                for geom_id in range(base_env.sim.model.ngeom):
                    if base_env.sim.model.geom_bodyid[geom_id] == table_body_id:
                        # geom의 크기 정보 가져오기
                        geom_size = base_env.sim.model.geom_size[geom_id].copy()
                        # 가장 큰 geom을 테이블 크기로 사용
                        if sum(geom_size) > sum(table_size):
                            table_size = geom_size
                
                # CUROBO 장애물 목록 형식: [x, y, z, qx, qy, qz, qw, size_x, size_y, size_z]
                table_obstacle = [
                    table_position[0], table_position[1], table_position[2],  # 위치
                    table_quaternion[1], table_quaternion[2], table_quaternion[3], table_quaternion[0],  # 방향
                    table_size[0], table_size[1], table_size[2]  # 크기
                ]
                
                print(f"Step {timestep:3d} - CUROBO Obstacle (Table): {table_obstacle}")
                print(f"  Position: [{table_position[0]:.3f}, {table_position[1]:.3f}, {table_position[2]:.3f}]")
                print(f"  Orientation: [{table_quaternion[1]:.3f}, {table_quaternion[2]:.3f}, {table_quaternion[3]:.3f}, {table_quaternion[0]:.3f}]")
                print(f"  Size: [{table_size[0]:.3f}, {table_size[1]:.3f}, {table_size[2]:.3f}]")
            else:
                print(f"Step {timestep:3d} - Table body not found. Available bodies: {[name for name in base_env.sim.model.body_names if 'table' in name.lower() or 'counter' in name.lower()]}")
                
        except Exception as e:
            print(f"Error getting table obstacle info: {e}")

    def run_simulation(self, config: SimulationConfig) -> None:
        """Run the simulation for the specified number of episodes."""
        start_time = time.time()
        print(f"Running {config.n_episodes} episodes for {config.env_name}")

        self._total_steps = config.multistep.max_episode_steps
        self.env = self.setup_environment(config)

        # 로봇 정보 출력
        if hasattr(self.env, 'envs') and len(self.env.envs) > 0:
            base_env = self.env.envs[0]
            while hasattr(base_env, 'env'):
                base_env = base_env.env
            if hasattr(base_env, 'robots') and len(base_env.robots) > 0:
                robot = base_env.robots[0]
                print(f"✓ Robot loaded: {robot.robot_model.naming_prefix}")
                print(f"  - Total DOF: {robot.dof}")
        
        print(f"Action space: {self.env.single_action_space}")

        for ep_idx in range(config.n_episodes):
            print(f"\n--- Starting Episode {ep_idx + 1}/{config.n_episodes} ---")
            obs, _ = self.env.reset()
            print("✓ Environment reset completed.")
            
            # 초기 상태 출력
            self._print_states(0)

            max_timesteps = config.multistep.max_episode_steps
            for timestep in range(max_timesteps):
                actions = self.get_gripper_action(self.env, timestep, config.multistep.n_action_steps)
                try:
                    obs, rewards, terminations, truncations, env_infos = self.env.step(actions)
                    
                    # 매 step마다 모든 31 DOF, 로봇 base 좌표, 테이블 장애물 정보 출력
                    self._print_all_joints_dof(timestep)
                    self._print_robot_base_pose(timestep)
                    self._print_table_info(timestep)
                    
                    # 50 timestep마다 상태 출력
                    if timestep % 50 == 0:
                        self._print_states(timestep)

                    if terminations[0] or truncations[0]:
                        break
                except Exception as e:
                    print(f"✗ Environment step failed: {e}")
                    raise
        
        print("\nFinalizing simulation...")
        self.env.close()
        self.env = None
        print(f"✓ Simulation completed in {time.time() - start_time:.2f} seconds")
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


def main():
    """Main function to run the G1 tabletop simulation."""
    parser = argparse.ArgumentParser(description="G1 Robot Tabletop Simulation with Natural Gripper Movement")
    parser.add_argument("--steps", type=int, default=500, help="Number of simulation steps per episode (default: 500)")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run (default: 1)")
    parser.add_argument("--robot", type=str, default="G1ArmsAndWaistDex31Hands", help="G1 robot variant to use")
    parser.add_argument("--env", type=str, default="g1_unified/PnPCupToPlateNoDistractors_G1ArmsAndWaistDex31Hands_Env", help="Tabletop environment to use")
    parser.add_argument("--video_dir", type=str, default=None, help="Directory to save video recordings")
    parser.add_argument("--multiview", action="store_true", help="Enable multi-view video recording")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS (default: 30)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(parser.description)
    print("=" * 60)
    print(f"Robot: {args.robot}")
    print(f"Environment: {args.env}")
    print(f"Episodes: {args.episodes}")
    print(f"Steps per episode: {args.steps}")
    print(f"Video directory: {args.video_dir if args.video_dir else 'Not set'}")
    print(f"Multi-view recording: {'Enabled' if args.multiview else 'Disabled'}")
    print(f"Video FPS: {args.fps}")
    print("---")
    print("Gripper Motion Sequence:")
    print(f"  Steps 1-{args.steps//3}: Left hand closing")
    print(f"  Steps {args.steps//3+1}-{2*args.steps//3}: Left hand opening")
    print(f"  Steps {2*args.steps//3+1}-{args.steps}: Left hand stays open")
    print("=" * 60)
    
    video_config = VideoConfig(video_dir=args.video_dir, fps=args.fps)
    config = SimulationConfig(
        env_name=args.env, n_episodes=args.episodes,
        video=video_config,
        multistep=MultiStepConfig(max_episode_steps=args.steps),
        multi_video=args.multiview
    )
    
    simulation = G1TabletopSimulationWithTrajectory(robot_name=args.robot)
    
    try:
        simulation.run_simulation(config)
    except KeyboardInterrupt:
        print("\n⚠ Simulation interrupted by user")
    except Exception as e:
        print(f"\n✗ An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if hasattr(simulation, 'env') and simulation.env is not None:
            simulation.env.close()
        print("\nSimulation ended.")


if __name__ == "__main__":
    main()