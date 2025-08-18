#!/usr/bin/env python3
"""
G1 Robot Tabletop Environment Simulation with CUROBO-driven Motion

This script creates a simulation where the G1 robot's left arm moves to a
target object using a trajectory planned by CUROBO. The simulation is
recorded as a multi-view video.

The motion sequence:
- At step 0, get the world state (robot, cup, obstacles).
- Plan a trajectory with CUROBO to move the left hand above the cup.
- Execute the planned trajectory step-by-step in the simulation.

Usage:
    python main.py --video_dir ./videos_curobo --multiview
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

# CUROBO Imports
import curobo
from curobo.geom.types import WorldConfig, Pose
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
from curobo.types.robot import RobotConfig
import torch

# Add the robocasa and robosuite paths to enable imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'robocasa_g1'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'robosuite_g1'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'gr00t'))

import robocasa
import robosuite
from robocasa.utils.gym_utils import GrootRoboCasaEnv

# Import video recording utilities
from gr00t.eval.wrappers.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from gr00t.eval.wrappers.multi_video_recording_wrapper import MultiVideoRecordingWrapper, MultiVideoRecorder
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
    n_action_steps: int = 1
    max_episode_steps: int = 500

@dataclass
class SimulationConfig:
    """Main configuration for simulation environment."""
    env_name: str
    n_episodes: int = 1
    n_envs: int = 1
    video: VideoConfig = field(default_factory=VideoConfig)
    multistep: MultiStepConfig = field(default_factory=MultiStepConfig)
    multi_video: bool = False

class G1SimulationWithCurobo:
    def __init__(self, robot_name="G1ArmsAndWaistDex31Hands"):
        self.robot_name = robot_name
        self.env = None
        self.curobo_planner = None
        print(f"Initializing G1 Simulation with CUROBO integration")

    def setup_curobo_planner(self):
        """Initializes and returns a CUROBO motion generator."""
        print("Setting up CUROBO planner...")
        try:
            config_file = os.path.join(os.path.dirname(__file__), "curobo_config.yaml")
            import yaml
            with open(config_file, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
            
            robot_cfg = RobotConfig.from_dict(config_dict)
            
            motion_gen_config = MotionGenConfig.load_from_robot_config(
                robot_cfg,
                world_model=None,
                use_cuda_graph=True,
                num_ik_seeds=500,
                num_graph_seeds=200,
                num_trajopt_seeds=200,
            )
            
            planner = MotionGen(motion_gen_config)
            print("✓ CUROBO planner initialized successfully.")
            return planner
        except Exception as e:
            print(f"✗ ERROR: Failed to initialize CUROBO: {e}")
            raise

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
            layout_ids=0,
            style_ids=0,
        )
        print(f"✓ Base environment created successfully")
        
        if hasattr(env, 'envs') and len(env.envs) > 0:
            base_env = env.envs[0]
            while hasattr(base_env, 'env'):
                base_env = base_env.env
            
            if hasattr(base_env, 'robots') and len(base_env.robots) > 0:
                robot = base_env.robots[0]
                robot.initialization_noise = {"magnitude": 0.0, "type": "gaussian"}
                robot.reset(deterministic=True)
                print(f"✓ Robot deterministic reset applied")
        
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

    def get_base_environment(self):
        """Get the base robosuite environment."""
        base_env = self.env.envs[0]
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        return base_env

    def get_current_state(self, base_env):
        """Extracts all necessary state information from the simulation."""
        current_q = self.get_planning_joint_positions(base_env)
        robot_base_pose = self.get_robot_base_pose(base_env)
        cup_pose = self.get_cup_pose(base_env)
        table_obstacle = self.get_table_obstacle(base_env)
        
        return {
            "joint_positions": current_q,
            "base_pose": robot_base_pose,
            "cup_pose": cup_pose,
            "obstacles": [table_obstacle] if table_obstacle else []
        }

    def get_planning_joint_positions(self, base_env):
        """Gets the positions of the joints that CUROBO will plan for (10 DOF)."""
        planning_joints = [
            'robot0_waist_yaw_joint', 'robot0_waist_roll_joint', 'robot0_waist_pitch_joint',
            'robot0_left_shoulder_pitch_joint', 'robot0_left_shoulder_roll_joint', 'robot0_left_shoulder_yaw_joint',
            'robot0_left_elbow_joint', 'robot0_left_wrist_roll_joint', 'robot0_left_wrist_pitch_joint', 'robot0_left_wrist_yaw_joint'
        ]
        
        joint_positions = []
        for joint_name in planning_joints:
            try:
                joint_id = base_env.sim.model.joint_name2id(joint_name)
                joint_positions.append(base_env.sim.data.qpos[joint_id])
            except:
                joint_positions.append(0.0)
        
        return np.array(joint_positions)

    def get_robot_base_pose(self, base_env):
        """Get robot base world pose."""
        try:
            base_body_name = 'robot0_base'
            base_body_id = base_env.sim.model.body_name2id(base_body_name)
            base_position = base_env.sim.data.body_xpos[base_body_id].copy()
            base_quaternion = base_env.sim.data.body_xquat[base_body_id].copy()
            
            position_tensor = torch.tensor(base_position, dtype=torch.float32).unsqueeze(0).cuda()
            quaternion_tensor = torch.tensor(base_quaternion, dtype=torch.float32).unsqueeze(0).cuda()
            
            return Pose(position=position_tensor, quaternion=quaternion_tensor)
        except Exception as e:
            print(f"Error getting robot base pose: {e}")
            return None

    def get_cup_pose(self, base_env):
        """Get cup world pose."""
        try:
            target_obj_name = None
            
            if hasattr(base_env, 'objects') and base_env.objects:
                try:
                    if isinstance(base_env.objects, dict):
                        if len(base_env.objects) > 0:
                            first_obj = list(base_env.objects.values())[0]
                        else:
                            first_obj = None
                    elif isinstance(base_env.objects, list):
                        first_obj = base_env.objects[0] if len(base_env.objects) > 0 else None
                    else:
                        first_obj = None
                        
                    if first_obj:
                        potential_body_name = first_obj.name + "_main"
                    else:
                        potential_body_name = None
                except (KeyError, IndexError, AttributeError) as e:
                    first_obj = None
                    potential_body_name = None
                
                if potential_body_name in base_env.sim.model.body_names:
                    target_obj_name = potential_body_name
                else:
                    if first_obj.name in base_env.sim.model.body_names:
                         target_obj_name = first_obj.name
            
            if target_obj_name:
                cup_body_id = base_env.sim.model.body_name2id(target_obj_name)
                cup_pos = base_env.sim.data.xpos[cup_body_id].copy()
                cup_quat = base_env.sim.data.xquat[cup_body_id].copy()
                
                position_tensor = torch.tensor(cup_pos, dtype=torch.float32).unsqueeze(0).cuda()
                quaternion_tensor = torch.tensor(cup_quat, dtype=torch.float32).unsqueeze(0).cuda()
                
                return Pose(position=position_tensor, quaternion=quaternion_tensor)
            else:
                print("Target object not found in the simulation bodies.")
                return None
        except Exception as e:
            print(f"Error getting cup pose: {e}")
            return None

    def get_table_obstacle(self, base_env):
        """Get table obstacle for CUROBO."""
        try:
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
                
                table_size = [0.61, 0.375, 0.92]
                
                for geom_id in range(base_env.sim.model.ngeom):
                    if base_env.sim.model.geom_bodyid[geom_id] == table_body_id:
                        geom_size = base_env.sim.model.geom_size[geom_id].copy()
                        if sum(geom_size) > sum(table_size):
                            table_size = geom_size
                
                from scipy.spatial.transform import Rotation
                
                pelvis_body_id = base_env.sim.model.body_name2id('robot0_base')
                pelvis_position = base_env.sim.data.body_xpos[pelvis_body_id].copy()
                pelvis_quaternion = base_env.sim.data.body_xquat[pelvis_body_id].copy()
                
                table_rot_world = Rotation.from_quat([table_quaternion[1], table_quaternion[2], table_quaternion[3], table_quaternion[0]])
                pelvis_rot_world = Rotation.from_quat([pelvis_quaternion[1], pelvis_quaternion[2], pelvis_quaternion[3], pelvis_quaternion[0]])
                
                table_pos_relative = table_position - pelvis_position
                table_pos_pelvis_frame = pelvis_rot_world.as_matrix().T @ table_pos_relative
                
                relative_table_rot = table_rot_world * pelvis_rot_world.inv()
                table_quat_pelvis_frame = relative_table_rot.as_quat()
                
                return WorldConfig.from_dict({
                    "table": {
                        "type": "box", 
                        "pose": [table_pos_pelvis_frame[0], table_pos_pelvis_frame[1], table_pos_pelvis_frame[2], 
                                 table_quat_pelvis_frame[0], table_quat_pelvis_frame[1], table_quat_pelvis_frame[2], table_quat_pelvis_frame[3]], 
                        "dims": list(table_size)
                    }
                })
            else:
                return None
                
        except Exception as e:
            print(f"Error getting table obstacle info: {e}")
            return None

    def run_simulation(self, config: SimulationConfig) -> None:
        """Run the simulation for the specified number of episodes."""
        start_time = time.time()
        print(f"Running {config.n_episodes} episodes for {config.env_name}")

        # 1. 환경 및 플래너 설정
        self.env = self.setup_environment(config)
        self.curobo_planner = self.setup_curobo_planner()
        self.kin_model = self.curobo_planner.kinematics

        for ep_idx in range(config.n_episodes):
            print(f"\n--- Starting Episode {ep_idx + 1}/{config.n_episodes} ---")
            obs, _ = self.env.reset()
            base_env = self.get_base_environment()
            print("✓ Environment reset completed.")
            
            if hasattr(base_env, 'robots') and len(base_env.robots) > 0:
                robot = base_env.robots[0]
                robot.reset(deterministic=True)
                print("✓ Robot deterministic reset applied after environment reset")
            
            # 2. 초기 상태 얻기 (Step 0)
            print("\n--- Extracting Initial State (Step 0) ---")
            initial_state = self.get_current_state(base_env)
            
            if not initial_state["cup_pose"]:
                print("Cannot proceed without cup pose. Ending episode.")
                continue

            if not initial_state["base_pose"]:
                print("Cannot proceed without robot base pose. Ending episode.")
                continue

            # 3. 목표 포즈 계산 (pelvis 기준 좌표계로 변환)
            target_pose = initial_state["cup_pose"].clone()
            target_position = target_pose.position.clone()

            cup_pos_world = target_position[0].cpu().numpy()
            cup_quat_world = target_pose.quaternion[0].cpu().numpy()

            pelvis_pos_world = initial_state["base_pose"].position[0].cpu().numpy()
            pelvis_quat_world = initial_state["base_pose"].quaternion[0].cpu().numpy()

            print(f"Robot (pelvis) world coordinates: {pelvis_pos_world}")
            print(f"Robot (pelvis) world quaternion: {pelvis_quat_world}")
            print(f"Cup world coordinates: {cup_pos_world}")
            print(f"Cup world quaternion: {cup_quat_world}")

            from scipy.spatial.transform import Rotation

            # [w, x, y, z] 순서로 올바르게 변환
            # pelvis_quat_world는 [w, x, y, z] 순서이므로 그대로 사용
            pelvis_rotation = Rotation.from_quat([pelvis_quat_world[1], pelvis_quat_world[2], pelvis_quat_world[3], pelvis_quat_world[0]])
            pelvis_rotation_matrix = pelvis_rotation.as_matrix()

            # 월드 → pelvis 좌표계 변환
            cup_pos_relative = cup_pos_world - pelvis_pos_world
            cup_pos_pelvis_frame = pelvis_rotation_matrix.T @ cup_pos_relative

            # 회전도 변환 - [w, x, y, z] 순서 사용
            cup_rotation_world = Rotation.from_quat([cup_quat_world[1], cup_quat_world[2], cup_quat_world[3], cup_quat_world[0]])
            relative_rotation = cup_rotation_world * pelvis_rotation.inv()
            cup_quat_pelvis_frame = relative_rotation.as_quat()

            print(f"Cup coordinates relative to robot: {cup_pos_pelvis_frame}")

            # 직접 컵 위치 사용
            target_pos_pelvis_frame = cup_pos_pelvis_frame

            # CUROBO용 quaternion 순서로 변환 - [w, x, y, z] 순서 유지
            target_quaternion_pelvis_frame = torch.tensor([
                cup_quat_pelvis_frame[3], cup_quat_world[0], 
                cup_quat_world[1], cup_quat_world[2]
            ], dtype=torch.float32).unsqueeze(0).cuda()

            target_position_pelvis_frame = torch.tensor(target_pos_pelvis_frame, dtype=torch.float32).unsqueeze(0).cuda()
            target_pose_pelvis_frame = Pose(position=target_position_pelvis_frame, quaternion=target_quaternion_pelvis_frame)

            # 4. Curobo로 경로 계획
            print("\n--- Planning trajectory with CUROBO... ---")
            world_model = initial_state["obstacles"][0] if initial_state["obstacles"] else None

            from curobo.types.state import JointState

            # Waist 관절을 0으로 고정하지 말고 실제 값 사용
            full_joint_positions = initial_state['joint_positions'].copy()
            # full_joint_positions[0:3] = 0.0  # 이 줄 제거

            joint_positions_tensor = torch.tensor(full_joint_positions, dtype=torch.float32).unsqueeze(0).cuda()

            # Orientation 조건 제거 - 현재 EEF 방향을 그대로 유지 (방향 제약 없음)
            current_ee_pose = self.kin_model.forward(joint_positions_tensor)
            if hasattr(current_ee_pose, 'quaternion'):
                target_quaternion_pelvis_frame = current_ee_pose.quaternion  # 현재 EEF 방향 유지
            else:
                target_quaternion_pelvis_frame = current_ee_pose[1]  # 현재 EEF 방향 유지

            target_position_pelvis_frame = torch.tensor(target_pos_pelvis_frame, dtype=torch.float32).unsqueeze(0).cuda()
            target_pose_pelvis_frame = Pose(position=target_position_pelvis_frame, quaternion=target_quaternion_pelvis_frame)

            planning_joint_names = [
                'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
                'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
                'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint'
            ]

            start_state = JointState.from_position(joint_positions_tensor, joint_names=planning_joint_names)

            result = self.curobo_planner.plan_single(
                start_state,
                target_pose_pelvis_frame
            )

            if not result.success.item():
                print("✗ CUROBO failed to find a solution. Skipping episode.")
                continue

            planned_trajectory = result.get_interpolated_plan()
            print(f"✓ CUROBO found a solution with {len(planned_trajectory.position)} steps.")

            # 7DOF 결과를 10DOF로 확장하는 부분 제거 - 원래대로
            # for step_idx in range(len(planned_trajectory.position)):
            #     # 7DOF left arm 결과
            #     left_arm_result = planned_trajectory.position[step_idx].cpu().numpy()
            #     
            #     # 10DOF로 확장 (waist = 0, left arm = 결과)
            #     full_result = np.zeros(10)
            #     full_result[0:3] = 0.0  # waist joints = 0
            #     full_result[3:10] = left_arm_result  # left arm joints = CUROBO 결과
            #     
            #     # planned_trajectory를 10DOF로 업데이트
            #     planned_trajectory.position[step_idx] = torch.tensor(full_result, dtype=torch.float32).cuda()

            # Compute achieved EEF pose from last planned joint
            q_last = planned_trajectory.position[-1].unsqueeze(0)
            achieved_ee_pose = self.kin_model.forward(q_last)
            
            if hasattr(achieved_ee_pose, 'position'):
                achieved_pos = achieved_ee_pose.position.cpu().numpy()[0]
                achieved_quat = achieved_ee_pose.quaternion.cpu().numpy()[0]
            else:
                achieved_pos = achieved_ee_pose[0].cpu().numpy()[0]
                achieved_quat = achieved_ee_pose[1].cpu().numpy()[0]

            pos_diff = np.linalg.norm(achieved_pos - target_pose_pelvis_frame.position.cpu().numpy()[0])
            print(f"CUROBO planned EEF position: {achieved_pos}")
            print(f"Target position: {target_pose_pelvis_frame.position.cpu().numpy()[0]}")
            print(f"Position difference: {pos_diff:.6f}m")
            
            # Waist 관절을 0으로 강제 설정하는 부분도 제거
            # for step_idx in range(len(planned_trajectory.position)):
            #     for waist_idx in [0, 1, 2]:  # waist joint indices
            #         planned_trajectory.position[step_idx, waist_idx] = 0.0

            # 5. CUROBO trajectory를 직접 적용
            print(f"\n--- Executing {len(planned_trajectory.position)} planned steps directly... ---")
            
            joint_mapping = [
                ('waist_yaw_joint', 'robot0_waist_yaw_joint'),
                ('waist_roll_joint', 'robot0_waist_roll_joint'), 
                ('waist_pitch_joint', 'robot0_waist_pitch_joint'),
                ('left_shoulder_pitch_joint', 'robot0_left_shoulder_pitch_joint'),
                ('left_shoulder_roll_joint', 'robot0_left_shoulder_roll_joint'),
                ('left_shoulder_yaw_joint', 'robot0_left_shoulder_yaw_joint'),
                ('left_elbow_joint', 'robot0_left_elbow_joint'),
                ('left_wrist_roll_joint', 'robot0_left_wrist_roll_joint'),
                ('left_wrist_pitch_joint', 'robot0_left_wrist_pitch_joint'),
                ('left_wrist_yaw_joint', 'robot0_left_wrist_yaw_joint')
            ]
            
            for timestep in range(len(planned_trajectory.position)):
                try:
                    current_base_env = self.get_base_environment()
                    
                    planned_joints = planned_trajectory.position[timestep].cpu().numpy()
                    
                    # CUROBO 계획된 joint 값을 직접 설정
                    for i, (curobo_name, sim_name) in enumerate(joint_mapping):
                        try:
                            joint_id = current_base_env.sim.model.joint_name2id(sim_name)
                            current_base_env.sim.data.qpos[joint_id] = planned_joints[i]
                            current_base_env.sim.data.qvel[joint_id] = 0.0
                        except Exception as e:
                            print(f"    Warning: Failed to set {sim_name}: {e}")
                    
                    # 시뮬레이션 forward pass 수행
                    current_base_env.sim.forward()
                    
                    # 비디오 녹화를 위해 더미 액션으로 step 수행하되, joint 값은 다시 강제 설정
                    n_action_steps = config.multistep.n_action_steps
                    dummy_action = {k: np.zeros((1, n_action_steps, v.shape[-1]), dtype=np.float32) 
                                   for k, v in self.env.single_action_space.spaces.items() if hasattr(v, 'shape')}
                    
                    obs, rewards, terminations, truncations, env_infos = self.env.step(dummy_action)
                    
                    # env.step() 후에 joint 값이 변경되었을 수 있으므로 다시 강제 설정
                    current_base_env = self.get_base_environment()
                    for i, (curobo_name, sim_name) in enumerate(joint_mapping):
                        try:
                            joint_id = current_base_env.sim.model.joint_name2id(sim_name)
                            current_base_env.sim.data.qpos[joint_id] = planned_joints[i]
                            current_base_env.sim.data.qvel[joint_id] = 0.0
                        except Exception as e:
                            pass
                    
                    current_base_env.sim.forward()
                    
                    if timestep % 10 == 0 or timestep == len(planned_trajectory.position) - 1:
                        print(f"  Step {timestep+1}/{len(planned_trajectory.position)}")
                        print(f"    CUROBO planned joint values: {[f'{x:.3f}' for x in planned_joints]}")
                        
                        # 시뮬레이션에서 실제 joint 값 확인
                        sim_joint_values = []
                        for i, (curobo_name, sim_name) in enumerate(joint_mapping):
                            try:
                                joint_id = current_base_env.sim.model.joint_name2id(sim_name)
                                sim_joint_values.append(current_base_env.sim.data.qpos[joint_id])
                            except:
                                sim_joint_values.append(0.0)
                        
                        print(f"    Simulation actual joint values: {[f'{x:.3f}' for x in sim_joint_values]}")
                        
                        # 값이 일치하는지 확인
                        max_diff = max(abs(planned_joints[i] - sim_joint_values[i]) for i in range(len(planned_joints)))
                        print(f"    Maximum difference: {max_diff:.6f}")
                    
                    if terminations[0] or truncations[0]:
                        print("  Episode terminated early.")
                        break
                        
                except Exception as e:
                    print(f"✗ Step {timestep} failed: {e}")
                    continue

            # 6. 최종 EEF 위치 확인
            print(f"\n--- Final EEF Position Analysis ---")
            try:
                palm_body_id = current_base_env.sim.model.body_name2id('robot0_left_eef')
                final_palm_position = current_base_env.sim.data.body_xpos[palm_body_id].copy()
                
                distance_to_cup = np.linalg.norm(final_palm_position - cup_pos_world)
                print(f"Final EEF position (world): {final_palm_position}")
                print(f"Cup position (world): {cup_pos_world}")
                print(f"Final distance to cup: {distance_to_cup:.6f}m")
                
                if distance_to_cup < 0.05:
                    print(f"🎉 SUCCESS: EEF reached cup! (distance: {distance_to_cup:.6f}m)")
                elif distance_to_cup < 0.1:
                    print(f"✅ CLOSE: EEF near cup (distance: {distance_to_cup:.6f}m)")
                else:
                    print(f"📍 INFO: EEF distance from cup: {distance_to_cup:.6f}m")
                    
            except Exception as e:
                print(f"Error calculating final EEF position: {e}")

        print("\nFinalizing simulation...")
        self.env.close()
        self.env = None
        print(f"✓ Simulation completed in {time.time() - start_time:.2f} seconds")
        if config.video.video_dir:
            print(f"✓ Videos saved to: {config.video.video_dir}")

def main():
    """Main function to run the G1 tabletop simulation with CUROBO."""
    parser = argparse.ArgumentParser(description="G1 Robot Tabletop Simulation with CUROBO Motion Planning")
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
    print("=" * 60)
    
    video_config = VideoConfig(video_dir=args.video_dir, fps=args.fps)
    config = SimulationConfig(
        env_name=args.env, n_episodes=args.episodes,
        video=video_config,
        multistep=MultiStepConfig(max_episode_steps=args.steps),
        multi_video=args.multiview
    )
    
    simulation = G1SimulationWithCurobo(robot_name=args.robot)
    
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