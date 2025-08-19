# src/env_utils.py
import datetime, uuid
from copy import deepcopy
import gymnasium as gym
import numpy as np
import os
import sys

# Add robocasa and robosuite paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'robosuite_g1'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'robocasa_g1'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'gr00t'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'curobo', 'src'))


import robocasa  # we need this to register environments  # noqa: F401
import robosuite
from robocasa.utils.gym_utils import GrootRoboCasaEnv  # 이게 g1_unified 환경들을 등록함
from gymnasium import spaces
from robocasa.environments.tabletop.tabletop import Tabletop
from robocasa.models.robots import (
    GROOT_ROBOCASA_ENVS_GR1_FULL,
    GROOT_ROBOCASA_ENVS_GR1_ARMS_ONLY,
    GROOT_ROBOCASA_ENVS_GR1_ARMS_AND_WAIST,
    GROOT_ROBOCASA_ENVS_GR1_FIXED_LOWER_BODY,
    GROOT_ROBOCASA_ENVS_G1_FULL,
    GROOT_ROBOCASA_ENVS_G1_ARMS_ONLY,
    GROOT_ROBOCASA_ENVS_G1_ARMS_AND_WAIST,
    GROOT_ROBOCASA_ENVS_G1_FIXED_LOWER_BODY,
    gather_robot_observations,
    make_key_converter,
)
from robosuite.controllers import load_composite_controller_config
from robosuite.controllers.parts.arm.osc import OperationalSpaceController
from robosuite.controllers.composite.composite_controller import HybridMobileBase
from robosuite.environments.base import REGISTERED_ENVS


ALLOWED_LANGUAGE_CHARSET = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.\n\t[]{}()!?'_:"
)


def create_env_robosuite(
    env_name,
    # robosuite-related configs
    robots="PandaOmron",
    controller_configs=None,
    camera_names=[
        "egoview",
        "robot0_eye_in_left_hand",
        "robot0_eye_in_right_hand",
    ],
    camera_widths=128,
    camera_heights=128,
    enable_render=True,
    seed=None,
    # robocasa-related configs
    obj_instance_split=None,
    generative_textures=None,
    randomize_cameras=False,
    layout_and_style_ids=None,
    layout_ids=None,
    style_ids=None,
    **kwargs,  # Accept additional kwargs for environment
):
    if controller_configs is None:
        controller_configs = load_composite_controller_config(
            controller=None,
            robot=robots if isinstance(robots, str) else robots[0],
        )
    env_kwargs = dict(
        env_name=env_name,
        robots=robots,
        controller_configs=controller_configs,
        camera_names=camera_names,
        camera_widths=camera_widths,
        camera_heights=camera_heights,
        has_renderer=False,
        has_offscreen_renderer=enable_render,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=enable_render,
        camera_depths=False,
        seed=seed,
        translucent_robot=False,
        **kwargs,  # Include additional kwargs
    )
    env_class = REGISTERED_ENVS[env_name]

    env = robosuite.make(**env_kwargs)
    return env, env_kwargs


class RoboCasaEnv(gym.Env):
    def __init__(
        self,
        env_name=None,
        robots_name=None,
        camera_names=None,
        camera_widths=None,
        camera_heights=None,
        enable_render=True,
        dump_rollout_dataset_dir=None,
        **kwargs,  # Accept additional kwargs
    ):
        self.key_converter = make_key_converter(robots_name)
        (
            _,
            camera_names,
            default_camera_widths,
            default_camera_heights,
        ) = self.key_converter.get_camera_config()

        if camera_widths is None:
            camera_widths = default_camera_widths
        if camera_heights is None:
            camera_heights = default_camera_heights

        controller_configs = load_composite_controller_config(
            controller=None,
            robot=robots_name.split("_")[0],
        )
        if (
            robots_name in GROOT_ROBOCASA_ENVS_GR1_FULL
            or robots_name in GROOT_ROBOCASA_ENVS_GR1_ARMS_ONLY
            or robots_name in GROOT_ROBOCASA_ENVS_GR1_ARMS_AND_WAIST
            or robots_name in GROOT_ROBOCASA_ENVS_GR1_FIXED_LOWER_BODY
            or robots_name in GROOT_ROBOCASA_ENVS_G1_FULL
            or robots_name in GROOT_ROBOCASA_ENVS_G1_ARMS_ONLY
            or robots_name in GROOT_ROBOCASA_ENVS_G1_ARMS_AND_WAIST
            or robots_name in GROOT_ROBOCASA_ENVS_G1_FIXED_LOWER_BODY
        ):
            controller_configs["type"] = "BASIC"
            controller_configs["composite_controller_specific_configs"] = {}
            controller_configs["control_delta"] = False

        self.env, self.env_kwargs = create_env_robosuite(
            env_name=env_name,
            robots=robots_name.split("_"),
            controller_configs=controller_configs,
            camera_names=camera_names,
            camera_widths=camera_widths,
            camera_heights=camera_heights,
            enable_render=enable_render,
            **kwargs,  # Forward kwargs to create_env_robosuite
        )

        # TODO: the following info should be output by grootrobocasa
        self.camera_names = camera_names
        self.camera_widths = camera_widths
        self.camera_heights = camera_heights
        self.enable_render = enable_render
        self.render_obs_key = f"{camera_names[0]}_image"
        self.render_cache = None

        # setup spaces
        action_space = spaces.Dict()
        for robot in self.env.robots:
            cc = robot.composite_controller
            pf = robot.robot_model.naming_prefix
            for part_name, controller in cc.part_controllers.items():
                min_value, max_value = -1, 1
                start_idx, end_idx = cc._action_split_indexes[part_name]
                shape = [end_idx - start_idx]
                this_space = spaces.Box(
                    low=min_value, high=max_value, shape=shape, dtype=np.float32
                )
                action_space[f"{pf}{part_name}"] = this_space
            if isinstance(cc, HybridMobileBase):
                this_space = spaces.Discrete(2)
                action_space[f"{pf}base_mode"] = this_space

            action_space = spaces.Dict(action_space)
            self.action_space = action_space

        obs = (
            self.env.viewer._get_observations(force_update=True)
            if self.env.viewer_get_obs
            else self.env._get_observations(force_update=True)
        )
        obs.update(gather_robot_observations(self.env))
        observation_space = spaces.Dict()
        for obs_name, obs_value in obs.items():
            shape = list(obs_value.shape)
            if obs_name.endswith("_image"):
                continue
            min_value, max_value = -1, 1
            this_space = spaces.Box(
                low=min_value, high=max_value, shape=shape, dtype=np.float32
            )
            observation_space[obs_name] = this_space

        for camera_name in camera_names:
            shape = [camera_heights, camera_widths, 3]
            this_space = spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
            observation_space[f"{camera_name}_image"] = this_space

        observation_space["language"] = spaces.Text(
            max_length=256, charset=ALLOWED_LANGUAGE_CHARSET
        )

        self.observation_space = observation_space

        self.dump_rollout_dataset_dir = dump_rollout_dataset_dir
        self.groot_exporter = None
        self.np_exporter = None

    def get_basic_observation(self, raw_obs):
        raw_obs.update(gather_robot_observations(self.env))

        # Image are in (H, W, C), flip it upside down
        def process_img(img):
            return np.copy(img[::-1, :, :])

        for obs_name, obs_value in raw_obs.items():
            if obs_name.endswith("_image"):
                # image observations
                raw_obs[obs_name] = process_img(obs_value)
            else:
                # non-image observations
                raw_obs[obs_name] = obs_value.astype(np.float32)

        # Return black image if rendering is disabled
        if not self.enable_render:
            for name in self.camera_names:
                raw_obs[f"{name}_image"] = np.zeros(
                    (self.camera_heights, self.camera_widths, 3), dtype=np.uint8
                )

        self.render_cache = raw_obs[self.render_obs_key]
        raw_obs["language"] = self.env.get_ep_meta().get("lang", "")

        return raw_obs

    def reset(self, seed=None, options=None):
        np.random.seed(seed)
        raw_obs = self.env.reset()
        # return obs
        obs = self.get_basic_observation(raw_obs)

        info = {}
        info["success"] = False
        info["grasp_distractor_obj"] = False

        return obs, info

    def step(self, action_dict):
        env_action = []
        for robot in self.env.robots:
            cc = robot.composite_controller
            pf = robot.robot_model.naming_prefix
            action = np.zeros(cc.action_limits[0].shape)
            for part_name, controller in cc.part_controllers.items():
                start_idx, end_idx = cc._action_split_indexes[part_name]
                act = action_dict.pop(f"{pf}{part_name}")
                action[start_idx:end_idx] = act
            if isinstance(cc, HybridMobileBase):
                action[-1] = action_dict.pop(f"{pf}base_mode")
            env_action.append(action)

        assert len(action_dict) == 0, f"Unprocessed actions: {action_dict}"
        env_action = np.concatenate(env_action)

        raw_obs, reward, done, info = self.env.step(env_action)

        obs = self.get_basic_observation(raw_obs)

        truncated = False

        info["success"] = reward > 0
        info["grasp_distractor_obj"] = False
        if hasattr(self, "_check_grasp_distractor_obj"):
            info["grasp_distractor_obj"] = self._check_grasp_distractor_obj()

        return obs, reward, done, truncated, info

    def render(self):
        if self.render_cache is None:
            raise RuntimeError("Must run reset or step before render.")
        return self.render_cache

    def close(self):
        self.env.close()

import gymnasium as gym
from pathlib import Path
from gr00t.eval.wrappers import VideoRecordingWrapper, MultiVideoRecordingWrapper, VideoRecorder, MultiVideoRecorder
from gr00t.eval.wrappers.multistep_wrapper import MultiStepWrapper
import numpy as np
from curobo.geom.types import Pose, WorldConfig
import torch
from functools import partial
import logging
from config import SimulationConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def setup_environment(config: SimulationConfig):
    env_fns = [partial(create_single_env, config=config, idx=i) for i in range(config.n_envs)]
    env = gym.vector.SyncVectorEnv(env_fns) if config.n_envs == 1 else gym.vector.AsyncVectorEnv(env_fns, shared_memory=False, context="spawn")
    logging.info(f"Environment setup: {config.env_name}")
    return env

def create_single_env(config, idx):
    # demo.py처럼 gym.make() 직접 사용
    env_kwargs = {
        'enable_render': True,
        'layout_ids': 0,
        'style_ids': 0
    }
    
    # 컵 위치가 설정된 경우 전달
    if config.cup_table_pos is not None:
        env_kwargs['cup_table_pos'] = config.cup_table_pos
        logging.info(f"Creating environment with cup_table_pos: {config.cup_table_pos}")
    
    env = gym.make(config.env_name, **env_kwargs)
    logging.info(f"✓ Base environment created successfully")
    
    # demo.py처럼 robot deterministic reset 적용
    if hasattr(env, 'envs') and len(env.envs) > 0:
        base_env = env.envs[0]
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        
        if hasattr(base_env, 'robots') and len(base_env.robots) > 0:
            robot = base_env.robots[0]
            robot.initialization_noise = {"magnitude": 0.0, "type": "gaussian"}
            robot.reset(deterministic=True)
            logging.info(f"✓ Robot deterministic reset applied")

    if config.video.video_dir:
        video_path = Path(config.video.video_dir)
        video_path.mkdir(parents=True, exist_ok=True)
        recorder_class = MultiVideoRecorder if config.multi_video else VideoRecorder
        wrapper_class = MultiVideoRecordingWrapper if config.multi_video else VideoRecordingWrapper
        video_recorder = recorder_class.create_h264(
            fps=config.video.fps, codec=config.video.codec, input_pix_fmt=config.video.input_pix_fmt,
            crf=config.video.crf, thread_type=config.video.thread_type, thread_count=config.video.thread_count,
        )
        env = wrapper_class(env, video_recorder, video_dir=video_path, steps_per_render=config.video.steps_per_render)
    
    env = MultiStepWrapper(
        env, video_delta_indices=config.multistep.video_delta_indices,
        state_delta_indices=config.multistep.state_delta_indices, n_action_steps=config.multistep.n_action_steps
    )
    return env

def get_base_environment(env):
    base_env = env.envs[0] if hasattr(env, 'envs') else env
    while hasattr(base_env, 'env'): base_env = base_env.env
    return base_env

def get_current_state(base_env):
    current_q = get_planning_joint_positions(base_env)
    robot_base_pose = get_robot_base_pose(base_env)
    cup_pose = get_cup_pose(base_env)
    table_obstacle = get_table_obstacle(base_env)
    state = {"joint_positions": current_q, "base_pose": robot_base_pose, "cup_pose": cup_pose, "obstacles": [table_obstacle] if table_obstacle else []}
    logging.info(f"State: cup_pose={cup_pose.position.cpu().numpy()[0] if cup_pose else None}")
    return state

def get_planning_joint_positions(base_env):
    planning_joints = [
        'robot0_waist_yaw_joint', 'robot0_waist_roll_joint', 'robot0_waist_pitch_joint',
        'robot0_left_shoulder_pitch_joint', 'robot0_left_shoulder_roll_joint', 'robot0_left_shoulder_yaw_joint',
        'robot0_left_elbow_joint', 'robot0_left_wrist_roll_joint', 'robot0_left_wrist_pitch_joint', 'robot0_left_wrist_yaw_joint'
    ]
    joint_positions = [base_env.sim.data.qpos[base_env.sim.model.joint_name2id(j)] if j in base_env.sim.model.joint_names else 0.0 for j in planning_joints]
    return np.array(joint_positions)

def get_robot_base_pose(base_env):
    try:
        base_body_id = base_env.sim.model.body_name2id('robot0_base')
        position = torch.tensor(base_env.sim.data.body_xpos[base_body_id], dtype=torch.float32).unsqueeze(0).cuda()
        quaternion = torch.tensor(base_env.sim.data.body_xquat[base_body_id], dtype=torch.float32).unsqueeze(0).cuda()
        return Pose(position=position, quaternion=quaternion)
    except Exception as e:
        logging.error(f"Robot base pose error: {e}")
        return None

def get_cup_pose(base_env):
    try:
        first_obj = base_env.objects[0] if base_env.objects else None
        target_obj_name = first_obj.name + "_main" if first_obj and first_obj.name + "_main" in base_env.sim.model.body_names else first_obj.name
        cup_body_id = base_env.sim.model.body_name2id(target_obj_name)
        pos = torch.tensor(base_env.sim.data.xpos[cup_body_id], dtype=torch.float32).unsqueeze(0).cuda()
        quat = torch.tensor(base_env.sim.data.xquat[cup_body_id], dtype=torch.float32).unsqueeze(0).cuda()
        logging.info(f"Cup pose: pos={pos.cpu().numpy()[0]}, quat={quat.cpu().numpy()[0]}")
        return Pose(position=pos, quaternion=quat)
    except Exception as e:
        logging.error(f"Cup pose error: {e}")
        return None

def get_table_obstacle(base_env):
    from scipy.spatial.transform import Rotation
    try:
        table_body_names = ['table_main_group_main', 'table', 'table_main', 'counter', 'counter_main']
        table_body_name = next((name for name in table_body_names if name in base_env.sim.model.body_names), None)
        if not table_body_name:
            return None
        table_body_id = base_env.sim.model.body_name2id(table_body_name)
        table_position = base_env.sim.data.body_xpos[table_body_id].copy()
        table_quaternion = base_env.sim.data.body_xquat[table_body_id].copy()
        table_size = [0.61, 0.375, 0.92]
        for geom_id in range(base_env.sim.model.ngeom):
            if base_env.sim.model.geom_bodyid[geom_id] == table_body_id:
                geom_size = base_env.sim.model.geom_size[geom_id].copy()
                if sum(geom_size) > sum(table_size): table_size = geom_size
        pelvis_body_id = base_env.sim.model.body_name2id('robot0_base')
        pelvis_position = base_env.sim.data.body_xpos[pelvis_body_id].copy()
        pelvis_quaternion = base_env.sim.data.body_xquat[pelvis_body_id].copy()
        table_rot = Rotation.from_quat([table_quaternion[1], table_quaternion[2], table_quaternion[3], table_quaternion[0]])
        pelvis_rot = Rotation.from_quat([pelvis_quaternion[1], pelvis_quaternion[2], pelvis_quaternion[3], pelvis_quaternion[0]])
        table_pos_relative = table_position - pelvis_position
        table_pos_pelvis_frame = pelvis_rot.as_matrix().T @ table_pos_relative
        relative_table_rot = table_rot * pelvis_rot.inv()
        table_quat_pelvis_frame = relative_table_rot.as_quat()
        return WorldConfig.from_dict({
            "table": {
                "type": "box", 
                "pose": list(table_pos_pelvis_frame) + list(table_quat_pelvis_frame),
                "dims": list(table_size)
            }
        })
    except Exception as e:
        logging.error(f"Table obstacle error: {e}")
        return None

def set_cup_position(base_env, world_pos):
    try:
        first_obj = base_env.objects[0] if base_env.objects else None
        target_obj_name = first_obj.name + "_main" if first_obj and first_obj.name + "_main" in base_env.sim.model.body_names else first_obj.name
        cup_body_id = base_env.sim.model.body_name2id(target_obj_name)
        base_env.sim.data.xpos[cup_body_id] = world_pos
        base_env.sim.forward()
        logging.info(f"Cup moved to world pos: {world_pos}")
    except Exception as e:
        logging.error(f"Set cup position error: {e}")