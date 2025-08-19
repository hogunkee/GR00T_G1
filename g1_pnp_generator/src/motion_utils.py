# src/motion_utils.py
import sys
import os

# Add curobo path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'curobo', 'src'))

import curobo
from curobo.geom.types import Pose
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
from curobo.types.robot import RobotConfig
from curobo.types.state import JointState
import torch
import numpy as np
from scipy.spatial.transform import Rotation
from coordinate_utils import table_to_world
import logging
import yaml
from config import SimulationConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def _ease_in_out_cosine(t):
    return -0.5 * (np.cos(np.pi * t) - 1)

def setup_curobo_planner():
    config_file = os.path.join(os.path.dirname(__file__), "../configs/curobo_config.yaml")
    with open(config_file, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    robot_cfg = RobotConfig.from_dict(config_dict)
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg, world_model=None, use_cuda_graph=True,
        num_ik_seeds=500, num_graph_seeds=200, num_trajopt_seeds=200
    )
    planner = MotionGen(motion_gen_config)
    logging.info("CUROBO planner initialized")
    return planner

def convert_curobo_to_action(env, planned_joints, n_action_steps, grip_state='open'):
    action_space = env.single_action_space
    action = {k: np.zeros((1, n_action_steps, v.shape[-1]), dtype=np.float32) 
              for k, v in action_space.spaces.items() if hasattr(v, 'shape')}
    waist_joints = planned_joints[0:3]
    left_arm_joints = planned_joints[3:10]
    open_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    close_pose = np.array([-0.8, -0.8, -0.8, -0.8, 0.8, 0.4, 0.4])
    gripper_pose = close_pose if grip_state == 'close' else open_pose
    for sub_step in range(n_action_steps):
        if 'action.waist' in action:
            action['action.waist'][0, sub_step, :] = waist_joints
        if 'action.left_arm' in action:
            action['action.left_arm'][0, sub_step, :] = left_arm_joints
        if 'action.right_arm' in action:
            action['action.right_arm'][0, sub_step, :] = np.array([0.0, -0.1, 0.0, -0.2, 0.0, 0.0, 0.0])
        if 'action.left_hand' in action:
            action['action.left_hand'][0, sub_step, :] = gripper_pose
    return action

def grip(env, duration=2):
    base_env = get_base_environment(env)
    state = get_current_state(base_env)
    waist = state['joint_positions'][0:3]
    left_arm = state['joint_positions'][3:10]
    right_arm = np.array([0.0, -0.1, 0.0, -0.2, 0.0, 0.0, 0.0])
    n_action_steps = env.n_action_steps if hasattr(env, 'n_action_steps') else 16  # MultiStepWrapper에서 가져옴
    total_sub_steps = duration * n_action_steps
    close_phase_end = total_sub_steps // 3
    open_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    close_pose = np.array([-0.8, -0.8, -0.8, -0.8, 0.8, 0.4, 0.4])
    
    for step in range(duration):
        action = {k: np.zeros((1, n_action_steps, v.shape[-1]), dtype=np.float32) 
                  for k, v in env.single_action_space.spaces.items() if hasattr(v, 'shape')}
        for sub_step in range(n_action_steps):
            current_sub = step * n_action_steps + sub_step
            if current_sub < close_phase_end:
                linear_progress = current_sub / close_phase_end
                eased_progress = _ease_in_out_cosine(linear_progress)
                gripper_action = open_pose + eased_progress * (close_pose - open_pose)
            else:
                gripper_action = close_pose
            if 'action.left_hand' in action:
                action['action.left_hand'][0, sub_step, :] = gripper_action
            if 'action.waist' in action:
                action['action.waist'][0, sub_step, :] = waist
            if 'action.left_arm' in action:
                action['action.left_arm'][0, sub_step, :] = left_arm
            if 'action.right_arm' in action:
                action['action.right_arm'][0, sub_step, :] = right_arm
        env.step(action)
    logging.info("Gripper closed smoothly")

def release(env, duration=2):
    base_env = get_base_environment(env)
    state = get_current_state(base_env)
    waist = state['joint_positions'][0:3]
    left_arm = state['joint_positions'][3:10]
    right_arm = np.array([0.0, -0.1, 0.0, -0.2, 0.0, 0.0, 0.0])
    n_action_steps = env.n_action_steps if hasattr(env, 'n_action_steps') else 16
    total_sub_steps = duration * n_action_steps
    open_phase_end = total_sub_steps // 3
    close_pose = np.array([-0.8, -0.8, -0.8, -0.8, 0.8, 0.4, 0.4])
    open_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    for step in range(duration):
        action = {k: np.zeros((1, n_action_steps, v.shape[-1]), dtype=np.float32) 
                  for k, v in env.single_action_space.spaces.items() if hasattr(v, 'shape')}
        for sub_step in range(n_action_steps):
            current_sub = step * n_action_steps + sub_step
            if current_sub < open_phase_end:
                linear_progress = current_sub / open_phase_end
                eased_progress = _ease_in_out_cosine(linear_progress)
                gripper_action = close_pose + eased_progress * (open_pose - close_pose)
            else:
                gripper_action = open_pose
            if 'action.left_hand' in action:
                action['action.left_hand'][0, sub_step, :] = gripper_action
            if 'action.waist' in action:
                action['action.waist'][0, sub_step, :] = waist
            if 'action.left_arm' in action:
                action['action.left_arm'][0, sub_step, :] = left_arm
            if 'action.right_arm' in action:
                action['action.right_arm'][0, sub_step, :] = right_arm
        env.step(action)
    logging.info("Gripper opened smoothly")

def pick_and_place(env, A_table, B_table, height_offset=0.1):
    base_env = get_base_environment(env)
    planner = setup_curobo_planner()
    state = get_current_state(base_env)
    if not state["cup_pose"] or not state["base_pose"]:
        logging.error("Missing cup or robot pose")
        return

    # A로 이동 (grip open)
    A_world = table_to_world(A_table, height_offset)
    pelvis_pos = state["base_pose"].position[0].cpu().numpy()
    pelvis_quat = state["base_pose"].quaternion[0].cpu().numpy()
    pelvis_rot = Rotation.from_quat([pelvis_quat[1], pelvis_quat[2], pelvis_quat[3], pelvis_quat[0]])
    A_pos_relative = np.array(A_world) - pelvis_pos
    A_pos_pelvis_frame = pelvis_rot.as_matrix().T @ A_pos_relative
    target_pose = Pose(
        position=torch.tensor(A_pos_pelvis_frame, dtype=torch.float32).unsqueeze(0).cuda(),
        quaternion=state["cup_pose"].quaternion  # 현재 EEF 방향 유지
    )
    joint_positions = torch.tensor(state["joint_positions"], dtype=torch.float32).unsqueeze(0).cuda()
    result = planner.plan_single(
        JointState.from_position(joint_positions, joint_names=[
            'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
            'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
            'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint'
        ]), target_pose
    )
    if not result.success.item():
        logging.error("Planning to A failed")
        return
    traj_A = result.get_interpolated_plan()
    logging.info(f"Planned to A: {len(traj_A.position)} steps")
    
    for t, pos in enumerate(traj_A.position):
        action = convert_curobo_to_action(env, pos.cpu().numpy(), env.single_action_space['action.waist'].shape[1] if 'action.waist' in env.single_action_space else 3, grip_state='open')
        env.step(action)
        if t % 10 == 0:
            logging.info(f"Step {t+1}/{len(traj_A.position)}: joints={pos.cpu().numpy()}")
    
    # Grip close (2 steps)
    grip(env, duration=2)
    
    # B로 이동 (grip close 유지)
    B_world = table_to_world(B_table, height_offset)
    B_pos_relative = np.array(B_world) - pelvis_pos
    B_pos_pelvis_frame = pelvis_rot.as_matrix().T @ B_pos_relative
    target_pose = Pose(
        position=torch.tensor(B_pos_pelvis_frame, dtype=torch.float32).unsqueeze(0).cuda(),
        quaternion=state["cup_pose"].quaternion
    )
    joint_positions = torch.tensor(get_current_state(base_env)["joint_positions"], dtype=torch.float32).unsqueeze(0).cuda()
    result = planner.plan_single(JointState.from_position(joint_positions), target_pose)
    if not result.success.item():
        logging.error("Planning to B failed")
        return
    traj_B = result.get_interpolated_plan()
    logging.info(f"Planned to B: {len(traj_B.position)} steps")
    
    for t, pos in enumerate(traj_B.position):
        action = convert_curobo_to_action(env, pos.cpu().numpy(), env.single_action_space['action.waist'].shape[1] if 'action.waist' in env.single_action_space else 3, grip_state='close')
        env.step(action)
        # 컵 위치 EEF에 동기화 (grip 중)
        palm_body_id = base_env.sim.model.body_name2id('robot0_left_eef')
        eef_pos = base_env.sim.data.body_xpos[palm_body_id].copy()
        set_cup_position(base_env, eef_pos)
        if t % 10 == 0:
            logging.info(f"Step {t+1}/{len(traj_B.position)}: joints={pos.cpu().numpy()}")
    
    # Release open (2 steps)
    release(env, duration=2)
    
    # 최종 EEF-컵 거리 확인
    palm_body_id = base_env.sim.model.body_name2id('robot0_left_eef')
    final_palm_pos = base_env.sim.data.body_xpos[palm_body_id].copy()
    distance = np.linalg.norm(final_palm_pos - np.array(B_world))
    logging.info(f"Final EEF pos: {final_palm_pos}, Cup pos: {B_world}, Distance: {distance:.6f}m")

def get_base_environment(env):
    """Get the base robosuite environment."""
    base_env = env.envs[0] if hasattr(env, 'envs') else env
    while hasattr(base_env, 'env'): 
        base_env = base_env.env
    return base_env

def get_current_state(base_env):
    """Extracts all necessary state information from the simulation."""
    current_q = get_planning_joint_positions(base_env)
    robot_base_pose = get_robot_base_pose(base_env)
    cup_pose = get_cup_pose(base_env)
    table_obstacle = get_table_obstacle(base_env)
    state = {"joint_positions": current_q, "base_pose": robot_base_pose, "cup_pose": cup_pose, "obstacles": [table_obstacle] if table_obstacle else []}
    logging.info(f"State: cup_pose={cup_pose.position.cpu().numpy()[0] if cup_pose else None}")
    return state

def get_planning_joint_positions(base_env):
    """Gets the positions of the joints that CUROBO will plan for (10 DOF)."""
    planning_joints = [
        'robot0_waist_yaw_joint', 'robot0_waist_roll_joint', 'robot0_waist_pitch_joint',
        'robot0_left_shoulder_pitch_joint', 'robot0_left_shoulder_roll_joint', 'robot0_left_shoulder_yaw_joint',
        'robot0_left_elbow_joint', 'robot0_left_wrist_roll_joint', 'robot0_left_wrist_pitch_joint', 'robot0_left_wrist_yaw_joint'
    ]
    joint_positions = [base_env.sim.data.qpos[base_env.sim.model.joint_name2id(j)] if j in base_env.sim.model.joint_names else 0.0 for j in planning_joints]
    return np.array(joint_positions)

def get_robot_base_pose(base_env):
    """Get robot base world pose."""
    try:
        base_body_id = base_env.sim.model.body_name2id('robot0_base')
        position = torch.tensor(base_env.sim.data.body_xpos[base_body_id], dtype=torch.float32).unsqueeze(0).cuda()
        quaternion = torch.tensor(base_env.sim.data.body_xquat[base_body_id], dtype=torch.float32).unsqueeze(0).cuda()
        return Pose(position=position, quaternion=quaternion)
    except Exception as e:
        logging.error(f"Robot base pose error: {e}")
        return None

def get_cup_pose(base_env):
    """Get cup world pose."""
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
    """Get table obstacle for CUROBO."""
    from curobo.geom.types import WorldConfig
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
                if sum(geom_size) > sum(table_size): 
                    table_size = geom_size
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
    """Set cup position in world coordinates."""
    try:
        first_obj = base_env.objects[0] if base_env.objects else None
        target_obj_name = first_obj.name + "_main" if first_obj and first_obj.name + "_main" in base_env.sim.model.body_names else first_obj.name
        cup_body_id = base_env.sim.model.body_name2id(target_obj_name)
        base_env.sim.data.xpos[cup_body_id] = world_pos
        base_env.sim.forward()
        logging.info(f"Cup moved to world pos: {world_pos}")
    except Exception as e:
        logging.error(f"Set cup position error: {e}")