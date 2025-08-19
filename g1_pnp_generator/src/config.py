# src/config.py
from dataclasses import dataclass, field
import argparse
import numpy as np

@dataclass
class VideoConfig:
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
    video_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    state_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    n_action_steps: int = 16
    max_episode_steps: int = 500

@dataclass
class SimulationConfig:
    env_name: str
    n_episodes: int = 1
    n_envs: int = 1
    video: VideoConfig = field(default_factory=VideoConfig)
    multistep: MultiStepConfig = field(default_factory=MultiStepConfig)
    multi_video: bool = False
    grip_height_offset: float = 0.1  # 그리핑 오프셋 (m)
    robot: str = "G1ArmsAndWaistDex31Hands"  # 기본 로봇 이름
    cup_table_pos: tuple = None  # 컵 테이블 좌표 (x, y)

    @classmethod
    def from_args(cls, args):
        video_config = VideoConfig(video_dir=args.video_dir, fps=args.fps)
        multistep_config = MultiStepConfig(max_episode_steps=args.steps)
        return cls(
            env_name=args.env,
            n_episodes=args.episodes,
            video=video_config,
            multistep=multistep_config,
            multi_video=args.multiview,
            robot=args.robot
        )

def parse_args():
    parser = argparse.ArgumentParser(description="G1 Robot Tabletop PnP Simulation")
    parser.add_argument("--steps", type=int, default=500, help="Steps per episode")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes")
    parser.add_argument("--robot", type=str, default="G1ArmsAndWaistDex31Hands", help="Robot variant")
    parser.add_argument("--env", type=str, default="g1_unified/PnPCupToPlateNoDistractors_G1ArmsAndWaistDex31Hands_Env", help="Environment")
    parser.add_argument("--video_dir", type=str, default=None, help="Video save directory")
    parser.add_argument("--multiview", action="store_true", help="Enable multi-view video")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")
    return parser.parse_args()