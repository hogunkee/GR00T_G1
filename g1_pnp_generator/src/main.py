# src/main.py
#!/usr/bin/env python3
"""
Main script for G1 Robot Tabletop PnP Simulation with CUROBO.
"""
import argparse
from config import parse_args, SimulationConfig
from env_utils import setup_environment, get_base_environment, set_cup_position
from motion_utils import pick_and_place
from coordinate_utils import table_to_world
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    args = parse_args()
    
    # A_table 위치 정의 (초기 컵 위치)
    A_table = (-0.5, -0.5)  # 초기 컵 위치
    B_table = (-0.3, -0)  # 최종 목표 컵 위치
    
    # config 생성 시 컵 위치 전달
    config = SimulationConfig.from_args(args)
    config.cup_table_pos = A_table  # A_table 위치로 컵 초기 위치 설정
    
    env = setup_environment(config)
    logging.info(f"Starting simulation: {config.env_name}, {config.n_episodes} episodes")
    logging.info(f"Cup will be placed at table position: {A_table}")

    for ep in range(config.n_episodes):
        logging.info(f"Episode {ep+1}/{config.n_episodes}")
        obs, _ = env.reset()
        base_env = get_base_environment(env)
        
        # demo.py처럼 robot reset을 환경 reset 후에 다시 한 번 실행
        if hasattr(base_env, 'robots') and len(base_env.robots) > 0:
            robot = base_env.robots[0]
            robot.reset(deterministic=True)
            logging.info("✓ Robot deterministic reset applied after environment reset")
        
        pick_and_place(env, A_table, B_table)
    
    env.close()
    logging.info("Simulation completed")

if __name__ == "__main__":
    main()