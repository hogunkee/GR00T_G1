#!/bin/bash

GPU_ID=$1
TASK_NAME=$2

CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/simulation_service.py --client --env_name ${TASK_NAME} --video_dir ./videos/${TASK_NAME} --max_episode_steps 720 --n_envs 5 --n_episodes 25 --log
