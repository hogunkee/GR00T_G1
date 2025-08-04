#!/bin/bash

GPU_ID=$1
TASK_NAME=$2
TAG=$3

CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/simulation_service.py --client --env_name ${TASK_NAME} --video_dir ./videos/${TAG}_${TASK_NAME} --max_episode_steps 720 --n_envs 1 --n_episodes 5 --log
