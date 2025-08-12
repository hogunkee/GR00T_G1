#!/bin/bash

GPU_ID=$1
TASK_NAME=$2
TAG=$3

CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/simulation_service.py --client --env_name ${TASK_NAME} --video_dir ./videos/${TAG}/${TASK_NAME} --max_episode_steps 360 --n_envs 5 --n_episodes 10 --log --multi_video --embodiment_tag g1
