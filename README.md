<div align="center">
<img src="media/g1_gr00t_transfer_frontview.gif" width="45%">
<img src="media/g1_gr00t_transfer_egoview.gif" width="45%">
</div>

# GR00T-G1

**GR00T-G1** is an adaptation of NVIDIA's [Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T) project for the **Unitree G1 humanoid robot**.
This repository modifies the original GR00T framework to integrate G1's kinematics, dynamics, and control interface for simulation and potential real-world deployment.

### Sim Env: [robocasa-g1-tabletop-tasks](https://github.com/hogunkee/robocasa_g1)

For Simulation Evaluation, please refer to [robocasa-g1-tabletop-tasks](https://github.com/hogunkee/robocasa_g1)

---

## Scripts

### Training

```bash
dataset_list=("/data1/hogun/dataset/1130_Kitchen_LocoManip")
CUDA_VISIBLE_DEVICES=0 python scripts/gr00t_finetune.py \
    --dataset-path ${dataset_list[@]} \
    --num-gpus 1 --batch-size 8 \
    --tune_llm --tune_visual --tune_projector --tune_diffusion_model \
    --output-dir <OUTPUT_DIR> \
    --data-config unitree_g1 --embodiment_tag g1 \
    --max-steps 30000 --save-steps 10000
```

### Inference

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/inference_service.py --server \
--model-path <CKPT_DIR> \
--data-config unitree_g1 --embodiment-tag g1
```

---

## Overview

GR00T-G1 enables:
- Simulation of Unitree G1 in Robocasa and Robosuite with GR00T’s learning & control framework
- Compatibility with G1’s URDF and mesh models
- Motion retargetting to control G1 with GR00T-N1.5.

---

## Key Changes from Original GR00T
- Adjusted control API for G1 joint naming & limits
- Updated motion policies to match G1's DoF and body proportions
- Added example configs for G1 manipulation tasks

