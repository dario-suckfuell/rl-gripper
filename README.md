# Master’s Thesis: End-to-End Reinforcement Learning for Robotic Grasping

This repository contains the code, datasets, and experiments conducted as part of my master’s thesis. The work focuses on developing and evaluating an **end-to-end reinforcement learning framework** for robotic grasping using **vision-only inputs** processed through a pre-trained model.

---

## Project Overview

The goal of this thesis is to create a robust reinforcement learning system capable of:
- Learning directly from **RGB image streams** (perception-only) without explicit object-specific information.
- Tackling challenges like **generalization**, **stability**, and **sample efficiency** through innovative architectures and techniques.

### Key Contributions
- **Decoupling policy and representation learning** by using a pre-trained model for feature extraction.
- Experimentation with various **adapter network architectures**.
- Insights into the impact of **data augmentation** and **curriculum learning** for robotic grasping.
  
---

## Repository Structure

- **`rl_gripper/`**: Core implementation of the reinforcement learning pipeline.
  - **`config/`**: Configuration file for experiments and training.
  - **`envs/`**: Environment definitions and setup files.
  - **`resources/`**: Classes, utility functions, URDF model files and datasets.
  - **`training/`**: Training artifacts, including TensorBoard logs, checkpoints, and final model files.

---

## Getting Started

### Clone the Repository
```bash
git clone https://github.com/dario-suckfuell/rl-gripper.git
cd rl-gripper
# rl-gripper
