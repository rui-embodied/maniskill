import gymnasium as gym
import argparse
import yaml
import threading
import time
import numpy as np
from sim.tasks.env import TestEnv
from sim.tasks.test import StairEnv
from sim.tasks.taskA import TaskAEnv
import select
import sys
import pickle
from sim import CONFIG_DIR
from PIL import Image
import os
import math
GRID_SIZE = 0.5

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default=f"{CONFIG_DIR}/test_env.yaml", help='Configuration to build scenes, assets and agents.')
    return parser.parse_args()

def run_env(args):
    # 加载配置
    env_id = "stair"
    robot_uids = ("aliengoZ1")
    env = gym.make(
        id=env_id,  
        robot_uids=robot_uids,
        obs_mode="rgbd",
        control_mode="pd_joint_pos",
        render_mode="human",
        viewer_camera_configs=dict(shader_pack="rt-med")
    )
    obs = env.reset()
    viewer = env.unwrapped.viewer
    i, n = 0, 0
    while not viewer.closed:
        action = env.action_space.sample() if env.action_space is not None else None
        action = np.zeros_like(action)   # 停止运动
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        n += 1
        if not viewer.closed and n > 15 and viewer.window.key_down("q"): 
            extrinsic = viewer.window.get_camera_pose().to_transformation_matrix()
            fovy = viewer.window.fovy
            width, height = viewer.window.size
            fx = fy = (height / 2) / math.tan(fovy / 2)
            cx, cy = width / 2, height/ 2
            intrinsic = np.array(
                [
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ]
            )
            rgba = viewer.window.get_picture("Color")
            rgba_img = (rgba[..., :3] * 255).clip(0, 255).astype("uint8")
            rgba_pil = Image.fromarray(rgba_img)
            image_name = f"output/images/{i}_iter.png"
            folder_name = f"output"
            os.makedirs(folder_name, exist_ok=True)
            os.makedirs("output/images", exist_ok=True)
            if i == 0 or i == 1:
                np.savetxt(f"{folder_name}/intrinsic_{i}.txt", intrinsic)
                np.savetxt(f"{folder_name}/extrinsic_{i}.txt", extrinsic)
            rgba_pil.save(image_name)
            i += 1
            n = 0

if __name__ == "__main__":
    run_env(parse_args())