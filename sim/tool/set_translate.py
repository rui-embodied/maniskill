import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import gymnasium as gym
import argparse
import yaml
import threading
import time
import numpy as np
from sim.tasks.env import TestEnv
from sim.tasks.test import StairEnv
from sim.tasks.taskA import TaskAEnv
from sim.tasks.scene119 import Scene119Env
import select
import sys
import pickle
from sim import CONFIG_DIR

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default=f"{CONFIG_DIR}/test_env.yaml", help='Configuration to build scenes, assets and agents.')
    return parser.parse_args()

def run_env(args):
    # 加载配置
    env_id = "scene119"
    robot_uids = ("aliengoZ1")
    env = gym.make(
        id=env_id,  
        robot_uids=robot_uids,
        obs_mode="rgbd",
        control_mode="pd_joint_pos",
        render_mode="human",
        viewer_camera_configs=dict(shader_pack="rt-fast")
    )
    obs = env.reset()
    viewer = env.unwrapped.viewer
    while not viewer.closed:
        action = env.action_space.sample() if env.action_space is not None else None
        action = np.zeros_like(action)   # 停止运动
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        # print(dir(viewer.window))
        if not viewer.closed and viewer.window.mouse_click(0):
            print("=" * 80)
            try:
                entity = viewer.selected_entity
                print(entity.name)
                print(entity.pose.p)
                print(entity.pose.q)              
                                  
            except AttributeError:
                print("⚠ No entity selected (viewer.selected_entity is None).")


if __name__ == "__main__":
    run_env(parse_args())
