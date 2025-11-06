import gymnasium as gym
import argparse
import yaml
import threading
import time
import numpy as np
from sim.tasks.env import TestEnv
from sim.tasks.stair import StairEnv
from sim.tasks.taskA import TaskAEnv
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
    env_id = "taskA"
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

    while True:
        action = env.action_space.sample() if env.action_space is not None else None
        action = np.zeros_like(action)   # 停止运动
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

if __name__ == "__main__":
    run_env(parse_args())
