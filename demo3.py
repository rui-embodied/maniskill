import gymnasium as gym
import argparse
import yaml
import threading
import time
import numpy as np
from sim.tasks.scene119 import Scene119Env
import select
import sys
import pickle
from sim import CONFIG_DIR

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default=f"{CONFIG_DIR}/scene119.yaml", help='Configuration to build scenes, assets and agents.')
    return parser.parse_args()

def run_env(args):
    # 加载配置
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        env_id = config['task_name']
    robot_uids = [agent_cfg['robot_type'] for agent_cfg in config['agents']]
    robot_uids = tuple(robot_uids)

    env = gym.make(
        id=env_id,  
        config=args.config,
        robot_uids=robot_uids,
        obs_mode="rgbd",
        control_mode="pd_joint_pos",
        render_mode="human",
        viewer_camera_configs=dict(shader_pack="rt-fast")
    )
    obs = env.reset()

    zero_action = {}
    for agent_id, space in env.action_space.spaces.items():
        zero_action[agent_id] = np.zeros(space.shape, dtype=space.dtype)

    zero_action['b2z1'] = np.array([0, 0])
    while True:
        action = zero_action
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

if __name__ == "__main__":
    run_env(parse_args())
