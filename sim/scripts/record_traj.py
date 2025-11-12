import gymnasium as gym
import argparse
import time
import numpy as np
import math
import yaml
from pathlib import Path
from sim import CONFIG_DIR
from sim.tasks.env import TestEnv
from sim.tasks.stair import StairEnv
from sim.tasks.taskA import TaskAEnv

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=f"{CONFIG_DIR}/test_env.yaml",
        help='Configuration to build scenes, assets and agents.'
    )
    return parser.parse_args()

def run_env(args):
    env_id = "stair"
    robot_uids = ("b2z1",)
    env = gym.make(
        id=env_id,
        robot_uids=robot_uids,
        obs_mode="rgbd",
        control_mode="pd_joint_pos",
        render_mode="human",
    )
    env.reset()
    viewer = env.unwrapped.viewer

    # 输出路径
    output_dir = Path("")
    output_dir.mkdir(parents=True, exist_ok=True)

    traj_file = output_dir / "traj.txt"
    with open(traj_file, "w") as f:
        f.write("# frame_id followed by pose\n")

    recording = False
    frame_id = 0
    save_interval = 1.0 / 10.0  # 每秒存一次，也可以改成其他间隔
    last_save_time = time.time()
    q_prev_state = False

    while not viewer.closed:
        action = np.zeros_like(env.action_space.sample())
        env.step(action)
        env.render()

        # 按 q 控制录制开关
        q_current_state = viewer.window.key_down("q")
        if q_current_state and not q_prev_state:
            recording = not recording
            print("[INFO] 开始录制..." if recording else "[INFO] 停止录制.")
            frame_id = 0 if recording else frame_id
        q_prev_state = q_current_state

        if recording and (time.time() - last_save_time >= save_interval):
            # 获取外参矩阵（相机位姿）
            extrinsic = viewer.window.get_camera_pose()
            # 展平为 16 个数并存入 traj.txt
            line = f"{frame_id} " + " ".join(f"{v:.6f}" for v in extrinsic.p) + " " + " ".join(f"{v:.6f}" for v in extrinsic.q) + "\n"
            with open(traj_file, "a") as f:
                f.write(line)

            frame_id += 1
            last_save_time = time.time()

if __name__ == "__main__":
    run_env(parse_args())