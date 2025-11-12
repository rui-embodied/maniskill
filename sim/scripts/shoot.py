import gymnasium as gym
import argparse
import time
import numpy as np
from PIL import Image
import os
import math
import torch
import yaml
import cv2
from pathlib import Path
from sim import CONFIG_DIR
from sim.tasks.env import TestEnv
from sim.tasks.stair import StairEnv
from sim.tasks.taskA import TaskAEnv
from sapien import Pose

GRID_SIZE = 0.5
DESIRED_WIDTH = 1440
DESIRED_HEIGHT = 1920
PNG_DEPTH_SCALE = 1000.0  # 深度存到png时的比例（米→毫米）

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=f"{CONFIG_DIR}/test_env.yaml",
        help='Configuration to build scenes, assets and agents.'
    )
    return parser.parse_args()

def adjust_intrinsics(fx, fy, cx, cy, orig_w, orig_h, new_w, new_h):
    sx = new_w / orig_w
    sy = new_h / orig_h
    return fx * sx, fy * sy, cx * sx, cy * sy

def load_poses_txt(file_path):
    poses = []
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue  # 忽略注释和空行
            parts = line.strip().split()
            frame_id = int(parts[0])
            # 位置
            px, py, pz = map(float, parts[1:4])
            # 四元数 w,x,y,z
            qw, qx, qy, qz = map(float, parts[4:8])

            poses.append(Pose(
                p=np.array([px, py, pz], dtype=np.float32), 
                q=np.array([qw, qx, qy, qz], dtype=np.float32)
                )
            )
    return poses

def run_env(args):
    env_id = "stair"
    robot_uids = ("b2z1")
    env = gym.make(
        id=env_id,
        robot_uids=robot_uids,
        obs_mode="rgbd",
        control_mode="pd_joint_pos",
        render_mode="human",
    )
    obs = env.reset()
    viewer = env.unwrapped.viewer

    # 输出文件夹（直接生成 ConceptGraphs 格式）
    output_dir = Path("output_preprocessed/record3d_scans/co_store_processed")
    rgb_out = output_dir / "rgb"
    depth_out = output_dir / "depth"
    conf_out = output_dir / "conf"
    conf_img_out = output_dir / "conf_images"
    high_conf_depth_out = output_dir / "high_conf_depth"
    poses_out = output_dir / "poses"
    for p in [rgb_out, depth_out, conf_out, conf_img_out, high_conf_depth_out, poses_out]:
        p.mkdir(parents=True, exist_ok=True)

    intrinsics_path = output_dir / "dataconfig.yaml"
    poses = load_poses_txt("traj.txt")
    cam = viewer.cameras[0]

    frame_id = 0
    intrinsics_saved = False

    pose_len = len(poses)

    while not viewer.closed and frame_id < pose_len:
        cam.set_local_pose(poses[frame_id])
        action = env.action_space.sample() if env.action_space is not None else None
        action = np.zeros_like(action)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        extrinsic = obs['sensor_param']['m_cam']['extrinsic_cv'].cpu().numpy()
        extrinsic_4x4 = np.eye(4, dtype=extrinsic.dtype)
        extrinsic_4x4[:3, :] = extrinsic
        # extrinsic = extrinsic_4x4
        extrinsic=np.linalg.inv(extrinsic_4x4)
        intrinsic = obs['sensor_param']['m_cam']['intrinsic_cv'].cpu().squeeze(0).numpy()
        data = obs['sensor_data']['m_cam']
        rgb = data['rgb'].squeeze(0).cpu().numpy()
        depth = data['depth'].squeeze(0).squeeze(-1).cpu().numpy().astype(np.uint16)
        if not intrinsics_saved:
            cfg = {
                "dataset_name": "record3d",
                "camera_params": {
                    "image_height": DESIRED_HEIGHT,
                    "image_width": DESIRED_WIDTH,
                    "fx": float(intrinsic[0, 0]),
                    "fy": float(intrinsic[1, 1]),
                    "cx": float(intrinsic[0, 2]),
                    "cy": float(intrinsic[1, 2]),
                    "png_depth_scale": PNG_DEPTH_SCALE
                }
            }
            with open(intrinsics_path, "w") as f:
                yaml.dump(cfg, f)
            intrinsics_saved = True

        rgba_img = rgb.astype(np.uint8)
        bgr_img = cv2.cvtColor(rgba_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(rgb_out / f"frame{frame_id:06d}.png"), bgr_img)
        depth_image = Image.fromarray(depth)
        depth_image.save(depth_out / f"frame{frame_id:06d}.png")

        # 置信度图（这里生成全2）
        conf_map = np.ones((DESIRED_HEIGHT, DESIRED_WIDTH), dtype=np.uint8) * 2
        np.save(conf_out / f"frame{frame_id:06d}.npy", conf_map)
        Image.fromarray(conf_map).save(conf_img_out / f"frame{frame_id:06d}.png")

        # 高置信度深度（conf==2保留）
        high_conf_depth = np.where(conf_map == 2, depth / PNG_DEPTH_SCALE, 0)
        high_conf_depth_mm = (high_conf_depth * PNG_DEPTH_SCALE).astype(np.uint16)
        Image.fromarray(high_conf_depth_mm).save(high_conf_depth_out / f"frame{frame_id:06d}.png")

        np.save(poses_out / f"frame{frame_id:06d}.npy", extrinsic)

        frame_id += 1

if __name__ == "__main__":
    run_env(parse_args())
