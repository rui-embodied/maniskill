import sys, os, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import gymnasium as gym
import argparse
import yaml
import threading
import time
import numpy as np
from sim.tasks.env import TestEnv
from sim.tasks.stair import StairEnv
from sim.tasks.taskA import TaskAEnv
from sim.tasks.scene119 import Scene119Env
import select
import sys
import pickle
from sim import CONFIG_DIR
from translate import sapien_to_json
from translate_inv import json_to_sapien

# ===================== JSON 读写相关 =====================
def find_closest_object_by_prev_pose(json_data, pos_sapien_now, quat_sapien_now):
    """根据上一次保存的 JSON 位姿（逆变换后）在 Sapien 坐标系下匹配当前实体"""
    def quat_dist(q1, q2):
        q1 = np.array(q1) / np.linalg.norm(q1)
        q2 = np.array(q2) / np.linalg.norm(q2)
        return 1 - np.abs(np.dot(q1, q2))

    all_objs = []
    if "object_instances" in json_data:
        all_objs.extend(json_data["object_instances"])
    if "articulated_object_instances" in json_data:
        all_objs.extend(json_data["articulated_object_instances"])

    best_obj, best_score = None, float("inf")
    for obj in all_objs:
        pos_json = obj.get("translation", None)
        quat_json = obj.get("rotation", None)
        if pos_json is None or quat_json is None:
            continue

        # 把上一次 JSON 坐标转成 Sapien 坐标
        pos_prev_sapien, quat_prev_sapien = json_to_sapien(pos_json, quat_json)

        pos_diff = np.linalg.norm(pos_prev_sapien - pos_sapien_now)
        quat_diff = quat_dist(quat_prev_sapien, quat_sapien_now)
        score = pos_diff + 0.1 * quat_diff

        if score < best_score:
            best_obj, best_score = obj, score
    return best_obj


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str,
        default=f"{CONFIG_DIR}/test_env.yaml"
    )
    parser.add_argument(
        "--scene", type=str, default="scene119.json",
        help="Path to the scene JSON file"
    )
    return parser.parse_args()


# ===================== 主运行逻辑 =====================
def run_env(args):
    env_id = "scene119"
    robot_uids = ("aliengoZ1")
    env = gym.make(
        id=env_id,
        robot_uids=robot_uids,
        obs_mode="rgbd",
        control_mode="pd_joint_pos",
        render_mode="human",
        viewer_camera_configs=dict(shader_pack="rt-fast"),
    )

    obs = env.reset()
    viewer = env.unwrapped.viewer

    scene_path = "/home/sutai/.maniskill/data/scene_datasets/replica_cad_dataset/configs/scenes/scene119.json"
    with open(scene_path, "r") as f:
        json_data = json.load(f)

    while not viewer.closed:
        action = np.zeros_like(env.action_space.sample()) if env.action_space is not None else None
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if not viewer.closed and viewer.window.mouse_click(0):
            print("=" * 80)
            try:
                entity = viewer.selected_entity
                if entity is None:
                    print("⚠ No entity selected.")
                    continue

                # 当前 Sapien 实体的位姿
                pos_sapien = np.array(entity.pose.p)
                quat_sapien = np.array(entity.pose.q)

                # 根据“上一次的 JSON 坐标（转为 Sapien 系）”匹配物体
                closest_obj = find_closest_object_by_prev_pose(json_data, pos_sapien, quat_sapien)
                if closest_obj:
                    print(f"→ Matched object: {closest_obj['template_name']}")

                    # 将当前位姿转成 JSON 系并写回
                    pos_json_new, quat_json_new = sapien_to_json(pos_sapien, quat_sapien)
                    closest_obj["translation"] = [float(x) for x in pos_json_new]
                    closest_obj["rotation"] = [float(x) for x in quat_json_new]

                    with open(scene_path, "w") as f:
                        json.dump(json_data, f, indent=2)
                    print(f"✅ Updated {scene_path}")
                else:
                    print("❌ No matching object found.")

            except Exception as e:
                print(f"⚠ Error: {e}")


if __name__ == "__main__":
    run_env(parse_args())
