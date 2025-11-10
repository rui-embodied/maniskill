from typing import Any, Dict, Union
import numpy as np
import sapien
import torch
import time
import random
import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import SO100, Fetch, Panda, XArm6Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


@register_env("StaticConveyor-v1", max_episode_steps=200)
class StaticConveyorBeltEnv(BaseEnv):
    """
    **Task Description:**
    A data collection environment for VLA training. The robot arm needs to grasp moving objects 
    on the conveyor belt and place them in a box. This environment is designed for imitation 
    learning data collection rather than reinforcement learning.

    **Features:**
    - Conveyor belt with moving objects
    - Box for placing objects
    - Simplified reward function for VLA training
    - Extended episode length for human demonstration
    """

    SUPPORTED_ROBOTS = [
        "panda",
        "fetch", 
        "xarm6_robotiq",
        "so100",
    ]
    
    agent: Union[Panda, Fetch, XArm6Robotiq, SO100]
    cube_half_size = 0.02
    goal_thresh = 0.025
    cube_spawn_half_size = 0.05
    cube_spawn_center = (0, 0)
    
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.0,**kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        if robot_uids in PICK_CUBE_CONFIGS:
            cfg = PICK_CUBE_CONFIGS[robot_uids]
        else:
            cfg = PICK_CUBE_CONFIGS["panda"]
        self.cube_half_size = cfg["cube_half_size"]
        self.goal_thresh = cfg["goal_thresh"]
        self.cube_spawn_half_size = cfg["cube_spawn_half_size"]
        self.cube_spawn_center = cfg["cube_spawn_center"]
        self.max_goal_height = cfg["max_goal_height"]
        self.sensor_cam_eye_pos = cfg["sensor_cam_eye_pos"]
        self.sensor_cam_target_pos = cfg["sensor_cam_target_pos"]
        self.human_cam_eye_pos = cfg["human_cam_eye_pos"]
        self.human_cam_target_pos = cfg["human_cam_target_pos"]
        # VLA环境不需要复杂的成功条件判断
        super().__init__(*args, robot_uids=robot_uids, **kwargs)


    @property
    def _default_sensor_configs(self):
        # Top-down camera: positioned above the workspace looking down
        top_down_pose = sapien_utils.look_at(
            eye=[-0.820 - 0.6, 0.025, 0.586 + 0.4],  # 将摄像头移至机械臂前方，并略微升高
            target=[-0.820, 0.025, 0.586]  # 对准机械臂中心
        )
        
        return [
            CameraConfig(
                uid="top_down_camera",
                pose=top_down_pose,
                width=224,
                height=224,
                fov=np.pi / 2,
                near=0.01,
                far=100
            )
        ]
    
    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.45, 0, 0]))
        
        # Add wrist-mounted camera configuration after agent is loaded
        # This is needed because we need access to the robot's links
        if self.robot_uids == "panda":
            # Define the wrist camera pose relative to the end-effector
            # Camera looking downward (along negative Z in world frame)
            # Rotation: X=90° to point camera downward
            wrist_cam_pose = sapien.Pose(
                p=[0.03, 0, -0.03],  # Slightly offset from the TCP
                # q=[0.5, 0.5, -0.5, 0.5]  # Rotated 90° around X to look downward
                q = [0, 0.7071, 0, 0.7071]
            )
            
            # Get the end-effector link
            tcp_link = sapien_utils.get_obj_by_name(
                self.agent.robot.get_links(), "panda_hand_tcp"
            )
            
            # Create the wrist camera config with the mount set directly
            self._wrist_camera_config = CameraConfig(
                uid="wrist_camera",
                pose=wrist_cam_pose,
                width=224,
                height=224,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                mount=tcp_link
            )

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=self.human_cam_eye_pos, target=self.human_cam_target_pos
        )
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)
    
    def _setup_sensors(self, options: dict):
        # Call parent to setup base sensors
        super()._setup_sensors(options)
        
        # Add wrist camera if it was created
        if hasattr(self, '_wrist_camera_config'):
            from mani_skill.sensors.camera import Camera
            self._sensors['wrist_camera'] = Camera(
                self._wrist_camera_config,
                self.scene,
                articulation=None  # Already set mount directly
            )
            self.scene.sensors = self._sensors

    def _load_scene(self, options: dict):
        # 加载传送带URDF
        conveyor_urdf_path = "assets/my_assets/conveyor_belt/conveyor.urdf"
        self.conveyor_loader = self.scene.create_urdf_loader()
        self.conveyor_loader.fix_root_link = True
        
        # 加载传送带并设置位置 - 传送带在场景中心
        self.conveyor = self.conveyor_loader.load(conveyor_urdf_path, name="conveyor")
        conveyor_pose = sapien.Pose(p=[1, 0, 0.0], q=[0, 0, 0, 1])
        self.conveyor.set_root_pose(conveyor_pose)
        
        # 构建桌子场景
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # 在桌子上放置一个箱子（调小尺寸）
        box_urdf_path = "assets/my_assets/box.urdf"
        self.box_loader = self.scene.create_urdf_loader()
        self.box_loader.fix_root_link = True
        self.box_loader.scale = 0.2  # 将box缩小到30%
        self.box = self.box_loader.load(box_urdf_path, name="box")
        box_pose = sapien.Pose(p=[1, 0, 0.0], q=[1, 0, 0, 0])
        self.box.set_root_pose(box_pose)

        # 生成随机颜色
        colors = [
            [1, 0, 0, 1],  # 红色
            [0, 1, 0, 1],  # 绿色
            [0, 0, 1, 1],  # 蓝色
            [1, 1, 0, 1],  # 黄色
            [1, 0, 1, 1],  # 紫色
            [0, 1, 1, 1],  # 青色
        ]
        color = random.choice(colors)

        arm_pose = self.agent.robot.get_pose()  # 获取机械臂当前的位置
        arm_position = arm_pose.p[0]  # 机械臂的当前位置

        
        # 创建方块
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=color,
            name="cube",
            initial_pose=sapien.Pose(p=[-0.6 + random.uniform(-0.5, -0.4), random.uniform(-0.05, 0.05), -0.05]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.agent.robot.set_pose(sapien.Pose([-0.45, 0, 0], [0, 0, 0, 1]))

            conveyor_pose = sapien.Pose(p=[-0.75, -1.9, 0.0], q=[0.707, 0, 0, 0.707])
            self.conveyor.set_root_pose(conveyor_pose)

            box_pose = sapien.Pose(p=[-0.55, 0.5, 0.0], q=[1, 0, 0, 0])
            self.box.set_root_pose(box_pose)



            arm_pose = self.agent.robot.get_pose()  # 获取机械臂当前的位置
            arm_position = arm_pose.p[0]  # 机械臂的当前位置
            cube_position = [-0.6 + random.uniform(-0.5, -0.4), random.uniform(-0.05, 0.05), -0.05]
            self.cube.set_pose(sapien.Pose(p=cube_position))


    def evaluate(self):
        # 调整篮子中心位置 - 根据 _load_scene 中的设置
        box_center = torch.tensor([-0.55, 0.3, 0.0], device=self.device)  # 传送带和桌子的中心位置

        # 判断物体是否在篮子内
        cube_pos = self.cube.pose.p
        in_box_xy = (
            (torch.abs(cube_pos[:, 0] - box_center[0]) < 0.08) &  # x方向范围
            (torch.abs(cube_pos[:, 1] - box_center[1]) < 0.08)    # y方向范围
        )
        # z方向：物体应该在篮子底部附近（允许一定高度差）
        in_box_z = (cube_pos[:, 2] > 0.02) & (cube_pos[:, 2] < 0.10)

        is_obj_in_box = in_box_xy & in_box_z
        is_grasped = self.agent.is_grasping(self.cube)
        is_robot_static = self.agent.is_static(0.2)

        return {
            "success": is_obj_in_box & is_robot_static & (~is_grasped),
            "is_obj_in_box": is_obj_in_box,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        batch_size = action.shape[0]
        return torch.zeros(batch_size, device=action.device)


