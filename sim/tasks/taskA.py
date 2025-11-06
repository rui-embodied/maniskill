from typing import Any, Dict, Union, Tuple
import numpy as np
import torch
import sapien
import yaml
import copy
import os.path as osp
from transforms3d.euler import euler2quat
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose
from mani_skill.utils.sapien_utils import look_at
from mani_skill.agents.multi_agent import MultiAgent
import sim.utils.scenes as scenes
from sim import CONFIG_DIR
from sim.utils.nested_dict_utils import nested_yaml_map, replace_dir
from sim.robots.apollo import Apollo
from sim.robots.drone import Drone
from sim.robots.aliengoZ1 import AliengoZ1
from sim.robots.b2z1 import B2z1
from sim.robots.R1_Lite import StarSeaMap
from sim.robots.tracer_mini import Tracer_mini
from sim.utils.scenes.robocasa.scene_builder import RoboCasaSceneBuilder
from mani_skill.sensors.camera import (
    Camera,
    CameraConfig,
    parse_camera_configs,
    update_camera_configs_from_dict,
)
from mani_skill.sensors.depth_camera import StereoDepthCamera, StereoDepthCameraConfig


@register_env("taskA", max_episode_steps=100)
class TaskAEnv(BaseEnv):
    def __init__(
            self, *args, robot_uids=None, robot_init_qpos_noise=0.02, **kwargs
            # self, *args, robot_uids=("aliengoZ1", "apollo", "panda", "drone"), robot_init_qpos_noise=0.02, **kwargs
    ):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
    
    # @property
    # def _default_sensor_configs(self):
    #     cfg = copy.deepcopy(self.cfg)
    #     camera_cfg = cfg.get('cameras', {})
    #     sensor_cfg = camera_cfg.get('sensor', [])
    #     all_camera_configs =[]
    #     for sensor in sensor_cfg:
    #         pose = sensor['pose']
    #         if pose['type'] == 'pose':
    #             sensor['pose'] = sapien.Pose(*pose['params'])
    #         elif pose['type'] == 'look_at':
    #             sensor['pose'] = sapien_utils.look_at(*pose['params'])
    #         all_camera_configs.append(CameraConfig(**sensor))
        
    #     return all_camera_configs

    # @property
    # def _default_human_render_camera_configs(self):
    #     cfg = copy.deepcopy(self.cfg)
    #     camera_cfg = cfg.get('cameras', {})
    #     render_cfg = camera_cfg.get('human_render', [])
    #     all_camera_configs =[]
    #     for render in render_cfg:
    #         pose = render['pose']
    #         if pose['type'] == 'pose':
    #             render['pose'] = sapien.Pose(*pose['params'])
    #         elif pose['type'] == 'look_at':
    #             render['pose'] = sapien_utils.look_at(*pose['params'])
    #         all_camera_configs.append(CameraConfig(**render))
    #     return all_camera_configs

    def _load_agent(self, options: dict):
        init_poses = []
        init_poses.append(sapien.Pose(p=[0., 0., 0.]))
        super()._load_agent(options, init_poses)

    def _load_scene(self, options: dict):
        self.scene_builder = RoboCasaSceneBuilder(env=self)
        self.scene_builder.build()

    def _setup_sensors(self, options: dict):
        """Setup sensor configurations and the sensor objects in the scene. Called by `self._reconfigure`"""

        # First create all the configurations
        self._sensor_configs = dict()

        # Add task/external sensors
        self._sensor_configs.update(parse_camera_configs(self._default_sensor_configs))

        # Add agent sensors
        self._agent_sensor_configs = dict()
        if self.agent is not None:
            if isinstance(self.agent, MultiAgent):
                self._agent_sensor_configs = parse_camera_configs(self.agent.sensor_configs)
            else:
                self._agent_sensor_configs = parse_camera_configs(self.agent._sensor_configs)
            self._sensor_configs.update(self._agent_sensor_configs)

        # Add human render camera configs
        self._human_render_camera_configs = parse_camera_configs(
            self._default_human_render_camera_configs
        )

        self._viewer_camera_config = parse_camera_configs(
            self._default_viewer_camera_configs
        )

        # Override camera configurations with user supplied configurations
        if self._custom_sensor_configs is not None:
            update_camera_configs_from_dict(
                self._sensor_configs, self._custom_sensor_configs
            )
        if self._custom_human_render_camera_configs is not None:
            update_camera_configs_from_dict(
                self._human_render_camera_configs,
                self._custom_human_render_camera_configs,
            )
        if self._custom_viewer_camera_configs is not None:
            update_camera_configs_from_dict(
                self._viewer_camera_config,
                self._custom_viewer_camera_configs,
            )
        self._viewer_camera_config = self._viewer_camera_config["viewer"]

        # Now we instantiate the actual sensor objects
        self._sensors = dict()

        for uid, sensor_config in self._sensor_configs.items():
            if uid in self._agent_sensor_configs:
                if isinstance(self.agent, MultiAgent):
                    articulation = sensor_config.mount.articulation
                else:
                    articulation = self.agent.robot
            else:
                articulation = None
            if isinstance(sensor_config, StereoDepthCameraConfig):
                sensor_cls = StereoDepthCamera
            elif isinstance(sensor_config, CameraConfig):
                sensor_cls = Camera
            self._sensors[uid] = sensor_cls(
                sensor_config,
                self.scene,
                articulation=articulation,
            )

        # Cameras for rendering only
        self._human_render_cameras = dict()
        for uid, camera_config in self._human_render_camera_configs.items():
            self._human_render_cameras[uid] = Camera(
                camera_config,
                self.scene,
            )

        self.scene.sensors = self._sensors
        self.scene.human_render_cameras = self._human_render_cameras


    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        pass

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        pass