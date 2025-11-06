from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Union, Optional
from functools import cached_property
import copy

import torch
import numpy as np
from gymnasium import spaces
import sapien
from transforms3d.euler import euler2quat
from transforms3d.quaternions import axangle2quat
from pathlib import Path

if TYPE_CHECKING:
    from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs import Actor, Articulation
from mani_skill.utils.structs.types import Array
from mani_skill.utils.structs.pose import Pose
from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.envs.utils.randomization.pose import random_quaternions



class SceneBuilder:
    """Base class for defining scene builders that can be reused across tasks"""

    env: BaseEnv
    """Env which scenebuilder will build in."""

    robot_init_qpos_noise: float = 0.02
    """Robot init qpos noise"""

    builds_lighting: bool = False
    """Whether this scene builder will add its own lighting when build is called. If False, ManiSkill will add some default lighting"""

    build_configs: Optional[List[Any]] = None
    """List of scene configuration information that can be used to **build** scenes during reconfiguration (i.e. `env.reset(seed=seed, options=dict(reconfigure=True))`). Can be a dictionary, a path to a json file, or some other data. If a scene needs to load build config data, it will index/sample such build configs from this list."""
    init_configs: Optional[List[Any]] = None
    """List of scene configuration information that can be used to **init** scenes during reconfiguration (i.e. `env.reset()`). Can be a dictionary, a path to a json file, or some other data. If a scene needs to load init config data, it will index/sample such init configs from this list."""

    scene_objects: Optional[Dict[str, Actor]] = None
    """Scene objects are any dynamic, kinematic, or static Actor built by the scene builder. Useful for accessing objects in the scene directly."""
    movable_objects: Optional[Dict[str, Actor]] = None
    """Movable objects are any **dynamic** Actor built by the scene builder. movable_objects is a subset of scene_objects. Can be used to query dynamic objects for e.g. task initialization."""
    articulations: Optional[Dict[str, Articulation]] = None
    """Articulations are any articulation loaded in by the scene builder."""

    navigable_positions: Optional[List[Union[Array, spaces.Box]]] = None
    """Some scenes allow for mobile robots to move through these scene. In this case, a list of navigable positions per env_idx (e.g. loaded from a navmesh) should be provided for easy initialization. Can be a discretized list, range, spaces.Box, etc."""

    def __init__(self, env, robot_init_qpos_noise=0.02):
        self.env = env
        self.robot_init_qpos_noise = robot_init_qpos_noise

    def build(self, build_config_idxs: List[int] = None):
        """
        Should create actor/articulation builders and only build objects into the scene without initializing pose, qpos, velocities etc.
        """
        raise NotImplementedError()

    def initialize(self, env_idx: torch.Tensor, init_config_idxs: List[int] = None):
        """
        Should initialize the scene, which can include e.g. setting the pose of all objects, changing the qpos/pose of articulations/robots etc.
        """
        raise NotImplementedError()

    def sample_build_config_idxs(self) -> List[int]:
        """
        Sample idxs of build configs for easy scene randomization. Should be changed to fit shape of self.build_configs.
        """
        return torch.randint(
            low=0, high=len(self.build_configs), size=(self.env.num_envs,)
        ).tolist()

    def sample_init_config_idxs(self) -> List[int]:
        """
        Sample idxs of init configs for easy scene randomization. Should be changed to fit shape of self.init_configs.
        """
        return torch.randint(
            low=0, high=len(self.init_configs), size=(self.env.num_envs,)
        ).tolist()

    @cached_property
    def build_config_names_to_idxs(self) -> Dict[str, int]:
        return dict((v, i) for i, v in enumerate(self.build_configs))

    @cached_property
    def init_config_names_to_idxs(self) -> Dict[str, int]:
        return dict((v, i) for i, v in enumerate(self.init_configs))

    @property
    def scene(self):
        return self.env.scene


class RFSceneBuilder(SceneBuilder):
    def __init__(self, env, cfg, **kwargs):
        super().__init__(env, **kwargs)
        self.cfg = cfg
        self.env.annotation_data = {}

    def initialize(self, env_idx: torch.Tensor):
        b = len(env_idx)
        cfg = copy.deepcopy(self.cfg)
        scene_cfg = cfg['scene']

        # primitive
        if 'primitives' in scene_cfg:
            for primitive_cfg in scene_cfg['primitives']:
                asset = getattr(self.env, primitive_cfg['name'], None)
                if not asset:
                    raise AttributeError(f'Attribute "{primitive_cfg["name"]}" not found in SceneBuilder.')
                ppos = primitive_cfg['pos']['ppos']['p']
                if 'randp_scale' in primitive_cfg['pos']:
                    ppos = np.array(ppos) + np.array(primitive_cfg['pos']['randp_scale']) * np.random.rand((len(ppos)))
                    ppos = ppos.tolist()
                qpos = np.array(primitive_cfg['pos']['qpos'])
                if 'randq_scale' in primitive_cfg:
                    qpos = np.array(qpos) + np.array(primitive_cfg['pos']['randq_scale'])* np.random.rand((len(qpos)))
                if 'random_quaternions' in primitive_cfg['pos']:
                    qpos = random_quaternions(
                        b,
                        lock_x=primitive_cfg['pos']['random_quaternions'][0],
                        lock_y=primitive_cfg['pos']['random_quaternions'][1],
                        lock_z=primitive_cfg['pos']['random_quaternions'][2]
                    )
                asset.set_pose(Pose.create_from_pq(ppos, qpos))

        # objects
        if 'objects' in cfg:
            objects_cfg = cfg['objects']
            self.movable_objects = {}
            for asset_cfg in objects_cfg:
                asset = getattr(self.env, asset_cfg['name'], None)
                if not asset:
                    raise AttributeError(f'Attribute "{asset_cfg["name"]}" not found in SceneBuilder.')
                ppos = asset_cfg['pos']['ppos']['p']
                if 'randp_scale' in asset_cfg['pos']:
                    ppos = np.array(ppos) + np.array(asset_cfg['pos']['randp_scale']) * np.random.rand((len(ppos)))
                    ppos = ppos.tolist()
                qpos = asset_cfg['pos']['qpos']
                if 'randq_scale' in asset_cfg['pos']:
                    qpos = np.array(qpos) + np.array(asset_cfg['pos']['randq_scale'])* np.random.rand((len(qpos)))
                if 'random_quaternions' in asset_cfg['pos']:
                    qpos = random_quaternions(
                        b,
                        lock_x=asset_cfg['pos']['random_quaternions'][0],
                        lock_y=asset_cfg['pos']['random_quaternions'][1],
                        lock_z=asset_cfg['pos']['random_quaternions'][2]
                    )
                asset.set_pose(Pose.create_from_pq(ppos, qpos))
                self.movable_objects[asset_cfg['name']] = asset
        # agents
        if 'agents' in cfg:
            agents_cfg = cfg['agents']
            is_multi_agent = (len(agents_cfg) > 1)
            agent = self.env.agent
            self.articulations = {}
            for idx, agent_cfg in enumerate(agents_cfg):
                pos_cfg = agent_cfg['pos']
                ppos = pos_cfg['ppos']['p']
                if 'randp_scale' in pos_cfg:
                    ppos = np.array(ppos) + np.array(agent_cfg['pos']['randp_scale']) * np.random.rand((len(ppos)))
                    ppos = ppos.tolist()
                ppos = sapien.Pose(ppos, q=euler2quat(*pos_cfg['ppos']['q']))
                qpos = np.array((pos_cfg['qpos']))
                if 'randq_scale' in pos_cfg:
                    qpos = np.tile(qpos, (b, 1)) + np.tile(np.array(agent_cfg['pos']['randq_scale']), (b, 1)) * np.random.rand(b, (len(qpos)))
                if 'random_quaternions' in agent_cfg['pos']:
                    qpos = random_quaternions(
                        b,
                        lock_x=agent_cfg['pos']['random_quaternions'][0],
                        lock_y=agent_cfg['pos']['random_quaternions'][1],
                        lock_z=agent_cfg['pos']['random_quaternions'][2]
                    )
                if is_multi_agent:
                    agent.agents[idx].reset(qpos)
                    agent.agents[idx].robot.set_pose(ppos)
                    self.articulations[agent_cfg['robot_uid']] = agent.agents[idx]
                else:
                    agent.reset(qpos)
                    agent.robot.set_pose(ppos)
                    self.articulations[agent_cfg['robot_uid']] = agent



def degree2quat(x, y, z):
    return euler2quat(np.deg2rad(x), np.deg2rad(y), np.deg2rad(z))

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return [w, x, y, z]

class CustomSceneBuilder(SceneBuilder):
    def __init__(self, env, cfg, **kwargs):
        super().__init__(env, **kwargs)
        self.cfg = cfg

    def build(self, build_config_idxs: List[int] = None):
        q = axangle2quat(
            np.array([1, 0, 0]), theta=np.deg2rad(90)
        )
        cfg = copy.deepcopy(self.cfg)
        scene_cfg = cfg['scene']
        if 'stage' in scene_cfg:
            stage_cfg = scene_cfg['stage']
            stage_file_path = stage_cfg['file_path']
            stage_name = stage_cfg['name']
            transform = stage_cfg.get('transform', {})
            position = transform.get('position', [0.0, 0.0, 0.0])
            rotation = transform.get('rotation', [0.0, 0.0, 0.0])
            builder = self.scene.create_actor_builder()
            builder.initial_pose = sapien.Pose(position, degree2quat(*rotation))
            builder.add_visual_from_file(stage_file_path)
            builder.add_nonconvex_collision_from_file(stage_file_path)
            suffix = Path(stage_file_path).suffix[1:] if Path(stage_file_path).suffix else None
            setattr(self.env, stage_name, builder.build_static(name=stage_name))

        if 'actors' in cfg:
            for actor_cfg in cfg['actors']:
                actor_name = actor_cfg['name']
                render_file_path = actor_cfg.get('render_file_path', None)
                collision_file_path = actor_cfg.get('collision_file_path', render_file_path)
                body_type = actor_cfg.get('body_type', 'dynamic')
                transform = actor_cfg.get('transform', {})
                position = transform.get('position', [0.0, 0.0, 0.0])
                rotation = transform.get('rotation', [0.0, 0.0, 0.0])
                use_visual_file_as_bounding_box = actor_cfg.get('use_visual_as_bounding_box', False)
                builder = self.scene.create_actor_builder()
                if render_file_path:
                    builder.add_visual_from_file(render_file_path)
                else:
                    raise ValueError(f"Actor {actor_name} must have a render file path.")
                suffix = Path(render_file_path).suffix[1:] if Path(render_file_path).suffix else None
                if suffix in ['glb', 'gltf']:
                    builder.initial_pose = sapien.Pose(position, quaternion_multiply(degree2quat(*rotation), q))
                else:
                    builder.initial_pose = sapien.Pose(position, degree2quat(*rotation))
                if body_type == 'dynamic':
                    if use_visual_file_as_bounding_box:
                        builder.add_convex_collision_from_file(render_file_path)
                    else:
                        builder.add_multiple_convex_collisions_from_file(collision_file_path)
                    actor = builder.build(name=actor_name)
                elif body_type == 'kinematic':
                    raise NotImplementedError(
                        f"Kinematic body type is not supported for actor {actor_name}. Use dynamic or static instead."
                    )
                elif body_type == 'static':
                    builder.add_nonconvex_collision_from_file(render_file_path)
                    actor = builder.build_static(name=actor_name)
                else:
                    raise ValueError(f"Unknown body type {body_type} for actor {actor_name}.")
                setattr(self.env, actor_name, actor)
        
        if 'articulations' in cfg:
            artis_cfg = cfg['articulations']
            for arti_cfg in artis_cfg:
                arti_name = arti_cfg['name']
                arti_urdf_path = arti_cfg['urdf_path']
                uniform_scale = arti_cfg.get('uniform_scale', 1.0)
                fixed_base = arti_cfg.get('fixed_base', True)
                transform = arti_cfg.get('transform', {})
                position = transform.get('position', [0.0, 0.0, 0.0])
                rotation = transform.get('rotation', [0.0, 0.0, 0.0])
                urdf_loader = self.scene.create_urdf_loader()
                urdf_loader.name = arti_name
                urdf_loader.fix_root_link = fixed_base
                urdf_loader.disable_self_collisions = True
                urdf_loader.scale = uniform_scale
                builder = urdf_loader.parse(arti_urdf_path)["articulation_builders"][0]
                pose = sapien.Pose(position, degree2quat(*rotation))
                builder.initial_pose = pose
                articulation = builder.build(name=arti_name)
                setattr(self.env, arti_name, articulation)

    def initialize(self, env_idx: torch.Tensor):
        b = len(env_idx)
        cfg = copy.deepcopy(self.cfg)
        if 'articulations' in cfg:
            artis_cfg = cfg['articulations']
            for arti_cfg in artis_cfg:
                obj = getattr(self.env, arti_cfg['name'])
                transform = arti_cfg.get('transform', {})
                position = transform.get('position', [0.0, 0.0, 0.0])
                rotation = transform.get('rotation', [0.0, 0.0, 0.0])
                pose = sapien.Pose(position, self.degree2quat(*rotation))
                obj.set_pose(pose)
                if isinstance(obj, Articulation):
                    obj.set_qpos(obj.qpos[0] * 0)
                    obj.set_qvel(obj.qvel[0] * 0)
        if 'agents' in cfg:
            agents_cfg = cfg['agents']
            is_multi_agent = (len(agents_cfg) > 1)
            agent = self.env.agent
            self.articulations = {}
            for idx, agent_cfg in enumerate(agents_cfg):
                pos_cfg = agent_cfg['pos']
                ppos = pos_cfg['ppos']['p']
                if 'randp_scale' in pos_cfg:
                    ppos = np.array(ppos) + np.array(agent_cfg['pos']['randp_scale']) * np.random.rand((len(ppos)))
                    ppos = ppos.tolist()
                ppos = sapien.Pose(ppos, q=euler2quat(*pos_cfg['ppos']['q']))
                qpos = np.array((pos_cfg['qpos']))
                if 'randq_scale' in pos_cfg:
                    qpos = np.tile(qpos, (b, 1)) + np.tile(np.array(agent_cfg['pos']['randq_scale']), (b, 1)) * np.random.rand(b, (len(qpos)))
                if 'random_quaternions' in agent_cfg['pos']:
                    qpos = random_quaternions(
                        b,
                        lock_x=agent_cfg['pos']['random_quaternions'][0],
                        lock_y=agent_cfg['pos']['random_quaternions'][1],
                        lock_z=agent_cfg['pos']['random_quaternions'][2]
                    )
                if is_multi_agent:
                    agent.agents[idx].reset(qpos)
                    agent.agents[idx].robot.set_pose(ppos)
                    self.articulations[agent_cfg['robot_uid']] = agent.agents[idx]
                else:
                    agent.reset(qpos)
                    agent.robot.set_pose(ppos)
                    self.articulations[agent_cfg['robot_uid']] = agent
