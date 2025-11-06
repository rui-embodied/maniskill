import sapien
import numpy as np
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.structs import Pose
from sim import ASSET_DIR


@register_agent()
class Tracer_mini(BaseAgent):
    uid = "tracer_mini"
    urdf_path = f"{ASSET_DIR}/tracer_mini/urdf/tracer_mini.urdf"
    fix_root_link = False
    disable_self_collisions = True

    keyframes = dict(
        idle=Keyframe(
            pose=sapien.Pose(p=[0, 0, 1.1])
        )
    )