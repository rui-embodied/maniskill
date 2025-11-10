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
    fix_root_link = True
    disable_self_collisions = False

    keyframes = dict(
        idle=Keyframe(
            pose=sapien.Pose(p=[0, 0, 1.1])
        )
    )

    base_joint_names = [
        "root_x_axis_joint",
        "root_y_axis_joint",
        "root_z_rotation_joint",
    ]

    @property
    def _controller_configs(self):
        self.stiffness = 80.0
        self.damping = 2.0
        self.force_limit = 100

        base_pd_joint_vel = PDBaseForwardVelControllerConfig(
            self.base_joint_names,
            lower=[-1, -3.14],
            upper=[1, 3.14],
            damping=1000,
            force_limit=500,
        )

        controller_configs = dict(
            # pd_joint_pos=dict(body=pd_joint_pos, balance_passive_force=False)
            pd_joint_pos=dict(base=base_pd_joint_vel),
        )
        return deepcopy_dict(controller_configs)