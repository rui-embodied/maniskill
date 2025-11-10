import sapien
import numpy as np
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.structs import Pose
from sim import ASSET_DIR


@register_agent()
class Galaxea(BaseAgent):
    uid = "galaxea"
    urdf_path = f"{ASSET_DIR}/R1_Lite/urdf/mmp_revB_invconfig_upright_a1x.urdf"
    fix_root_link = True
    disable_self_collisions = True

    keyframes = dict(
        idle=Keyframe(
            pose=sapien.Pose(p=[0, 0, 10.3])
        )
    )

    left_arm_joint_names = [
        "left_arm_joint1",
        "left_arm_joint2",
        "left_arm_joint3",
        "left_arm_joint4",
        "left_arm_joint5",
        "left_arm_joint6",
        "left_gripper_finger_joint1",
        "left_gripper_finger_joint2",
    ]

    right_arm_joint_names = [
        "right_arm_joint1",
        "right_arm_joint2",
        "right_arm_joint3",
        "right_arm_joint4",
        "right_arm_joint5",
        "right_arm_joint6",
        "right_gripper_finger_joint1",
        "right_gripper_finger_joint2",
    ]

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
            pd_joint_delta_pos=dict(
                base=base_pd_joint_vel,
                left_arm=PDJointPosControllerConfig(
                    self.left_arm_joint_names,
                    lower=-0.225,
                    upper=0.225,
                    force_limit=40.0,
                    stiffness=25.0,
                    damping=0.5,
                    normalize_action=False,
                    use_delta=True,
                ),
                right_arm=PDJointPosControllerConfig(
                    self.right_arm_joint_names,
                    lower=-0.225,
                    upper=0.225,
                    force_limit=40.0,
                    stiffness=25.0,
                    damping=0.5,
                    normalize_action=False,
                    use_delta=True,
                ),
            ),
            pd_joint_pos=dict(
                base=base_pd_joint_vel,
                left_arm=PDJointPosControllerConfig(
                    self.left_arm_joint_names,
                    lower=-0.225,
                    upper=0.225,
                    force_limit=40.0,
                    stiffness=25.0,
                    damping=0.5,
                    normalize_action=False,
                    use_delta=False,
                ),
                right_arm=PDJointPosControllerConfig(
                    self.right_arm_joint_names,
                    lower=-0.225,
                    upper=0.225,
                    force_limit=40.0,
                    stiffness=25.0,
                    damping=0.5,
                    normalize_action=False,
                    use_delta=False,
                ),
            ),
            # pd_joint_pos=dict(body=pd_joint_pos, balance_passive_force=False)

        )
        return deepcopy_dict(controller_configs)