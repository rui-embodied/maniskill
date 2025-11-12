import sapien
import numpy as np
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.structs import Pose
from sim import ASSET_DIR


@register_agent()
class AliengoZ1(BaseAgent):
    uid = "aliengoZ1"
    urdf_path = f"{ASSET_DIR}/b2z1/urdf/b2z1_description.urdf"

    fix_root_link = True
    disable_self_collisions = True

    keyframes = dict(
        standing=Keyframe(
            pose=sapien.Pose(p=[0, 0, 0.5]),
            qpos=np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.6, 0.6, 0.6, 0.0, -1.2, -1.2, -1.2, -1.2, 0.0, 0.0, 0.0, 0.0, 0.0]
            ), # hip, joint1, thigh, joint2, calf, joint3~6, gripper FR/FL/RR/RL
        )
    )

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="spot_head",
                pose=Pose.create_from_pq([0.45, 0, 0], [1, 0, 0, 0]),
                width=256,
                height=256,
                fov=2,
                near=0.01,
                far=100,
                mount=self.robot.links_map["FR_hip_link"],
            ),
        ]

    arm_joint_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "jointGripper",
    ]
    dog_joint_names = [
        "FR_hip_joint",
        "FL_hip_joint",
        "RR_hip_joint",
        "RL_hip_joint",
        "FR_thigh_joint",
        "FL_thigh_joint",
        "RR_thigh_joint",
        "RL_thigh_joint",
        "FR_calf_joint",
        "FL_calf_joint",
        "RR_calf_joint",
        "RL_calf_joint",
    ]

    @property
    def _controller_configs(
        self,
    ):

        return dict(
            pd_joint_delta_pos=dict(
                dog=PDJointPosControllerConfig(
                    self.dog_joint_names,
                    lower=-0.225,
                    upper=0.225,
                    force_limit=40.0,
                    stiffness=25.0,
                    damping=0.5,
                    normalize_action=False,
                    use_delta=True,
                ),
                arm=PDJointPosControllerConfig(
                    self.arm_joint_names,
                    lower=-0.225,
                    upper=0.225,
                    force_limit=40.0,
                    stiffness=25.0,
                    damping=0.5,
                    normalize_action=False,
                    use_delta=True,
                ),
                balance_passive_force=False,
            ),
            pd_joint_pos=dict(
                dog=PDJointPosControllerConfig(
                    self.dog_joint_names,
                    lower=-0.225,
                    upper=0.225,
                    force_limit=40.0,
                    stiffness=1000,
                    damping=100,
                    normalize_action=False,
                    use_delta=False,
                ),
                arm=PDJointPosControllerConfig(
                    self.arm_joint_names,
                    lower=-0.225,
                    upper=0.225,
                    force_limit=40.0,
                    stiffness=1000,
                    damping=100,
                    normalize_action=False,
                    use_delta=False,
                ),
                balance_passive_force=False,
            ),
        )
