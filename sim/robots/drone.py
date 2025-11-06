import sapien
import numpy as np
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from sim import ASSET_DIR


@register_agent()
class Drone(BaseAgent):
    uid="drone"
    urdf_path=f"{ASSET_DIR}/drone/cf2x.urdf"
    fix_root_link = False
    disable_self_collisions = True

    keyframes = dict(
        idle=Keyframe(
            pose=sapien.Pose(p=[0, 0, 1.1])
        )
    )

    def land_on():
        # 直接生成一个垂直向下的path
        # 然后moveto过去
        pass

    def move_to_ward(x, y, z):
        """
        生成路径
        """
        pass

    def move(self, x, y, z):
        """
        移动到具体点
        """
        print(x, y, z)
        self.articulation.set_pose(sapien.Pose([x, y, z], self.articulation.pose.q))
        pass

    def take_off():
        # 生成一个垂直向上的path，并且不能碰到头上的东西
        # 然后moveto过去？
        pass

    def turn_on():
        # 开警报吗？
        pass
    
