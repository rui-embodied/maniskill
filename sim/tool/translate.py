import numpy as np
import transforms3d.quaternions as tq

q_rx90  = tq.axangle2quat([1, 0, 0], np.deg2rad(90))
q_rx180 = tq.axangle2quat([1, 0, 0], np.deg2rad(180))
q_rx90_inv = tq.qconjugate(q_rx90)

def sapien_to_json(pos_sapien, quat_sapien):
    pos_sapien = np.asarray(pos_sapien)
    quat_sapien = np.asarray(quat_sapien)

    # --- 平移 ---
    R = tq.quat2mat(q_rx90_inv)
    pos_json = R @ pos_sapien

    # --- 姿态 ---
    # 再加一个 Rx(180°) 翻转修正上下
    q_correction = tq.qmult(q_rx180, q_rx90_inv)
    quat_json_xyzw = tq.qmult(q_correction, quat_sapien)
    quat_json_xyzw /= np.linalg.norm(quat_json_xyzw)

    if quat_json_xyzw[3] < 0:
        quat_json_xyzw = -quat_json_xyzw

    # 输出 wxyz
    w, x, y, z = quat_json_xyzw[3], quat_json_xyzw[0], quat_json_xyzw[1], quat_json_xyzw[2]
    quat_json_wxyz = np.array([w, x, y, z])

    return pos_json, quat_json_wxyz

import numpy as np

def quat_inverse(q_out):
    """逆变换函数: 将 ManiSkill 输出的四元数转换回原始坐标"""
    w, x, y, z = q_out
    return np.array([x, w, y, z])



def main():
    # pos = [-1.779462, 4.550405, 0.975565]
    # quat = [0.0, -0.707, 0.707, -0.0]
    # pos_json, quat_json = sapien_to_json(pos, quat)
    # print("pos_json =", pos_json)
    # print("quat_json =", quat_json)
    print(quat_inverse([0, -0.707, 0.707, 0]))

if __name__ == "__main__":
    main()

# pos_json = [ 7.299  0.515 -4.696]
# quat_json = [ 0.00000000e+00 -1.11039071e-16  1.00000000e+00  0.00000000e+00]