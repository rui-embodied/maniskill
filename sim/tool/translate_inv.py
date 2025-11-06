# translate_inv.py
import numpy as np
import transforms3d.quaternions as tq

# 与原变换保持一致
q_rx90  = tq.axangle2quat([1, 0, 0], np.deg2rad(90))
q_rx180 = tq.axangle2quat([1, 0, 0], np.deg2rad(180))
q_rx90_inv = tq.qconjugate(q_rx90)


def json_to_sapien(pos_json, quat_json):
    """
    将 JSON 坐标系中的平移 + 四元数(w, x, y, z)
    转换为 Sapien 坐标系下的平移 + 四元数(x, y, z, w)
    """
    pos_json = np.asarray(pos_json)
    quat_json = np.asarray(quat_json)

    # --- 平移部分 ---
    R = tq.quat2mat(q_rx90)
    pos_sapien = R @ pos_json

    # --- 姿态部分 ---
    # JSON文件中保存的是 wxyz, 转换成 xyzw
    quat_xyzw = np.array([quat_json[1], quat_json[2], quat_json[3], quat_json[0]])

    # 正变换中使用了 q_correction = q_rx180 * q_rx90_inv
    # 所以这里使用它的逆 q_correction_inv
    q_correction_inv = tq.qconjugate(tq.qmult(q_rx180, q_rx90_inv))
    quat_sapien = tq.qmult(q_correction_inv, quat_xyzw)
    quat_sapien /= np.linalg.norm(quat_sapien)

    return pos_sapien, quat_sapien


def main():
    # 示例：输入一个 JSON 坐标
    pos_json = [-1.415,  1.989,  1.409]
    quat_json = [ 7.07106781e-01, -5.55111512e-17,  7.07106781e-01, -5.55111512e-17]  # wxyz

    pos_sapien, quat_sapien = json_to_sapien(pos_json, quat_json)
    print("pos_sapien =", pos_sapien)
    print("quat_sapien =", quat_sapien)


if __name__ == "__main__":
    main()
