import xml.etree.ElementTree as ET

# 加载URDF文件
file_path = 'sim/assets/StarSeaMap_R1_Lite/urdf/mmp_revA_v1_0_0.urdf'  # 你可以修改为文件的实际路径
tree = ET.parse(file_path)
root = tree.getroot()

# 用于检查小于0的惯性矩值
def check_inertia_values(urdf_root):
    count = 0
    print("----------------------------")
    for link in urdf_root.findall('.//link'):
        inertia = link.find('.//inertia')
        if inertia is not None:
            # 获取惯性矩的值
            try:
                ixx = float(inertia.get('ixx', 0))
                iyy = float(inertia.get('iyy', 0))
                izz = float(inertia.get('izz', 0))

                if ixx > 0 and iyy > 0 and izz > 0:
                    count += 1
                
                # 检查是否有负值
                if ixx < 0 or iyy < 0 or izz < 0:
                    print(f"Found negative inertia values in link '{link.get('name')}'")
                    print(f"ixx: {ixx}, iyy: {iyy}, izz: {izz}")
            except ValueError:
                print(f"Invalid inertia values in link '{link.get('name')}'")
    print(count)
# 检查并打印结果
check_inertia_values(root)
