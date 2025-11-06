import pyvhacd

pyvhacd.compute_vhacd(
    input_file="/home/sutai/Desktop/bed.glb",
    output_file="/home/sutai/Desktop/bed_convex.glb",
    resolution=200000,       # 体素分辨率
    max_convex_hulls=64,     # 最大凸体数
    concavity=0.0025,        # 凹陷程度
)
