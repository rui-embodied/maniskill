## 场景搭建
- **1. 场景配置文件位置**：在`configs/`下，`test_env.yaml`配置园区安防，`scene.yaml`配置灾备救援
- **2. 资产位置**：
  ```
  assets/
  ├── apl/                   *载人小车的模型和urdf
  ├── dog_with_arm/ 
  │   ├── meshes/
  │   │   ├── aliengo_description   * 机器狗本体的模型
  │   │   └── z1_descruotion        * 机器狗手臂的模型    
  │   └── robot.urdf                * 机器的urdf
  ├── drone/                 *无人机的模型和urdf
  ├── flames/                *火焰的模型和贴图，有两种
  ├── unnamed/               *灾备救援场景中，仓库附带门的模型和urdf
  ├── cell_0.glb             *园区安防场地
  ├── cell_1.glb             *灾备救援场地
  ├── male_03.glb
  ├── OldWarehouse.glb       *灾备救援场景中，仓库的模型
  └── table.glb              *测试用物品
  ```
- **3. 场景配置文件说明**：
  - `scene > stage`描述了场景中场地的信息，`file_path`描述场地模型的位置，`transform`描述位置和旋转
  - actors参数描述了场景中所有不会发生形变的物体，`render_file_path`描述其外观模型的位置，`use_visual_as_bounding_box`决定是否根据其外观模型生成碰撞，`collision_file_path`描述其碰撞箱的形状，`body_type`可选`static`（该物体在场景中静止）或`dynamic`（该物体可被agent移动），`transform`描述位置和旋转
  - articulations参数描述了场景中所有会发生形变的物体，必须用urdf格式描述，`urdf_path`决定urdf文件位置，`fixed_base`决定其root link是否固定在场景中，`uniform_scale`决定其缩放，`transform`描述位置和旋转
  - agents参数描述了场景中所有agents信息，`robot_type`说明agent的种类，可选择`aliengoZ1`(机械狗)，`drone`(无人机)，`apollo`(载人小车)，`robot_uid`说明agent在场景中的名称，`pos`参数中，`ppos > p`描述agent的初始位置，`ppos > q`描述agent的初始旋转，其余内容可不必在意
  - cameras中`human_render`决定场景镜头的初始位置
  - task_path决定了改场景待执行的任务队列

## 动作执行

### 1. 流程脚本说明

**更改动作流程只需要修改json文件即可**

- 行为脚本对于动作流程两个场景我们分别设计了$hw/cmd/fire.json$和$hw/cmd/rescue.json$对应第一个和第二个场景，为了能够突出其顺序执行的特点，如果需要更改指令，直接在该文件中修改即可。每一个都需要有$agent\_id$来表示调用的机器人，用$name$表示执行的动作，如果是$move\_toward$需要添加目的地以及用到的$voxel\_map$[在hw下面的三个pkl文件中]，这里像素图是因为两个场景需要两个像素图，第二个场景因为涉及室内所以分成了0.5和1两种分辨率，如果是0.5就是三维路径规划，需要用到$astar3d$处理室内问题。

  示例

  ```json
  {
        "agent_id": "aliengoZ1-agent-0", //需要调用哪个agent
        "name": "move_toward", //动作函数
        "position": [90, 50, 0.697], //目标点
        "voxel_path": "hw/voxel_3d_1_1.pkl", //需要的像素图
        "surface_layer_index": 1 //是按二维走还是三维走。这个主要是根据agent还有是否是室内来决定，是为了减少计算时间
  },
  ```

  

- 对于脚本的处理用的是$hw/utils$，也就是分析是哪个$agent$以去调用对应的类

- 任务调度在$hw/run$中，利用队列进行处理，每个任务完成后$done$变为$true$，再去执行下一个任务。

- 整体代码在$hw/ennrities.py$中，设计了基类$Task$和$ActorTask$两种，然后设计了狗，无人机，无人车，人，火五个子类，并且分别实现各自所需要的行为。



### 2. 动作函数列表

| 函数名            | 作用                                                  | 调用对象             | 思想                                                         |
| ----------------- | ----------------------------------------------------- | -------------------- | ------------------------------------------------------------ |
| `move_toward`     | 移动到目标点，需要指定目标点坐标，以及所需的voxel_map | 在全部子类中均有调用 | 利用$astar2d$或者$astar3d$算法进行路径规划，利用插值算法进行插值，利用set_pose方法对位置进行更新，旋转角度利用相邻点的旋转角进行旋转。该部分封装函数都在$planner/utils.py$。<br />另外在狗中还设计开门判断，自动感应开门，所以实际上狗的`open_door`实际也是在`move_toward`中处理。另外狗的固定步态设计也是放在这里。 |
| `turnon_camera`   | 开启镜头跟随                                          | 三个agent子类        | 直接计算当前位置来跟随镜头，内嵌在`move_toward`中，并未单独处理 |
| `solve_exception` | 特殊情况处理，如灭火                                  | 无人机               | 对于灭火我们采用让火`move_toward`方法，每次让火往下移动直到全部消失。 |


## 使用示例
- 使用`python -m hw.run -c ./configs/test_env.yaml`运行园区安防场景
- 使用`python -m hw.run -c ./configs/scene.yaml`运行灾备救援场景