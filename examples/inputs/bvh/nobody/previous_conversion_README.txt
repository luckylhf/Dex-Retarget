1013_nobody_chr01 转换文件与WBC回放数据

本包是此前三栏对比视频实际使用的数据。没有针对本动作训练或微调WBC。
原始文件保持不变，路径及SHA256见 records/input_manifest.json。
成功视频使用云端 experiments/nobody_demo_v1/motion_only_v2 的产物；最初保留标定帧的失败尝试仍保留在其父目录，未混入本包。

转换步骤
1. 原始BVH：59关节，所有关节六通道，2086帧，Frame Time=0.011111，约90Hz，厘米制Y-up。
   第0帧是全零旋转T-pose标定；到第1帧根部跳约2.2米。仅去掉该标定帧，保留原始第1至2085帧。
   nobody_without_calibration.bvh 是清理后的2085帧人体骨架，未调速、未补站立过渡。
2. 用现有GMR的LAFAN兼容骨架读取和 bvh_lafan1_to_dex_evt.json 映射，转为米制Z-up，求解Dex 29关节IK。
   为初始化IK先重复首个有效姿态60帧，转换结束后去掉这60帧；实际动作仍是2085帧。
   预热前3帧使用过IK恢复；保留的2085帧没有IK恢复事件。详见 records/nobody.ik.json。
   nobody.pkl 包含 root_pos、root_rot、dof_pos、joint_names 和模型/映射身份。
   注意：PKL的 root_rot 四元数为 XYZW；fps=90.000900009。
3. 经既有 Dex-Mimic/scripts/pkl_to_npz.py 重采样到50Hz，并计算39个机器人刚体的FK及速度。
   nobody_robot.npz：1158帧，29关节，39刚体，四元数为 WXYZ。
   主要字段为 joint_pos、joint_vel、body_pos_w、body_quat_w、body_lin_vel_w、body_ang_vel_w。
   时间跨度按(N-1)/50计算是23.14秒；日志中的N/50=23.16秒是帧数口径。
   既有check_motion检查通过，抽样FK最大误差为0。
4. 将同一条机器人轨迹打包为WBC目标特征。
   reference.npz：1158帧，q/qd各29维，14个跟踪刚体，goal为1158×730。
   它是WBC播放输入，包含10帧未来目标；此步是特征打包，没有再次求IK。

文件用途
- nobody.pkl：GMR重定向结果，适合GMR动作查看/后续转换。
- nobody_robot.npz：现有Dex机器人动作库/NPZ查看器使用的标准参考动作。
- reference.npz：本次SuperMimic WBC运行时读取的参考目标。
- rollouts/night_10440 与 rollouts/morning_12341：WBC真实ROS+MuJoCo闭环执行的轨迹和逐帧指标，不是参考动作。
  nobody_ros.trajectory.npz中的qpos是MuJoCo原始广义坐标顺序；查看时配合单独保存的 model.mjb 和各自contract.json，不能直接当标准机器人参考NPZ读取。
  编译模型体积较大，未放入本ZIP；本地位置与校验值见 rollouts/model_manifest.json。PKL、机器人参考NPZ和WBC参考NPZ均已完整打包。
- records：清理说明、IK审计、转换与检查日志、实际映射和执行程序。

回放结果
昨晚10440：完整23.14秒，最大位置偏差1.574米。
今早12341：完整23.14秒，最大位置偏差0.258米。
两者原生MuJoCo与ROS的完整qpos数组均一致，无物理支撑、无中途重置。
外部评测口径为height_diagnostic_v1：高度偏差记录但不停止，倾斜偏差仍停止。本动作两者均未触发严格高度或倾斜标准。
今早12341的位置反馈来自仿真真值，未在此提供真机位置估计能力。
MuJoCo模型文件版本为3.16.0；模型SHA256和关节名称见对应contract.json。
