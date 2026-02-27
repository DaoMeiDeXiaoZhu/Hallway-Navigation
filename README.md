# 走廊避障导航 （ROS2、SAC、Lidar、静态+动态障碍物避障）

**版本说明：**

Ubuntu22.04 + ROS2(Humble) + python3.10.12

**文件结构说明：**

<img width="3735" height="730" alt="yuque_diagram" src="https://github.com/user-attachments/assets/57ea2699-7720-4e5d-967f-9ba64bd90cf0" />

**文件作用说明：**

1. launch 文件夹：用于存放仿真训练和真机部署时的启动文件。
2. hallway_pkg 文件夹：文件夹名与源代码名相同（ROS2 项目配置），这里存放的 py 文件用于在 setup.py 中声明并在 .launch.py 文件中启动。
3. checkpoint 文件夹：存放仿真环境中训练的模型文件、奖励曲线等。
4. urdf 文件夹：用于存放机器人模型文件，包括 xacro 、urdf。
5. world 文件夹：用于存放我们在 gazebo 中手动搭建的走廊仿真场景。
6. setup.py 文件：用于讲我们所有需要用到的文件安装到 install 的 share 目录下，并且要保持相同的目录结构，否则可能找不到文件，例如 launch、rviz、urdf、world 文件夹（heallway_pkg 是项目自动生成的，因此会自动安装）。
7. obstacle_move.py：仿真场景中的移动障碍物的移动逻辑。
8. sac_simulation.py：仿真场景中用到的 sac 模型（包括 replay_buffer、networks、agent）。
9. nodes_simulation.py：仿真中用到的节点，完成话题订阅和发布功能。gazebo 中的雷达插件会使用 inf 和 0 区分无穷远和贴脸的情况，但是真实小车面对这两种情况返回值都是 0，因此这里的py文件是用来将 gazebo 中返回的 inf 也变为 0，接着采用启发式算法来推算出当前的 " 0 " 是无穷远还是贴脸，是无穷远则修改为range_max，否则仍然保持为 0。
10. run_simulation.py：在仿真环境中训练 sac 模型的程序入口。
11. lidar_bridge.py：用于处理真实 ROS2 小车的/scan_raw话题，该话题雷达消息的 ranges 数组长度在 477 和 478 之间变化（有 nan 消息被自动过滤了），导致 SLAM 建图不稳定，且与 gazebo 中提供的 360 维雷达信息不匹配。该文件用于将/scan_raw中的雷达消息均匀采样为 360 并且重新发布/scan话题保证雷达数据的稳定性。
12. nodes_reality.py：用于订阅和发布话题。
13. sac_reality.py：真实场景部署时用到的 sac 模型，包含 replay_buffer、networks、agent。
14. run_reality.py：真实场景中接收雷达消息并输出动作的程序入口。

**程序运行说明(提前退出anaconda环境)：**

仿真训练：colcon build --packages-select hallway_pkg && source install/setup.bash && ros2 launch hallway_pkg simulation.launch.py

真机演示：colcon build --packages-select hallway_pkg && source install/setup.bash && ros2 launch hallway_pkg reality.launch.py

**项目不足：**

1. 训练时使用的场地摩擦力、雷达延迟、精度与现实有一定差距。
2. 训练时奖励函数存在一定的震荡，后期需要对奖励函数进行精准调优。
3. 训练时的话题接收和发送默认没有延迟，部署时小车的数据需要传回服务器，服务器输出动作再发送给小车存在延迟。（可以ssh登录小车后将项目部署到小车内部）
4. 训练时使用的gazebo定位真值，而真实部署的时候使用的是slam，两者在精度和延迟上有差距。（gazebo训练时可以对位置添加高斯噪声和延迟，或者训练时也使用slam定位）
5. 由于真机雷达统一用0表示“无穷远”和“贴脸”，所以仿真中也把“无穷远”的雷达信息inf改为了0,用启发式算法实时计算当前的“0”代表的是“无穷远”还是“贴脸”，如果是“无穷远”则修改为range_max，这样做本身就会导致训练不稳定。

**效果演示：**

https://github.com/user-attachments/assets/5e40f69c-c284-4839-8dde-206e5527fed6

