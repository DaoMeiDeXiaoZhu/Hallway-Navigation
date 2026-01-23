# 走廊避障导航
ROS2、SAC、Lidar、静态+动态障碍物避障

这是一个基于Ubuntu22.04+ROS2的Humble版本的项目，按照以下指令运行即可：
1.在文件终端中启动仿真训练：colcon build + source install/setup.bash + ros2 launch hallway_pkg/hallway.launch.py
2.在文件终端启动真实场景部署：ros2 launch real_env_naivgation_pkg/star_natigation.launch.py
