import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, GroupAction, ExecuteProcess, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, SetRemap
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # ==========================================================
    # 1. 基础配置
    # ==========================================================
    pkg_slam = get_package_share_directory('slam_toolbox')

    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_link_to_lidar',
        arguments=['0', '0', '0.05', '0', '0', '0', 'base_link', 'lidar_frame'],
        output='screen'
    )

    # ==========================================================
    # 2. 定义各个任务模块
    # ==========================================================

    # [A] 重置 Odom 指令
    reset_odom_cmd = ExecuteProcess(
        cmd=[
            'ros2', 'topic', 'pub', '--once',
            '/set_pose',
            'geometry_msgs/msg/PoseWithCovarianceStamped',
            "{header: {frame_id: 'odom'}, pose: {pose: {position: {x: 0.0, y: 0.0, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}}"
        ],
        output='screen'
    )

    # [B] 感知节点 (Lidar Processor)
    # 负责：scan_raw -> 降采样/清洗 -> scan
    lidar_node = Node(
        package='hallway_pkg',
        executable='inference_lidar',  # 请确保 setup.py 里注册了这个名字
        name='lidar_processor',
        output='screen',
        emulate_tty=True
    )

    # [C] SLAM 节点
    # 负责：scan -> Map/TF
    slam_launch = GroupAction(
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(pkg_slam, 'launch', 'online_async_launch.py')
                ),
                launch_arguments={'use_sim_time': 'false'}.items()
            )
        ]
    )

    # [D] 决策节点 (RL Actor)
    # 负责：Scan + TF -> SAC Agent -> cmd_vel
    actor_node = Node(
        package='hallway_pkg',
        executable='inference_actor',  # 请确保 setup.py 里注册了这个名字
        name='rl_controller',
        output='screen',
        emulate_tty=True,
        parameters=[{'use_sim_time': False}]
    )

    # ==========================================================
    # 3. 编排时间轴 (Timeline)
    # ==========================================================

    # 阶段 1: 启动雷达处理 (T=1.0s)
    # 给重置指令 1秒 时间执行，然后立刻启动雷达，确保 /scan 话题有数据
    start_lidar_event = TimerAction(
        period=1.0,
        actions=[
            LogInfo(msg=">>> [Step 1] 启动雷达处理节点 (发布 /scan)..."),
            lidar_node
        ]
    )

    # 阶段 2: 启动 SLAM (T=4.0s)
    # 雷达跑了3秒后，数据流非常稳定了，此时启动 SLAM 不会报错
    start_slam_event = TimerAction(
        period=4.0,
        actions=[
            LogInfo(msg=">>> [Step 2] 雷达数据就绪，启动 SLAM 建图..."),
            slam_launch
        ]
    )

    # 阶段 3: 启动 RL 决策 (T=8.0s)
    # SLAM 跑了4秒后，地图和 TF 树应该建立完毕，此时启动大脑
    start_actor_event = TimerAction(
        period=8.0,
        actions=[
            LogInfo(msg=">>> [Step 3] 系统完全就绪，启动 RL 智能体..."),
            actor_node
        ]
    )

    return LaunchDescription([
        static_tf_node,
        reset_odom_cmd,     # 0. 先归零
        start_lidar_event,  # 1. 启雷达
        start_slam_event,   # 2. 启SLAM
        start_actor_event   # 3. 启大脑
    ])