from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, TimerAction
from ament_index_python.packages import get_package_share_directory
import os
import xacro

def generate_launch_description():
    pkg_name = 'hallway_pkg'
    pkg_path = get_package_share_directory(pkg_name)
    
    # 1. 解析 URDF
    urdf_file = os.path.join(pkg_path, 'urdf', 'mecanum.xacro')
    doc = xacro.process_file(urdf_file, mappings={'use_sim_collision': 'true'})
    robot_description_content = doc.toxml()

    # 2. State Publisher
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[
            {'robot_description': robot_description_content},
            {'use_sim_time': True}
        ],
        output='screen'
    )

    # 3. Gazebo
    world_file = os.path.join(pkg_path, 'world', 'hallway.world')
    gazebo = ExecuteProcess(
        # 加载 ros_factory 插件即可，物理系统由 Gazebo 内部接管
        cmd=['gazebo', '--verbose', world_file, '-s', 'libgazebo_ros_init.so', '-s', 'libgazebo_ros_factory.so'],
        output='screen'
    )

    # 4. 生成机器人
    spawn_robot_node = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', '/robot_description', '-entity', 'my_robot', '-x', '0', '-y', '0', '-z', '0.1'],
        output='screen'
    )

    # 5. 障碍物
    motion_node = Node(
        package=pkg_name, executable='obstacle_mover', output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # 5.启动强化学习训练
    rl_node = Node(
        package='hallway_pkg',      # 你的包名
        executable='run_simulation', # setup.py 中设置的入口点名称
        name='rl_agent_node',       # 节点运行时的名称
        output='screen',            # 重要：这样你才能在终端看到训练的打印信息(Reward, Episode等)
        emulate_tty=True            # 让打印输出带有颜色（如果代码支持）
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher_node,
        spawn_robot_node,
        TimerAction(period=5.0, actions=[motion_node]),
        TimerAction(period=2.0, actions=[rl_node]),
    ])