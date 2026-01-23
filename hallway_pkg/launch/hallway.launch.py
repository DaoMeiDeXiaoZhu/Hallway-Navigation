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

    # 5. 障碍物 & RViz
    motion_node = Node(
        package=pkg_name, executable='obstacle_mover', output='screen',
        parameters=[{'use_sim_time': True}]
    )
    rviz_node = Node(
        package='rviz2', executable='rviz2',
        arguments=['-d', os.path.join(pkg_path, 'rviz', 'display.launch.rviz')],
        parameters=[{'use_sim_time': True}], output='screen'
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher_node,
        spawn_robot_node,
        TimerAction(period=5.0, actions=[motion_node]),
        rviz_node
    ])