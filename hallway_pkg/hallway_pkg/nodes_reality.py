import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import numpy as np
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Pose2D
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

class RobotEnvNode(Node):
    def __init__(self):
        super().__init__('robot_env_node')

        # 1. 基础订阅和发布
        # 注意：这里订阅的是 /scan (已经由 inference_lidar 处理过的 360 点数据)
        # 或者是原始 /scan_raw 也没关系，反正下面会重采样为 90 点
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # 2. 初始化数据容器
        self.latest_scan = None
        self.latest_pose = None   # 格式: (rel_x, rel_y, z, roll, pitch, rel_yaw)
        self.latest_twist = None  # 格式: (vx, vy, wz)
        
        # 范围初始化 (后续会被 msg 覆盖)
        self.range_min = 0.0 
        self.range_max = 0.0

        # [核心修改] 初始位姿偏移量 (用于自动归零)
        self.initial_offset = None 

        # TF 初始化
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # 20Hz 频率查询坐标
        self.create_timer(0.05, self.update_pose_from_tf)

    # --- 辅助函数：四元数转欧拉角 ---
    def euler_from_quaternion(self, quaternion):
        x = quaternion.x
        y = quaternion.y
        z = quaternion.z
        w = quaternion.w
        
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        sinp = 2 * (w * y - z * x)
        pitch = math.asin(sinp) if abs(sinp) <= 1 else math.copysign(math.pi / 2, sinp)
        
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw

    # [核心修改] 从 TF 获取坐标并自动减去初始偏移
    def update_pose_from_tf(self):
        try:
            # 查询从 'map' 到 'base_link' 的变换
            t = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time())

            p = t.transform.translation
            q = t.transform.rotation
            roll, pitch, yaw = self.euler_from_quaternion(q)

            # 1. 记录初始偏移 (只在程序启动后的第一帧有效数据时执行)
            if self.initial_offset is None:
                self.initial_offset = {
                    'x': p.x,
                    'y': p.y,
                    'yaw': yaw
                }
                self.get_logger().info(f">>> [Auto-Zero] 归零成功！将SLAM坐标 ({p.x:.2f}, {p.y:.2f}) 设为逻辑原点 (0,0)")

            # 2. 计算相对坐标 (核心逻辑)
            # 当前绝对位置 - 初始位置 = 相对位置 (Relative Position)
            rel_x = p.x - self.initial_offset['x']
            rel_y = p.y - self.initial_offset['y']
            
            # 3. 计算相对角度 (Relative Yaw)
            # 让 RL 认为机器人初始朝向也是 0 度
            rel_yaw = yaw - self.initial_offset['yaw']
            
            # 规范化角度到 -pi ~ pi
            while rel_yaw > math.pi: rel_yaw -= 2 * math.pi
            while rel_yaw < -math.pi: rel_yaw += 2 * math.pi

            # 更新给 RL 的状态 (使用相对值)
            self.latest_pose = (rel_x, rel_y, p.z, roll, pitch, rel_yaw)

        except TransformException as ex:
            # TF 还没建立好，忽略
            self.latest_pose = None

    def scan_callback(self, msg: LaserScan):
        # ------------------------------------------------------------------
        # 1. 动态获取参数
        # ------------------------------------------------------------------
        self.range_min = msg.range_min
        self.range_max = msg.range_max

        # ------------------------------------------------------------------
        # 2. 数据清洗
        # ------------------------------------------------------------------
        ranges = np.array(msg.ranges)
        total_points = len(ranges)

        # 清洗 inf 和 nan
        ranges[np.isinf(ranges)] = 0
        ranges[np.isnan(ranges)] = 0
        
        processed_ranges = ranges.copy()
        
        # 区分 0 是盲区还是无限远
        zero_indices = np.where(processed_ranges == 0.0)[0]
        valid_indices = np.where(processed_ranges > 0.0)[0]

        if len(valid_indices) == 0:
            processed_ranges[:] = self.range_max
        else:
            CLOSE_THRESHOLD = 0.5 
            for idx in zero_indices:
                # 寻找最近邻居
                dist_to_valid = np.abs(valid_indices - idx)
                dist_to_valid = np.minimum(dist_to_valid, total_points - dist_to_valid)
                nearest_valid_idx = valid_indices[np.argmin(dist_to_valid)]
                nearest_val = ranges[nearest_valid_idx]

                if nearest_val > CLOSE_THRESHOLD:
                    processed_ranges[idx] = self.range_max
                else:
                    processed_ranges[idx] = self.range_min

        # 截断数据
        processed_ranges = np.clip(processed_ranges, self.range_min, self.range_max)

        # ------------------------------------------------------------------
        # 3. 采样 90 个点 (Resampling)
        # ------------------------------------------------------------------
        target_len = 90
        
        if total_points >= target_len:
            # 均匀生成 90 个索引
            indices = np.linspace(0, total_points - 1, target_len, dtype=int)
            self.latest_scan = processed_ranges[indices]
        else:
            # 极少见情况：原始数据不够90个，进行插值
            x_old = np.linspace(0, 1, total_points)
            x_new = np.linspace(0, 1, target_len)
            self.latest_scan = np.interp(x_new, x_old, processed_ranges)

    def odom_callback(self, msg: Odometry):
        # 获取速度 (Odom 速度是相对车身坐标系的，不需要做 Offset)
        v_x = msg.twist.twist.linear.x
        v_y = msg.twist.twist.linear.y
        w_z = msg.twist.twist.angular.z
        self.latest_twist = (v_x, v_y, w_z)

    def move(self, linear_x=0.0, angular_z=0.0):
        cmd = Twist()
        cmd.linear.x = float(linear_x)
        cmd.angular.z = float(angular_z)
        self.cmd_pub.publish(cmd)

    def stop(self):
        self.move(0.0, 0.0)

    # 机器人话题和数据自检
    def is_ready(self):
        topic_list = self.get_topic_names_and_types()
        active_topic_names = [name for (name, types) in topic_list]

        if '/scan' not in active_topic_names:
            return False, "错误: 未检测到 '/scan' 话题"
        
        if '/cmd_vel' not in active_topic_names:
            return False, "错误: 未检测到 '/cmd_vel' 话题"
        
        if self.latest_scan is None:
            return False, "警告: 话题存在，但尚未收到雷达数据"
        
        # [逻辑验证] 
        # 现在 latest_pose 已经是(0,0)起步的相对坐标了
        if self.latest_pose is not None:
            curr_x = self.latest_pose[0]
            curr_y = self.latest_pose[1]
            
            # 计算距离逻辑原点的偏移
            # 理论上启动瞬间这里应该是 0.0
            error_pos = math.sqrt(curr_x**2 + curr_y**2)
            
            # 如果 error_pos 很大，说明机器人在归零后又跑远了
            # 0.1m 是一个合理的容差
            if error_pos > 0.1: 
                return False, f"警告: 机器人发生漂移 (距初始点: {error_pos:.3f}m). 请重启程序或检查Odom."
        else:
            return False, "错误: 无法获取 SLAM 坐标 (TF map->base_link 尚未建立)"
        
        return True, "系统就绪，坐标已归零"