import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetEntityState
import numpy as np
import math

class RobotEnvNode(Node):
    def __init__(self):
        super().__init__('robot_env_node')

        # === 1. 参数与状态变量 ===
        self.robot_name = 'my_robot' 
        self.range_min, self.range_max = 0.05, 12.0
        
        # 噪声参数
        self.noise_pos_std = 0.0
        self.noise_ori_std = 0.0
        self.noise_vel_lin = 0.0
        self.noise_vel_ang = 0.0

        self.latest_scan = None
        self.latest_twist = None
        self.latest_pose = None

        # === 2. 通信接口 ===
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(ModelStates, '/model_states', self.model_states_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.set_entity_client = self.create_client(SetEntityState, '/set_entity_state')

    # --- 数据处理逻辑 (保持你的原始实现，已包含噪声和雷达预处理) ---
    def model_states_callback(self, msg: ModelStates):
        try:
            if self.robot_name in msg.name:
                idx = msg.name.index(self.robot_name)
                p = msg.pose[idx].position
                q = msg.pose[idx].orientation
                t = msg.twist[idx]
                roll, pitch, yaw = self.quaternion_to_euler(q)
                
                # 注入高斯噪声
                n_x, n_y = np.random.normal(0, self.noise_pos_std, 2)
                n_yaw = np.random.normal(0, self.noise_ori_std)
                n_vx, n_vy = np.random.normal(0, self.noise_vel_lin, 2)
                n_wz = np.random.normal(0, self.noise_vel_ang)

                self.latest_pose = (p.x + n_x, p.y + n_y, p.z, roll, pitch, yaw + n_yaw)
                self.latest_twist = (t.linear.x + n_vx, t.linear.y + n_vy, t.angular.z + n_wz)
        except Exception as e:
            self.get_logger().warn(f'获取状态异常: {e}')

    def scan_callback(self, msg: LaserScan):
        # ------------------------------------------------------------------
        # 1. 第一步：清洗所有“脏数据”，统一归为 0.0
        # ------------------------------------------------------------------
        raw_ranges = np.array(msg.ranges)
        total_points = len(raw_ranges)
        
        # 必须先处理 NaN，防止后续计算报错
        raw_ranges[np.isnan(raw_ranges)] = 0.0
        raw_ranges[np.isinf(raw_ranges)] = 0.0
        # 模拟真机：小于物理最小距离的也归为 0.0
        raw_ranges[raw_ranges < self.range_min] = 0.0
        
        # 此时 raw_ranges 里只有两种值：
        #   A. 有效测量值 (e.g. 0.5, 3.0)
        #   B. 0.0 (代表 盲区 或 超量程)
        
        processed_ranges = raw_ranges.copy()
        
        # 获取 0 值和 非0 值 的索引
        zero_indices = np.where(processed_ranges == 0.0)[0]
        valid_indices = np.where(processed_ranges > 0.0)[0]

        # ------------------------------------------------------------------
        # 2. 第二步：先进启发式算法 (基于空间连续性)
        # ------------------------------------------------------------------
        
        # 情况 A: 极端情况，整圈都没有有效数据 (比如在一个巨大的空房间中心)
        if len(valid_indices) == 0:
            # 全部视为无限远
            processed_ranges[:] = self.range_max
            
        # 情况 B: 正常情况，混合了 0 和有效值
        else:
            # 算法核心参数：判定阈值
            # 如果最近的邻居小于 0.4m，我们认为当前的 0 是因为贴脸导致的盲区
            # 否则，我们认为当前的 0 是因为太远导致的
            BLIND_SPOT_THRESHOLD = 0.4 
            
            # --- 向量化计算 (比 for 循环更快) ---
            # 我们需要为每一个 0 点找到最近的有效点
            # 这里使用 numpy 的广播机制计算环形距离
            
            # 1. 扩展维度以构建距离矩阵 (Zero x Valid)
            z_idx = zero_indices[:, np.newaxis]  # Shape: (N_zeros, 1)
            v_idx = valid_indices[np.newaxis, :] # Shape: (1, N_valid)
            
            # 2. 计算直接距离
            abs_dist = np.abs(z_idx - v_idx)
            
            # 3. 计算环形距离 (处理首尾相接)
            # 例如：总长360，索引0和索引359的距离应该是1，而不是359
            circular_dist = np.minimum(abs_dist, total_points - abs_dist)
            
            # 4. 找到每个 0 点对应的最近有效点的索引位置 (argmin 返回的是 v_idx 的下标)
            nearest_valid_arg = np.argmin(circular_dist, axis=1)
            
            # 5. 获取这些最近邻居的真实距离值
            # valid_indices[nearest_valid_arg] 找到了原始 ranges 里的索引
            nearest_vals = raw_ranges[valid_indices[nearest_valid_arg]]
            
            # --- 核心分类逻辑 ---
            # 创建掩码：哪些 0 对应的邻居是“远”的
            is_far_mask = nearest_vals > BLIND_SPOT_THRESHOLD
            
            # 应用逻辑：
            # 1. 邻居很远 -> 说明这个 0 是超量程 -> 设为 range_max (Inf)
            processed_ranges[zero_indices[is_far_mask]] = self.range_max
            
            # 2. 邻居很近 (is_far_mask 为 False) -> 说明这个 0 是盲区
            #    策略：保持为 0.0 或者设为 range_min。
            #    为了让 RL 明确知道这是“危险”，建议保持 0.0 或设为非常小的值
            #    这里我们不修改它，让他保持 0.0，等待后续处理
            pass 

        # ------------------------------------------------------------------
        # 3. 第三步：采样 (注意：这里不要盲目 clip)
        # ------------------------------------------------------------------
        
        # 只有当你确定网络不能接受 0.0 时才做 clip。
        # 如果你的网络希望 0.0 代表“碰撞”，就不要 clip 下限。
        # 这里为了安全，我们只限制上限，保留 0.0 作为“贴脸/碰撞”的特征
        processed_ranges = np.clip(processed_ranges, 0.0, self.range_max)
        
        # 均匀降采样
        indices = np.linspace(0, total_points - 1, 90, dtype=int)
        self.latest_scan = processed_ranges[indices]

    def quaternion_to_euler(self, q):
        sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    def move(self, linear_x=0.0, angular_z=0.0):
        cmd = Twist()
        cmd.linear.x = float(linear_x)
        cmd.angular.z = float(angular_z)
        self.cmd_pub.publish(cmd)

    # === 3. 核心改进：系统就绪检查 ===
    def wait_for_system_ready(self, timeout=15.0):
        """
        启动时阻塞等待，直到 Gazebo 服务可用且传感器有数据。
        """
        self.get_logger().info("正在检查系统就绪状态...")
        start_time = time.time()
        
        # 1. 检查服务
        while not self.set_entity_client.wait_for_service(timeout_sec=1.0):
            if time.time() - start_time > timeout:
                self.get_logger().error("服务等待超时！")
                return False
            self.get_logger().info('等待 /set_entity_state 服务中...')

        # 2. 检查数据回传 (核心：必须通过 spin 让回调执行)
        while rclpy.ok():
            # 必须调用 spin_once，否则 callback 永远不会被触发，latest_scan 永远是 None
            rclpy.spin_once(self, timeout_sec=0.1)
            
            if self.latest_pose is not None and self.latest_scan is not None:
                self.get_logger().info(">>> 系统已就绪，传感器数据正常。")
                return True
            
            if time.time() - start_time > timeout:
                self.get_logger().error("传感器数据接收超时！请检查主题发布是否正常。")
                return False
            
            self.get_logger().info("等待数据回传中...", throttle_duration_sec=2.0)
        return False

    # === 4. 核心改进：重置逻辑 ===
    def reset(self):
        self.get_logger().info(">>> 触发重置...")

        # 1. 停止机器人并确保 Service 在线
        self.move(0.0, 0.0) 
        if not self.set_entity_client.service_is_ready():
            self.set_entity_client.wait_for_service()

        # 2. 构造重置请求
        req = SetEntityState.Request()
        req.state.name = self.robot_name
        req.state.pose.position.x = 0.0
        req.state.pose.position.y = 0.0
        req.state.pose.position.z = 0.02 # 略微悬空防止卡地壳
        req.state.pose.orientation.w = 1.0
        # 必须把所有速度清零，防止重置后机器人带着惯性飞出去
        req.state.twist.linear.x = 0.0
        req.state.twist.linear.y = 0.0
        req.state.twist.angular.z = 0.0

        success = False
        for i in range(5): # 最多尝试 5 次重置服务调用
            self.latest_scan = None
            self.latest_pose = None
            
            future = self.set_entity_client.call_async(req)
            
            # 手动轮询，不使用嵌套的 spin_until_future_complete
            while rclpy.ok() and not future.done():
                # 使用 node 自己的 executor 执行一次任务队列
                rclpy.spin_once(self, timeout_sec=0.1)
            
            # 等待数据刷新，确保拿到的 Scan 是重置后的数据
            for _ in range(20):
                rclpy.spin_once(self, timeout_sec=0.05)
                if self.latest_pose is not None and self.latest_scan is not None:
                    # 检查位置是否接近原点
                    dist = math.sqrt(self.latest_pose[0]**2 + self.latest_pose[1]**2)
                    if dist < 0.1:
                        success = True
                        break
            if success: break
            self.get_logger().warn(f"重置尝试 {i+1} 失败，正在重试...")

        self.get_logger().info(f"<<< 重置完成 (成功: {success})")
        return self.latest_scan, self.latest_pose