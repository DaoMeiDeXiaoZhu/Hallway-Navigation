import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState
import math
import time

class SceneAnimator(Node):
    def __init__(self):
        super().__init__('scene_animator_node')
        
        self.client = self.create_client(SetEntityState, '/set_entity_state')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('正在连接 Gazebo 服务 /set_entity_state ...')
            
        self.get_logger().info('服务已连接，开始执行轨迹控制...')
        self.start_time = time.time()
        
        # 50Hz 更新频率
        self.timer = self.create_timer(0.02, self.update_all_positions)

    def set_model_pose(self, name, x, y, z, yaw=0.0):
        req = SetEntityState.Request()
        req.state.name = name
        req.state.pose.position.x = float(x)
        req.state.pose.position.y = float(y)
        req.state.pose.position.z = float(z)
        req.state.pose.orientation.z = math.sin(yaw / 2.0)
        req.state.pose.orientation.w = math.cos(yaw / 2.0)
        req.state.reference_frame = 'world'
        self.client.call_async(req)

    def update_all_positions(self):
        t = time.time() - self.start_time
        
        # ========================================================
        # 【修改点】 速度参数设置
        # 1.0 m/s = 正常步行
        # 0.5 m/s = 老年人/慢速散步
        # 1.5 m/s = 快走 (你之前的速度)
        # ========================================================
        
        # --------------------------------------------------------
        # 1. cylinder_1: 正常速度行人 (约 0.75 m/s)
        # --------------------------------------------------------
        # 速度系数 0.5 -> V_max = 1.5 * 0.5 = 0.75 m/s
        s1 = 0.6
        cyl1_x = 1.0
        cyl1_y = -1.5 * math.cos(t * s1)
        # 简单的朝向计算：根据往返方向改变朝向 (0 或 3.14)
        yaw1 = 0.0 if math.sin(t * s1) > 0 else 3.14 
        self.set_model_pose("cylinder_1", cyl1_x, cyl1_y, 0.875, yaw=yaw1)

        # --------------------------------------------------------
        # 2. cylinder_2: 稍慢的行人 (约 0.6 m/s)
        # --------------------------------------------------------
        # 速度系数 0.4 -> V_max = 1.5 * 0.4 = 0.6 m/s
        # 加上相位偏移 +1.0，避免和 cylinder_1 完全同步
        s2 = 0.8
        offset2 = 1.0 
        cyl2_x = 2.0
        cyl2_y = 1.5 * math.cos(t * s2 + offset2)
        yaw2 = 0.0 if math.sin(t * s2 + offset2) > 0 else 3.14
        self.set_model_pose("cylinder_2", cyl2_x, cyl2_y, 0.875, yaw=yaw2)

        # --------------------------------------------------------
        # 3. cylinder_3: 绕圈散步 (约 0.68 m/s)
        # --------------------------------------------------------
        # 半径 R=1.7，目标速度 V=0.7 m/s
        # 角速度 w = V / R = 0.7 / 1.7 ≈ 0.4
        s3 = 0.4
        radius = 1.7
        center_x = 4.7
        center_y = 0.0
        
        cyl3_x = center_x + radius * math.cos(t * s3 + math.pi)
        cyl3_y = center_y + radius * math.sin(t * s3 + math.pi)
        
        # 计算圆周运动的朝向 (切线方向)
        # 或者是简单的面向圆心/背向圆心，这里计算的是切线朝向
        yaw3 = (t * s3 + math.pi) + (math.pi / 2)
        
        self.set_model_pose("cylinder_3", cyl3_x, cyl3_y, 0.875, yaw=yaw3)

        # --------------------------------------------------------
        # 4. move_pillar: 极慢的障碍物 (约 0.3 m/s)
        # --------------------------------------------------------
        # s4 = 0.5
        # pillar_x = 4.7
        # pillar_y = 1.5 * math.cos(t * s4)
        # self.set_model_pose("move_pillar", pillar_x, pillar_y, 1.0)

def main(args=None):
    rclpy.init(args=args)
    node = SceneAnimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()