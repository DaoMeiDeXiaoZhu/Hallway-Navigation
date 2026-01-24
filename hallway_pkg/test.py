#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class YVelPublisher(Node):
    def __init__(self):
        super().__init__('y_vel_publisher')
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # 定时器：10Hz 发布
        self.timer = self.create_timer(0.1, self.timer_callback)

        # 你想测试的横向速度（m/s）
        self.y_speed = 0.2

        self.get_logger().info(f"Publishing y velocity: {self.y_speed} m/s")

    def timer_callback(self):
        msg = Twist()
        msg.linear.y = self.y_speed
        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = YVelPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
