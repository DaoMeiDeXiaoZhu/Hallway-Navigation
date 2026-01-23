import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import time

class ScanRawReader(Node):
    def __init__(self):
        super().__init__('scan_raw_reader')

        self.sub = self.create_subscription(
            LaserScan,
            '/scan_raw',
            self.scan_callback,
            10
        )

        self.last_print_time = time.time()
        self.get_logger().info("已启动 /scan_raw 订阅节点，每 0.5 秒输出一次 ranges[0]")

    def scan_callback(self, msg: LaserScan):
        now = time.time()

        # 每 0.5 秒输出一次
        if now - self.last_print_time >= 0.5:
            if len(msg.ranges) > 0:
                self.get_logger().info(f"ranges长度 = {len(msg.ranges)}")
            else:
                self.get_logger().warn("ranges 数组为空！")

            self.last_print_time = now

def main(args=None):
    rclpy.init(args=args)
    node = ScanRawReader()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
