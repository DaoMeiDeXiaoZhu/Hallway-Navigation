#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np

class LidarBridgeNode(Node):
    def __init__(self):
        super().__init__('lidar_bridge_node')

        # ---------------- 参数配置 ----------------
        self.target_points = 360  # 目标采样点数
        self.source_topic = '/scan_raw'
        self.target_topic = '/scan'
        # -----------------------------------------

        # 创建订阅者 (订阅原始数据)
        self.sub = self.create_subscription(
            LaserScan, 
            self.source_topic, 
            self.scan_callback, 
            10
        )

        # 创建发布者 (发布处理后的数据)
        self.pub = self.create_publisher(
            LaserScan, 
            self.target_topic, 
            10
        )

        self.get_logger().info(f'Lidar Bridge Started: {self.source_topic} -> {self.target_topic} ({self.target_points} points)')

    def scan_callback(self, msg: LaserScan):
        """
        回调函数：处理每一帧雷达数据
        """
        # 1. 将原始 ranges 转为 numpy 数组以便快速计算
        raw_ranges = np.array(msg.ranges)
        total_raw_points = len(raw_ranges)

        # ---------------------------------------------------------
        # 2. 数据清洗 (Data Cleaning)
        # ---------------------------------------------------------
        # 处理无穷大 (inf): 通常代表超出最大量程，设为 range_max
        raw_ranges[np.isinf(raw_ranges)] = msg.range_max
        # 处理非数字 (nan): 通常代表无效测量，设为 0.0 或 range_max
        raw_ranges[np.isnan(raw_ranges)] = 0.0

        # ---------------------------------------------------------
        # 3. 降采样 (Downsampling)
        # ---------------------------------------------------------
        # 生成均匀分布的索引。例如：从 0 到 1000 中均匀取 360 个整数索引
        if total_raw_points > 0:
            indices = np.linspace(0, total_raw_points - 1, self.target_points, dtype=int)
            sampled_ranges = raw_ranges[indices]
            
            # 如果有强度数据(intensities)，也要同步采样
            sampled_intensities = []
            if msg.intensities:
                raw_intensities = np.array(msg.intensities)
                # 检查强度数据长度是否匹配，防止越界
                if len(raw_intensities) == total_raw_points:
                    sampled_intensities = raw_intensities[indices].tolist()
        else:
            # 异常保护：如果原始数据为空
            sampled_ranges = np.zeros(self.target_points)
            sampled_intensities = []

        # ---------------------------------------------------------
        # 4. 数据限制 (Data Clipping)
        # ---------------------------------------------------------
        # 确保所有数据严格在 [min, max] 范围内
        sampled_ranges = np.clip(sampled_ranges, msg.range_min, msg.range_max)

        # ---------------------------------------------------------
        # 5. 构建新消息 (Repackaging)
        # ---------------------------------------------------------
        new_scan = LaserScan()
        new_scan.header = msg.header # 保持时间戳和 Frame ID 不变
        
        new_scan.angle_min = msg.angle_min
        new_scan.angle_max = msg.angle_max
        new_scan.range_min = msg.range_min
        new_scan.range_max = msg.range_max
        
        # [关键] 重新计算角度增量 (Angle Increment)
        # 因为点数变少了，每两个点之间的角度间隔变大了
        # 公式：总视场角 / (点数 - 1)
        if self.target_points > 1:
            new_scan.angle_increment = (msg.angle_max - msg.angle_min) / (self.target_points - 1)
        else:
            new_scan.angle_increment = 0.0
            
        # 重新估算时间增量 (按比例缩放)
        new_scan.time_increment = msg.time_increment * (total_raw_points / self.target_points) if self.target_points > 0 else 0.0
        new_scan.scan_time = msg.scan_time
        
        # 填入数据
        new_scan.ranges = sampled_ranges.tolist()
        new_scan.intensities = sampled_intensities

        # 发布
        self.pub.publish(new_scan)

def main(args=None):
    rclpy.init(args=args)
    node = LidarBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()