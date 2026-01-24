import math
import os
import threading
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from ament_index_python import get_package_share_directory
from .nodes_reality import RobotEnvNode
from sac_reality import *
import rclpy

def main(args=None):
    rclpy.init(args=args)
    
    # 1. 初始化主节点
    node = RobotEnvNode()
    
    # 2. 开启后台线程处理回调
    # 注意：这里 daemon=True 意味着主线程一退，它就会立刻死掉，所以我们在 finally 里要小心
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # 3. 加载模型
    actor = Actor(state_dim=95, action_dim=2) 
    actor.eval()
    
    if not load_actor(actor, 'actor_prama.pth'):
        print("请检查模型路径！")
        return

    print("等待传感器初始化...")
    try:
        # 4. 等待传感器就绪
        while rclpy.ok():
            ready, msg = node.is_ready()
            if ready:
                print("传感器就绪，开始推理！")
                break
            else:
                print(f"等待中: {msg}")
                time.sleep(0.5)

        # 5. 推理循环
        print(">>> 开始推理循环...")
        rate = node.create_rate(10)
        while rclpy.ok():
            # --- 增加空值检查 ---
            if node.latest_pose is None or node.latest_scan is None:
                print("警告: 传感器数据丢失，跳过本次推理")
                rate.sleep()
                continue
            
            # A. 获取状态
            state = get_state(node)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            # B. 推理动作
            with torch.no_grad():
                _, _, action_mean = actor.sample(state_tensor)
                action = action_mean.cpu().numpy()[0]

            # C. 发送指令
            # print(f"Action: v={action[0]:.2f}, w={action[1]:.2f}") # 调试时可以打开
            node.move(action[0] * 0.5, action[1])
            
            # D. 维持频率
            rate.sleep()

    except KeyboardInterrupt:
        print("\n!!! 检测到键盘中断 (Ctrl+C) !!!")
    
    # === 新增：捕获其他所有报错，让你知道为什么挂了 ===
    except Exception as e:
        import traceback
        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(">>> 程序发生运行错误 (Runtime Error) <<<")
        print(f"错误信息: {e}")
        print("详细堆栈:")
        traceback.print_exc()
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

    finally:
        print("--------------------------------")
        print(">>> 正在执行系统级强制停车...")

        # 【核武器】直接调用系统终端命令
        # 这相当于你在终端里手敲了一行命令，不依赖当前 Python 脚本的任何状态
        # 即使 Python 挂了，这行命令也会由操作系统独立执行
        
        cmd = "ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist \"{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}\""
        
        # 连发 3 次，确保停车
        for i in range(3):
            print(f"   [System Call] 发送停车指令 ({i+1}/3)...")
            os.system(cmd) 
            time.sleep(0.1)

        print(">>> 正在清理资源...")
        try:
            node.destroy_node()
        except:
            pass
            
        if rclpy.ok():
            rclpy.shutdown()
            
        # 强制杀掉所有后台线程，不再等待
        print("!!! 程序已退出 !!!")
        # 强制退出 Python 进程，防止 spin 线程卡死
        os._exit(0)

if __name__ == '__main__':
    main()