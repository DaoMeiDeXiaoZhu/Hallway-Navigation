from glob import glob
import os
from setuptools import find_packages, setup

package_name = 'real_env_navigation_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # 拷贝网络参数
        (os.path.join('share', package_name, 'checkpoint_pth'), glob('checkpoint_pth/*.pth')),
        
        # 拷贝所有的launch文件和目录结构
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),

        # 拷贝所有的rviz文件和目录结构
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='xiaozhu',
    maintainer_email='xiaozhu@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'inference_actor = real_env_navigation_pkg.actor:main',
            'inference_lidar = real_env_navigation_pkg.lidar_bridge:main',
        ],
    },
)
