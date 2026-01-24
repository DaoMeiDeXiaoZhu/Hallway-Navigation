from glob import glob
import os
from setuptools import find_packages, setup

package_name = 'hallway_pkg'

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

        # launch
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.py')),

        # urdf
        (os.path.join('share', package_name, 'urdf'),
         glob('urdf/*.xacro')),

        # world
        (os.path.join('share', package_name, 'world'),
         glob('world/*.world')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='xiaozhu',
    maintainer_email='xiaozhu@todo.todo',
    description='Hallway simulation package',
    license='Apache-2.0',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
        #  艺名 (Executable)     =    文件夹.文件名:入口函数
        'obstacle_mover = hallway_pkg.obstacle_move:main',
        'run_simulation = hallway_pkg.run_simulation:main',
        'inference_lidar = hallway_pkg.lidar_bridge:main',
        'run_reality = hallway_pkg.run_reality:main',
        ],
    },
)
