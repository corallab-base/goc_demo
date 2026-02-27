import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'goc_demo'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # include config and launch files
        (os.path.join('share', package_name, 'urdf'),
         glob(os.path.join('urdf', '*'))),
        (os.path.join('share', package_name, 'config'),
         glob(os.path.join('config', '*'))),
        (os.path.join('share', package_name, 'launch'),
         glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tassos',
    maintainer_email='tassos.manganaris@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'goc_demo_node = goc_demo.goc_demo_node:main',
            'goc_cartesian_demo_node = goc_demo.goc_cartesian_demo_node:main',
            'goc_cartesian_demo_node_1_robot = goc_demo.goc_cartesian_demo_node_1_robot:main',
            'interactive_camera_tweaker = goc_demo.interactive_camera_tweaker:main',
            'tracker_node = goc_demo.keypoint_tracker_node:main',
            'demo_world_node = goc_demo.demo_world_node:main',
        ],
    },
)
