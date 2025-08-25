import os

from ament_index_python.packages import get_package_share_directory

# demo_world.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():


    urdf = PathJoinSubstitution([FindPackageShare('goc_demo'), 'urdf', 'workspace.urdf.xacro'])
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='rsp',
        parameters=[{'robot_description': ParameterValue(Command(['xacro ', urdf]), value_type=str)}],
        output='screen',
    )

    demo = Node(
        package='goc_demo',
        executable='demo_world_node',  # or `python3 demo_world_node.py`
        name='demo_world_node',
        parameters=[{
            'world_frame': 'world',
            'camera_frame': 'camera_color_optical_frame',
            'centroids_px_topic': '/sam2_click_tracker_node/centroids_px',
            'centroids_3d_topic': '/sam2_click_tracker_node/centroids_3d',
            'depth_topic': '/camera/camera/aligned_depth_to_color/image_raw',
            'camera_info_topic': '/camera/camera/color/camera_info',
            'publish_markers': True,
        }],
        output='screen',
    )

    # --- RViz2 with preloaded config ---
    rviz_cfg = PathJoinSubstitution([FindPackageShare('goc_demo'), "config", "centroids_world.rviz"])
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_cfg],
    )
    return LaunchDescription([rsp, demo, rviz])
