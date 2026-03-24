import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, TextSubstitution, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource


def _topic(*parts):
    # Build "/<ns>/<parts...>" using launch substitutions
    return PathJoinSubstitution([TextSubstitution(text="/"), *parts])


def generate_launch_description():
    declare_camera_ns = DeclareLaunchArgument(
        "camera_ns", default_value="camera",
        description="Namespace/name for the RealSense camera node (topics will be under /<camera_ns>/...)"
    )
    declare_cam_serial = DeclareLaunchArgument('serial', default_value='',
                                               description='Specific RealSense serial (optional)')
    declare_width  = DeclareLaunchArgument('width',  default_value='1280')
    declare_height = DeclareLaunchArgument('height', default_value='720')
    declare_fps    = DeclareLaunchArgument('fps',    default_value='30')

    camera_ns = LaunchConfiguration("camera_ns")

    # --- RealSense (use official launch, easy align_depth) ---
    rs_launch_path = os.path.join(
        get_package_share_directory("realsense2_camera"), "launch", "rs_launch.py"
    )
    realsense = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(rs_launch_path),
        launch_arguments={
            "camera_name": camera_ns,   # topics under /<camera_ns>
            "enable_rgbd": "true",
            "enable_sync": "true",
            "align_depth.enable": "true",
            "enable_color": "true",
            "enable_depth": "true",
            # optional: set profiles if you want
            "rgb_camera.color_profile": "640x360x15",
            "rgb_camera.enable_auto_exposure": "false",
            "rgb_camera.exposure": "600",
            "depth_module.depth_profile": "640x360x15",
        }.items(),
    )

    # --- Tracker node (your package/executable) ---
    color_topic = _topic("camera", camera_ns, "color", "image_raw")
    depth_topic = _topic("camera", camera_ns, "aligned_depth_to_color", "image_raw")
    info_topic  = _topic("camera", camera_ns, "color", "camera_info")

    colors_tracker = Node(
        package='coral_trackers',
        executable='colors_tracker',
        name='colors_tracker',
        remappings=[
            ('image',        color_topic),
            ('camera_info',  info_topic),
            ('depth',        depth_topic),
        ],
        output='screen',
        # parameters=[{'queue_size': 20, 'approximate': True}],
    )

    return LaunchDescription([
        declare_camera_ns,
        declare_cam_serial,
        declare_width,
        declare_height,
        declare_fps,
        realsense,
        colors_tracker
    ])
