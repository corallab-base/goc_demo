# goc_demo/launch/tracker_with_realsense.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, TextSubstitution, PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
import os


def _topic(*parts):
    # Build "/<ns>/<parts...>" using launch substitutions
    return PathJoinSubstitution([TextSubstitution(text="/"), *parts])


def generate_launch_description():
    # --- Arguments ---
    camera_ns = LaunchConfiguration("camera_ns")
    publish_annotated = LaunchConfiguration("publish_annotated")
    depth_unit_scale = LaunchConfiguration("depth_unit_scale")

    declare_camera_ns = DeclareLaunchArgument(
        "camera_ns", default_value="camera",
        description="Namespace/name for the RealSense camera node (topics will be under /<camera_ns>/...)"
    )
    declare_publish_ann = DeclareLaunchArgument(
        "publish_annotated", default_value="true",
        description="Publish annotated images from the tracker"
    )
    declare_depth_scale = DeclareLaunchArgument(
        "depth_unit_scale", default_value="0.001",
        description="Scale for depth if using 16UC1 (e.g., 0.001 for mm->m)"
    )

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
            # "depth_module.enable_depth": "true",
            # "align_depth": "true",      # we need /aligned_depth_to_color/image_raw
            # optional: set profiles if you want
            "rgb_camera.color_profile": "640x360x15",
            "depth_module.depth_profile": "640x360x15",
        }.items(),
    )

    # --- Tracker node (your package/executable) ---
    color_topic = _topic("camera", camera_ns, "color", "image_raw")
    depth_topic = _topic("camera", camera_ns, "aligned_depth_to_color", "image_raw")
    info_topic  = _topic("camera", camera_ns, "color", "camera_info")

    tracker = Node(
        package="goc_demo",
        executable="tracker_node",
        name="sam2_click_tracker_node",
        output="screen",
        parameters=[{
            "color_topic": color_topic,
            "depth_topic": depth_topic,
            "camera_info_topic": info_topic,
            "publish_annotated": publish_annotated,
            "depth_unit_scale": depth_unit_scale,
        }],
        # If you prefer to place the tracker in the same namespace:
        # namespace=camera_ns,
    )

    return LaunchDescription([
        declare_camera_ns,
        declare_publish_ann,
        declare_depth_scale,
        realsense,
        tracker,
    ])
