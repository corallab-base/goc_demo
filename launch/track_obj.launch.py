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

    declare_obj_names = DeclareLaunchArgument(
        "obj_names",
        default_value="['cheezit']",
        description="Name of objects to use in the foundation pose tracker."
    )
    declare_mesh_file = DeclareLaunchArgument(
        "mesh_file",
        default_value="/home/tassos/phd/data/ycb/003_cracker_box_google_16k/003_cracker_box/google_16k/textured.obj",
        description="Path to mesh file to give to foundation pose."
    )

    camera_ns = LaunchConfiguration("camera_ns")
    obj_names = LaunchConfiguration("obj_names")
    mesh_file = LaunchConfiguration("mesh_file")

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

    object_tracker = Node(
        package='coral_trackers',
        executable='foundation_pose_tracker',
        name='foundation_pose_tracker',
        parameters=[{
            "rgb_topic": "/camera/camera/color/image_raw",
            "depth_topic": "/camera/camera/aligned_depth_to_color/image_raw",
            "camera_info_topic": "/camera/camera/aligned_depth_to_color/camera_info",
            # "mesh_file_path": "/home/tassos/phd/data/ycb/010_potted_meat_can_google_16k/010_potted_meat_can/google_16k/textured.obj",
            "mesh_file_path": mesh_file,
            "objects": obj_names,
        }],
        output='screen',
        # parameters=[{'queue_size': 20, 'approximate': True}],
    )

    return LaunchDescription([
        declare_camera_ns,
        declare_cam_serial,
        declare_width,
        declare_height,
        declare_fps,
        declare_mesh_file,
        declare_obj_names,
        realsense,
        object_tracker
    ])
