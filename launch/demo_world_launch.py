import os

from ament_index_python.packages import get_package_share_directory

# demo_world.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, IncludeLaunchDescription, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, PushRosNamespace
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterFile, ParameterValue
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
)

def make_dual_ur5e_launch_items(robot_description):
    # Declare arguments
    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            "use_fake_hardware",
            default_value="false",
            description="Start robot with fake hardware mirroring command to its states.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "fake_sensor_commands",
            default_value="false",
            description="Enable fake command interfaces for sensors used for simple simulations. "
            "Used only if 'use_fake_hardware' parameter is true.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "headless_mode",
            default_value="false",
            description="Enable headless mode for robot control",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "controller_spawner_timeout",
            default_value="10",
            description="Timeout used when spawning controllers.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "initial_joint_controller",
            default_value="scaled_joint_trajectory_controller",
            description="Initially loaded robot controller.",
            choices=[
                "scaled_joint_trajectory_controller",
                "joint_trajectory_controller",
                "forward_velocity_controller",
                "forward_position_controller",
                "freedrive_mode_controller",
                "passthrough_trajectory_controller",
            ],
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "activate_joint_controller",
            default_value="false",
            description="Activate loaded joint controller.",
        )
    )

    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    fake_sensor_commands = LaunchConfiguration("fake_sensor_commands")
    initial_joint_controller = LaunchConfiguration("initial_joint_controller")
    activate_joint_controller = LaunchConfiguration("activate_joint_controller")
    headless_mode = LaunchConfiguration("headless_mode")
    controller_spawner_timeout = LaunchConfiguration("controller_spawner_timeout")

    launches = []

    # "left_", "right_"

    update_rate_config_file = PathJoinSubstitution(
        [FindPackageShare("ur_robot_driver"), "config", "ur5e_update_rate.yaml"]
    )

    # FOR THE CONTROLLER
    goc_demo_share = FindPackageShare("goc_demo")
    common_params_file = PathJoinSubstitution(
        [goc_demo_share, "config", "common_ur5e_control_params.yaml"]
    )
    left_params_file = PathJoinSubstitution(
        [goc_demo_share, "config", "left_ur5e_control_params.yaml"]
    )
    right_params_file = PathJoinSubstitution(
        [goc_demo_share, "config", "right_ur5e_control_params.yaml"]
    )
    single_params_file = PathJoinSubstitution(
        [goc_demo_share, "config", "single_ur5e_control_params.yaml"]
    )

    ur_control_node = Node(
        package="ur_robot_driver",
        executable="ur_ros2_control_node",
        parameters=[
            robot_description,
            update_rate_config_file,
            # ParameterFile(common_params_file, allow_substs=True),
            # ParameterFile(single_params_file, allow_substs=True),
            ParameterFile(left_params_file, allow_substs=True),
            ParameterFile(right_params_file, allow_substs=True),
        ],
        output="screen",
    )

    # Spawn controllers
    def controller_spawner(controllers, active=True):
        inactive_flags = ["--inactive"] if not active else []
        return Node(
            package="controller_manager",
            executable="spawner",
            arguments=[
                "--controller-manager",
                "/controller_manager",
                "--controller-manager-timeout",
                controller_spawner_timeout,
            ]
            + inactive_flags
            + controllers,
        )

    # , "right_"

    for i, prefix in enumerate(["left_", "right_"]):
        declared_arguments.append(
            DeclareLaunchArgument(
                prefix+"robot_ip",
                description=f"IP address by which robot {prefix} can be reached.",
            )
        )

        # Initialize Arguments
        robot_ip = LaunchConfiguration(prefix+"robot_ip")

        dashboard_client_node = Node(
            package="ur_robot_driver",
            executable="dashboard_client",
            name="dashboard_client",
            output="screen",
            emulate_tty=True,
            parameters=[{"robot_ip": robot_ip}],
        )

        robot_state_helper_node = Node(
            package="ur_robot_driver",
            executable="robot_state_helper",
            name="ur_robot_state_helper",
            output="screen",
            parameters=[
                {"headless_mode": headless_mode},
                {"robot_ip": robot_ip},
            ],
        )

        urscript_interface = Node(
            package="ur_robot_driver",
            executable="urscript_interface",
            parameters=[{"robot_ip": robot_ip}],
            output="screen",
        )

        controller_stopper_node = Node(
            package="ur_robot_driver",
            executable="controller_stopper_node",
            name="controller_stopper",
            output="screen",
            emulate_tty=True,
            parameters=[
                {"headless_mode": headless_mode},
                {"joint_controller_active": activate_joint_controller},
                {
                    "consistent_controllers": [
                        prefix+"joint_state_broadcaster",
                        prefix+"io_and_status_controller",
                        prefix+"speed_scaling_state_broadcaster",
                        prefix+"force_torque_sensor_broadcaster",
                        prefix+"tcp_pose_broadcaster",
                        prefix+"ur_configuration_controller",
                    ]
                },
            ],
        )

        controllers_active = [
            prefix+"joint_state_broadcaster",
            prefix+"io_and_status_controller",
            prefix+"speed_scaling_state_broadcaster",
            prefix+"force_torque_sensor_broadcaster",
            prefix+"tcp_pose_broadcaster",
            prefix+"ur_configuration_controller",
            prefix+"scaled_joint_trajectory_controller",
        ]
        controllers_inactive = [
            prefix+"joint_trajectory_controller",
            prefix+"forward_velocity_controller",
            prefix+"forward_position_controller",
            prefix+"force_mode_controller",
            prefix+"passthrough_trajectory_controller",
            prefix+"freedrive_mode_controller",
            prefix+"tool_contact_controller",
        ]
        controller_spawners = [
            controller_spawner(controllers_active),
            controller_spawner(controllers_inactive, active=False),
        ]

        launches.extend([dashboard_client_node,
                         robot_state_helper_node,
                         urscript_interface,
                         controller_stopper_node] + controller_spawners)

    return LaunchDescription(declared_arguments + [ur_control_node] + launches)


def generate_launch_description():
    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument("launch_rviz", default_value="true", description="Launch RViz?")
    )

    urdf = PathJoinSubstitution([FindPackageShare('goc_demo'), 'urdf', 'workspace.urdf.xacro'])
    # urdf = PathJoinSubstitution([FindPackageShare('goc_demo'), 'urdf', 'single_robot_workspace.urdf.xacro'])
    robot_description_content = Command(['xacro ', urdf])
    robot_description = {
        "robot_description": ParameterValue(value=robot_description_content, value_type=str)
    }
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='both',
        name='rsp',
        parameters=[robot_description],
    )

    demo = Node(
        package='goc_demo',
        executable='demo_world_node',
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

    robot_launch_items = make_dual_ur5e_launch_items(robot_description)

    launch_rviz = LaunchConfiguration("launch_rviz")

    # --- RViz2 with preloaded config ---
    rviz_cfg = PathJoinSubstitution([FindPackageShare('goc_demo'), "config", "centroids_world.rviz"])
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_cfg],
        condition=IfCondition(launch_rviz)
    )

    return LaunchDescription(declared_arguments + [rsp, demo, robot_launch_items, rviz])
