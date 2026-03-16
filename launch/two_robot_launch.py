from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    GroupAction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import (
    AndSubstitution,
    LaunchConfiguration,
    NotSubstitution,
    PathJoinSubstitution,
)
from launch_ros.actions import Node, PushROSNamespace
from launch_ros.parameter_descriptions import ParameterFile
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    left_robot_ip_arg = DeclareLaunchArgument(
        "left_robot_ip", description="IP address by which the left robot can be reached."
    )
    right_robot_ip_arg = DeclareLaunchArgument(
        "right_robot_ip", description="IP address by which the right robot can be reached."
    )
    launch_rviz_arg = DeclareLaunchArgument("launch_rviz", default_value="true", description="Launch RViz?")

    left_robot_ip = LaunchConfiguration("left_robot_ip")
    right_robot_ip = LaunchConfiguration("right_robot_ip")

    one_robot_launch_file = PathJoinSubstitution(
        [FindPackageShare("goc_demo"), "launch", "one_robot_launch.py"]
    )

    left_controller_params_file = PathJoinSubstitution(
        [FindPackageShare("goc_demo"), "config", "left_controllers.yaml"]
    )
    right_controller_params_file = PathJoinSubstitution(
        [FindPackageShare("goc_demo"), "config", "right_controllers.yaml"]
    )

    left_robot_launch = GroupAction(actions=[
        PushROSNamespace("left"),
        IncludeLaunchDescription(
            one_robot_launch_file,
            launch_arguments={
                "robot_ip": left_robot_ip,
                "ur_type": "ur5e",
                "tf_prefix": "left_",
                "launch_rviz": "false",
                "controllers_file": left_controller_params_file,
                "reverse_port": "50001",
                "script_sender_port": "50002",
                "trajectory_port": "50003",
                "script_command_port": "50004",
            }.items(),
        )
    ])
    right_robot_launch = GroupAction(actions=[
        PushROSNamespace("right"),
        IncludeLaunchDescription(
            one_robot_launch_file,
            launch_arguments={
                "robot_ip": right_robot_ip,
                "ur_type": "ur5e",
                "tf_prefix": "right_",
                "launch_rviz": "false",
                "controllers_file": right_controller_params_file,
                "reverse_port": "50005",
                "script_sender_port": "50006",
                "trajectory_port": "50007",
                "script_command_port": "50008",
            }.items(),
        )
    ])

    launch_rviz = LaunchConfiguration("launch_rviz")
    rviz_config_file = PathJoinSubstitution(
        [FindPackageShare("goc_demo"), "config", "view_two_robots.rviz"]
    )

    rviz_node = Node(
        package="rviz2",
        condition=IfCondition(launch_rviz),
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
    )

    return LaunchDescription([
        left_robot_ip_arg,
        right_robot_ip_arg,
        launch_rviz_arg,
        left_robot_launch,
        right_robot_launch,
        rviz_node,
    ])
