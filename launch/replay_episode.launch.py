import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    GroupAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushROSNamespace
from launch_ros.substitutions import FindPackageShare

_SPAM_DATA_DIR = os.path.join(
    os.path.dirname(__file__),          # …/goc_demo/launch/
    "..", "..", "..", "..",              # up to workspace root
    "saved_data", "pick_and_place_spam",
)
_SPAM_DATA_DIR = os.path.normpath(_SPAM_DATA_DIR)


def generate_launch_description():
    data_dir_arg = DeclareLaunchArgument(
        "data_dir",
        default_value=_SPAM_DATA_DIR,
        description="Directory containing episode .pkl files",
    )
    episode_file_arg = DeclareLaunchArgument(
        "episode_file",
        description="Filename of the episode .pkl to replay (relative to data_dir)",
    )
    rate_hz_arg = DeclareLaunchArgument(
        "rate_hz", default_value="10.0",
        description="Playback rate in Hz",
    )
    loop_arg = DeclareLaunchArgument(
        "loop", default_value="true",
        description="Loop the episode continuously",
    )
    stage_arg = DeclareLaunchArgument(
        "stage", default_value="0",
        description=(
            "Restrict playback to one episode stage: "
            "0=all, 1=pre-grasp, 2=carry (grasp→release), 3=post-release"
        ),
    )
    launch_rviz_arg = DeclareLaunchArgument(
        "launch_rviz", default_value="true",
        description="Launch RViz?",
    )
    ur_type_arg = DeclareLaunchArgument(
        "ur_type", default_value="ur5e",
        description="UR robot type for URDF generation",
    )

    kinematics_params = PathJoinSubstitution(
        [FindPackageShare("ur_data_collection"), "config", "kinematics_params.yaml"]
    )

    episode_path = PathJoinSubstitution(
        [LaunchConfiguration("data_dir"), LaunchConfiguration("episode_file")]
    )

    rsp = GroupAction(actions=[
        PushROSNamespace("fake"),
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(
                PathJoinSubstitution(
                    [FindPackageShare("ur_robot_driver"), "launch", "ur_rsp.launch.py"]
                )
            ),
            launch_arguments={
                "robot_ip": "10.168.4.249",
                "ur_type": LaunchConfiguration("ur_type"),
                "use_fake_hardware": "true",
                "kinematics_params_file": kinematics_params,
                "tf_prefix": "replay_",
            }.items(),
        )
    ])

    replay_node = Node(
        package="goc_demo",
        executable="replay_pick_and_place",
        parameters=[{
            "episode_path": episode_path,
            "rate_hz": LaunchConfiguration("rate_hz"),
            "loop": LaunchConfiguration("loop"),
            "tf_prefix": "replay_",
            "stage": LaunchConfiguration("stage"),
        }],
        output="screen",
    )

    rviz_config = PathJoinSubstitution(
        [FindPackageShare("ur_data_collection"), "config", "replay_episode.rviz"]
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", rviz_config],
        condition=IfCondition(LaunchConfiguration("launch_rviz")),
        output="screen",
    )

    return LaunchDescription([
        data_dir_arg,
        episode_file_arg,
        rate_hz_arg,
        loop_arg,
        stage_arg,
        launch_rviz_arg,
        ur_type_arg,
        rsp,
        replay_node,
        rviz,
    ])
