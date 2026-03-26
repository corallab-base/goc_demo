""" Static transform publisher acquired via MoveIt 2 hand-eye calibration """
""" EYE-TO-HAND: camera_color_optical_frame -> right_world """
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    nodes = [
        Node(
            package="goc_demo",
            executable="tf_tweaker",
            namespace="right",
            output="log",
            arguments=[
                "--parent",
                "camera_color_optical_frame",
                "--frame",
                "right_world",
                "--translation", "0.371261", "-0.0465642", "1.71467",
                "--quaternion", "-0.563191", "0.582802", "-0.418913", "-0.409475",
                # "--roll",
                # "1.91024",
                # "--pitch",
                # "0.0202526",
                # "--yaw",
                # "0.679201",
            ],
        ),
    ]
    return LaunchDescription(nodes)
