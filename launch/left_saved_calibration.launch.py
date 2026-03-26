""" Static transform publisher acquired via MoveIt 2 hand-eye calibration """
""" EYE-TO-HAND: left_world -> camera_color_optical_frame """
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    nodes = [
        Node(
            package="goc_demo",
            executable="tf_tweaker",
            namespace="left",
            output="log",
            arguments=[
                "--parent",
                "left_world",
                "--frame",
                "camera_color_optical_frame",
                "--translation", "-1.62998", "-0.321902", "0.52115",
                "--quaternion", "-0.568393", "0.58145", "-0.415384", "0.407801",
                # "--roll",
                # "1.11791",
                # "--pitch",
                # "2.47401",
                # "--yaw",
                # "-2.869",
            ],
        ),
    ]
    return LaunchDescription(nodes)
