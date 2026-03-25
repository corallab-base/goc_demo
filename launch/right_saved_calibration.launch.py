""" Static transform publisher acquired via MoveIt 2 hand-eye calibration """
""" EYE-TO-HAND: camera_color_optical_frame -> right_world """
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    nodes = [
        Node(
            package="goc_demo",
            executable="tf_tweaker",
            output="log",
            arguments=[
                "--parent",
                "camera_color_optical_frame",
                "--frame",
                "right_world",
                "--translation", "0.595444", "0.0586775", "1.36572",
                "--quaternion", "0.771667", "-0.266418", "0.200159", "0.541745",
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
