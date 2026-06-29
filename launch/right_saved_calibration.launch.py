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
                "--translation", "0.4265", "0.0929", "1.9393",
                "--quaternion",  "-0.5427", "0.5744", "-0.4582", "-0.4070",
                # "--quaternion",  "-0.5373", "0.5793", "-0.4568", "-0.4089",
                # ORIGINAL
                # "--translation", "0.457214", "0.0456883", "1.97405",
                # "--quaternion", "-0.563436", "0.576038", "-0.421077", "-0.416429",
                # "--roll",
                # "1.87399",
                # "--pitch",
                # "-0.00525849",
                # "--yaw",
                # "1.58905",
            ],
        ),
    ]
    return LaunchDescription(nodes)
