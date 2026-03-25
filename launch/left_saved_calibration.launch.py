""" Static transform publisher acquired via MoveIt 2 hand-eye calibration """
""" EYE-TO-HAND: left_world -> camera_color_optical_frame """
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
                "left_world",
                "--frame",
                "camera_color_optical_frame",
                "--translation", "-1.30206", "-1.29951", "0.519321",
                "--quaternion", "-0.769982", "0.280968", "-0.207111", "0.534125",
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
