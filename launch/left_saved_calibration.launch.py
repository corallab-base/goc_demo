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
                "--translation", "-1.8246", "-0.2100", "0.5692", 
                "--quaternion",  "-0.5427", "0.5744", "-0.4582", "0.4070",
                # ORIGINAL:
                # "--translation", "-1.78208", "-0.20803", "0.496675",
                # "--quaternion", "-0.542661", "0.574435", "-0.458176", "0.406962",
                # "--roll",
                # "2.8136",
                # "--pitch",
                # "1.30474",
                # "--yaw",
                # "1.87917",
            ],
        ),
    ]
    return LaunchDescription(nodes)
