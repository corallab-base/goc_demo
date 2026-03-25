#!/usr/bin/env python3

import sys
import argparse
from typing import Tuple

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose, TransformStamped
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, Marker
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from interactive_markers.menu_handler import MenuHandler

from tf2_ros import TransformBroadcaster, Buffer, TransformListener

class TfTweakerNode(Node):
    """
    Grabs an existing transform from the TF tree, spawns a 6-DOF interactive marker
    at that pose, and allows you to tweak it visually in RViz.
    Publishes the new pose as a Static TF only when triggered via the menu.
    """

    def __init__(self, parent_frame: str, target_frame: str, init_trans=None, init_quat=None):
        super().__init__("tf_tweaker_node")

        self.parent_frame = parent_frame
        self.target_frame = target_frame
        self.marker_scale = 0.25

        # State
        self.current_pose = Pose()
        self.initialized = False

        if init_trans and init_quat:
            self.current_pose.position.x, self.current_pose.position.y, self.current_pose.position.z = init_trans
            self.current_pose.orientation.x = init_quat[0]
            self.current_pose.orientation.y = init_quat[1]
            self.current_pose.orientation.z = init_quat[2]
            self.current_pose.orientation.w = init_quat[3]

            self.initialized = True
            self.get_logger().info("Using user-provided initial transform. Skipping TF lookup.")
        else:
            # 2. Fallback to dynamic lookup
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
            self.server = InteractiveMarkerServer(self, "tf_tweaker_server")
            self.init_timer = self.create_timer(0.5, self._try_init_from_tf)
            self.get_logger().info(f"Waiting for TF {parent_frame} -> {target_frame}...")

        # TF publication setup
        self.tf_pub = TransformBroadcaster(self)
        self.is_broadcasting = False
        self.pub_timer = None

        # Interactive marker server + menu
        self.server = InteractiveMarkerServer(self, "tf_tweaker_server")
        self.menu_handler = MenuHandler()
        self._make_interactive_marker()

        # Set up our right-click menu options
        self.menu_handler.insert("1. Publish as Static TF", callback=self._on_menu_publish_static)
        self.menu_handler.insert("2. Print Current Transform", callback=self._on_menu_print)

    def _try_init_from_tf(self):
        """Polls the TF tree until the target transform is available."""
        if self.initialized:
            return

        try:
            tf = self.tf_buffer.lookup_transform(
                self.parent_frame,
                self.target_frame,
                rclpy.time.Time()
            )

            # Seed our pose from the TF tree
            self.current_pose.position.x = tf.transform.translation.x
            self.current_pose.position.y = tf.transform.translation.y
            self.current_pose.position.z = tf.transform.translation.z
            self.current_pose.orientation = tf.transform.rotation

            self.initialized = True
            self.init_timer.cancel() # Stop polling

            self.get_logger().info(f"Success! Captured initial pose for '{self.target_frame}'. Spawning marker.")
            self._make_interactive_marker()

        except Exception as e:
            # Silently wait until the TF is published by the original source
            pass

    # -------- Interactive marker creation --------
    def _make_interactive_marker(self):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = self.parent_frame
        int_marker.name = f"tweaker_{self.target_frame}"
        int_marker.description = f"Tweak: {self.target_frame}"
        int_marker.scale = self.marker_scale
        int_marker.pose = self.current_pose

        # Add a central visual block
        control = InteractiveMarkerControl()
        control.always_visible = True
        control.markers.append(self._make_body_marker(self.marker_scale))
        int_marker.controls.append(control)

        # Append standard RViz 6-DOF rings and arrows
        int_marker.controls.extend(self._make_6dof_controls())

        self.server.insert(int_marker, feedback_callback=self._process_feedback)
        self.menu_handler.apply(self.server, int_marker.name)
        self.server.applyChanges()

    def _make_body_marker(self, s: float) -> Marker:
        m = Marker()
        m.type = Marker.CUBE
        m.scale.x = s * 0.3
        m.scale.y = s * 0.3
        m.scale.z = s * 0.3
        m.color.a = 0.8
        m.color.r = 0.2; m.color.g = 0.8; m.color.b = 0.2
        return m

    def _make_6dof_controls(self):
        controls = []

        def make_control(name, mode, x, y, z, w):
            c = InteractiveMarkerControl()
            c.name = name
            c.interaction_mode = mode
            c.orientation.x = float(x)
            c.orientation.y = float(y)
            c.orientation.z = float(z)
            c.orientation.w = float(w)
            return c

        controls.append(make_control("move_x",   InteractiveMarkerControl.MOVE_AXIS,   1, 0, 0, 1))
        controls.append(make_control("rotate_x", InteractiveMarkerControl.ROTATE_AXIS, 1, 0, 0, 1))
        controls.append(make_control("move_y",   InteractiveMarkerControl.MOVE_AXIS,   0, 0, 1, 1))
        controls.append(make_control("rotate_y", InteractiveMarkerControl.ROTATE_AXIS, 0, 0, 1, 1))
        controls.append(make_control("move_z",   InteractiveMarkerControl.MOVE_AXIS,   0, 1, 0, 1))
        controls.append(make_control("rotate_z", InteractiveMarkerControl.ROTATE_AXIS, 0, 1, 0, 1))

        return controls

    # -------- Feedback & menu --------
    def _process_feedback(self, feedback):
        # Update our internal state when the marker is dragged, but DO NOT publish TF yet.
        self.current_pose = feedback.pose

    def _on_menu_publish_static(self, feedback):
        """Starts a continuous dynamic broadcast of the current tweaked pose."""
        self.is_broadcasting = True

        # If a timer already exists, cancel it first to reset
        if self.pub_timer:
            self.pub_timer.cancel()

        # Start the 30Hz broadcast loop
        self.pub_timer = self.create_timer(1.0/30.0, self._publish_loop)
        self.get_logger().info(f"ACTIVE: Now broadcasting {self.parent_frame} -> {self.target_frame}")

    def _publish_loop(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.parent_frame
        t.child_frame_id = self.target_frame

        t.transform.translation.x = self.current_pose.position.x
        t.transform.translation.y = self.current_pose.position.y
        t.transform.translation.z = self.current_pose.position.z
        t.transform.rotation = self.current_pose.orientation

        self.tf_pub.sendTransform(t)

    def _on_menu_print(self, feedback):
        """Prints the current translation and quaternion."""
        p = self.current_pose.position
        o = self.current_pose.orientation
        self.get_logger().info("\n--- Current Tweaked Transform ---")
        self.get_logger().info(f"(Expressed in frame {self.parent_frame})")
        self.get_logger().info(f"Translation: [{p.x:.4f}, {p.y:.4f}, {p.z:.4f}]")
        self.get_logger().info(f"Quaternion:  [{o.x:.4f}, {o.y:.4f}, {o.z:.4f}, {o.w:.4f}]")
        self.get_logger().info("---------------------------------\n")

def main():
    parser = argparse.ArgumentParser(description="Interactively tweak a TF frame statically.")
    parser.add_argument("--frame", type=str, required=True, help="The target frame to tweak")
    parser.add_argument("--parent", type=str, required=True, help="The parent frame against which to publish the transform")

    # for passing in an initial transform
    parser.add_argument("--translation", type=float, nargs=3, help="Initial [x, y, z]")
    parser.add_argument("--quaternion", type=float, nargs=4, help="Initial [qx, qy, qz, qw]")

    args, ros_args = parser.parse_known_args(sys.argv)

    rclpy.init(args=ros_args)
    node = TfTweakerNode(
        parent_frame=args.parent,
        target_frame=args.frame,
        init_trans=args.translation,
        init_quat=args.quaternion
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
