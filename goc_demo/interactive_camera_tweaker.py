import math
from typing import Tuple

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose, PoseStamped
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, Marker
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from interactive_markers.menu_handler import MenuHandler

from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster, Buffer, TransformListener
from geometry_msgs.msg import TransformStamped


# ---------- Small math helpers ----------
def quat_from_yaw_pitch_roll(yaw, pitch, roll) -> Tuple[float,float,float,float]:
    cy, sy = math.cos(yaw*0.5), math.sin(yaw*0.5)
    cp, sp = math.cos(pitch*0.5), math.sin(pitch*0.5)
    cr, sr = math.cos(roll*0.5), math.sin(roll*0.5)
    qw = cr*cp*cy + sr*sp*sy
    qx = sr*cp*cy - cr*sp*sy
    qy = cr*sp*cy + sr*cp*sy
    qz = cr*cp*sy - sr*sp*cy
    return (qx,qy,qz,qw)

def pose_to_transform(p: Pose, parent: str, child: str, stamp) -> TransformStamped:
    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = parent
    t.child_frame_id = child
    t.transform.translation.x = p.position.x
    t.transform.translation.y = p.position.y
    t.transform.translation.z = p.position.z
    t.transform.rotation = p.orientation
    return t

# ---------- Node ----------
class InteractiveCameraPoseNode(Node):
    """
    Publishes TF(world -> camera_frame) driven by a 6-DOF interactive marker.
    Right-click menu has 'Publish Static TF' to freeze current pose as static.
    """

    def __init__(self):
        super().__init__("interactive_camera_pose_node")

        # Params
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("camera_frame", "camera_color_optical_frame")
        self.declare_parameter("marker_name", "camera_pose_marker")
        self.declare_parameter("init_xyz", [0.0, 0.0, 0.5])         # meters
        self.declare_parameter("init_rpy_deg", [0.0, 0.0, 0.0])     # roll, pitch, yaw in deg
        self.declare_parameter("marker_scale", 0.25)                # length of marker axes (m)
        self.declare_parameter("publish_rate_hz", 30.0)             # re-publish dynamic TF

        self.world_frame = self.get_parameter("world_frame").get_parameter_value().string_value
        self.camera_frame = self.get_parameter("camera_frame").get_parameter_value().string_value
        self.marker_name = self.get_parameter("marker_name").get_parameter_value().string_value
        self.marker_scale = float(self.get_parameter("marker_scale").value)
        init_xyz = [float(v) for v in self.get_parameter("init_xyz").value]
        init_rpy_deg = [float(v) for v in self.get_parameter("init_rpy_deg").value]
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)

        # TF broadcasters
        self.tf_pub = TransformBroadcaster(self)
        self.static_tf_pub = StaticTransformBroadcaster(self)

        # TF buffer+listener to query initial camera pose
        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Try to seed initial pose from TF
        self.current_pose = Pose()
        if self._try_seed_from_tf():
            self.get_logger().info(f"Seeded marker from TF {self.world_frame}->{self.camera_frame}")
        else:
            # fallback to init params
            init_xyz = [float(v) for v in self.get_parameter("init_xyz").value]
            init_rpy_deg = [float(v) for v in self.get_parameter("init_rpy_deg").value]
            self.current_pose.position.x, self.current_pose.position.y, self.current_pose.position.z = init_xyz
            roll, pitch, yaw = [math.radians(v) for v in init_rpy_deg]
            qx, qy, qz, qw = quat_from_yaw_pitch_roll(yaw, pitch, roll)
            self.current_pose.orientation.x = qx
            self.current_pose.orientation.y = qy
            self.current_pose.orientation.z = qz
            self.current_pose.orientation.w = qw
            self.get_logger().warn("No TF available yet, using parameter defaults.")

        # Interactive marker server + menu
        self.server = InteractiveMarkerServer(self, "interactive_camera_pose_server")
        self.menu_handler = MenuHandler()
        self.menu_handle_static = self.menu_handler.insert("Publish Static TF", callback=self._on_menu_static)
        self.menu_handler.insert("Delete Static TF (revert to dynamic)", callback=self._on_menu_clear_static)

        # Create marker
        self.current_pose = Pose()
        self.current_pose.position.x, self.current_pose.position.y, self.current_pose.position.z = init_xyz
        roll, pitch, yaw = [math.radians(v) for v in init_rpy_deg]
        qx, qy, qz, qw = quat_from_yaw_pitch_roll(yaw, pitch, roll)  # ZYX intrinsic
        self.current_pose.orientation.x = qx
        self.current_pose.orientation.y = qy
        self.current_pose.orientation.z = qz
        self.current_pose.orientation.w = qw

        self._static_sent = False
        self._make_interactive_marker()

        # Timer for dynamic TF (so RViz keeps seeing it even without interaction)
        period = 1.0 / max(1e-3, self.publish_rate_hz)
        self.timer = self.create_timer(period, self._publish_dynamic_tf)

        self.get_logger().info(f"Interactive marker '{self.marker_name}' in frame '{self.world_frame}'.")
        self.get_logger().info(f"Publishing TF {self.world_frame} -> {self.camera_frame} (dynamic until you publish static).")

    def _try_seed_from_tf(self) -> bool:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.world_frame, self.camera_frame, rclpy.time.Time(), rclpy.time.Duration(seconds=1))
        except Exception as e:
            return False
        # copy into self.current_pose
        self.current_pose.position.x = tf.transform.translation.x
        self.current_pose.position.y = tf.transform.translation.y
        self.current_pose.position.z = tf.transform.translation.z
        self.current_pose.orientation = tf.transform.rotation
        return True

    # -------- Interactive marker creation --------
    def _make_interactive_marker(self):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = self.world_frame
        int_marker.name = self.marker_name
        int_marker.description = f"Set pose of {self.camera_frame}"
        int_marker.scale = self.marker_scale
        int_marker.pose = self.current_pose

        # A visual body (camera frustum-like cube)
        control = InteractiveMarkerControl()
        control.always_visible = True
        control.markers.append(self._make_body_marker(self.marker_scale * 0.4))
        int_marker.controls.append(control)

        # 6-DOF controls (move/rotate in world)
        int_marker.controls.extend(self._make_6dof_controls())

        # Insert + menu
        self.server.insert(int_marker, feedback_callback=self._process_feedback)
        self.menu_handler.apply(self.server, int_marker.name)
        self.server.applyChanges()

    def _make_body_marker(self, s: float) -> Marker:
        m = Marker()
        m.type = Marker.CUBE
        m.scale.x = s * 0.6
        m.scale.y = s * 0.4
        m.scale.z = s * 0.3
        m.color.a = 0.7
        m.color.r = 0.2; m.color.g = 0.6; m.color.b = 0.9
        return m

    def _axis_control(self, name: str, axis: str, mode: int) -> InteractiveMarkerControl:
        c = InteractiveMarkerControl()
        c.name = name
        c.interaction_mode = mode
        # orientation fields specify along which axis this control acts
        if axis == 'x':
            c.orientation.w = 1.0; c.orientation.x = 1.0; c.orientation.y = 0.0; c.orientation.z = 0.0
        elif axis == 'y':
            c.orientation.w = 1.0; c.orientation.x = 0.0; c.orientation.y = 1.0; c.orientation.z = 0.0
        else:  # 'z'
            c.orientation.w = 1.0; c.orientation.x = 0.0; c.orientation.y = 0.0; c.orientation.z = 1.0
        return c

    def _make_6dof_controls(self):
        ctrl = []
        # Rotate around XYZ (in world)
        c = self._axis_control("rotate_x", 'x', InteractiveMarkerControl.ROTATE_AXIS); ctrl.append(c)
        c = self._axis_control("move_x",   'x', InteractiveMarkerControl.MOVE_AXIS);   ctrl.append(c)
        c = self._axis_control("rotate_y", 'y', InteractiveMarkerControl.ROTATE_AXIS); ctrl.append(c)
        c = self._axis_control("move_y",   'y', InteractiveMarkerControl.MOVE_AXIS);   ctrl.append(c)
        c = self._axis_control("rotate_z", 'z', InteractiveMarkerControl.ROTATE_AXIS); ctrl.append(c)
        c = self._axis_control("move_z",   'z', InteractiveMarkerControl.MOVE_AXIS);   ctrl.append(c)
        return ctrl

    # -------- Feedback & menu --------
    def _process_feedback(self, feedback):
        # Update internal pose from marker feedback
        self.current_pose = feedback.pose
        self._static_sent = False  # touching it invalidates previous static TF
        # Also publish immediately for snappy UX
        self._publish_dynamic_tf()

    def _on_menu_static(self, feedback):
        # Publish current pose as STATIC TF (one-shot)
        ts = pose_to_transform(self.current_pose, self.world_frame, self.camera_frame, self.get_clock().now().to_msg())
        self.static_tf_pub.sendTransform(ts)
        self._static_sent = True
        self.get_logger().info(f"Published STATIC TF {self.world_frame} -> {self.camera_frame}")

    def _on_menu_clear_static(self, feedback):
        # There is no "delete static" in TF2—just resume dynamic publishing and ignore previous static.
        self._static_sent = False
        self.get_logger().info("Reverting to dynamic TF publishing (existing static TF will still exist in tree).")

    # -------- TF publishing --------
    def _publish_dynamic_tf(self):
        # If static was published we still publish dynamic (helps override apps that prefer latest)
        ts = pose_to_transform(self.current_pose, self.world_frame, self.camera_frame, self.get_clock().now().to_msg())
        self.tf_pub.sendTransform(ts)

def main():
    rclpy.init()
    node = InteractiveCameraPoseNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
