#!/usr/bin/env python3
import os
import sys
from typing import Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from sensor_msgs.msg import PointCloud, ChannelFloat32, Image, CameraInfo
from geometry_msgs.msg import Point32
from visualization_msgs.msg import Marker, MarkerArray

import cv2

import tf2_ros


def quat_to_rotm(qx, qy, qz, qw):
    """Quaternion (x,y,z,w) -> 3x3 rotation matrix."""
    x, y, z, w = qx, qy, qz, qw
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.array([[1 - 2*(yy+zz),     2*(xy - wz),     2*(xz + wy)],
                  [    2*(xy + wz), 1 - 2*(xx+zz),     2*(yz - wx)],
                  [    2*(xz - wy),     2*(yz + wx), 1 - 2*(xx+yy)]], dtype=np.float64)
    return R


def transform_points_cam_to_world(points_cam_xyz: np.ndarray, tf) -> np.ndarray:
    """
    points_cam_xyz: (N,3) in source frame.
    tf: geometry_msgs/TransformStamped from source->target (camera->world)
    returns (N,3) in target frame.
    """
    t = tf.transform.translation
    q = tf.transform.rotation
    R = quat_to_rotm(q.x, q.y, q.z, q.w)
    tvec = np.array([t.x, t.y, t.z], dtype=np.float64)
    return (points_cam_xyz @ R.T) + tvec  # note: p_world = R * p_cam + t

# ---------- Image helpers (cv_bridge-free) ----------
def imgmsg_to_numpy(msg: Image) -> np.ndarray:
    enc = msg.encoding.lower()
    if enc in ("rgb8", "bgr8"):
        ch, dtype = 3, np.uint8
    elif enc in ("mono8",):
        ch, dtype = 1, np.uint8
    elif enc in ("16uc1", "mono16"):
        ch, dtype = 1, np.uint16
    elif enc in ("32fc1",):
        ch, dtype = 1, np.float32
    else:
        raise NotImplementedError(f"Unsupported encoding: {msg.encoding}")

    row_stride = msg.step // np.dtype(dtype).itemsize
    arr = np.frombuffer(msg.data, dtype=dtype)
    if ch == 1:
        img = arr.reshape((msg.height, row_stride))[:, :msg.width]
    else:
        img = arr.reshape((msg.height, row_stride // ch, ch))[:, :msg.width, :]
    return img

def numpy_to_imgmsg(img: np.ndarray, frame_id: str, encoding="bgr8") -> Image:
    msg = Image()
    msg.header.frame_id = frame_id
    msg.height, msg.width = img.shape[:2]
    msg.encoding = encoding
    msg.is_bigendian = 0
    step_channels = (img.shape[2] if img.ndim == 3 else 1)
    msg.step = msg.width * step_channels * img.dtype.itemsize
    msg.data = memoryview(img.tobytes())
    return msg


class DemoWorldNode(Node):
    """
    Subscribes to tracker centroids and transforms them into a world frame using TF2.

    Supports two input modes:
      1) Pixels + Depth:  subscribe to centroids_px (PointCloud with pixel coords), depth image, and camera info.
         -> deproject to camera optical frame -> TF to world -> publish ~/centroids_world (PointCloud)
      2) 3D in camera:   subscribe to centroids_3d (PointCloud with meters in camera optical frame).
         -> TF to world -> publish ~/centroids_world

    Also publishes an optional MarkerArray for RViz preview of the world points.
    """

    def __init__(self):
        super().__init__('demo_world_node')

        # ---- Parameters ----
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('centroids_px_topic', '/sam2_click_tracker_node/centroids_px')
        self.declare_parameter('centroids_3d_topic', '/sam2_click_tracker_node/centroids_3d')
        self.declare_parameter('depth_topic', '/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('depth_scale', 0.001)  # if depth is 16UC1 in millimeters, 0.001 -> meters
        self.declare_parameter('depth_window', 5)     # median window (px) around centroid
        self.declare_parameter('publish_markers', True)
        self.declare_parameter('marker_ns', 'tracked_points')
        self.declare_parameter('marker_scale', 0.03)  # sphere radius (m)

        self.world_frame = self.get_parameter('world_frame').get_parameter_value().string_value
        self.camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value
        self.depth_scale = float(self.get_parameter('depth_scale').value)
        self.depth_window = int(self.get_parameter('depth_window').value)
        self.publish_markers = bool(self.get_parameter('publish_markers').value)
        self.marker_ns = self.get_parameter('marker_ns').get_parameter_value().string_value
        self.marker_scale = float(self.get_parameter('marker_scale').value)

        centroids_px_topic = self.get_parameter('centroids_px_topic').get_parameter_value().string_value
        centroids_3d_topic = self.get_parameter('centroids_3d_topic').get_parameter_value().string_value
        depth_topic        = self.get_parameter('depth_topic').get_parameter_value().string_value
        cam_info_topic     = self.get_parameter('camera_info_topic').get_parameter_value().string_value

        # ---- TF2 ----
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---- Publishers ----
        self.pub_world = self.create_publisher(PointCloud, '~/centroids_world', 10)
        self.pub_markers = self.create_publisher(MarkerArray, '~/centroids_markers', 10) if self.publish_markers else None

        # ---- Subscriptions ----
        self.sub_px = None
        self.sub_3d = None
        if centroids_px_topic:
            self.sub_px = self.create_subscription(PointCloud, centroids_px_topic, self._px_cb, 10)
        if centroids_3d_topic:
            self.sub_3d = self.create_subscription(PointCloud, centroids_3d_topic, self._points3d_cb, 10)

        # Depth + camera info (only needed for pixel mode)
        self.depth_image: Optional[np.ndarray] = None
        self.last_depth_stamp: Optional[rclpy.time.Time] = None
        self.K = None  # fx, fy, cx, cy
        self.create_subscription(Image, depth_topic, self._depth_cb, 10)
        self.create_subscription(CameraInfo, cam_info_topic, self._caminfo_cb, 10)

        self.get_logger().info("demo_world_node started.")
        self.get_logger().info(f"Expecting camera frame: '{self.camera_frame}', world frame: '{self.world_frame}'")
        if self.sub_px:
            self.get_logger().info(f"Pixel centroids: {centroids_px_topic} + depth: {depth_topic}, camera_info: {cam_info_topic}")
        if self.sub_3d:
            self.get_logger().info(f"3D centroids: {centroids_3d_topic}")

    # ---- Callbacks ----

    def _caminfo_cb(self, msg: CameraInfo):
        # Get intrinsics (assume rectified pinhole)
        if msg.k is not None and len(msg.k) == 9:
            fx = msg.k[0]; fy = msg.k[4]; cx = msg.k[2]; cy = msg.k[5]
            self.K = (fx, fy, cx, cy)

    def _depth_cb(self, msg: Image):
        try:
            # Try to get a 16-bit or 32-bit depth (no color conversion)
            img = imgmsg_to_numpy(msg)
            self.depth_image = img
            self.last_depth_stamp = Time.from_msg(msg.header.stamp)
        except Exception as e:
            self.get_logger().warn(f"Depth convert failed: {e}")

    def _px_cb(self, cloud: PointCloud):
        """Centroids in pixels (x=u, y=v). Lift to 3D with depth+intrinsics, TF to world, republish."""
        if self.depth_image is None or self.K is None:
            self.get_logger().warning(
                "Waiting for depth image and camera info...",
                throttle_duration_sec=2.0)
            return

        # Build id channel lookup (optional)
        ids = None
        if cloud.channels:
            for ch in cloud.channels:
                if ch.name == 'id':
                    ids = ch.values
                    break

        # Depth stamp might not match centroids stamp (since tracker is separate). We'll just use the latest depth.
        depth = self.depth_image
        fx, fy, cx, cy = self.K

        # Deproject pixels -> camera optical frame (meters)
        pts_cam = []
        out_ids = []
        H, W = depth.shape[:2]
        k = max(1, self.depth_window)
        half = k // 2

        for i, p in enumerate(cloud.points):
            u = int(round(p.x)); v = int(round(p.y))
            if u < 0 or v < 0 or u >= W or v >= H:
                continue

            # median depth in kxk window
            u0 = max(0, u - half); v0 = max(0, v - half)
            u1 = min(W, u + half + 1); v1 = min(H, v + half + 1)
            patch = depth[v0:v1, u0:u1]

            z_m = None
            if patch.size > 0:
                if patch.dtype == np.uint16:
                    # assume millimeters by default
                    z_m = float(np.median(patch[patch > 0])) * self.depth_scale
                else:
                    # float depth in meters (e.g., 32FC1)
                    zvals = patch[np.isfinite(patch)]
                    if zvals.size > 0:
                        z_m = float(np.median(zvals))

            if z_m is None or z_m <= 0.0:
                continue

            X = (u - cx) / fx * z_m
            Y = (v - cy) / fy * z_m
            Z = z_m
            pts_cam.append([X, Y, Z])
            out_ids.append(ids[i] if ids and i < len(ids) else float(i))

        if not pts_cam:
            return

        pts_cam = np.asarray(pts_cam, dtype=np.float64)

        # TF: camera -> world
        try:
            tf = self.tf_buffer.lookup_transform(self.world_frame, self.camera_frame, rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed ({self.world_frame} <- {self.camera_frame}): {e}")
            return

        pts_world = transform_points_cam_to_world(pts_cam, tf)

        # Publish
        out = PointCloud()
        out.header.frame_id = self.world_frame
        # stamp left empty (this is a fused product); you can copy depth stamp if you prefer
        id_channel = ChannelFloat32()
        id_channel.name = 'id'
        for pi, pid in zip(pts_world, out_ids):
            out.points.append(Point32(x=float(pi[0]), y=float(pi[1]), z=float(pi[2])))
            id_channel.values.append(float(pid))
        out.channels.append(id_channel)
        self.pub_world.publish(out)

        if self.pub_markers:
            self.pub_markers.publish(self._make_marker_array(pts_world, out_ids))

    def _points3d_cb(self, cloud: PointCloud):
        """Centroids already in camera optical frame (meters). Just TF->world and republish."""
        pts_cam = np.array([[p.x, p.y, p.z] for p in cloud.points], dtype=np.float64)
        if pts_cam.size == 0:
            return

        # read ids if present
        ids = list(range(len(cloud.points)))
        if cloud.channels:
            for ch in cloud.channels:
                if ch.name == 'id':
                    ids = [int(v) for v in ch.values]
                    break

        try:
            tf = self.tf_buffer.lookup_transform(self.world_frame, self.camera_frame, rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed ({self.world_frame} <- {self.camera_frame}): {e}")
            return

        pts_world = transform_points_cam_to_world(pts_cam, tf)

        out = PointCloud()
        out.header.frame_id = self.world_frame
        id_channel = ChannelFloat32(); id_channel.name = 'id'
        for pi, pid in zip(pts_world, ids):
            out.points.append(Point32(x=float(pi[0]), y=float(pi[1]), z=float(pi[2])))
            id_channel.values.append(float(pid))
        out.channels.append(id_channel)
        self.pub_world.publish(out)

        if self.pub_markers:
            self.pub_markers.publish(self._make_marker_array(pts_world, ids))

    # ---- Markers ----

    def _make_marker_array(self, pts_world: np.ndarray, ids):
        ma = MarkerArray()
        now = self.get_clock().now().to_msg()
        for i, (p, pid) in enumerate(zip(pts_world, ids)):
            m = Marker()
            m.header.frame_id = self.world_frame
            m.header.stamp = now
            m.ns = self.marker_ns
            m.id = int(pid)  # stable id per object
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(p[0])
            m.pose.position.y = float(p[1])
            m.pose.position.z = float(p[2])
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = self.marker_scale
            m.color.r = 1.0; m.color.g = 0.2; m.color.b = 0.3; m.color.a = 0.9
            m.lifetime.sec = 0  # persistent
            ma.markers.append(m)
        return ma


def main():
    rclpy.init()
    node = DemoWorldNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
