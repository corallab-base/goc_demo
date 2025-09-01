#!/usr/bin/env python3
import os
import sys
from typing import Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from sensor_msgs.msg import PointCloud, ChannelFloat32, Image, CameraInfo
from geometry_msgs.msg import Point32, Pose
from visualization_msgs.msg import Marker, MarkerArray

import cv2

import tf2_ros
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

# NEW import our helper and the service
from goc_demo.checkerboard_pose import detect_checkerboard_pose_T_GC, T_to_transform_stamped
from goc_demo_interfaces.srv import ComputeAndSetCameraPose


def rpy_deg_to_rotm(roll_deg, pitch_deg, yaw_deg):
    r, p, y = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    # ZYX intrinsic (yaw, then pitch, then roll), standard in ROS
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    return Rz @ Ry @ Rx

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

def make_T(R, t_xyz):
    T = np.eye(4); T[:3,:3] = R; T[:3,3] = np.asarray(t_xyz, float)
    return T

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

def _rotz(deg):
    r = np.deg2rad(deg); c, s = np.cos(r), np.sin(r)
    return np.array([[c,-s,0],
                     [s, c,0],
                     [0, 0,1]], dtype=float)

def _rotx(deg):
    r = np.deg2rad(deg); c, s = np.cos(r), np.sin(r)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], float)

def _premul_grid_R(T_GC, Rg):
    """Apply a pure rotation in the GRID frame: T' = S * T, with S = [Rg | 0; 0 0 0 1]."""
    S = np.eye(4); S[:3,:3] = Rg
    return S @ T_GC

def _premul_grid_Rz(T_GC, deg):
    """Return S * T_GC where S is a rotation about +Z of the GRID frame by 'deg' degrees."""
    S = np.eye(4)
    S[:3,:3] = _rotz(deg)
    return S @ T_GC

def _project_grid_step_du_dv(T_GC, K, delta=np.array([1.0,0.0,0.0]), pG=np.array([0.0,0.0,0.0])):
    """
    Compute image delta (du,dv) for a small move in GRID by 'delta'.
    Uses current camera pose in GRID frame T_GC and intrinsics K.
    """
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

    R_GC = T_GC[:3,:3]
    t_GC = T_GC[:3, 3:4]  # 3x1

    # Invert: grid->camera (R_CG, t_CG)
    R_CG = R_GC.T
    t_CG = -R_CG @ t_GC

    p0G = pG.reshape(3,1)
    p1G = (pG + delta).reshape(3,1)

    p0C = R_CG @ p0G + t_CG
    p1C = R_CG @ p1G + t_CG

    X0,Y0,Z0 = p0C.flatten()
    X1,Y1,Z1 = p1C.flatten()

    # pinhole projection (ignore distortion; direction is all we need)
    u0 = fx * (X0 / Z0) + cx; v0 = fy * (Y0 / Z0) + cy
    u1 = fx * (X1 / Z1) + cx; v1 = fy * (Y1 / Z1) + cy
    return (u1 - u0, v1 - v0)

def enforce_grid_x_projects_right(T_GC, K):
    """
    Return an adjusted T_GC' so that:
      1) motion along +X_grid projects predominantly horizontally, and
      2) specifically projects RIGHTWARD (du > 0).
    """
    # Step 0: evaluate current direction
    du, dv = _project_grid_step_du_dv(T_GC, K)

    T_best = T_GC
    best_du, best_dv = du, dv

    # Step 1: if it's more vertical than horizontal, try ±90° swaps and pick the more horizontal
    if abs(du) < abs(dv):
        cand = []
        for ang in (+90, -90):
            T_try = _premul_grid_Rz(T_GC, ang)
            du_try, dv_try = _project_grid_step_du_dv(T_try, K)
            cand.append((T_try, du_try, dv_try))
        # choose the one with larger |du|
        T_best, best_du, best_dv = max(cand, key=lambda tpl: abs(tpl[1]))

    # Step 2: ensure leftward (du < 0). If not, flip 180°
    if best_du <= 0:
        T_best = _premul_grid_Rz(T_best, 180)
        best_du, best_dv = _project_grid_step_du_dv(T_best, K)

    # (Optional) small safety: if still borderline vertical, you can re-run Step 1,
    # but with the square board and the above logic it should be stable.

    return T_best

def enforce_camera_above_grid(T_GC):
    """
    Ensure camera lies at positive +Z in GRID frame: if z_cam < 0, flip grid by 180° about X.
    This reverses the grid normal (+Z) while keeping +X unchanged (so your 'leftward' rule survives).
    """
    z_cam_in_G = T_GC[2, 3]
    if z_cam_in_G < 0:
        T_GC = _premul_grid_R(T_GC, _rotx(180))  # flips Y and Z of the GRID frame
    return T_GC

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
        self.declare_parameter('color_topic', '/camera/color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('depth_scale', 0.001)  # if depth is 16UC1 in millimeters, 0.001 -> meters
        self.declare_parameter('depth_window', 5)     # median window (px) around centroid
        self.declare_parameter('publish_markers', True)
        self.declare_parameter('marker_ns', 'tracked_points')
        self.declare_parameter('marker_scale', 0.03)  # sphere radius (m)
        self.declare_parameter('publish_calib_tf', True)             # publish after service is called
        self.declare_parameter('calibrated_camera_frame', '')        # optional override; default uses self.camera_frame

        color_topic = self.get_parameter('color_topic').get_parameter_value().string_value
        self.publish_calib_tf = bool(self.get_parameter('publish_calib_tf').value)
        self.calibrated_camera_frame = self.get_parameter('calibrated_camera_frame').get_parameter_value().string_value

        # Keep last color image
        self.color_image = None
        self.color_stamp = None
        if color_topic:
            self.create_subscription(Image, color_topic, self._color_cb, 10)
            self.get_logger().info(f"Color image: {color_topic}")

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

        # TF broadcasters for calibration result
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

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

        # Service: compute pose & publish TF
        self.srv = self.create_service(
            ComputeAndSetCameraPose,
            '~/compute_and_set_camera_pose',
            self._srv_compute_and_set_pose
        )
        self.get_logger().info("Service '~/compute_and_set_camera_pose' ready.")

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
            tf = self.tf_buffer.lookup_transform(self.world_frame, self.camera_frame, rclpy.time.Time(), rclpy.time.Duration(seconds=1))
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
        """Centroids already in camera optical frame (meters). TF -> world and republish."""
        if not cloud.points:
            return

        # 1) Read source frame & time from the message
        src_frame = cloud.header.frame_id or self.camera_frame
        stamp = Time.from_msg(cloud.header.stamp) if cloud.header.stamp.sec or cloud.header.stamp.nanosec else rclpy.time.Time()

        # 2) Gather points
        pts_cam = np.array([[p.x, p.y, p.z] for p in cloud.points], dtype=np.float64)

        # IDs (optional)
        ids = list(range(len(cloud.points)))
        for ch in (cloud.channels or []):
            if ch.name == 'id' and len(ch.values) == len(cloud.points):
                ids = [int(v) for v in ch.values]
                break

        # 3) Get the TF at the cloud timestamp with a short timeout
        try:
            if not self.tf_buffer.can_transform(self.world_frame, src_frame, stamp, rclpy.duration.Duration(seconds=0.25)):
                self.get_logger().warn(f"TF not available: {self.world_frame} <- {src_frame} at {stamp.nanoseconds}", throttle_duration_sec=2.0)
                return
            tf = self.tf_buffer.lookup_transform(self.world_frame, src_frame, stamp)
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed ({self.world_frame} <- {src_frame}): {e}", throttle_duration_sec=2.0)
            return

        # 4) Transform
        pts_world = transform_points_cam_to_world(pts_cam, tf)  # p_W = R p_src + t

        # 5) Republish with proper header (propagate stamp)
        out = PointCloud()
        out.header.frame_id = self.world_frame
        out.header.stamp = cloud.header.stamp  # keep timing
        id_channel = ChannelFloat32(); id_channel.name = 'id'
        for pi, pid in zip(pts_world, ids):
            out.points.append(Point32(x=float(pi[0]), y=float(pi[1]), z=float(pi[2])))
            id_channel.values.append(float(pid))
        if id_channel.values:
            out.channels.append(id_channel)

        self.pub_world.publish(out)

        if self.pub_markers:
            self.pub_markers.publish(self._make_marker_array(pts_world, ids))

    # def _points3d_cb(self, cloud: PointCloud):
    #     """Centroids already in camera optical frame (meters). Just TF->world and republish."""
    #     pts_cam = np.array([[p.x, p.y, p.z] for p in cloud.points], dtype=np.float64)
    #     if pts_cam.size == 0:
    #         return

    #     # read ids if present
    #     ids = list(range(len(cloud.points)))
    #     if cloud.channels:
    #         for ch in cloud.channels:
    #             if ch.name == 'id':
    #                 ids = [int(v) for v in ch.values]
    #                 break

    #     try:
    #         tf = self.tf_buffer.lookup_transform(self.world_frame, self.camera_frame, rclpy.time.Time())
    #     except Exception as e:
    #         self.get_logger().warn(f"TF lookup failed ({self.world_frame} <- {self.camera_frame}): {e}")
    #         return

    #     pts_world = transform_points_cam_to_world(pts_cam, tf)

    #     out = PointCloud()
    #     out.header.frame_id = self.world_frame
    #     id_channel = ChannelFloat32(); id_channel.name = 'id'
    #     for pi, pid in zip(pts_world, ids):
    #         out.points.append(Point32(x=float(pi[0]), y=float(pi[1]), z=float(pi[2])))
    #         id_channel.values.append(float(pid))
    #     out.channels.append(id_channel)
    #     self.pub_world.publish(out)

    #     if self.pub_markers:
    #         self.pub_markers.publish(self._make_marker_array(pts_world, ids))

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

    # --- Getting images and Pose Computation Service ----

    def _color_cb(self, msg: Image):
        try:
            self.color_image = imgmsg_to_numpy(msg)  # BGR8 expected
            self.color_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().warn(f"Color convert failed: {e}")

    def _srv_compute_and_set_pose(self, req: ComputeAndSetCameraPose.Request,
                                  res: ComputeAndSetCameraPose.Response):
        # Preconditions
        if self.color_image is None:
            res.success = False
            res.message = "No color image received yet"
            return res
        if self.K is None:
            res.success = False
            res.message = "No CameraInfo received yet"
            return res

        fx, fy, cx, cy = self.K
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]], dtype=np.float64)

        # Distortion: if you want exact, read from CameraInfo.D or image_proc rectified topic.
        # Here we assume rectified or negligible distortion.
        dist = np.zeros(5, dtype=np.float64)

        board_size = (int(req.board_cols), int(req.board_rows))
        square_m = float(req.square_size_m)

        # center of board at world center, but grid frame is located at upper
        # left corner. To move from world to grid, move left (-x) and forward (-y)
        board_w = board_size[0] * square_m
        board_h = board_size[1] * square_m
        x_offset = 0 # board_w / 2.0
        y_offset = 0 # -board_h / 2.0
        
        try:
            # Detect pose: returns T_GC (camera pose in Grid frame).
            # Assumption: Grid frame == World frame (board at world origin, normal = +Z_world).
            T_GC = detect_checkerboard_pose_T_GC(self.color_image, K, dist, board_size, square_m)
            T_GC = enforce_grid_x_projects_right(T_GC, K)
            T_GC = enforce_camera_above_grid(T_GC)

            # world to camera grid is -90 degree about z and 180 degree about y rotation
            T_WC = make_T(rpy_deg_to_rotm(0, 0, 0), [x_offset, y_offset, 0.0]) @ T_GC

            # rotate final frame by 180 degrees about its own z axis
            T_WC = T_WC @ make_T(rpy_deg_to_rotm(0, 0, 180), [0.0, 0.0, 0.0])

            # Publish TF if requested
            if self.publish_calib_tf:
                cam_child = self.calibrated_camera_frame or self.camera_frame
                ts = T_to_transform_stamped(T_WC, self.color_stamp or self.get_clock().now().to_msg(),
                                            self.world_frame, cam_child)
                if req.use_static:
                    self.static_tf_broadcaster.sendTransform(ts)
                    pub_kind = "static"
                else:
                    self.tf_broadcaster.sendTransform(ts)
                    pub_kind = "dynamic (one-shot)"
                self.get_logger().info(f"Published {pub_kind} TF {self.world_frame} -> {cam_child}")

            # Fill response Pose (geometry_msgs/Pose)
            res.success = True
            res.message = "Pose estimated and TF published" if self.publish_calib_tf else "Pose estimated"
            res.camera_pose_world = Pose()
            res.camera_pose_world.position.x = float(T_WC[0,3])
            res.camera_pose_world.position.y = float(T_WC[1,3])
            res.camera_pose_world.position.z = float(T_WC[2,3])

            # matrix -> quaternion
            R = T_WC[:3,:3]
            qw = np.sqrt(max(0.0, 1.0 + R[0,0] + R[1,1] + R[2,2])) / 2.0
            qx = (R[2,1] - R[1,2]) / (4.0*qw) if qw != 0 else 0.0
            qy = (R[0,2] - R[2,0]) / (4.0*qw) if qw != 0 else 0.0
            qz = (R[1,0] - R[0,1]) / (4.0*qw) if qw != 0 else 0.0
            res.camera_pose_world.orientation.x = float(qx)
            res.camera_pose_world.orientation.y = float(qy)
            res.camera_pose_world.orientation.z = float(qz)
            res.camera_pose_world.orientation.w = float(qw)
        except Exception as e:
            res.success = False
            res.message = f"Failed: {e}"

        return res

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
