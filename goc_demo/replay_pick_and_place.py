#!/usr/bin/env python3
"""ROS 2 node that replays a pick-and-place episode recorded by ur_data_collection.

Expected episode fields (defaultdict of lists, one entry per timestep):
  q           – (6,)   joint angles
  spam_pose   – (4,)   object pose as [x, y, z, yaw]  (world frame)
  termination – scalar 0/1; playback stops at the first step where this is 1

Publishes:
  /fake/joint_states          – sensor_msgs/JointState
  /tf                         – dynamic: world → spam
  /replay/spam_marker         – visualization_msgs/MarkerArray  (axes triad + sphere)
"""

from __future__ import annotations

import pickle
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Point, TransformStamped, Vector3
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import ColorRGBA
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

_UR5E_JOINT_NAMES = [
    'elbow_joint',
    'shoulder_lift_joint',
    'shoulder_pan_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]

GRIPPER_OPEN_MAX = 20  # gripper_pos values ≤ this are considered open

STAGE_LABELS = {
    0: 'all',
    1: 'pre-grasp',
    2: 'carry (grasp→release)',
    3: 'post-release',
}


def _stage_boundaries(gripper_pos: np.ndarray):
    """Return (first_grasp, last_release) indices, or (None, None) if no grasp found.

    first_grasp  – first step where gripper is closed
    last_release – first step that is open after the last closed block
    """
    is_closed = gripper_pos > GRIPPER_OPEN_MAX
    closed_idx = np.where(is_closed)[0]
    if len(closed_idx) == 0:
        return None, None
    return int(closed_idx[0]), int(closed_idx[-1]) + 1


def _apply_stage(data: list, first_grasp: int, last_release: int, stage: int) -> list:
    if stage == 1:
        return data[:first_grasp]
    elif stage == 2:
        return data[first_grasp:last_release]
    else:  # stage 3
        return data[last_release:]


def _spam_pose_to_matrix(spam_pose: np.ndarray) -> np.ndarray:
    """Convert [x, y, z, yaw] → 4×4 SE(3) matrix (yaw = rotation about world Z)."""
    x, y, z, yaw = float(spam_pose[0]), float(spam_pose[1]), float(spam_pose[2]), float(spam_pose[3])
    c, s = np.cos(yaw), np.sin(yaw)
    mat = np.eye(4)
    mat[0, 0], mat[0, 1] = c, -s
    mat[1, 0], mat[1, 1] = s,  c
    mat[0, 3], mat[1, 3], mat[2, 3] = x, y, z
    return mat


def _mat_to_transform_stamped(mat: np.ndarray, parent: str, child: str, stamp) -> TransformStamped:
    from geometry_msgs.msg import Quaternion
    from scipy.spatial.transform import Rotation

    ts = TransformStamped()
    ts.header.stamp = stamp
    ts.header.frame_id = parent
    ts.child_frame_id = child
    ts.transform.translation.x = float(mat[0, 3])
    ts.transform.translation.y = float(mat[1, 3])
    ts.transform.translation.z = float(mat[2, 3])
    q = Rotation.from_matrix(mat[:3, :3]).as_quat()  # (x, y, z, w)
    ts.transform.rotation.x, ts.transform.rotation.y = float(q[0]), float(q[1])
    ts.transform.rotation.z, ts.transform.rotation.w = float(q[2]), float(q[3])
    return ts


def _make_markers(mat: np.ndarray, stamp, axis_len: float = 0.05, axis_diam: float = 0.005) -> MarkerArray:
    """Axes triad (X=red, Y=green, Z=blue) plus a small sphere at the origin."""
    origin = mat[:3, 3].astype(float)
    axes = mat[:3, :3].T  # rows: X, Y, Z directions
    axis_colors = [
        ColorRGBA(r=0.9, g=0.1, b=0.1, a=1.0),
        ColorRGBA(r=0.1, g=0.9, b=0.1, a=1.0),
        ColorRGBA(r=0.1, g=0.1, b=0.9, a=1.0),
    ]
    markers = []

    for i, (direction, color) in enumerate(zip(axes, axis_colors)):
        tip = origin + axis_len * direction
        m = Marker()
        m.header.frame_id = 'world'
        m.header.stamp = stamp
        m.ns = 'spam_axes'
        m.id = i
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.scale = Vector3(x=axis_diam, y=axis_diam * 2, z=0.0)
        m.color = color
        m.points = [
            Point(x=float(origin[0]), y=float(origin[1]), z=float(origin[2])),
            Point(x=float(tip[0]),    y=float(tip[1]),    z=float(tip[2])),
        ]
        markers.append(m)

    sphere = Marker()
    sphere.header.frame_id = 'world'
    sphere.header.stamp = stamp
    sphere.ns = 'spam_body'
    sphere.id = 0
    sphere.type = Marker.SPHERE
    sphere.action = Marker.ADD
    sphere.pose.position.x = float(origin[0])
    sphere.pose.position.y = float(origin[1])
    sphere.pose.position.z = float(origin[2])
    sphere.pose.orientation.w = 1.0
    sphere.scale = Vector3(x=0.06, y=0.04, z=0.10)   # approximate spam-can dimensions
    sphere.color = ColorRGBA(r=0.8, g=0.5, b=0.1, a=0.7)
    markers.append(sphere)

    return MarkerArray(markers=markers)


class ReplayPickAndPlaceNode(Node):

    def __init__(self):
        super().__init__('replay_pick_and_place')

        self.declare_parameter('episode_path', '')
        self.declare_parameter('rate_hz', 10.0)
        self.declare_parameter('loop', True)
        self.declare_parameter('tf_prefix', 'replay_')
        self.declare_parameter('stage', 0)

        episode_path = self.get_parameter('episode_path').value
        rate_hz      = float(self.get_parameter('rate_hz').value)
        self._loop   = bool(self.get_parameter('loop').value)
        tf_prefix    = self.get_parameter('tf_prefix').value
        stage        = int(self.get_parameter('stage').value)

        if not episode_path:
            self.get_logger().fatal("'episode_path' parameter is required but not set.")
            raise RuntimeError("episode_path not set")

        self.get_logger().info(f"Loading episode: {episode_path}")
        with open(episode_path, 'rb') as f:
            episode = pickle.load(f)

        # Truncate at first termination==1
        termination = np.asarray(episode.get('termination', []), dtype=float)
        term_indices = np.where(termination == 1)[0]
        n_frames = int(term_indices[0]) if len(term_indices) else len(termination)
        if len(term_indices):
            self.get_logger().info(f"Truncating at termination step {n_frames}.")

        q_all       = list(episode.get('q', []))[:n_frames]
        spam_all    = list(episode.get('spam_pose', []))[:n_frames]
        gripper_all = np.asarray(episode.get('gripper_pos', []))[:n_frames]

        if stage != 0:
            if stage not in STAGE_LABELS:
                raise RuntimeError(f"'stage' must be 0–3, got {stage}")
            fg, lr = _stage_boundaries(gripper_all)
            if fg is None:
                self.get_logger().warn(
                    f"No grasp detected; stage {stage} filter has no effect.")
            else:
                q_all    = _apply_stage(q_all,    fg, lr, stage)
                spam_all = _apply_stage(spam_all, fg, lr, stage)
                self.get_logger().info(
                    f"Stage {stage} ({STAGE_LABELS[stage]}): "
                    f"steps {fg if stage>=2 else 0}–"
                    f"{lr if stage<=2 else n_frames} "
                    f"→ {len(q_all)} frames."
                )

        self._q_list = q_all
        self._spam_poses: list[Optional[np.ndarray]] = [
            np.asarray(p, dtype=float) for p in spam_all
        ]

        if not self._spam_poses:
            self.get_logger().warn("No 'spam_pose' data — object TF/markers will not be published.")

        raw_names = episode.get('joint_names') or _UR5E_JOINT_NAMES
        self._joint_names = [tf_prefix + n for n in raw_names]

        self._n_frames = n_frames
        self._frame_idx = 0
        self._done = False

        self._js_pub      = self.create_publisher(JointState,   '/fake/joint_states',   10)
        self._marker_pub  = self.create_publisher(MarkerArray,  '/replay/spam_marker',  10)
        self._tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info(
            f"Replaying {n_frames} frames at {rate_hz:.1f} Hz "
            f"({'looping' if self._loop else 'once'})."
        )
        self.create_timer(1.0 / rate_hz, self._step)

    def _step(self):
        if self._done or self._n_frames == 0:
            return

        stamp = self.get_clock().now().to_msg()
        idx = self._frame_idx

        if idx < len(self._q_list):
            js = JointState()
            js.header.stamp = stamp
            js.name = self._joint_names
            js.position = [float(v) for v in self._q_list[idx]]
            self._js_pub.publish(js)

        if self._spam_poses and idx < len(self._spam_poses):
            mat = _spam_pose_to_matrix(self._spam_poses[idx])
            tf = _mat_to_transform_stamped(mat, 'world', 'spam', stamp)
            self._tf_broadcaster.sendTransform(tf)
            self._marker_pub.publish(_make_markers(mat, stamp))

        self._frame_idx += 1
        if self._frame_idx >= self._n_frames:
            if self._loop:
                self._frame_idx = 0
            else:
                self._done = True
                self.get_logger().info("Replay finished.")


def main(args=None):
    rclpy.init(args=args)
    try:
        node = ReplayPickAndPlaceNode()
        rclpy.spin(node)
    except (RuntimeError, KeyboardInterrupt):
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
