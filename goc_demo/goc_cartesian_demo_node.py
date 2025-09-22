#!/usr/bin/env python3
from __future__ import annotations

import os
import argparse
import numpy as np
from typing import List, Optional, Tuple, Sequence, Union
from collections import namedtuple

import pickle
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from rclpy.action import ActionClient
from sensor_msgs.msg import JointState, PointCloud
from geometry_msgs.msg import PoseStamped, TwistStamped, Pose, Twist
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from nav_msgs.msg import Path
from control_msgs.action import FollowJointTrajectory
from builtin_interfaces.msg import Duration as RosDuration

from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from tf2_geometry_msgs import do_transform_pose  # applies TransformStamped to Pose/PoseStamped

from pydrake.math import RollPitchYaw
from pydrake.common.eigen_geometry import Quaternion

from goc_mpc.splines import Block
from goc_mpc.goc_mpc import GraphOfConstraints, GraphOfConstraintsMPC
from goc_mpc.simple_drake_env import SimpleDrakeGym

from goc_demo import robotiq
from goc_demo.plans import *


WORLD_FRAME = "world"

Task = namedtuple('Task', ["builder"])


def translational_curvature_xyz(X, eps=1e-9):
    """
    X: (N, >=3) array of poses; xyz are in columns 0:3.
    Returns: (N,) curvature array (1/m). Endpoints use one-sided diffs.
    """
    P = np.asarray(X)[:, :3]                       # (N,3)
    N = P.shape[0]
    if N < 3:
        return np.zeros(N)

    # Arc length parameter s
    dP = np.linalg.norm(np.diff(P, axis=0), axis=1)         # (N-1,)
    s = np.zeros(N)
    s[1:] = np.cumsum(dP)

    # First derivative dr/ds
    r_s = np.zeros_like(P)
    # central differences for interior
    ds_c = (s[2:] - s[:-2])[:, None]                        # (N-2,1)
    r_s[1:-1] = (P[2:] - P[:-2]) / np.maximum(ds_c, eps)
    # one-sided at ends
    r_s[0]  = (P[1]  - P[0])  / max(s[1]-s[0], eps)
    r_s[-1] = (P[-1] - P[-2]) / max(s[-1]-s[-2], eps)

    # Second derivative d2r/ds2 (curvature vector)
    r_ss = np.zeros_like(P)
    # interior: nonuniform spacing formula via flux form
    ds_fwd = (s[2:] - s[1:-1])[:, None]                     # (N-2,1)
    ds_bwd = (s[1:-1] - s[:-2])[:, None]
    r_ss[1:-1] = 2.0 * ( (P[2:] - P[1:-1]) / np.maximum(ds_fwd, eps)
                         - (P[1:-1] - P[:-2]) / np.maximum(ds_bwd, eps) ) \
                         / np.maximum(ds_fwd + ds_bwd, eps)
    # ends: simple one-sided second diffs
    r_ss[0]  = (P[2]  - 2*P[1]  + P[0])  / max((s[2]-s[0])*(s[1]-s[0]) + eps, eps)
    r_ss[-1] = (P[-1] - 2*P[-2] + P[-3]) / max((s[-1]-s[-3])*(s[-1]-s[-2]) + eps, eps)

    # Curvature κ = ||r' × r''|| / ||r'||^3  (with derivatives w.r.t. s)
    cross = np.cross(r_s, r_ss)                             # (N,3)
    num = np.linalg.norm(cross, axis=1)
    den = np.maximum(np.linalg.norm(r_s, axis=1)**3, eps)
    kappa = num / den
    return kappa


class GocMpcCartesianNode(Node):
    """
    Subscribes to TCP Poses, calls goc_mpc.step(t, x, x_dot), and streams tiny
    FollowCartesianTrajectory goals to the cartesian_motion_controller.
    """

    def __init__(self, task_name: str):
        super().__init__("goc_mpc_cartesian_node")
        
        # --- Parameters (your snippet + a couple extra) ---
        self.declare_parameter("left_pose_topic", "/left_cartesian_motion_controller/current_pose")
        self.declare_parameter("left_twist_topic", "/left_cartesian_motion_controller/current_twist")
        self.declare_parameter("right_pose_topic", "/right_cartesian_motion_controller/current_pose")
        self.declare_parameter("right_twist_topic", "/right_cartesian_motion_controller/current_twist")
        self.declare_parameter("keypoints_topic", "/demo_world_node/centroids_world")
        self.declare_parameter("rate_hz", 30.0)
        self.declare_parameter("dry_run", False)
        self.declare_parameter("mpc_output_mode", "position")  # or "velocity"
        self.declare_parameter("preview_points", 1)            # 1 is fine for most
        self.declare_parameter("dt_scale", 1.0)                # stretch/shrink dt used in goal points
        self.declare_parameter("goal_time_tolerance_sec", 0.05)
        self.declare_parameter("stop_with_zero_velocity", True)

        # Read params
        self._left_pose_topic: str = self.get_parameter("left_pose_topic").value
        self._left_twist_topic: str = self.get_parameter("left_twist_topic").value
        self._right_pose_topic: str = self.get_parameter("right_pose_topic").value
        self._right_twist_topic: str = self.get_parameter("right_twist_topic").value
        self._keypoints_topic: str = self.get_parameter("keypoints_topic").value

        self._rate_hz: float = float(self.get_parameter("rate_hz").value)
        self._preview_points: int = int(self.get_parameter("preview_points").value)
        self._dt_scale: float = float(self.get_parameter("dt_scale").value)
        self._goal_time_tol: float = float(self.get_parameter("goal_time_tolerance_sec").value)
        self._stop_with_zero_velocity: bool = bool(self.get_parameter("stop_with_zero_velocity").value)
        self._dry_run = bool(self.get_parameter("dry_run").value)

        if self._rate_hz <= 0.0:
            self.get_logger().warn("rate_hz must be > 0; defaulting to 100.0")
            self._rate_hz = 100.0

        self._period_sec = 1.0 / self._rate_hz

        # --- TF stuff ---
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        # --- Sub/Pub QoS ---
        pose_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        keypoints_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # --- Visualization Publications ---
        self._left_short_path_publisher = self.create_publisher(Path, "/left_short_path", 10)
        self._right_short_path_publisher = self.create_publisher(Path, "/right_short_path", 10)
        self._left_long_path_publisher = self.create_publisher(Path, "/left_long_path", 10)
        self._right_long_path_publisher = self.create_publisher(Path, "/right_long_path", 10)

        # --- Subscriptions ---
        self._latest_left_pose: Optional[PoseStamped] = None
        self.create_subscription(PoseStamped, self._left_pose_topic, self._on_left_pose, pose_qos)
        self._latest_left_twist: Optional[TwistStamped] = None
        self.create_subscription(TwistStamped, self._left_twist_topic, self._on_left_twist, pose_qos)
        self._latest_right_pose: Optional[PoseStamped] = None
        self.create_subscription(PoseStamped, self._right_pose_topic, self._on_right_pose, pose_qos)
        self._latest_right_twist: Optional[TwistStamped] = None
        self.create_subscription(TwistStamped, self._right_twist_topic, self._on_right_twist, pose_qos)

        self._latest_keypoints: np.ndarray = np.zeros((0, 3))
        self.create_subscription(PointCloud, self._keypoints_topic, self._on_keypoints, keypoints_qos)

        # Publisher to send the target pose to the robot
        if not self._dry_run:
            left_target_topic_name = "/left_target_frame"
            self.left_target_pose_publisher = self.create_publisher(
                PoseStamped, left_target_topic_name, 10
            )
            right_target_topic_name = "/right_target_frame"
            self.right_target_pose_publisher = self.create_publisher(
                PoseStamped, right_target_topic_name, 10
            )

        # instatiate real grippers (not the cleanest, but has to be done)
        left_ip_address = "10.164.8.235"
        self.left_real_gripper = robotiq.RobotiqGripper(disabled=False)
        self.left_real_gripper.connect(left_ip_address, 63352)
        self.left_real_gripper.activate(auto_calibrate=False)
        self.left_real_gripper.open()

        right_ip_address = "10.164.8.222"
        self.right_real_gripper = robotiq.RobotiqGripper(disabled=False)
        self.right_real_gripper.connect(right_ip_address, 63352)
        self.right_real_gripper.activate(auto_calibrate=False)
        self.right_real_gripper.open()

        self.left_robot_paused = False
        self.right_robot_paused = False
        self._left_pre_grasp_timer = None
        self._right_pre_grasp_timer = None
        self._left_resume_timer = None
        self._right_resume_timer = None

        # Pending gripper cmds (latched until pre-delay expires)
        self._left_pending_gripper_cmd = None
        self._right_pending_gripper_cmd = None

        # Tunables
        self._grasp_settle_sec = 1.00          # wait before actuating gripper
        self._grasp_pause_after_cmd_sec = 1.00 # time to remain paused after actuation

        # --- Controller ---

        tasks = {
            "move_in_circles": Task(builder=move_in_circles_builder),
            "track_above": Task(builder=track_above_builder),
            # "stack_blocks": Task(builder=stack_blocks_builder),
            "pick_and_pour": Task(builder=pick_and_pour_builder),
            # "folding": Task(builder=stack_blocks_builder),
        }

        self.task = tasks[task_name]
        self.n_agents = 2
        self.n_keypoints = 0
        self.goc_mpc = self._setup_goc_mpc(self.task)
        self._obs = None

        # metrics
        self.waypoint_solve_times = []
        self.timing_solve_times = []
        self.short_path_solve_times = []

        # --- Timing ---
        self.fake_time = 0.0
        self._start_time = self.get_clock().now()
        self.end_elapsed_time = None
        self._timer = self.create_timer(self._period_sec, self._on_timer)

        # Track last goal handle (optional)
        self._last_goal_handle = None

        self.get_logger().info(
            f"Streaming pose goals at {self._rate_hz:.1f} Hz"
        )

    def _setup_goc_mpc(self, task):
        env, graph, goc_mpc = task.builder()

        self._env = env
        self.n_keypoints = graph.num_objects

        self.get_logger().info(f"n_keypoints: {self.n_keypoints}")

        return goc_mpc

    # --- Callbacks ---
    def _on_left_pose(self, msg: PoseStamped):
        # msg is in left_base_link; convert to world
        pw = self._to_world(msg)
        if pw is not None:
            self._latest_left_pose = pw

    def _on_left_twist(self, msg: TwistStamped):
        # msg is in left_base_link; convert to world
        tw = self._twist_to_world(msg)
        if tw is not None:
            self._latest_left_twist = tw

    def _on_right_pose(self, msg: PoseStamped):
        # msg is in right_base_link; convert to world
        pw = self._to_world(msg)
        if pw is not None:
            self._latest_right_pose = pw

    def _on_right_twist(self, msg: TwistStamped):
        # msg is in right_base_link; convert to world
        tw = self._twist_to_world(msg)
        if tw is not None:
            self._latest_right_twist = tw

    def _on_keypoints(self, msg: PointCloud):
        self._latest_keypoints = np.array([(p.x, p.y, p.z) for p in msg.points])

    def _extract_state(self,
                       left_pose: Pose,
                       left_twist: Twist,
                       right_pose: Pose,
                       right_twist: Twist,
                       kps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        def pose_to_arr(pose: Pose):
            return np.array([pose.position.x,
                             pose.position.y,
                             pose.position.z,
                             pose.orientation.w,
                             pose.orientation.x,
                             pose.orientation.y,
                             pose.orientation.z])

        def twist_to_arr(twist: Twist):
            return np.array([twist.linear.x,
                             twist.linear.y,
                             twist.linear.z,
                             twist.angular.x,
                             twist.angular.y,
                             twist.angular.z])

        # Reorder according to self._joints
        left_x = pose_to_arr(left_pose)
        left_x_dot = twist_to_arr(left_twist)

        right_x = pose_to_arr(right_pose)
        right_x_dot = twist_to_arr(right_twist)

        if kps.shape[0] != self.n_keypoints:
            raise ValueError("Not enough keypoints yet")

        kp_x = kps[:self.n_keypoints].flatten()
        kp_x_dot = np.zeros((self.n_keypoints, 3)).flatten()
        
        # x, x_dot
        x = np.concatenate((left_x, right_x, kp_x))
        x_dot = np.concatenate((left_x_dot, right_x_dot, kp_x_dot))
        return x, x_dot

    def _on_timer(self):
        if self._latest_left_pose is None:
            return
        if self._latest_right_pose is None:
            return
        if self._latest_left_twist is None:
            return
        if self._latest_right_twist is None:
            return
        if self._latest_keypoints is None:
            return

        now = self.get_clock().now()
        t = (now - self._start_time).nanoseconds * 1e-9

        try:
            if self._dry_run:
                if self._obs is None:
                    self._obs, _ = self._env.reset()
                    x, x_dot = self._extract_state(self._latest_left_pose,
                                                   self._latest_left_twist,
                                                   self._latest_right_pose,
                                                   self._latest_right_twist,
                                                   self._latest_keypoints)
                    self._env._set_controlled_q(x)
                    self._env._set_controlled_qdot(x_dot)
                    self._env.render()
                else:
                    x, x_dot = self._obs
            else:
                x, x_dot = self._extract_state(self._latest_left_pose,
                                               self._latest_left_twist,
                                               self._latest_right_pose,
                                               self._latest_right_twist,
                                               self._latest_keypoints)

        except Exception as e:
            self.get_logger().warn(f"Bad State: {e}")
            return

        # MPC step
        try:
            xi_h, _, _ = self.goc_mpc.step(self.fake_time, x, x_dot)
            
            self.waypoint_solve_times.append(self.goc_mpc.waypoint_mpc.get_last_solve_time())
            self.timing_solve_times.append(self.goc_mpc.timing_mpc.get_last_solve_time())
            self.short_path_solve_times.append(self.goc_mpc.short_path_mpc.get_last_solve_time())

            # with open("./goc_mpc_state.pkl", "wb") as f:
            #     self.goc_mpc.dump(f, x, x_dot)
            # breakpoint()
        except Exception as e:
            self.get_logger().error(f"goc_mpc.step failed: {e}")
            return

        h, d = xi_h.shape
        xi_h = xi_h.reshape(h, self.n_agents, d // self.n_agents).transpose(1, 0, 2)

        # publish short path for visualization
        self._publish_paths(xi_h)
        self._publish_long_paths(*self.goc_mpc.timing_mpc.view_wps_list())

        nodes_and_taus = list(zip(self.goc_mpc.timing_mpc.get_next_nodes(), self.goc_mpc.timing_mpc.get_next_taus()))
        # self.get_logger().info(f"next waypoints in: {nodes_and_taus}")

        if len(nodes_and_taus) == 0 and self.end_elapsed_time is None:
            self.end_elapsed_time = t

        target_min = 3
        target_max = 5
        gamma = 16.0

        left_xi_h = xi_h[0]
        left_curvature = translational_curvature_xyz(left_xi_h)[1]
        left_radius = 1 / (left_curvature + 0.001)
        left_target = int(np.clip(gamma * left_radius, target_min, target_max).item())
        left_target_pose = left_xi_h[left_target]

        left_target_pose_stamped = PoseStamped()
        left_target_pose_stamped.header.frame_id = "world"
        left_target_pose_stamped.header.stamp = self.get_clock().now().to_msg()
        left_target_pose_stamped.pose.position.x = left_target_pose[0]
        left_target_pose_stamped.pose.position.y = left_target_pose[1]
        left_target_pose_stamped.pose.position.z = left_target_pose[2]
        left_target_pose_stamped.pose.orientation.w = left_target_pose[3]
        left_target_pose_stamped.pose.orientation.x = left_target_pose[4]
        left_target_pose_stamped.pose.orientation.y = left_target_pose[5]
        left_target_pose_stamped.pose.orientation.z = left_target_pose[6]

        right_xi_h = xi_h[1]
        right_curvature = translational_curvature_xyz(right_xi_h)[1]
        right_radius = 1 / (right_curvature + 0.001)
        right_target = int(np.clip(gamma * right_radius, target_min, target_max).item())
        right_target_pose = right_xi_h[right_target]

        right_target_pose_stamped = PoseStamped()
        right_target_pose_stamped.header.frame_id = "world"
        right_target_pose_stamped.header.stamp = self.get_clock().now().to_msg()
        right_target_pose_stamped.pose.position.x = right_target_pose[0]
        right_target_pose_stamped.pose.position.y = right_target_pose[1]
        right_target_pose_stamped.pose.position.z = right_target_pose[2]
        right_target_pose_stamped.pose.orientation.w = right_target_pose[3]
        right_target_pose_stamped.pose.orientation.x = right_target_pose[4]
        right_target_pose_stamped.pose.orientation.y = right_target_pose[5]
        right_target_pose_stamped.pose.orientation.z = right_target_pose[6]

        self.fake_time += 4 * self.goc_mpc.short_path_time_per_step

        qpos = np.concatenate((left_target_pose, right_target_pose))
        if self._dry_run:
            self._obs, _, _, _, _ = self._env.step(qpos, grasp_cmds=self.goc_mpc.last_grasp_commands)
        else:
            self._obs, _, _, _, _ = self._env.step(qpos, grasp_cmds=self.goc_mpc.last_grasp_commands)

            if len(self.goc_mpc.last_grasp_commands) > 0:
                self.get_logger().info(f"Grasp Commands! {self.goc_mpc.last_grasp_commands}")
                for cmd, robot, point in self.goc_mpc.last_grasp_commands:
                    if robot == "free_body_0":
                        side = "left"
                    elif robot == "free_body_1":
                        side = "right"
                    else:
                        continue
                    self.get_logger().info(f"Paused {side}!")
                    self._pause_robot_delayed(
                        side=side,
                        pre_delay=self._grasp_settle_sec,
                        post_delay=self._grasp_pause_after_cmd_sec,
                        gripper_cmd=cmd
                    )

            if len(self.goc_mpc.last_cycle_backtracked_phases) > 0:
                for agent_idx, new_phase in self.goc_mpc.last_cycle_backtracked_phases.items():
                    if agent_idx == 0:
                        side = "left"
                    elif agent_idx == 1:
                        side = "right"
                    else:
                        continue
                    self.get_logger().info(f"Paused {side}!")
                    self._pause_robot_delayed(
                        side=side,
                        pre_delay=self._grasp_settle_sec,
                        post_delay=self._grasp_pause_after_cmd_sec,
                        gripper_cmd="release"
                    )

            if not self.left_robot_paused:
                self.left_target_pose_publisher.publish(left_target_pose_stamped)

            if not self.right_robot_paused:
                self.right_target_pose_publisher.publish(right_target_pose_stamped)

    # --- Helpers ---

    def _publish_paths(self, xi_h):
        left_xi_h = xi_h[0]
        left_path_msg = Path()
        left_path_msg.header.frame_id = "world"   # or "map", depending on your TF setup
        left_path_msg.header.stamp = self.get_clock().now().to_msg()

        for row in left_xi_h:
            pose = PoseStamped()
            pose.header = left_path_msg.header
            # take the first 7 elements of the row (first pose)
            x, y, z, qw, qx, qy, qz = row[:7]
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = float(z)
            pose.pose.orientation.w = float(qw)
            pose.pose.orientation.x = float(qx)
            pose.pose.orientation.y = float(qy)
            pose.pose.orientation.z = float(qz)
            left_path_msg.poses.append(pose)

        self._left_short_path_publisher.publish(left_path_msg)

        right_xi_h = xi_h[1]
        right_path_msg = Path()
        right_path_msg.header.frame_id = "world"   # or "map", depending on your TF setup
        right_path_msg.header.stamp = self.get_clock().now().to_msg()

        for row in right_xi_h:
            pose = PoseStamped()
            pose.header = right_path_msg.header
            # take the first 7 elements of the row (first pose)
            x, y, z, qw, qx, qy, qz = row[:7]
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = float(z)
            pose.pose.orientation.w = float(qw)
            pose.pose.orientation.x = float(qx)
            pose.pose.orientation.y = float(qy)
            pose.pose.orientation.z = float(qz)
            right_path_msg.poses.append(pose)

        self._right_short_path_publisher.publish(right_path_msg)

    def _publish_long_paths(self, left_xi_l, right_xi_l):
        left_path_msg = Path()
        left_path_msg.header.frame_id = "world"   # or "map", depending on your TF setup
        left_path_msg.header.stamp = self.get_clock().now().to_msg()

        for row in left_xi_l:
            pose = PoseStamped()
            pose.header = left_path_msg.header
            # take the first 7 elements of the row (first pose)
            x, y, z, qw, qx, qy, qz = row[:7]
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = float(z)
            pose.pose.orientation.w = float(qw)
            pose.pose.orientation.x = float(qx)
            pose.pose.orientation.y = float(qy)
            pose.pose.orientation.z = float(qz)
            left_path_msg.poses.append(pose)

        self._left_long_path_publisher.publish(left_path_msg)

        right_path_msg = Path()
        right_path_msg.header.frame_id = "world"   # or "map", depending on your TF setup
        right_path_msg.header.stamp = self.get_clock().now().to_msg()

        for row in right_xi_l:
            pose = PoseStamped()
            pose.header = right_path_msg.header
            # take the first 7 elements of the row (first pose)
            x, y, z, qw, qx, qy, qz = row[:7]
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = float(z)
            pose.pose.orientation.w = float(qw)
            pose.pose.orientation.x = float(qx)
            pose.pose.orientation.y = float(qy)
            pose.pose.orientation.z = float(qz)
            right_path_msg.poses.append(pose)

        self._right_long_path_publisher.publish(right_path_msg)


    def _do_gripper_cmd(self, side: str, cmd: str):
        try:
            gr = self.left_real_gripper if side == 'left' else self.right_real_gripper
            if cmd == 'grab':
                gr.close()
            elif cmd == 'release':
                gr.open()
            else:
                self.get_logger().warn(f"Unknown gripper cmd: {cmd}")
        except Exception as e:
            self.get_logger().error(f"Gripper {side} command '{cmd}' failed: {e}")

    def _resume_robot_left(self):
        self.left_robot_paused = False
        if self._left_resume_timer is not None:
            self._left_resume_timer.cancel()
            self._left_resume_timer = None
        self.get_logger().info("Left robot resumed after grasp pause.")

    def _resume_robot_right(self):
        self.right_robot_paused = False
        if self._right_resume_timer is not None:
            self._right_resume_timer.cancel()
            self._right_resume_timer = None
        self.get_logger().info("Right robot resumed after grasp pause.")

    def _on_left_pre_grasp(self):
        """Fires after settle delay: actuate gripper then start resume timer."""
        if self._left_pre_grasp_timer is not None:
            self._left_pre_grasp_timer.cancel()
            self._left_pre_grasp_timer = None
        cmd = self._left_pending_gripper_cmd
        self._left_pending_gripper_cmd = None
        if cmd is not None:
            self._do_gripper_cmd('left', cmd)
        # chain the resume one-shot
        if self._left_resume_timer is not None:
            self._left_resume_timer.cancel()
            self._left_resume_timer = None
        self._left_resume_timer = self.create_timer(self._grasp_pause_after_cmd_sec,
                                                    self._resume_robot_left)

    def _on_right_pre_grasp(self):
        if self._right_pre_grasp_timer is not None:
            self._right_pre_grasp_timer.cancel()
            self._right_pre_grasp_timer = None
        cmd = self._right_pending_gripper_cmd
        self._right_pending_gripper_cmd = None
        if cmd is not None:
            self._do_gripper_cmd('right', cmd)
        if self._right_resume_timer is not None:
            self._right_resume_timer.cancel()
            self._right_resume_timer = None
        self._right_resume_timer = self.create_timer(self._grasp_pause_after_cmd_sec,
                                                     self._resume_robot_right)

    def _pause_robot_delayed(self, side: str, pre_delay: float, post_delay: float, gripper_cmd: str):
        """
        Immediately pause 'side', wait pre_delay, then execute gripper_cmd, then
        wait post_delay and resume. If re-triggered, refresh the sequence.
        """
        if side == 'left':
            self.left_robot_paused = True
            self._left_pending_gripper_cmd = gripper_cmd

            # refresh pre-grasp one-shot
            if self._left_pre_grasp_timer is not None:
                self._left_pre_grasp_timer.cancel()
                self._left_pre_grasp_timer = None
            self._left_pre_grasp_timer = self.create_timer(pre_delay, self._on_left_pre_grasp)

            # cancel any existing resume timer; it will be reset after actuation
            if self._left_resume_timer is not None:
                self._left_resume_timer.cancel()
                self._left_resume_timer = None

        elif side == 'right':
            self.right_robot_paused = True
            self._right_pending_gripper_cmd = gripper_cmd

            if self._right_pre_grasp_timer is not None:
                self._right_pre_grasp_timer.cancel()
                self._right_pre_grasp_timer = None
            self._right_pre_grasp_timer = self.create_timer(pre_delay, self._on_right_pre_grasp)

            if self._right_resume_timer is not None:
                self._right_resume_timer.cancel()
                self._right_resume_timer = None
        else:
            self.get_logger().warn(f"_pause_robot_delayed: unknown side '{side}'")

    def _to_world(self, pose_msg: PoseStamped, timeout_sec: float = 0.05) -> Optional[Pose]:
        """Transform a PoseStamped from its header.frame_id to WORLD_FRAME."""
        if pose_msg is None:
            return None
        src_frame = pose_msg.header.frame_id
        if not src_frame:
            self.get_logger().warn("Incoming PoseStamped has empty header.frame_id")
            return None
        if src_frame == WORLD_FRAME:
            return pose_msg.pose  # already in world

        try:
            # Get transform: target <- source (i.e., world <- src_frame)
            tf: 'TransformStamped' = self.tf_buffer.lookup_transform(
                WORLD_FRAME,                # target frame
                src_frame,                  # source frame
                pose_msg.header.stamp,      # use the pose time if timestamps are reasonable
                timeout=rclpy.duration.Duration(seconds=timeout_sec)
            )
            pose_world: Pose = do_transform_pose(pose_msg.pose, tf)
            # pose_world.header.frame_id = WORLD_FRAME  # make sure it says 'world'
            # keep the original timestamp (or set to now() if you prefer)
            return pose_world
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(
                f"TF transform failed ({WORLD_FRAME} <- {src_frame}) at t={pose_msg.header.stamp.sec}.{pose_msg.header.stamp.nanosec}: {e}"
            )
            return None


    def _twist_to_world(self, twist_msg: TwistStamped, timeout_sec: float = 0.05) -> Optional[Twist]:
        """
        Transform a TwistStamped into the world frame.
        """
        if twist_msg is None:
            return None
        src_frame = twist_msg.header.frame_id
        if not src_frame:
            self.get_logger().warn("Incoming PoseStamped has empty header.frame_id")
            return None
        if src_frame == WORLD_FRAME:
            return twist_msg.twist  # already in world

        raise NotImplementedError()


def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='stack_blocks', help='task to perform')
    # parser.add_argument('--save_path', type=str, help='path to save files and data')
    args = parser.parse_args()

    node = GocMpcCartesianNode(task_name=args.task)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:

        metrics = {
            "total_time": node.end_elapsed_time,
            "waypoint_solve_times": node.waypoint_solve_times,
            "timing_solve_times": node.timing_solve_times,
            "short_path_solve_times": node.short_path_solve_times,
        }

        current_datetime = datetime.now()

        # results_dir = "experiment_results/folding_trial1"
        # results_dir = "experiment_results/pick_and_pour_trial1"
        results_dir = "experiment_results/block_stacking_trial2"
        with open(os.path.join(results_dir, f"log_file_{current_datetime}.pkl"), "wb") as f:
            pickle.dump(metrics, f)

        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
