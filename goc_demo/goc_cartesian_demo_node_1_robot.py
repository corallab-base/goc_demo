#!/usr/bin/env python3
from __future__ import annotations

import os
import argparse
import numpy as np
from typing import Optional, Tuple
from collections import namedtuple
import pickle
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import PoseStamped, TwistStamped, Pose, Twist
from nav_msgs.msg import Path

from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from tf2_geometry_msgs import do_transform_pose

from goc_demo import robotiq
from goc_demo.new_plans import move_in_circles_single_builder
from geometry_msgs.msg import PoseStamped, TwistStamped

WORLD_FRAME = "world"
Task = namedtuple("Task", ["builder"])

def slerp(q0, q1, alpha):
    """
    spherical linear interpolation for quaternions
    """
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)

    dot = np.dot(q0, q1)

    if dot < 0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        result = (1.0 - alpha) * q0 + alpha * q1
        return result / np.linalg.norm(result)

    theta_0 = np.arccos(dot)
    theta = theta_0 * alpha
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return s0 * q0 + s1 * q1


def hermite_blend(t, duration, p0, v0, p1, v1):
    """
    cubic hermite spline matching position and velocity at both endpoints
    returns (position, velocity) at time t
    """
    t_norm = t / duration

    h00 = 2*t_norm**3 - 3*t_norm**2 + 1
    h10 = t_norm**3 - 2*t_norm**2 + t_norm
    h01 = -2*t_norm**3 + 3*t_norm**2
    h11 = t_norm**3 - t_norm**2

    pos = h00*p0 + h10*duration*v0 + h01*p1 + h11*duration*v1

    dh00 = (6*t_norm**2 - 6*t_norm) / duration
    dh10 = 3*t_norm**2 - 4*t_norm + 1
    dh01 = (-6*t_norm**2 + 6*t_norm) / duration
    dh11 = 3*t_norm**2 - 2*t_norm

    vel = dh00*p0 + dh10*v0 + dh01*p1 + dh11*v1

    return pos, vel


def smoothstep(t):
    """
    hermite interpolation with zero derivatives at endpoints
    """
    return 3*t**2 - 2*t**3
def translational_curvature_xyz(X, eps=1e-9):
    """
    X: (N, >=3) array of poses; xyz are in columns 0:3.
    Returns: (N,) curvature array (1/m). Endpoints use one-sided diffs.
    """
    P = np.asarray(X)[:, :3]
    N = P.shape[0]
    if N < 3:
        return np.zeros(N)

    dP = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.zeros(N)
    s[1:] = np.cumsum(dP)

    r_s = np.zeros_like(P)
    ds_c = (s[2:] - s[:-2])[:, None]
    r_s[1:-1] = (P[2:] - P[:-2]) / np.maximum(ds_c, eps)
    r_s[0] = (P[1] - P[0]) / max(s[1] - s[0], eps)
    r_s[-1] = (P[-1] - P[-2]) / max(s[-1] - s[-2], eps)

    r_ss = np.zeros_like(P)
    ds_fwd = (s[2:] - s[1:-1])[:, None]
    ds_bwd = (s[1:-1] - s[:-2])[:, None]
    r_ss[1:-1] = (
        2.0
        * (
            (P[2:] - P[1:-1]) / np.maximum(ds_fwd, eps)
            - (P[1:-1] - P[:-2]) / np.maximum(ds_bwd, eps)
        )
        / np.maximum(ds_fwd + ds_bwd, eps)
    )
    r_ss[0] = (P[2] - 2 * P[1] + P[0]) / max((s[2] - s[0]) * (s[1] - s[0]) + eps, eps)
    r_ss[-1] = (P[-1] - 2 * P[-2] + P[-3]) / max((s[-1] - s[-3]) * (s[-1] - s[-2]) + eps, eps)

    cross = np.cross(r_s, r_ss)
    num = np.linalg.norm(cross, axis=1)
    den = np.maximum(np.linalg.norm(r_s, axis=1) ** 3, eps)
    return num / den


class GocMpcCartesianNode(Node):
    """
    Single-robot version.

    Subscribes to current pose/twist, calls goc_mpc.step(t, x, x_dot),
    and publishes PoseStamped targets to /target_frame.
    """

    def __init__(self, task_name: str):
        super().__init__("goc_mpc_cartesian_node")

        # --- Parameters ---

        self.declare_parameter("pose_topic", "/cartesian_compliance_controller/current_pose")
        self.declare_parameter("twist_topic", "/cartesian_compliance_controller/current_twist")
        self.declare_parameter("keypoints_topic", "/demo_world_node/centroids_world")
        self.declare_parameter("rate_hz", 30.0) # from 30
        self.declare_parameter("dry_run", False)

        self.declare_parameter("use_real_gripper", False)
        self.declare_parameter("gripper_ip_address", "")
        self.declare_parameter("gripper_port", 63352)

        self.declare_parameter("target_topic", "/target_frame")

        # self.declare_parameter("target_min_idx", 3)
        # self.declare_parameter("target_max_idx", 3)
        # self.declare_parameter("curvature_gamma", 16.0)

        self._filter_alpha = 0.3  #tunable: 0.1 (very smooth/laggy) to 1.0 (raw/jittery)
        self._last_filtered_pos = None
        self._last_filtered_ori = None

        self.declare_parameter("grasp_settle_sec", 1.00)
        self.declare_parameter("grasp_pause_after_cmd_sec", 1.00)

        # phase transition blending state
        self._prev_remaining_phases = None
        self._blend_start_time = None
        self._blend_duration = 0.3
        self._old_target_pose = None
        self._old_target_vel = None
        self._blend_start_pose = None
        self._blend_start_vel = None

        # Read params
        self._pose_topic: str = self.get_parameter("pose_topic").value
        self._twist_topic: str = self.get_parameter("twist_topic").value
        self._keypoints_topic: str = self.get_parameter("keypoints_topic").value
        self._target_topic: str = self.get_parameter("target_topic").value

        self._rate_hz: float = float(self.get_parameter("rate_hz").value)
        self._dry_run: bool = bool(self.get_parameter("dry_run").value)

        # self._target_min_idx: int = int(self.get_parameter("target_min_idx").value)
        # self._target_max_idx: int = int(self.get_parameter("target_max_idx").value)
        # self._gamma: float = float(self.get_parameter("curvature_gamma").value)

        self._grasp_settle_sec: float = float(self.get_parameter("grasp_settle_sec").value)
        self._grasp_pause_after_cmd_sec: float = float(self.get_parameter("grasp_pause_after_cmd_sec").value)

        if self._rate_hz <= 0.0:
            self.get_logger().warn("rate_hz must be > 0; defaulting to 100.0")
            self._rate_hz = 100.0

        self._period_sec = 1.0 / self._rate_hz

        # --- TF ---
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        # --- QoS ---
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

        # --- Visualization ---
        self._short_path_publisher = self.create_publisher(Path, "/short_path", 10)
        self._long_path_publisher = self.create_publisher(Path, "/long_path", 10)

        # --- Subscriptions ---
        self._latest_pose: Optional[Pose] = None
        self._latest_twist: Optional[Twist] = None

        self.create_subscription(PoseStamped, self._pose_topic, self._on_pose, pose_qos)
        self.create_subscription(TwistStamped, self._twist_topic, self._on_twist, pose_qos)

        self._latest_keypoints: np.ndarray = np.zeros((0, 3))
        self.create_subscription(PointCloud, self._keypoints_topic, self._on_keypoints, keypoints_qos)

        # --- Target publisher ---
        if not self._dry_run:
            self.target_pose_publisher = self.create_publisher(PoseStamped, self._target_topic, 10)
            self.target_twist_publisher = self.create_publisher(TwistStamped, 'target_twist', 10)

        # --- Optional real gripper ---
        self.use_real_gripper = bool(self.get_parameter("use_real_gripper").value)
        self.real_gripper = None
        if self.use_real_gripper:
            ip = str(self.get_parameter("gripper_ip_address").value)
            port = int(self.get_parameter("gripper_port").value)
            if ip:
                try:
                    self.real_gripper = robotiq.RobotiqGripper(disabled=False)
                    self.real_gripper.connect(ip, port)
                    self.real_gripper.activate(auto_calibrate=False)
                    self.real_gripper.open(speed=2, force=2)
                    self.get_logger().info(f"Real gripper connected: {ip}:{port}")
                except Exception as e:
                    self.get_logger().error(f"Failed to init real gripper: {e}")
                    self.real_gripper = None
            else:
                self.get_logger().warn("use_real_gripper=True but gripper_ip_address is empty; disabling gripper.")
                self.use_real_gripper = False

        # --- Pause/grasp state ---
        self.robot_paused = False
        self._pre_grasp_timer = None
        self._resume_timer = None
        self._pending_gripper_cmd = None

        # --- Task / Controller ---
        tasks = {
            "move_in_circles": Task(builder=move_in_circles_single_builder),
        }
        if task_name not in tasks:
            raise KeyError(f"Unknown task '{task_name}'. Options: {list(tasks.keys())}")

        self.task = tasks[task_name]
        self.n_agents = 1
        self.n_keypoints = 0
        self.goc_mpc = self._setup_goc_mpc(self.task)

        # metrics
        self.waypoint_solve_times = []
        self.timing_solve_times = []
        self.short_path_solve_times = []

        # Timing
        self.fake_time = 0.0
        self._start_time = self.get_clock().now()
        self.end_elapsed_time = None
        self._timer = self.create_timer(self._period_sec, self._on_timer)

        self.get_logger().info(
            f"Streaming single-robot pose goals at {self._rate_hz:.1f} Hz -> {self._target_topic}"
        )

    def _setup_goc_mpc(self, task):
        env, graph, goc_mpc = task.builder()
        self._env = env
        self.n_keypoints = graph.num_objects
        self.get_logger().info(f"n_keypoints: {self.n_keypoints}")
        return goc_mpc

    # --- Callbacks ---
    def _on_pose(self, msg: PoseStamped):
        pw = self._to_world(msg)
        if pw is not None:
            self._latest_pose = pw

    def _on_twist(self, msg: TwistStamped):
        tw = self._twist_to_world(msg)
        if tw is not None:
            self._latest_twist = tw

    def _on_keypoints(self, msg: PointCloud):
        self._latest_keypoints = np.array([(p.x, p.y, p.z) for p in msg.points], dtype=float)

    def _extract_state(self, pose: Pose, twist: Twist, kps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        robot_x = np.array([
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.w,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
        ], dtype=float)

        robot_x_dot = np.array([
            twist.linear.x,
            twist.linear.y,
            twist.linear.z,
            twist.angular.x,
            twist.angular.y,
            twist.angular.z,
        ], dtype=float)

        if self.n_keypoints > 0:
            if kps is None or kps.shape[0] < self.n_keypoints:
                kp_x = np.zeros(self.n_keypoints * 3, dtype=float)
            else:
                kp_x = kps[: self.n_keypoints].flatten()

            kp_x_dot = np.zeros(self.n_keypoints * 3, dtype=float)

            x = np.concatenate((robot_x, kp_x))
            x_dot = np.concatenate((robot_x_dot, kp_x_dot))
        else:
            x = robot_x
            x_dot = robot_x_dot

        return x, x_dot


    def _on_timer(self):
        if self._latest_pose is None:
            return
        if self._latest_twist is None:
            return

        now = self.get_clock().now()
        t = (now - self._start_time).nanoseconds * 1e-9

        try:
            x, x_dot = self._extract_state(
                self._latest_pose,
                self._latest_twist,
                self._latest_keypoints,
            )
        except Exception as e:
            self.get_logger().warn(f"Bad State: {e}")
            return

        if not hasattr(self, '_last_wall_time'):
            self._last_wall_time = t

        wall_dt = t - self._last_wall_time
        self._last_wall_time = t
        self.fake_time += wall_dt

        try:
            xi_h, vi_h, _ = self.goc_mpc.step(self.fake_time, x, x_dot)

            self.waypoint_solve_times.append(self.goc_mpc.waypoint_mpc.get_last_solve_time())
            self.timing_solve_times.append(self.goc_mpc.timing_mpc.get_last_solve_time())
            self.short_path_solve_times.append(self.goc_mpc.short_path_mpc.get_last_solve_time())
        except Exception as e:
            self.get_logger().error(f"goc_mpc.step failed: {e}")
            return

        h, d_pos = xi_h.shape
        _, d_vel = vi_h.shape

        agent_xi_h = xi_h.reshape(h, self.n_agents, d_pos // self.n_agents)[:, 0, :]
        agent_vi_h = vi_h.reshape(h, self.n_agents, d_vel // self.n_agents)[:, 0, :]

        self._publish_short_path(agent_xi_h)

        nodes_and_taus = list(zip(
            self.goc_mpc.timing_mpc.get_next_nodes(),
            self.goc_mpc.timing_mpc.get_next_taus()
        ))

        time_deltas_list = self.goc_mpc.timing_mpc.view_time_deltas_list()

        if nodes_and_taus:
            next_node, next_tau = nodes_and_taus[0]
            near_threshold = 0.15 < next_tau < 0.25
            agent_deltas = time_deltas_list[0] if time_deltas_list else []
            delta_0 = agent_deltas[0] if len(agent_deltas) > 0 else -1

            self.get_logger().info(
                f"node={next_node}, tau={next_tau:.3f}, delta[0]={delta_0:.3f}, "
                f"NEAR_THRESH={near_threshold}, remaining={self.goc_mpc.remaining_phases}\n"
                f"Current pos: [{x[0]:.3f}, {x[1]:.3f}, {x[2]:.3f}]\n"
                f"Target waypoint 0: {self.goc_mpc.waypoint_mpc.view_waypoints()[0][:3]}"
            )

        if len(nodes_and_taus) == 0 and self.end_elapsed_time is None:
            self.end_elapsed_time = t

        dt_lookahead = 0.35

        try:
            sp_times = np.array(self.goc_mpc.short_path_mpc.view_times())
        except Exception:
            sp_times = np.arange(h) * self.goc_mpc.short_path_time_per_step

        def select_target_time_based(agent_xi_h, agent_vi_h, sp_times, dt_lookahead):
            n = agent_xi_h.shape[0]
            t_target = sp_times[0] + dt_lookahead

            if t_target >= sp_times[-1]:
                return agent_xi_h[-1], agent_vi_h[-1]

            idx = int(np.searchsorted(sp_times[:n], t_target, side='right')) - 1
            idx = max(0, min(idx, n - 2))

            t0 = sp_times[idx]
            t1 = sp_times[idx + 1]
            alpha = (t_target - t0) / max(t1 - t0, 1e-9)
            alpha = float(np.clip(alpha, 0.0, 1.0))

            p0 = agent_xi_h[idx]
            p1 = agent_xi_h[idx + 1]

            pos = (1.0 - alpha) * p0[:3] + alpha * p1[:3]
            ori = (1.0 - alpha) * p0[3:7] + alpha * p1[3:7]
            ori /= np.linalg.norm(ori)

            v0 = agent_vi_h[idx]
            v1 = agent_vi_h[idx + 1]
            vel = (1.0 - alpha) * v0 + alpha * v1

            return np.concatenate((pos, ori)), vel

        target_pose, target_vel = select_target_time_based(agent_xi_h, agent_vi_h, sp_times, dt_lookahead)

        current_remaining = list(self.goc_mpc.remaining_phases)

        phase_changed = (
            self._prev_remaining_phases is not None and
            len(current_remaining) < len(self._prev_remaining_phases)
        )

        if phase_changed:
            if self._old_target_pose is not None and self._old_target_vel is not None:
                self._blend_start_time = t
                self._blend_start_pose = self._old_target_pose.copy()
                self._blend_start_vel = self._old_target_vel.copy()
                self.get_logger().info(
                    f"phase transition: {self._prev_remaining_phases} -> {current_remaining}, starting blend"
                )

        self._prev_remaining_phases = current_remaining

        if self._blend_start_time is not None:
            blend_elapsed = t - self._blend_start_time

            if blend_elapsed < self._blend_duration:
                t_norm = blend_elapsed / self._blend_duration
                alpha_smooth = smoothstep(t_norm)
                # 0 to 1 with an s curve kinda like

                # position uses cubic hermite matching pos and vel at both endpoints
                blended_pos, blended_lin_vel = hermite_blend(
                    blend_elapsed,
                    self._blend_duration,
                    self._blend_start_pose[:3],
                    self._blend_start_vel[:3],
                    target_pose[:3],
                    target_vel[:3]
                )

                # orientation uses slerp with smoothstep alpha
                blended_quat = slerp(
                    self._blend_start_pose[3:7],
                    target_pose[3:7],
                    alpha_smooth
                )

                # angular velocity uses smoothstep blend which is equivalent to hermite with zero acceleration
                blended_ang_vel = (1.0 - alpha_smooth) * self._blend_start_vel[3:6] + alpha_smooth * target_vel[3:6]

                target_pose = np.concatenate([blended_pos, blended_quat])
                target_vel = np.concatenate([blended_lin_vel, blended_ang_vel])

                self.get_logger().debug(f"blending: t_norm={t_norm:.3f}")
            else:
                self._blend_start_time = None
                self._blend_start_pose = None
                self._blend_start_vel = None
                self.get_logger().info("blend complete")

        if self._blend_start_time is None:
            self._old_target_pose = target_pose.copy()
            self._old_target_vel = target_vel.copy()

        target_pose_stamped = PoseStamped()
        target_pose_stamped.header.frame_id = "world"
        target_pose_stamped.header.stamp = now.to_msg()
        target_pose_stamped.pose.position.x = float(target_pose[0])
        target_pose_stamped.pose.position.y = float(target_pose[1])
        target_pose_stamped.pose.position.z = float(target_pose[2])
        target_pose_stamped.pose.orientation.w = float(target_pose[3])
        target_pose_stamped.pose.orientation.x = float(target_pose[4])
        target_pose_stamped.pose.orientation.y = float(target_pose[5])
        target_pose_stamped.pose.orientation.z = float(target_pose[6])

        target_twist_stamped = TwistStamped()
        target_twist_stamped.header.frame_id = "world"
        target_twist_stamped.header.stamp = now.to_msg()
        target_twist_stamped.twist.linear.x = float(target_vel[0])
        target_twist_stamped.twist.linear.y = float(target_vel[1])
        target_twist_stamped.twist.linear.z = float(target_vel[2])
        target_twist_stamped.twist.angular.x = float(target_vel[3])
        target_twist_stamped.twist.angular.y = float(target_vel[4])
        target_twist_stamped.twist.angular.z = float(target_vel[5])

        if not self._dry_run and not self.robot_paused:
            self.target_pose_publisher.publish(target_pose_stamped)
            self.target_twist_publisher.publish(target_twist_stamped)

            if not hasattr(self, '_logged_first_twist'):
                self.get_logger().info(
                    f"publishing first twist: lin=[{target_vel[0]:.3f}, {target_vel[1]:.3f}, {target_vel[2]:.3f}], "
                    f"ang=[{target_vel[3]:.3f}, {target_vel[4]:.3f}, {target_vel[5]:.3f}]"
                )
                self._logged_first_twist = True

    # --- Visualization ---
    def _publish_short_path(self, xi_h):
        path_msg = Path()
        path_msg.header.frame_id = WORLD_FRAME
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for row in xi_h:
            pose = PoseStamped()
            pose.header = path_msg.header
            x, y, z, qw, qx, qy, qz = row[:7]
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = float(z)
            pose.pose.orientation.w = float(qw)
            pose.pose.orientation.x = float(qx)
            pose.pose.orientation.y = float(qy)
            pose.pose.orientation.z = float(qz)
            path_msg.poses.append(pose)

        self._short_path_publisher.publish(path_msg)

    def _publish_long_path(self, xi_l):
        path_msg = Path()
        path_msg.header.frame_id = WORLD_FRAME
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for row in xi_l:
            pose = PoseStamped()
            pose.header = path_msg.header
            x, y, z, qw, qx, qy, qz = row[:7]
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = float(z)
            pose.pose.orientation.w = float(qw)
            pose.pose.orientation.x = float(qx)
            pose.pose.orientation.y = float(qy)
            pose.pose.orientation.z = float(qz)
            path_msg.poses.append(pose)

        self._long_path_publisher.publish(path_msg)

    # --- Gripper / Pause helpers ---
    def _do_gripper_cmd(self, cmd: str):
        if not self.use_real_gripper or self.real_gripper is None:
            return
        try:
            if cmd == "grab":
                self.real_gripper.close(speed=2, force=2)
            elif cmd == "release":
                self.real_gripper.open(speed=2, force=2)
            else:
                self.get_logger().warn(f"Unknown gripper cmd: {cmd}")
        except Exception as e:
            self.get_logger().error(f"Gripper command '{cmd}' failed: {e}")

    def _resume_robot(self):
        self.robot_paused = False
        if self._resume_timer is not None:
            self._resume_timer.cancel()
            self._resume_timer = None
        self.get_logger().info("Robot resumed after grasp pause.")

    def _on_pre_grasp(self):
        if self._pre_grasp_timer is not None:
            self._pre_grasp_timer.cancel()
            self._pre_grasp_timer = None
        cmd = self._pending_gripper_cmd
        self._pending_gripper_cmd = None
        if cmd is not None:
            self._do_gripper_cmd(cmd)

        if self._resume_timer is not None:
            self._resume_timer.cancel()
            self._resume_timer = None
        self._resume_timer = self.create_timer(self._grasp_pause_after_cmd_sec, self._resume_robot)

    def _pause_robot_delayed(self, pre_delay: float, post_delay: float, gripper_cmd: str):
        self.robot_paused = True
        self._pending_gripper_cmd = gripper_cmd

        if self._pre_grasp_timer is not None:
            self._pre_grasp_timer.cancel()
            self._pre_grasp_timer = None
        self._pre_grasp_timer = self.create_timer(pre_delay, self._on_pre_grasp)

        if self._resume_timer is not None:
            self._resume_timer.cancel()
            self._resume_timer = None

    # --- TF helpers ---
    def _to_world(self, pose_msg: PoseStamped, timeout_sec: float = 0.05) -> Optional[Pose]:
        if pose_msg is None:
            return None
        src_frame = pose_msg.header.frame_id
        if not src_frame:
            self.get_logger().warn("Incoming PoseStamped has empty header.frame_id")
            return None
        if src_frame == WORLD_FRAME:
            return pose_msg.pose

        try:
            tf = self.tf_buffer.lookup_transform(
                WORLD_FRAME,
                src_frame,
                pose_msg.header.stamp,
                timeout=rclpy.duration.Duration(seconds=timeout_sec),
            )
            return do_transform_pose(pose_msg.pose, tf)
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(
                f"TF transform failed ({WORLD_FRAME} <- {src_frame}) at "
                f"t={pose_msg.header.stamp.sec}.{pose_msg.header.stamp.nanosec}: {e}"
            )
            return None

    def _twist_to_world(self, twist_msg: TwistStamped, timeout_sec: float = 0.05) -> Optional[Twist]:
        if twist_msg is None:
            return None
        src_frame = twist_msg.header.frame_id
        if not src_frame:
            self.get_logger().warn("Incoming TwistStamped has empty header.frame_id")
            return None
        if src_frame == WORLD_FRAME:
            return twist_msg.twist

        self.get_logger().warn(f"Twist in non-world frame '{src_frame}' not supported; dropping.")
        return None


def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="move_in_circles", help="task to perform")
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
        results_dir = "experiment_results/single_robot_trial"
        os.makedirs(results_dir, exist_ok=True)

        with open(os.path.join(results_dir, f"log_file_{current_datetime}.pkl"), "wb") as f:
            pickle.dump(metrics, f)

        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()