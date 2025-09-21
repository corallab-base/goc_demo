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
            # "stack_blocks": Task(builder=stack_blocks_builder),
            # "pick_and_pour": Task(builder=stack_blocks_builder),
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

        # publish short path for visualization
        self._publish_paths(xi_h)

        nodes_and_taus = list(zip(self.goc_mpc.timing_mpc.get_next_nodes(), self.goc_mpc.timing_mpc.get_next_taus()))
        self.get_logger().info(f"next waypoints in: {nodes_and_taus}")

        if len(nodes_and_taus) == 0 and self.end_elapsed_time is None:
            self.end_elapsed_time = t

        target = 4
        self.fake_time += target * self.goc_mpc.short_path_time_per_step

        left_target_pose_stamped = PoseStamped()
        left_target_pose_stamped.header.frame_id = "world"
        left_target_pose_stamped.header.stamp = self.get_clock().now().to_msg()
        left_target_pose_stamped.pose.position.x = xi_h[target, 0]
        left_target_pose_stamped.pose.position.y = xi_h[target, 1]
        left_target_pose_stamped.pose.position.z = xi_h[target, 2]
        left_target_pose_stamped.pose.orientation.w = xi_h[target, 3]
        left_target_pose_stamped.pose.orientation.x = xi_h[target, 4]
        left_target_pose_stamped.pose.orientation.y = xi_h[target, 5]
        left_target_pose_stamped.pose.orientation.z = xi_h[target, 6]

        right_target_pose_stamped = PoseStamped()
        right_target_pose_stamped.header.frame_id = "world"
        right_target_pose_stamped.header.stamp = self.get_clock().now().to_msg()
        right_target_pose_stamped.pose.position.x = xi_h[target, 7+0]
        right_target_pose_stamped.pose.position.y = xi_h[target, 7+1]
        right_target_pose_stamped.pose.position.z = xi_h[target, 7+2]
        right_target_pose_stamped.pose.orientation.w = xi_h[target, 7+3]
        right_target_pose_stamped.pose.orientation.x = xi_h[target, 7+4]
        right_target_pose_stamped.pose.orientation.y = xi_h[target, 7+5]
        right_target_pose_stamped.pose.orientation.z = xi_h[target, 7+6]

        if self._dry_run:
            qpos = xi_h[target]
            self._obs, _, _, _, _ = self._env.step(qpos, grasp_cmds=self.goc_mpc.last_grasp_commands)
        else:
            qpos = xi_h[0]
            self._obs, _, _, _, _ = self._env.step(qpos, grasp_cmds=self.goc_mpc.last_grasp_commands)

            if len(self.goc_mpc.last_grasp_commands) > 0:
                # self.get_logger().info(f"Grasp Commands! {self.goc_mpc.last_grasp_commands}")
                for cmd, robot, point in self.goc_mpc.last_grasp_commands:
                    if robot == "free_body_0":
                        side = "left"
                    elif robot == "free_body_1":
                        side = "right"
                    else:
                        continue
                    self._pause_robot_delayed(
                        side=side,
                        pre_delay=self._grasp_settle_sec,
                        post_delay=self._grasp_pause_after_cmd_sec,
                        gripper_cmd=cmd
                    )

            if not self.left_robot_paused:
                self.left_target_pose_publisher.publish(left_target_pose_stamped)

            if not self.right_robot_paused:
                self.right_target_pose_publisher.publish(right_target_pose_stamped)

    # --- Helpers ---

    def _publish_paths(self, xi_h):
        h, d = xi_h.shape
        xi_h = xi_h.reshape(h, self.n_agents, d // self.n_agents).transpose(1, 0, 2)

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
        results_dir = "experiment_results/block_stacking_trial1"
        with open(os.path.join(results_dir, f"log_file_{current_datetime}.pkl"), "wb") as f:
            pickle.dump(metrics, f)

        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
