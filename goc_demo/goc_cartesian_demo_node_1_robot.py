#!/usr/bin/env python3
from __future__ import annotations

import os
import argparse
import numpy as np
from typing import List, Optional, Tuple, Sequence, Union
from collections import namedtuple
from collections import defaultdict

import pickle
from datetime import datetime

import rclpy
from rclpy.time import Time
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from rclpy.action import ActionClient
from sensor_msgs.msg import Image, JointState, PointCloud
from geometry_msgs.msg import PointStamped, PoseStamped, TwistStamped, Pose, Twist
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from nav_msgs.msg import Path
from control_msgs.action import FollowJointTrajectory
from builtin_interfaces.msg import Duration as RosDuration
from cv_bridge import CvBridge
import cv2

from tf2_ros import (
    Buffer,
    TransformListener,
    TransformException,
    LookupException,
    ConnectivityException,
    ExtrapolationException
)
from tf2_geometry_msgs import do_transform_pose_stamped, do_transform_point
from tf_transformations import quaternion_matrix, euler_from_quaternion, quaternion_from_euler

from pydrake.math import RollPitchYaw
from pydrake.common.eigen_geometry import Quaternion

from goc_mpc.splines import Block
from goc_mpc.goc_mpc import GraphOfConstraints, GraphOfConstraintsMPC
from goc_mpc.simple_drake_env import SimpleDrakeGym

from goc_demo import robotiq
from goc_demo.plans import (
    one_robot_move_in_circles_builder,
    pick_and_place_builder,
    test_yaw_builder,
    move_spam_builder,
)


WORLD_FRAME = "world"

Task = namedtuple('Task', ["builder", "points", "objects", "needs_yaw"])


class GocMpcCartesianNode(Node):
    """
    Runs GoC-MPC with one robot for a given task.
    """

    def __init__(self, task_name: str):
        super().__init__("goc_mpc_cartesian_node")

        # --- Parameters (your snippet + a couple extra) ---
        self.declare_parameter("pose_topic", "/cartesian_motion_controller/current_pose")
        self.declare_parameter("twist_topic", "/cartesian_motion_controller/current_twist")
        self.declare_parameter("rate_hz", 30.0)
        self.declare_parameter('target_img_dim', 128) # Default 224x224

        self.bridge = CvBridge()

        # Read params
        self._pose_topic: str = self.get_parameter("pose_topic").value
        self._twist_topic: str = self.get_parameter("twist_topic").value

        self._target_img_dim = self.get_parameter('target_img_dim').get_parameter_value().integer_value

        self._rate_hz: float = float(self.get_parameter("rate_hz").value)

        if self._rate_hz <= 0.0:
            self.get_logger().warn("rate_hz must be > 0; defaulting to 100.0")
            self._rate_hz = 100.0

        self._period_sec = 1.0 / self._rate_hz

        # --- TF stuff ---
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        # --- Sub/Pub QoS ---
        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # --- Visualization Publications ---
        self._short_path_publisher = self.create_publisher(Path, "/short_path", 10)
        self._long_path_publisher = self.create_publisher(Path, "/long_path", 10)
        self._waypoints_publisher = self.create_publisher(Path, "/waypoints", 10)

        # --- Subscriptions ---
        self._latest_pose: Optional[PoseStamped] = None
        self.create_subscription(PoseStamped, self._pose_topic, self._on_pose, best_effort_qos)
        self._latest_twist: Optional[TwistStamped] = None
        self.create_subscription(TwistStamped, self._twist_topic, self._on_twist, best_effort_qos)

        self._latest_q: Optional[np.ndarray] = None
        self._latest_qd: Optional[np.ndarray] = None
        self._latest_eff: Optional[np.ndarray] = None
        self.create_subscription(JointState, "/joint_states", self._on_joints, best_effort_qos)

        # self._latest_image = None
        # self.create_subscription(Image, '/camera/camera/color/image_raw', self._on_image, 10)

        # Publisher to send the target pose to the robot
        target_pose_topic_name = "/cartesian_motion_controller/target_frame"
        self.target_pose_publisher = self.create_publisher(
            PoseStamped, target_pose_topic_name, 10
        )

        # instatiate real grippers (not the cleanest, but has to be done)
        ip_address = "10.168.4.249"
        self._real_gripper = robotiq.RobotiqGripper(disabled=False)
        self._real_gripper.connect(ip_address, 63352)
        self._real_gripper.activate(auto_calibrate=True)
        self._real_gripper.open(speed=2, force=2)

        self._robot_paused = False
        self._pre_grasp_timer = None
        self._resume_timer = None

        # Pending gripper cmds (latched until pre-delay expires)
        self._pending_gripper_cmd = None

        # Tunables
        self._grasp_settle_sec = 1.00          # wait before actuating gripper
        self._grasp_pause_after_cmd_sec = 1.00 # time to remain paused after actuation

        # --- Controller ---

        tasks = {
            "move_in_circles": Task(builder=one_robot_move_in_circles_builder, points=[], objects=[], needs_yaw=False),
            "pick_and_place": Task(builder=pick_and_place_builder, points=["green", "red"], objects=[], needs_yaw=False),
            "test_yaw": Task(builder=test_yaw_builder, points=[], objects=[], needs_yaw=True),
            "move_spam": Task(builder=move_spam_builder, points=[], objects=["spam"], needs_yaw=True),
        }

        self._task = tasks[task_name]
        self._needs_yaw = self._task.needs_yaw

        self._latest_positions = {}
        self._latest_poses = {}

        self.subs = []
        for name in self._task.points:
            topic = f'/{name}/center'
            self.get_logger().info(f'Subscribing to {topic}')
            sub = self.create_subscription(
                PointStamped, topic,
                self._make_obj_point_callback(name),
                best_effort_qos
            )
            self.subs.append(sub)

        for name in self._task.objects:
            topic = f'/{name}/pose'
            self.get_logger().info(f'Subscribing to {topic}')
            sub = self.create_subscription(
                PoseStamped, topic,
                self._make_obj_pose_callback(name),
                best_effort_qos
            )
            self.subs.append(sub)

        self.n_agents = 1
        # self.n_keypoints = 0
        self.goc_mpc = self._setup_goc_mpc(self._task)
        self._obs = None

        # metrics
        self.waypoint_solve_times = []
        self.timing_solve_times = []
        self.short_path_solve_times = []

        # --- Timing ---
        self._start_time = self.get_clock().now()
        self.end_elapsed_time = None
        self._timer = self.create_timer(self._period_sec, self._on_timer)

        # Track last goal handle (optional)
        self._last_goal_handle = None

        self.get_logger().info(
            f"Streaming pose goals at {self._rate_hz:.1f} Hz"
        )

        self.recorded_data = defaultdict(list)

    def _setup_goc_mpc(self, task):
        graph, goc_mpc = task.builder()

        self.n_keypoints = graph.num_objects

        self.get_logger().info(f"n_keypoints: {self.n_keypoints}")

        return goc_mpc

    # --- Callbacks ---
    def _on_joints(self, msg: JointState):
        self._latest_q = np.array(msg.position)
        self._latest_qd = np.array(msg.velocity)
        self._latest_eff = np.array(msg.effort)

    def _on_pose(self, msg: PoseStamped):
        ps_w = self._to_world(msg)
        if ps_w is not None:
            self._latest_pose = ps_w.pose

    def _on_twist(self, msg: TwistStamped):
        tw = self._twist_to_world(msg)
        if tw is not None:
            self._latest_twist = tw

    # def _on_image(self, msg):
    #     # 1. Convert ROS Image message to OpenCV format
    #     cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    #     h, w = cv_img.shape[:2]

    #     # 2. Calculate cropping coordinates for a centered square
    #     # We find the shortest side to ensure the square fits
    #     size = min(h, w)
    #     start_x = (w - size) // 2
    #     start_y = (h - size) // 2

    #     # Crop using NumPy slicing: [y1:y2, x1:x2]
    #     square_crop = cv_img[start_y:start_y+size, start_x:start_x+size]

    #     # 3. Downscale to customizable resolution
    #     resized_img = cv2.resize(square_crop, (self._target_img_dim, self._target_img_dim), interpolation=cv2.INTER_AREA)

    #     # Optional: Do something with resized_img (e.g., publish it or run inference)
    #     self._latest_image = resized_img

    def _make_obj_point_callback(self, name: str):

        def callback(msg: PointStamped):
            """Transform incoming point into target_frame; store position only."""
            if not msg.header.frame_id:
                self.get_logger().warn(f'[{name}] Pose has empty frame_id; ignoring.')
                return

            try:
                # Get transform from pose frame to target_frame at the message time
                tf = self.tf_buffer.lookup_transform(
                    WORLD_FRAME,          # target
                    msg.header.frame_id,  # source
                    Time(), # rclpy.time.Time.from_msg(msg.header.stamp),
                    # timeout=rclpy.duration.Duration(seconds=0.2)
                )

                p = do_transform_point(msg, tf).point
                self._latest_positions[name] = (float(p.x), float(p.y), float(p.z))

            except TransformException as ex:
                # You might see this until TF is available / connected
                self.get_logger().debug(f'[{name}] TF error: {ex}')


        return callback

    def _make_obj_pose_callback(self, name: str):

        def callback(msg: PoseStamped):
            """Transform incoming pose into target_frame; store pose."""
            if not msg.header.frame_id:
                self.get_logger().warn(f'[{name}] Pose has empty frame_id; ignoring.')
                return

            try:
                # Get transform from pose frame to target_frame at the message time
                tf = self.tf_buffer.lookup_transform(
                    WORLD_FRAME,          # target
                    msg.header.frame_id,  # source
                    Time(), # rclpy.time.Time.from_msg(msg.header.stamp),
                    # timeout=rclpy.duration.Duration(seconds=0.2)
                )

                p = do_transform_pose_stamped(msg, tf).pose
                if self._needs_yaw:
                    yaw = self._quat_to_yaw(p.orientation)
                    self._latest_poses[name] = (float(p.position.x),
                                                float(p.position.y),
                                                float(p.position.z),
                                                float(yaw))
                else:
                    self._latest_poses[name] = (float(p.position.x),
                                                float(p.position.y),
                                                float(p.position.z),
                                                float(p.orientation.w),
                                                float(p.orientation.x),
                                                float(p.orientation.y),
                                                float(p.orientation.z))

            except TransformException as ex:
                # You might see this until TF is available / connected
                self.get_logger().debug(f'[{name}] TF error: {ex}')

        return callback


    def _extract_state(self,
                       pose: Pose,
                       twist: Twist,
                       latest_positions: dict[name, tuple[float, float, float]]) -> Tuple[np.ndarray, np.ndarray]:

        def pose_to_arr(pose: Pose):
            arr = np.array([pose.position.x,
                            pose.position.y,
                            pose.position.z])
            if self._needs_yaw:
                yaw = self._quat_to_yaw(pose.orientation)
                arr = np.append(arr, yaw)
            return arr

        def twist_to_arr(twist: Twist):
            arr = np.array([twist.linear.x,
                            twist.linear.y,
                            twist.linear.z])
            if self._needs_yaw:
                arr = np.append(arr, 0.0)  # yaw_dot not available
            return arr

        x = pose_to_arr(pose)
        x_dot = twist_to_arr(twist)

        if any([name not in latest_positions for name in self._task.points]):
            non_found = list(filter(lambda name: name not in latest_positions, self._task.points))
            raise ValueError(f"points {non_found} are not found")

        kp_x = np.array([latest_positions[name] for name in self._task.points]).flatten()
        if self._needs_yaw:
            kp_x_dot = np.zeros((self.n_keypoints, 4)).flatten()
        else:
            kp_x_dot = np.zeros((self.n_keypoints, 3)).flatten()

        # x, x_dot
        x = np.concatenate((x, kp_x))
        x_dot = np.concatenate((x_dot, kp_x_dot))
        return x, x_dot

    def _on_timer(self):
        if self._latest_pose is None:
            self.get_logger().info('_latest_pose is None')
            return
        if self._latest_twist is None:
            self.get_logger().info('_latest_twist is None')
            return
        if self._latest_positions is None:
            self.get_logger().info('_latest_positions is None')
            return
        if self._latest_poses is None:
            self.get_logger().info('_latest_poses is None')
            return

        if self._latest_q is None:
            self.get_logger().info('_latest_q is None')
            return
        if self._latest_qd is None:
            self.get_logger().info('_latest_qd is None')
            return
        if self._latest_eff is None:
            self.get_logger().info('_latest_eff is None')
            return

        # if self._latest_image is None:
        #     self.get_logger().info('_latest_image is None')
        #     return

        if len(self.goc_mpc.remaining_phases) <= 0:
            self.get_logger().info('Nothing left to do! Manually backtracking everything')

            # TODO: Fix this hack
            self.goc_mpc = self._setup_goc_mpc(self._task)

            current_datetime = datetime.now()
            results_dir = "saved_data"
            with open(os.path.join(results_dir, f"data_{current_datetime}.pkl"), "wb") as f:
                pickle.dump(self.recorded_data, f)

            del self.recorded_data
            self.recorded_data = defaultdict(list)
            
            return

        now = self.get_clock().now()
        t = (now - self._start_time).nanoseconds * 1e-9

        #######################################################################
        #                           GET OBSERVATION                           #
        #######################################################################

        try:
            x, x_dot = self._extract_state(self._latest_pose,
                                           self._latest_twist,
                                           self._latest_positions)
        except Exception as e:
            self.get_logger().warn(f"Bad State: {e}")
            return

        #######################################################################
        #                               MPC STEP                              #
        #######################################################################

        try:
            xi_h, xi_dot_h, _ = self.goc_mpc.step(t, x, x_dot)

            self.waypoint_solve_times.append(self.goc_mpc.waypoint_mpc.get_last_solve_time())
            self.timing_solve_times.append(self.goc_mpc.timing_mpc.get_last_solve_time())
            self.short_path_solve_times.append(self.goc_mpc.short_path_mpc.get_last_solve_time())
        except RuntimeError as e:
            self.get_logger().error(f"goc_mpc.step failed: {e}")
            print(e)
            return

        h, d_pos = xi_h.shape
        _, d_vel = xi_dot_h.shape

        xi_h = xi_h.reshape(h, self.n_agents, d_pos // self.n_agents)
        xi_dot_h = xi_dot_h.reshape(h, self.n_agents, d_vel // self.n_agents)

        #######################################################################
        #                            VISUALIZATION                            #
        #######################################################################

        # WPS VISUALIZATION

        agent_wps = self.goc_mpc.timing_mpc.view_wps_list()

        self._publish_paths(
            self._waypoints_publisher, agent_wps[0],
            pos_only=True,
        )

        # FULL SPLINE VISUALIZATION

        agent_xi_ls = []
        agent_spline = self.goc_mpc.last_cycle_splines[0]
        begin_time = agent_spline.begin()
        end_time = agent_spline.end()
        times = np.linspace(begin_time, end_time, 100)
        agent_xi_l, _ = agent_spline.eval_multiple(times)
        agent_xi_ls.append(agent_xi_l)

        self._publish_paths(
            self._long_path_publisher, agent_xi_ls[0],
            pos_only=True,
        )

        # SHORT SPLINE VISUALIZATION

        self._publish_paths(
            self._short_path_publisher, xi_h[:, 0],
            pos_only=True,
        )

        # LOGGING

        nodes_and_taus = list(zip(
            self.goc_mpc.timing_mpc.get_next_nodes(),
            self.goc_mpc.timing_mpc.get_next_taus()
        ))

        self.get_logger().info(f"next waypoints in: {nodes_and_taus}")

        #######################################################################
        #                            EXECUTE ACTION                           #
        #######################################################################

        target_pose = xi_h[3, 0]

        target_pose_stamped = PoseStamped()
        target_pose_stamped.header.frame_id = WORLD_FRAME
        target_pose_stamped.header.stamp = self.get_clock().now().to_msg()
        target_pose_stamped.pose.position.x = target_pose[0]
        target_pose_stamped.pose.position.y = target_pose[1]
        target_pose_stamped.pose.position.z = target_pose[2]

        if self._needs_yaw:
            yaw = target_pose[3]
            qw, qx, qy, qz = self._yaw_to_quat(yaw)
            target_pose_stamped.pose.orientation.w = qw
            target_pose_stamped.pose.orientation.x = qx
            target_pose_stamped.pose.orientation.y = qy
            target_pose_stamped.pose.orientation.z = qz
        else:
            target_pose_stamped.pose.orientation.w = 0.0
            target_pose_stamped.pose.orientation.x = 0.0
            target_pose_stamped.pose.orientation.y = 1.0
            target_pose_stamped.pose.orientation.z = 0.0

        if len(self.goc_mpc.last_grasp_commands) > 0:
            self.get_logger().info(f"Grasp Commands! {self.goc_mpc.last_grasp_commands}")
            for cmd, _, point in self.goc_mpc.last_grasp_commands:
                self.get_logger().info(f"Paused robot!")
                self._pause_robot_delayed(
                    pre_delay=self._grasp_settle_sec,
                    post_delay=self._grasp_pause_after_cmd_sec,
                    gripper_cmd=cmd
                )

        if len(self.goc_mpc.last_cycle_backtracked_phases) > 0:
            for _, new_phase in self.goc_mpc.last_cycle_backtracked_phases.items():
                self.get_logger().info(f"Paused robot to backtrack!")
                self._pause_robot_delayed(
                    pre_delay=0.0,
                    post_delay=0.0,
                    gripper_cmd="release"
                )

        if not self._robot_paused and target_pose_stamped is not None:
            self.target_pose_publisher.publish(target_pose_stamped)

        #######################################################################
        #                              RECORDING                              #
        #######################################################################

        # state information ###################################################
        self.recorded_data["img"].append(self._latest_image)

        self.recorded_data["q"].append(self._latest_q)
        self.recorded_data["qd"].append(self._latest_qd)
        self.recorded_data["eff"].append(self._latest_eff)

        self.recorded_data["ee_pos"].append(x[0:3])
        if self._needs_yaw:
            ee_yaw = self._quat_to_yaw(self._latest_pose.orientation)
            self.recorded_data["ee_yaw"].append(ee_yaw)
            qw, qx, qy, qz = self._yaw_to_quat(ee_yaw)
            self.recorded_data["ee_quat_wxyz"].append(np.array([qw, qx, qy, qz]))
        else:
            self.recorded_data["ee_quat_wxyz"].append(np.array([0.0, 0.0, 1.0, 0.0]))

        self.recorded_data["ee_vel"].append(x_dot[0:3])
        self.recorded_data["gripper_pos"].append(self._real_gripper.get_current_position())
        for name, pos in self._latest_positions.items():
            self.recorded_data[f"{name}_pos"].append(np.array(pos))

        if self._needs_yaw:
            action = target_pose[0:3] - x[0:3]
            action_yaw = target_pose[3] - x[3]
            self.recorded_data["action"].append(np.concatenate([action, [action_yaw]]))
            for name, pose in self._latest_poses.items():
                self.recorded_data[f"{name}_pose"].append(np.array(pose))
        else:
            self.recorded_data["action"].append(target_pose - x[0:3])
            for name, pose in self._latest_poses.items():
                self.recorded_data[f"{name}_pose"].append(np.array(pose))

        reward = 3 in self.goc_mpc.completed_phases
        termination = 3 in self.goc_mpc.completed_phases and 5 in self.goc_mpc.completed_phases
        # self.get_logger().info(f"reward: {reward}, termination: {termination}")
        self.recorded_data["reward"].append(0.0 if reward else -1.0)
        self.recorded_data["termination"].append(1.0 if termination else 0.0)


    # --- Helpers ---

    def _quat_to_yaw(self, quat):
        """Extract yaw (rotation around z-axis) from quaternion (x, y, z, w)."""
        roll, pitch, yaw = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
        return yaw

    def _yaw_to_quat(self, yaw):
        """Convert yaw angle to quaternion (w, x, y, z) with zero roll and pitch."""
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw)
        return (qw, qx, qy, qz)

    def _publish_paths(self, path_pub, xi, pos_only=True):
        path_msg = Path()
        path_msg.header.frame_id = WORLD_FRAME   # or "map", depending on your TF setup
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for row in xi:
            pose = PoseStamped()
            pose.header = path_msg.header
            if pos_only:
                x, y, z = row[:3]
                qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
            else:
                # take the first 7 elements of the row (first pose)
                x, y, z, qw, qx, qy, qz = row[:7]
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = float(z)
            pose.pose.orientation.w = float(qw)
            pose.pose.orientation.x = float(qx)
            pose.pose.orientation.y = float(qy)
            pose.pose.orientation.z = float(qz)
            path_msg.poses.append(pose)

        path_pub.publish(path_msg)

    def _do_gripper_cmd(self, cmd: str):
        try:
            if cmd == 'grab':
                self._real_gripper.close(speed=200, force=2)
            elif cmd == 'release':
                self._real_gripper.open(speed=200, force=2)
            else:
                self.get_logger().warn(f"Unknown gripper cmd: {cmd}")
        except Exception as e:
            self.get_logger().error(f"Gripper {side} command '{cmd}' failed: {e}")

    def _resume_robot(self):
        self._robot_paused = False
        if self._resume_timer is not None:
            self._resume_timer.cancel()
            self._resume_timer = None
        self.get_logger().info("Robot resumed after grasp pause.")

    def _on_pre_grasp(self):
        """Fires after settle delay: actuate gripper then start resume timer."""
        if self._pre_grasp_timer is not None:
            self._pre_grasp_timer.cancel()
            self._pre_grasp_timer = None

        cmd = self._pending_gripper_cmd
        self._pending_gripper_cmd = None

        if cmd is not None:
            self._do_gripper_cmd(cmd)

        # chain the resume one-shot
        if self._resume_timer is not None:
            self._resume_timer.cancel()
            self._resume_timer = None

        self._resume_timer = self.create_timer(self._grasp_pause_after_cmd_sec,
                                               self._resume_robot)

    def _pause_robot_delayed(self, pre_delay: float, post_delay: float, gripper_cmd: str):
        """
        Immediately pause 'side', wait pre_delay, then execute gripper_cmd, then
        wait post_delay and resume. If re-triggered, refresh the sequence.
        """
        self._robot_paused = True
        self._pending_gripper_cmd = gripper_cmd

        # refresh pre-grasp one-shot
        if self._pre_grasp_timer is not None:
            self._pre_grasp_timer.cancel()
            self._pre_grasp_timer = None
        self._pre_grasp_timer = self.create_timer(pre_delay, self._on_pre_grasp)

        # cancel any existing resume timer; it will be reset after actuation
        if self._resume_timer is not None:
            self._resume_timer.cancel()
            self._resume_timer = None

    def _to_world(self, pose_msg: PoseStamped, timeout_sec: float = 0.1, target_frame: str = WORLD_FRAME) -> Optional[PoseStamped]:
        """Turn a PoseStamped (using its header.frame_id) into a PoseStamped in the target frame."""
        if pose_msg is None:
            return None
        src_frame = pose_msg.header.frame_id
        if not src_frame:
            self.get_logger().warn("Incoming PoseStamped has empty header.frame_id")
            return None
        if src_frame == target_frame:
            return pose_msg  # already in target_frame

        try:
            # Get transform: target <- source (i.e., world <- src_frame)
            tf: 'TransformStamped' = self.tf_buffer.lookup_transform(
                target_frame,               # target frame
                src_frame,                  # source frame
                Time(), # pose_msg.header.stamp,      # Time(), # use the pose time if timestamps are reasonable
                timeout=rclpy.duration.Duration(seconds=timeout_sec)
            )
            pose_stamped_world: PoseStamped = do_transform_pose_stamped(pose_msg, tf)
            # pose_world.header.frame_id = WORLD_FRAME  # make sure it says 'world'
            # keep the original timestamp (or set to now() if you prefer)
            return pose_stamped_world
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(
                f"TF transform failed ({WORLD_FRAME} <- {src_frame}) at t={pose_msg.header.stamp.sec}.{pose_msg.header.stamp.nanosec}: {e}"
            )
            return None


    def _twist_to_world(self, twist_msg: TwistStamped, timeout_sec: float = 0.05) -> Optional[Twist]:
        """Turn a TwistStamped (using its header.frame_id) into a Twist in WORLD_FRAME."""
        if twist_msg is None:
            return None
        src_frame = twist_msg.header.frame_id
        if not src_frame:
            self.get_logger().warn("Incoming PoseStamped has empty header.frame_id")
            return None
        if src_frame == WORLD_FRAME:
            return twist_msg.twist  # already in world

        try:
            tf = self.tf_buffer.lookup_transform(
                WORLD_FRAME,
                src_frame,
                Time(), # twist_msg.header.stamp,
                timeout=rclpy.duration.Duration(seconds=timeout_sec),
            )

            p = tf.transform.translation
            q = tf.transform.rotation
            R = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]

            skew_symmetric_p =  np.array([
                [ 0,   -p.z,  p.y],
                [ p.z,  0,   -p.x],
                [-p.y,  p.x,  0]
            ])

            adjoint_T_ab = np.concatenate([
                np.concatenate([R, np.zeros((3,3))], axis=1),
                np.concatenate([np.matmul(skew_symmetric_p, R), R], axis=1)
            ], axis=0)


            twist_b = np.array([
                twist_msg.twist.linear.x,
                twist_msg.twist.linear.y,
                twist_msg.twist.linear.z,
                twist_msg.twist.angular.x,
                twist_msg.twist.angular.y,
                twist_msg.twist.angular.z,
            ])

            twist_a = np.matmul(adjoint_T_ab, twist_b)

            twist_world = Twist()
            twist_world.linear.x = twist_a[0]
            twist_world.linear.y = twist_a[1]
            twist_world.linear.z = twist_a[2]
            twist_world.angular.x = twist_a[3]
            twist_world.angular.y = twist_a[4]
            twist_world.angular.z = twist_a[5]
            return twist_world
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(
                f"TF transform failed ({WORLD_FRAME} <- {src_frame}) at "
                f"t={twist_msg.header.stamp.sec}.{twist_msg.header.stamp.nanosec}: {e}"
            )
            return None


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

        current_datetime = datetime.now()

        results_dir = "saved_data"
        with open(os.path.join(results_dir, f"data_{current_datetime}.pkl"), "wb") as f:
            pickle.dump(node.recorded_data, f)

        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
