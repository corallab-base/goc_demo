#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple, Sequence, Union

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from rclpy.action import ActionClient
from sensor_msgs.msg import JointState, PointCloud
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from builtin_interfaces.msg import Duration as RosDuration

from goc_mpc.goc_mpc import GraphOfConstraints, GraphOfConstraintsMPC
from goc_mpc.simple_drake_env import SimpleDrakeGym


class GocMpcFollowJTNode(Node):
    """
    Subscribes to JointState, calls goc_mpc.step(t, x, x_dot), and streams tiny
    FollowJointTrajectory goals to a ros2_control joint_trajectory_controller.
    """

    def __init__(self):
        super().__init__("goc_mpc_follow_jt_node")

        # --- Parameters (your snippet + a couple extra) ---
        self.declare_parameter("left_controller_name", "left_scaled_joint_trajectory_controller")
        self.declare_parameter("right_controller_name", "right_scaled_joint_trajectory_controller")
        self.declare_parameter("joints",
            [
                "left_shoulder_pan_joint",
                "left_shoulder_lift_joint",
                "left_elbow_joint",
                "left_wrist_1_joint",
                "left_wrist_2_joint",
                "left_wrist_3_joint",
                "right_shoulder_pan_joint",
                "right_shoulder_lift_joint",
                "right_elbow_joint",
                "right_wrist_1_joint",
                "right_wrist_2_joint",
                "right_wrist_3_joint",
            ],
        )
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("keypoints_topic", "/demo_world_node/centroids_world")
        self.declare_parameter("rate_hz", 30.0)
        self.declare_parameter("mpc_output_mode", "position")  # or "velocity"
        self.declare_parameter("preview_points", 1)            # 1 is fine for most
        self.declare_parameter("dt_scale", 1.0)                # stretch/shrink dt used in goal points
        self.declare_parameter("goal_time_tolerance_sec", 0.05)
        self.declare_parameter("stop_with_zero_velocity", True)

        # Read params
        self._joint_state_topic: str = self.get_parameter("joint_state_topic").value
        self._keypoints_topic: str = self.get_parameter("keypoints_topic").value
        self._rate_hz: float = float(self.get_parameter("rate_hz").value)
        self._joints: List[str] = list(self.get_parameter("joints").value)
        self._preview_points: int = int(self.get_parameter("preview_points").value)
        self._dt_scale: float = float(self.get_parameter("dt_scale").value)
        self._goal_time_tol: float = float(self.get_parameter("goal_time_tolerance_sec").value)
        self._stop_with_zero_velocity: bool = bool(self.get_parameter("stop_with_zero_velocity").value)

        if self._rate_hz <= 0.0:
            self.get_logger().warn("rate_hz must be > 0; defaulting to 100.0")
            self._rate_hz = 100.0

        self._period_sec = 1.0 / self._rate_hz

        # --- Sub/Pub QoS ---
        js_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        keypoints_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # --- Subscriptions ---
        self._latest_js: Optional[JointState] = None
        self.create_subscription(JointState, self._joint_state_topic, self._on_joint_state, js_qos)

        self._latest_keypoints: np.ndarray = np.zeros((0, 3))
        self.create_subscription(PointCloud, self._keypoints_topic, self._on_keypoints, keypoints_qos)

        # --- Action client ---
        controller_name = self.get_parameter("left_controller_name").value + "/follow_joint_trajectory"
        self._left_action_client = ActionClient(self, FollowJointTrajectory, controller_name)
        self.get_logger().info(f"Waiting for action server on {controller_name}")
        self._left_action_client.wait_for_server()
        self.get_logger().info(f"Action server on {controller_name} is ready")

        controller_name = self.get_parameter("right_controller_name").value + "/follow_joint_trajectory"
        self._right_action_client = ActionClient(self, FollowJointTrajectory, controller_name)
        self.get_logger().info(f"Waiting for action server on {controller_name}")
        self._right_action_client.wait_for_server()
        self.get_logger().info(f"Action server on {controller_name} is ready")

        # --- Controller (replace with your real object) ---
        self.n_keypoints = 0
        self.goc_mpc = self._setup_goc_mpc()

        # --- Timing ---
        self._start_time = self.get_clock().now()
        self._timer = self.create_timer(self._period_sec, self._on_timer)

        # Track last goal handle (optional)
        self._last_goal_handle = None

        self.get_logger().info(
            f"Streaming JT goals at {self._rate_hz:.1f} Hz, "
            f"joints={self._joints}"
        )

    def _setup_goc_mpc(self):
        # , "ur5e_1"

        # env and visualization
        env = SimpleDrakeGym(["ur5e_0", "ur5e_1"], ["cube_0"])

        # see if this can be improved
        state_lower_bound = -100.0
        state_upper_bound = 100.0

        symbolic_plant = env.plant.ToSymbolic()

        graph = GraphOfConstraints(symbolic_plant, ["ur5e_0", "ur5e_1"], ["cube_0"],
                                   state_lower_bound, state_upper_bound)

        graph.structure.add_nodes(2)
        graph.structure.add_edge(0, 1, True)
        
        goal_position_1 = np.array([-1.57, -1.57, 1.34, -1.34, -1.57, 0.0, -1.57, -1.57, 1.34, -1.34, -1.57, 0.0])
        goal_position_2 = np.array([-1.00, -2.00, 2.00, -2.00, -2.00, 0.0, -1.57, -1.57, 1.34, -1.34, -1.57, 0.0])

        joint_agent_dim = graph.num_agents * graph.dim;

        phi0 = graph.add_agents_linear_eq(0, np.eye(joint_agent_dim), goal_position_1)
        graph.add_grasp_change(phi0, "grab", 0, 0);

        phi1 = graph.add_agents_linear_eq(1, np.eye(joint_agent_dim), goal_position_2)
        graph.add_grasp_change(phi1, "release", 0, 0);

        # save intended number of keypoints
        self.n_keypoints = graph.num_objects

        # GoC-MPC
        goc_mpc = GraphOfConstraintsMPC(graph,
                                        short_path_time_per_step = 0.2)
                                        # max_vel = 0.05,  # maximum velocity for every joint
                                        # max_acc = 0.05,  # maximum acceleration for every joint
                                        # max_jerk = 0.05) # maximum jerk for every joint
        return goc_mpc

    # --- Callbacks ---
    def _on_joint_state(self, msg: JointState):
        self._latest_js = msg

    def _on_keypoints(self, msg: PointCloud):
        self._latest_keypoints = np.array([(p.x, p.y, p.z) for p in msg.points])

    def _extract_state(self, js: JointState, kps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Reorder according to self._joints
        name_to_idx = {n: i for i, n in enumerate(js.name)}
        idxs = []
        for j in self._joints:
            if j not in name_to_idx:
                raise KeyError(f"Joint '{j}' not in JointState: {js.name}")
            idxs.append(name_to_idx[j])
        robot_x = np.asarray(js.position, dtype=float)[idxs]
        kp_x = kps[:self.n_keypoints].flatten()
        robot_x_dot = np.asarray(js.velocity, dtype=float)[idxs]
        kp_x_dot = np.zeros((self.n_keypoints, 3)).flatten()

        # x, x_dot
        x = np.concatenate((robot_x, kp_x))
        x_dot = np.concatenate((robot_x_dot, kp_x_dot))
        return x, x_dot

    def _on_timer(self):
        if self._latest_js is None:
            return
        if self._latest_keypoints is None:
            return

        now = self.get_clock().now()
        t = (now - self._start_time).nanoseconds * 1e-9

        try:
            x, x_dot = self._extract_state(self._latest_js, self._latest_keypoints)
        except Exception as e:
            self.get_logger().warn(f"Bad JointState: {e}")
            return

        # MPC step
        try:
            xi_h, vels_h, ts = self.goc_mpc.step(t, x, x_dot)
        except Exception as e:
            self.get_logger().error(f"goc_mpc.step failed: {e}")
            return

        self.get_logger().info(f"next waypoints in: {self.goc_mpc.timing_mpc.get_next_taus()}")

        left_traj, right_traj = self._make_tiny_trajectories(x, x_dot, xi_h, vels_h, ts)

        # Build a tiny trajectory goal based on output mode
        left_goal = FollowJointTrajectory.Goal()
        left_goal.trajectory = left_traj
        left_goal.goal_time_tolerance = RosDuration(sec=int(self._goal_time_tol), nanosec=int((self._goal_time_tol % 1.0) * 1e9))

        right_goal = FollowJointTrajectory.Goal()
        right_goal.trajectory = right_traj
        right_goal.goal_time_tolerance = RosDuration(sec=int(self._goal_time_tol), nanosec=int((self._goal_time_tol % 1.0) * 1e9))

        # Send; this will preempt the previous goal in the controller
        left_send_future = self._left_action_client.send_goal_async(left_goal, feedback_callback=self._on_feedback)
        left_send_future.add_done_callback(self._on_goal_sent)
        right_send_future = self._right_action_client.send_goal_async(right_goal, feedback_callback=self._on_feedback)
        right_send_future.add_done_callback(self._on_goal_sent)

    def _make_tiny_trajectories(self, x: np.ndarray, x_dot: np.ndarray, xi_h: np.ndarray, vels: np.ndarry, ts: np.ndarray) -> JointTrajectory:
        left_traj = JointTrajectory()
        left_traj.joint_names = list(self._joints[:6])
        right_traj = JointTrajectory()
        right_traj.joint_names = list(self._joints[6:])
        
        for i, traj in enumerate([left_traj, right_traj]):
            agent_xi_h = xi_h[:, i*6:(i+1)*6]
            agent_vels = vels[:, i*6:(i+1)*6]
            for j in range(len(xi_h)):
                p_j = JointTrajectoryPoint()

                # Interpret u as absolute desired positions at t+dt
                p_j.positions = agent_xi_h[j].tolist()
                p_j.velocities = agent_vels[j].tolist()
                p_j.time_from_start = RosDuration(sec=int(ts[j]), nanosec=int((ts[j] % 1.0) * 1e9))
                traj.points.append(p_j)

        return left_traj, right_traj

    # --- Action plumbing (optional logging) ---
    def _on_feedback(self, feedback: FollowJointTrajectory.Feedback):
        # You can inspect tracking error here if desired
        # self.get_logger().warn("Got feedback: " + str(feedback))
        pass

    def _on_goal_sent(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Trajectory goal rejected by controller.")
            return
        self._last_goal_handle = goal_handle
        # Optionally check the result to detect controller issues (don’t block control loop)
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_result)

    def _on_result(self, future):
        # You can log errors or status codes if needed
        # _ = future.result()
        # self.get_logger().warn("Got result: " + str(result))
        pass


def main(args=None):
    rclpy.init(args=args)
    node = GocMpcFollowJTNode()

    # Replace stub with your real controller before spinning
    # from your_pkg.goc_mpc import GocMpc
    # node.goc_mpc = GocMpc(...)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


# import time
# import random
# import numpy as np
# import pybullet as p
# from importlib.resources import files

# import rclpy
# from rclpy.node import Node
# from rclpy.action import ActionClient

# from std_msgs.msg import String
# from builtin_interfaces.msg import Duration
# from action_msgs.msg import GoalStatus
# from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
# from control_msgs.action import FollowJointTrajectory
# from control_msgs.msg import JointTolerance
# from sensor_msgs.msg import JointState
# from std_msgs.msg import Float32MultiArray

# from corallab_lib.backends.pybullet.utils import setup_basic
# import corallab_lib.backends.pybullet.robots.ur5_robotiq85 as ur5

# from toposort import toposort

# from goc_mpc.goc_mpc import GraphOfConstraints, GraphOfConstraintsMPC
# # from goc_mpc.utils.mesh_cat_mirror import MeshCatMirror

# # from goc_demo import robotiq
# # from scl_demo.generation.common import Camera
# # from scl_demo.generation.target import (
# #     create_target_scene_loader,
# #     create_target_scene_loader_from_csv
# # )
# # from scl_demo.generation.initial import (
# #     create_fixed_initial_scene_loader,
# # )
# # from scl_demo.utilities.dependency_graph import dependence, dep_graph, dep_dict
# # from scl_demo.structural_planner import (
# #     plan_rearrangement,
# #     plan_reverse_rearrangement
# # )


# import goc_demo




# class GocMpcNode(Node):
#     """
#     A ROS2 node that:
#       - Listens to JointState for x (positions) and x_dot (velocities)
#       - Calls self.goc_mpc.step(elapsed_time, x, x_dot)
#       - Publishes the controller output as Float64MultiArray
#     """

#     def __init__(self):
#         super().__init__('goc_mpc_node')

#         # --- Parameters ---
#         self.declare_parameter("controller_name", "scaled_joint_trajectory_controller")
#         self.declare_parameter(
#             "joints",
#             [
#                 "shoulder_pan_joint",
#                 "shoulder_lift_joint",
#                 "elbow_joint",
#                 "wrist_1_joint",
#                 "wrist_2_joint",
#                 "wrist_3_joint",
#             ],
#         )
#         self.declare_parameter('joint_state_topic', '/joint_states')
#         self.declare_parameter('command_topic', '/goc_mpc/command')
#         self.declare_parameter('rate_hz', 10.0)  # control loop frequency
#         self.declare_parameter('joint_names', [])  # optional ordering filter

#         # action server
#         controller_name = self.get_parameter("controller_name").value + "/follow_joint_trajectory"
#         self.joints = self.get_parameter("joints").value

#         if self.joints is None or len(self.joints) == 0:
#             raise Exception('"joints" parameter is required')




#         joint_state_topic: str = self.get_parameter('joint_state_topic').get_parameter_value().string_value
#         command_topic: str = self.get_parameter('command_topic').get_parameter_value().string_value
#         rate_hz: float = float(self.get_parameter('rate_hz').get_parameter_value().double_value)
#         joint_names_param = self.get_parameter('joint_names').get_parameter_value().string_array_value
#         self._desired_joint_order: Optional[List[str]] = list(joint_names_param) if joint_names_param else None

#         if rate_hz <= 0.0:
#             self.get_logger().warn("rate_hz must be > 0; defaulting to 100.0")
#             rate_hz = 100.0
#         self._period_sec = 1.0 / rate_hz

#         # QoS suited to joint states (best-effort, keep last few)
#         js_qos = QoSProfile(
#             reliability=ReliabilityPolicy.BEST_EFFORT,
#             history=HistoryPolicy.KEEP_LAST,
#             depth=10,
#         )
#         cmd_qos = QoSProfile(
#             reliability=ReliabilityPolicy.RELIABLE,
#             history=HistoryPolicy.KEEP_LAST,
#             depth=10,
#         )

#         # --- I/O ---
#         self._js_sub = self.create_subscription(JointState, joint_state_topic, self._on_joint_state, js_qos)
#         self._cmd_pub = self.create_publisher(Float64MultiArray, command_topic, cmd_qos)

#         # --- State ---
#         self._latest_js: Optional[JointState] = None
#         self._start_time = self.get_clock().now()

#         # You should set/replace this with your actual controller instance before spinning,
#         # or subclass this node and set it in __init__.
#         self.goc_mpc = self._require_goc_mpc()

#         # Timer for control loop
#         self._timer = self.create_timer(self._period_sec, self._on_timer)

#         self.get_logger().info(
#             f"goc_mpc_node up. "
#             f"Subscribing: {joint_state_topic}, Publishing: {command_topic}, rate: {rate_hz} Hz, "
#             f"ordered joints: {self._desired_joint_order if self._desired_joint_order else 'as received'}"
#         )

#     # --- You can replace this with your real controller wiring ---
#     def _require_goc_mpc(self):
#         """
#         Replace this stub with your real controller object.
#         It must expose: step(elapsed_time: float, x: np.ndarray, x_dot: np.ndarray) -> Union[np.ndarray, Sequence[float]]
#         """
#         class _Stub:
#             def step(self, t: float, x: np.ndarray, x_dot: np.ndarray) -> np.ndarray:
#                 # Example: dumb PD -> command = -Kp*x - Kd*x_dot (for demo only)
#                 kp, kd = 1.0, 0.1
#                 return -kp * x - kd * x_dot
#         self.get_logger().warn("Using a stub goc_mpc. Replace self.goc_mpc with your real controller.")
#         return _Stub()
#     # -------------------------------------------------------------

#     def _on_joint_state(self, msg: JointState):
#         self._latest_js = msg

#     def _extract_x_xdot(self, js: JointState) -> Tuple[np.ndarray, np.ndarray, List[str]]:
#         names = list(js.name) if js.name else []
#         pos = np.array(js.position, dtype=float) if js.position else np.array([], dtype=float)
#         vel = np.array(js.velocity, dtype=float) if js.velocity else np.array([], dtype=float)

#         if self._desired_joint_order:
#             # Reorder (and filter) according to desired list
#             name_to_index = {n: i for i, n in enumerate(names)}
#             idxs = []
#             for jn in self._desired_joint_order:
#                 if jn not in name_to_index:
#                     raise KeyError(f"Desired joint '{jn}' not found in JointState: {names}")
#                 idxs.append(name_to_index[jn])
#             names = [names[i] for i in idxs]
#             pos = pos[idxs] if pos.size else pos
#             vel = vel[idxs] if vel.size else vel

#         if pos.size == 0 or vel.size == 0:
#             raise ValueError("JointState missing positions or velocities.")
#         if pos.shape != vel.shape:
#             raise ValueError(f"Position/velocity size mismatch: {pos.shape} vs {vel.shape}")

#         return pos, vel, names

#     def _on_timer(self):
#         if self._latest_js is None:
#             return  # wait for first joint state

#         # elapsed time since node start (seconds)
#         now = self.get_clock().now()
#         elapsed: Duration = now - self._start_time
#         t = elapsed.nanoseconds * 1e-9

#         try:
#             x, x_dot, names = self._extract_x_xdot(self._latest_js)
#         except Exception as e:
#             self.get_logger().throttle_warn(5.0, f"Cannot extract x/x_dot from JointState: {e}")
#             return

#         # --- Call controller ---
#         try:
#             u = self.goc_mpc.step(t, x, x_dot)
#         except Exception as e:
#             self.get_logger().error(f"goc_mpc.step failed: {e}")
#             return

#         # Normalize to 1D numpy array
#         if isinstance(u, (list, tuple)):
#             u = np.asarray(u, dtype=float)
#         elif not isinstance(u, np.ndarray):
#             try:
#                 u = np.asarray(u, dtype=float)
#             except Exception:
#                 self.get_logger().error(f"Controller output has unsupported type: {type(u)}")
#                 return

#         if u.ndim > 1:
#             u = u.reshape(-1)
#         if u.size != x.size:
#             self.get_logger().warn(
#                 f"Controller output length ({u.size}) != joint count ({x.size}); publishing anyway."
#             )

#         # --- Publish Float64MultiArray ---
#         msg = Float64MultiArray()
#         msg.data = u.astype(float).tolist()
#         # Optional: encode joint names and shape in layout
#         dim0 = MultiArrayDimension()
#         dim0.label = 'joints'
#         dim0.size = int(u.size)
#         dim0.stride = int(u.size)
#         msg.layout = MultiArrayLayout(dim=[dim0], data_offset=0)

#         self._cmd_pub.publish(msg)

#     # Utility for throttled warnings (rclpy Node has no throttle_* helpers by default)
#     def get_logger(self):
#         # Add simple throttling wrapper
#         logger = super().get_logger()
#         if not hasattr(logger, 'throttle_warn_state'):
#             logger.throttle_warn_state = {}
#         def throttle_warn(period_sec: float, text: str):
#             key = ('warn', text)
#             last = logger.throttle_warn_state.get(key, None)
#             now = rclpy.clock.Clock().now().nanoseconds * 1e-9
#             if last is None or (now - last) >= period_sec:
#                 logger.warn(text)
#                 logger.throttle_warn_state[key] = now
#         logger.throttle_warn = throttle_warn
#         return logger


# def main(args=None):
#     rclpy.init(args=args)
#     node = GocMpcNode()

#     # Example: REPLACE the stub with your real controller object before spinning:
#     # from your_pkg.goc_mpc import GocMpc
#     # node.goc_mpc = GocMpc(...)

#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()


# if __name__ == '__main__':
#     main()


# # class GoCDemoNode(Node):

# #     def __init__(self):
# #         super().__init__('goc_demo')


# #         self.position = None

#         self._action_client = ActionClient(self, FollowJointTrajectory, controller_name)
#         self.get_logger().info(f"Waiting for action server on {controller_name}")
#         self._action_client.wait_for_server()
#         self.get_logger().info(f"Action server on {controller_name} is ready")

# #         # Create a publisher for the command topic
# #         self.command_publisher = self.create_publisher(
# #             Float32MultiArray,
# #             'goc_mpc_short_path',  # Replace with your actual command topic name
# #             10  # QoS profile depth
# #         )


# #         # instatiate real gripper thing
# #         # ip_address = "10.164.8.235"
# #         # self.real_gripper = robotiq.RobotiqGripper(disabled=False)
# #         # self.real_gripper.connect(ip_address, 63352)
# #         # self.real_gripper.activate()

# #         # self.real_gripper.open()

# #         # setup goc mpc
# #         self.goc_mpc = self._setup_goc_mpc()

# #         # make trajectories from plan
# #         # trajs = self._make_trajectories(plan)

# #         # execute plan
# #         # self._send_goal_future = None
# #         # self._get_result_future = None
# #         self._execute()

# #     # def clear_obstacles(self):
# #     #     for obstacle in self.obstacles:
# #     #         p.removeBody(obstacle)

# #     def joint_state_callback(self, msg):
# #         # self.get_logger().info(f'Received Joint States: {msg.position}')
# #         self.position = msg.position

# #     def _setup_goc_mpc(self):
# #         # problem set-up
# #         num_agents = 1
# #         dim = 6
# #         state_lower_bound = np.ones(dim) * -100.0
# #         state_upper_bound = np.ones(dim) * 100.0

# #         graph = GraphOfConstraints(num_agents, dim, state_lower_bound, state_upper_bound)
# #         graph.structure.add_nodes(2)
# #         graph.structure.add_edge(0, 1, True)

# #         goal_position_1 = np.array([-1.57, -1.57, 1.34, -1.34, -1.57, 0.0])
# #         goal_position_2 = np.array([-1.00, -2.00, 2.00, -2.00, -2.00, 0.0])

# #         graph.add_linear_eq(0, np.eye(6), goal_position_1)
# #         graph.add_linear_eq(1, np.eye(6), goal_position_2)

# #         # GoC-MPC
# #         goc_mpc = GraphOfConstraintsMPC(graph,
# #                                         short_path_time_per_step = 0.1)
# #         # max_vel = 0.1,  # maximum velocity for every joint
# #         # max_acc = 0.1,  # maximum acceleration for every joint
# #         # max_jerk = 0.1) # maximum jerk for every joint
# #         return goc_mpc




# #     def _execute(self):

# #         dt = 1.0 / 30.0

# #         while not self.position:
# #             print("not received msg yet")
# #             time.sleep(1.0)

# #         for k in range(1500):
# #             print(self.position)

# #             # xi_h, ts = goc_mpc.step(k * dt, x, x_dot)

# #             # self._execute_trajectory(xi_h, ts)

# #             # obs, rew, done, trunc, info = env.step(qpos)


# #         # resp = input("Repeat?")
# #         # if resp == 'q':
# #         #     break



# #     # def _execute_trajectories(self, trajs):
# #     #     for i, (traj, grip) in enumerate(trajs):
# #     #         if grip == "close":
# #     #             self.real_gripper.open()

# #     #         self.get_logger().info(f"Starting traj execution #{i}")
# #     #         result = self._execute_trajectory(traj)
# #     #         self.get_logger().info(f"GRIP: {grip}")

# #     #         time.sleep(2.1)

# #     #         if grip is None:
# #     #             pass
# #     #         elif grip == "open":
# #     #             self.real_gripper.slightly_open()
# #     #         elif grip == "close":
# #     #             self.real_gripper.close()

# #     # def _execute_trajectory(self, traj):
# #     #     self.get_logger().info(f"Executing trajectory")
# #     #     goal = FollowJointTrajectory.Goal()
# #     #     goal.trajectory = traj

# #     #     goal.goal_time_tolerance = Duration(sec=0, nanosec=750000000)
# #     #     goal.goal_tolerance = [
# #     #         JointTolerance(position=0.005, velocity=0.01, name=self.joints[i]) for i in range(6)
# #     #     ]

# #     #     self._send_goal_future = self._action_client.send_goal_async(goal)
# #     #     self._send_goal_future.add_done_callback(self._goal_response_callback)

# #     # def _goal_response_callback(self, future):
# #     #     goal_handle = future.result()
# #     #     if not goal_handle.accepted:
# #     #         self.get_logger().error("Goal rejected :(")
# #     #         raise RuntimeError("Goal rejected :(")

# #     #     self.get_logger().debug("Goal accepted :)")

# #     #     self._get_result_future = goal_handle.get_result_async()
# #     #     self._get_result_future.add_done_callback(self._get_result_callback)

# #     # def _get_result_callback(self, future):
# #     #     result = future.result().result
# #     #     status = future.result().status
# #     #     self.get_logger().info(f"Done with result: {self.status_to_str(status)}")
# #     #     if status == GoalStatus.STATUS_SUCCEEDED:
# #     #         time.sleep(0.5)
# #     #     else:
# #     #         if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
# #     #             self.get_logger().error(
# #     #                 f"Done with result: {self.error_code_to_str(result.error_code)}"
# #     #             )
# #     #         raise RuntimeError("Executing trajectory failed. " + result.error_string)

# #     @staticmethod
# #     def error_code_to_str(error_code):
# #         if error_code == FollowJointTrajectory.Result.SUCCESSFUL:
# #             return "SUCCESSFUL"
# #         if error_code == FollowJointTrajectory.Result.INVALID_GOAL:
# #             return "INVALID_GOAL"
# #         if error_code == FollowJointTrajectory.Result.INVALID_JOINTS:
# #             return "INVALID_JOINTS"
# #         if error_code == FollowJointTrajectory.Result.OLD_HEADER_TIMESTAMP:
# #             return "OLD_HEADER_TIMESTAMP"
# #         if error_code == FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED:
# #             return "PATH_TOLERANCE_VIOLATED"
# #         if error_code == FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED:
# #             return "GOAL_TOLERANCE_VIOLATED"

# #     @staticmethod
# #     def status_to_str(error_code):
# #         if error_code == GoalStatus.STATUS_UNKNOWN:
# #             return "UNKNOWN"
# #         if error_code == GoalStatus.STATUS_ACCEPTED:
# #             return "ACCEPTED"
# #         if error_code == GoalStatus.STATUS_EXECUTING:
# #             return "EXECUTING"
# #         if error_code == GoalStatus.STATUS_CANCELING:
# #             return "CANCELING"
# #         if error_code == GoalStatus.STATUS_SUCCEEDED:
# #             return "SUCCEEDED"
# #         if error_code == GoalStatus.STATUS_CANCELED:
# #             return "CANCELED"
# #         if error_code == GoalStatus.STATUS_ABORTED:
# #             return "ABORTED"


# # def main(args=None):
# #     rclpy.init(args=args)

# #     goc_demo_node = GoCDemoNode()

# #     try:
# #         rclpy.spin(goc_demo_node)
# #     except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
# #         print("Keyboard interrupt received. Shutting down node.")
# #     except Exception as e:
# #         print(f"Unhandled exception: {e}")

# #     # Destroy the node explicitly
# #     # (optional - otherwise it will be done automatically
# #     # when the garbage collector destroys the node object)
# #     goc_demo_node.destroy_node()
# #     rclpy.shutdown()


# # if __name__ == '__main__':
# #     main()
