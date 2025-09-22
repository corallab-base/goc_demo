import os
import time
import imageio
import numpy as np
import mujoco as mj
import matplotlib.pyplot as plt

from mujoco import viewer

from pydrake.math import RollPitchYaw
from pydrake.geometry import Meshcat
from pydrake.common.eigen_geometry import Quaternion

from goc_mpc.splines import Block
from goc_mpc.systems import OnePointMassEnv
from goc_mpc.goc_mpc import GraphOfConstraints, GraphOfConstraintsMPC
from goc_mpc.utils.mesh_cat_mirror import MeshCatMirror
from goc_mpc.simple_drake_env import SimpleDrakeGym


TIME_DELTA_CUTOFF = 0.20
PHI_TOLERANCE = 0.03


# def two_gripper_simple_motion(n_points=3, quat=np.array([0.0, -0.70710678,  0.70710678, 0.0])):
#     env = SimpleDrakeGym(["free_body_0", "free_body_1"], [f"cube_{i}" for i in range(n_points)])

#     state_lower_bound = -10.0
#     state_upper_bound =  10.0

#     symbolic_plant = env.plant.ToSymbolic()
#     graph = GraphOfConstraints(symbolic_plant, ["free_body_0", "free_body_1"], [f"cube_{i}" for i in range(n_points)],
#                                state_lower_bound, state_upper_bound)

#     graph.structure.add_nodes(1)

#     phi0 = graph.add_robot_to_point_displacement_constraint(0, 0, 0, np.array([0.0, 0.0, -0.2]));
#     graph.add_robot_quat_linear_eq(0, 0, np.eye(4), quat)

#     # GoC-MPC
#     spline_spec = [Block.R(3), Block.SO3()]
#     goc_mpc = GraphOfConstraintsMPC(graph, spline_spec, short_path_time_per_step = 0.1,
#                                     solve_for_waypoints_once = False,
#                                     time_delta_cutoff = TIME_DELTA_CUTOFF,
#                                     phi_tolerance = PHI_TOLERANCE)
#     return env, graph, goc_mpc


# def two_gripper_block_stacking(n_points=3, quat=np.array([ 0.0,  0.0, -0.70710678,  0.70710678])):
#     env = SimpleDrakeGym(["free_body_0", "free_body_1"], [f"cube_{i}" for i in range(n_points)])

#     state_lower_bound = -10.0
#     state_upper_bound =  10.0

#     symbolic_plant = env.plant.ToSymbolic()
#     graph = GraphOfConstraints(symbolic_plant, ["free_body_0", "free_body_1"], [f"cube_{i}" for i in range(n_points)],
#                                state_lower_bound, state_upper_bound)

#     graph.structure.add_nodes(5)
#     graph.structure.add_edge(0, 1, True)
#     graph.structure.add_edge(2, 3, True)
#     graph.structure.add_edge(1, 3, True)
#     graph.structure.add_edge(1, 4, True)

#     phi0 = graph.add_robot_to_point_displacement_constraint(0, 0, 0, np.array([0.0, 0.0, -0.1]));
#     graph.add_robot_quat_linear_eq(0, 0, np.eye(4), quat)
#     graph.add_grasp_change(phi0, "grab", 0, 0);

#     graspPhi0 = graph.add_robot_holding_cube_constraint(0, 1, 0, 0, 0.1);

#     phi1 = graph.add_robot_to_point_displacement_constraint(1, 0, 1, np.array([0.0, 0.0, -0.2]));
#     graph.add_robot_quat_linear_eq(1, 0, np.eye(4), quat)
#     graph.add_grasp_change(phi1, "release", 0, 0);

#     phi2 = graph.add_robot_to_point_displacement_constraint(2, 1, 2, np.array([0.0, 0.0, -0.1]));
#     graph.add_robot_quat_linear_eq(2, 1, np.eye(4), quat)
#     graph.add_grasp_change(phi2, "grab", 1, 2);

#     graspPhi1 = graph.add_robot_holding_cube_constraint(2, 3, 1, 2, 0.1);

#     phi3 = graph.add_robot_to_point_displacement_constraint(3, 1, 0, np.array([0.0, 0.0, -0.2]));
#     graph.add_robot_quat_linear_eq(3, 1, np.eye(4), quat)
#     graph.add_grasp_change(phi3, "release", 1, 2);

#     phi4 = graph.add_robot_to_point_displacement_constraint(4, 0, 1, np.array([0.0, 0.0, -0.5]));
#     graph.add_robot_quat_linear_eq(4, 0, np.eye(4), quat)

#     # GoC-MPC
#     spline_spec = [Block.R(3), Block.SO3()]
#     goc_mpc = GraphOfConstraintsMPC(graph, spline_spec, short_path_time_per_step = 0.1,
#                                     solve_for_waypoints_once = False,
#                                     time_delta_cutoff = TIME_DELTA_CUTOFF,
#                                     phi_tolerance = PHI_TOLERANCE)
#     return env, graph, goc_mpc


# def two_gripper_pick_and_pour(n_points=3):
#     env = SimpleDrakeGym(["free_body_0", "free_body_1"], [f"cube_{i}" for i in range(n_points)])

#     state_lower_bound = -10.0
#     state_upper_bound =  10.0

#     symbolic_plant = env.plant.ToSymbolic()
#     graph = GraphOfConstraints(symbolic_plant, ["free_body_0", "free_body_1"], [f"cube_{i}" for i in range(n_points)],
#                                state_lower_bound, state_upper_bound)
#     graph.structure.add_nodes(5)
#     graph.structure.add_edge(0, 1, True)
#     graph.structure.add_edge(0, 2, True)
#     graph.structure.add_edge(1, 3, True)
#     graph.structure.add_edge(2, 3, True)
#     graph.structure.add_edge(3, 4, True)

#     joint_agent_dim = graph.num_agents * graph.dim;
#     home_position_1 = np.array([-0.30, -0.30, 1.0, 0.0, 1.0, 0.0, 0.0,
#                                 -0.30, 0.30, 1.0, 0.0, 0.0, -1.0, 0.0])
#     graph.add_robots_linear_eq(0, np.eye(joint_agent_dim), home_position_1)

#     # PICK UP PAPER CUP AT ANGLE FROM SIDE
#     graph.add_robot_to_point_alignment_cost(1, 0, 0, np.array([0.0, 1.0, 1.0]),
#                                             u_body_opt=np.array([1.0, 0.0, 0.0]),
#                                             roll_ref_flat=True,
#                                             w_flat=1.0)
#     # phi2 = graph.add_robot_to_point_displacement_constraint(1, 0, 0, np.array([0.0, 0.10, -0.03]));
#     phi2 = graph.add_robot_to_point_displacement_constraint(1, 0, 0, np.array([0.0, 0.05, -0.09]));
#     graph.add_grasp_change(phi2, "grab", 0, 0);

#     graspPhi0 = graph.add_robot_holding_cube_constraint(1, 3, 0, 0, 0.25);
#     graph.add_robot_relative_rotation_constraint(1, 3, 0, RollPitchYaw(0.0, 0.0, 0.0).ToQuaternion());

#     # PICK UP COFFEE CUP AT ANGLE FROM SIDE
#     graph.add_robot_to_point_alignment_cost(2, 1, 1, np.array([0.0, 0.0, 1.0]),
#                                             u_body_opt=np.array([1.0, 0.0, 0.0]),
#                                             roll_ref_flat=True,
#                                             w_flat=1.0)
#     phi4 = graph.add_robot_to_point_displacement_cost(2, 1, 1, np.array([0.0, -0.15, -0.04]))
#     graph.add_grasp_change(phi4, "grab", 1, 1);

#     graspPhi1 = graph.add_robot_holding_cube_constraint(2, 3, 1, 1, 0.25);
#     graph.add_robot_relative_rotation_constraint(2, 3, 1, RollPitchYaw(0.0, 0.0, 0.0).ToQuaternion());

#     # # BRING PITCHER AND CUP CLOSE TO EACH OTHER
#     # graph.add_robot_to_point_alignment_cost(2, 1, 1, np.array([0.0, 0.0, 1.0]),
#     #                                         u_body_opt=np.array([1.0, 0.0, 0.0]),
#     #                                         roll_ref_flat=True)
#     # graph.add_robot_to_point_displacement_cost(3, 1, 1, np.array([0.0, -0.25, -0.25]));

#     # graph.add_robot_to_point_displacement_cost(1, 0, 0, np.array([0.05, 0.0, -0.05]));
#     # graph.add_robot_to_point_displacement_cost(2, 1, 1, np.array([-0.05, 0.0, -0.05]));
#     graph.add_point_to_point_displacement_cost(3, 0, 1, np.array([0.0, 0.0, -0.1]));
#     graph.add_point_linear_eq(3, 0, np.array([[0.0, 0.0, 0.0],
#                                               [0.0, 0.0, 0.0],
#                                               [0.0, 0.0, 1.0]]), np.array([0.0, 0.0, 1.0]))

#     # # POUR PITCHER
#     graph.add_robot_holding_cube_constraint(3, 4, 0, 0, 0.25);
#     graph.add_robot_holding_cube_constraint(3, 4, 1, 1, 0.25);
#     graph.add_robot_relative_displacement_constraint(3, 4, 1, np.array([0.0, 0.0, 0.0]));
#     graph.add_point_to_point_displacement_cost(4, 0, 1, np.array([0.0, 0.01, -0.1]));
#     graph.add_robot_relative_rotation_constraint(3, 4, 0,
#                                                  RollPitchYaw(-np.pi/3, 0.0, 0.0).ToQuaternion());
#     graph.make_node_unpassable(4)

#     # GoC-MPC
#     spline_spec = [Block.R(3), Block.SO3()]
#     goc_mpc = GraphOfConstraintsMPC(graph, spline_spec, short_path_time_per_step = 0.1,
#                                     solve_for_waypoints_once = False,
#                                     time_delta_cutoff = TIME_DELTA_CUTOFF,
#                                     phi_tolerance = PHI_TOLERANCE)
#     return env, graph, goc_mpc


# def two_gripper_folding(n_points=4):
#     env = SimpleDrakeGym(["free_body_0", "free_body_1"], [f"cube_{i}" for i in range(n_points)])

#     state_lower_bound = -10.0
#     state_upper_bound =  10.0

#     symbolic_plant = env.plant.ToSymbolic()
#     graph = GraphOfConstraints(symbolic_plant, ["free_body_0", "free_body_1"], [f"cube_{i}" for i in range(n_points)],
#                                state_lower_bound, state_upper_bound)

#     graph.structure.add_nodes(4)
#     graph.structure.add_edge(0, 1, True)
#     graph.structure.add_edge(2, 3, True)

#     r1 = graph.add_variable();
#     r2 = graph.add_variable();
#     right_sleeve = 0
#     left_sleeve = 1

#     phi0 = graph.add_assignable_robot_to_point_displacement_constraint(0, r1, right_sleeve, np.array([0.0, 0.0, -0.2]));
#     # graph.add_robot_quat_linear_eq(0, 0, np.eye(4), quat)
#     # graph.add_grasp_change(phi0, "grab", 0, 0);

#     # graspPhi0 = graph.add_robot_holding_cube_constraint(0, 1, 0, 0, 0.1);

#     # phi1 = graph.add_robot_to_point_displacement_constraint(1, 0, 1, np.array([0.0, 0.0, -0.2]));
#     # graph.add_robot_quat_linear_eq(1, 0, np.eye(4), quat)
#     # graph.add_grasp_change(phi1, "release", 0, 0);

#     phi2 = graph.add_assignable_robot_to_point_displacement_constraint(2, r2, left_sleeve, np.array([0.0, 0.0, -0.2]));
#     # graph.add_robot_quat_linear_eq(2, 1, np.eye(4), quat)
#     # graph.add_grasp_change(phi2, "grab", 1, 2);

#     # graspPhi1 = graph.add_robot_holding_cube_constraint(2, 3, 1, 2, 0.1);

#     # phi3 = graph.add_robot_to_point_displacement_constraint(3, 1, 0, np.array([0.0, 0.0, -0.2]));
#     # graph.add_robot_quat_linear_eq(3, 1, np.eye(4), quat)
#     # graph.add_grasp_change(phi3, "release", 1, 2);

#     # phi4 = graph.add_robot_to_point_displacement_constraint(4, 0, 1, np.array([0.0, 0.0, -0.5]));
#     # graph.add_robot_quat_linear_eq(4, 0, np.eye(4), quat)


#     # phi0 = graph.add_robot_to_point_displacement_constraint(0, 0, 0, np.array([0.0, 0.0, -0.1]));
#     # graph.add_robot_quat_linear_eq(0, 0, np.eye(4), np.array([0.0, 0.0, 1.0, 0.0]))
#     # graph.add_grasp_change(phi0, "grab", 0, 0);

#     # graspPhi0 = graph.add_robot_holding_cube_constraint(0, 1, 0, 0, 0.1);

#     # phi1 = graph.add_robot_to_point_displacement_constraint(1, 0, 1, np.array([0.0, 0.0, -0.2]));
#     # graph.add_robot_quat_linear_eq(1, 0, np.eye(4), np.array([0.0, 0.0, 1.0, 0.0]))
#     # graph.add_grasp_change(phi1, "release", 0, 0);

#     # phi2 = graph.add_robot_to_point_displacement_constraint(2, 1, 2, np.array([0.0, 0.0, -0.1]));
#     # graph.add_robot_quat_linear_eq(2, 1, np.eye(4), np.array([0.0, 0.0, 1.0, 0.0]))
#     # graph.add_grasp_change(phi2, "grab", 1, 2);

#     # graspPhi1 = graph.add_robot_holding_cube_constraint(2, 3, 1, 2, 0.1);

#     # phi3 = graph.add_robot_to_point_displacement_constraint(3, 1, 0, np.array([0.0, 0.0, -0.2]));
#     # graph.add_robot_quat_linear_eq(3, 1, np.eye(4), np.array([0.0, 0.0, 1.0, 0.0]))
#     # graph.add_grasp_change(phi3, "release", 1, 2);

#     # phi4 = graph.add_robot_to_point_displacement_constraint(4, 0, 1, np.array([0.0, 0.0, -0.5]));
#     # graph.add_robot_quat_linear_eq(4, 0, np.eye(4), np.array([0.0, 0.0, 1.0, 0.0]))

#     # GoC-MPC
#     spline_spec = [Block.R(3), Block.SO3()]
#     goc_mpc = GraphOfConstraintsMPC(graph, spline_spec, short_path_time_per_step = 0.1,
#                                     solve_for_waypoints_once = False,
#                                     time_delta_cutoff = TIME_DELTA_CUTOFF,
#                                     phi_tolerance = PHI_TOLERANCE)
#     return env, graph, goc_mpc


# def two_gripper_assignable_move(n_points=3):
#     env = SimpleDrakeGym(["free_body_0", "free_body_1"], [f"cube_{i}" for i in range(n_points)])

#     state_lower_bound = -10.0
#     state_upper_bound =  10.0

#     symbolic_plant = env.plant.ToSymbolic()
#     graph = GraphOfConstraints(symbolic_plant, ["free_body_0", "free_body_1"], [f"cube_{i}" for i in range(n_points)],
#                                state_lower_bound, state_upper_bound)

#     graph.structure.add_nodes(2)
#     graph.structure.add_edge(0, 1, True)

#     r1 = graph.add_variable();
#     cube = 1

#     phi0 = graph.add_assignable_robot_to_point_displacement_constraint(0, r1, cube, np.array([0.0, 0.0, -0.1]))
#     graph.add_assignable_robot_quat_linear_eq(0, r1, np.eye(4), np.array([0.0, 0.0, 1.0, 0.0]))
#     graph.add_assignable_grasp_change(phi0, "grab", cube)

#     graspPhi0 = graph.add_assignable_robot_holding_point_constraint(0, 1, r1, cube, 0.1)

#     graph.add_assignable_robot_quat_linear_eq(1, r1, np.eye(4), np.array([0.0, 0.0, 1.0, 0.0]))
#     graph.add_point_linear_eq(1, cube, np.eye(3), np.array([0.0, 0.0, 0.1]))

#     # GoC-MPC
#     spline_spec = [Block.R(3), Block.SO3()]
#     goc_mpc = GraphOfConstraintsMPC(graph, spline_spec, short_path_time_per_step = 0.1,
#                                     solve_for_waypoints_once = True,
#                                     time_delta_cutoff = TIME_DELTA_CUTOFF,
#                                     phi_tolerance = PHI_TOLERANCE)
#     return env, graph, goc_mpc


def do_move_in_circles(graph):
    joint_agent_dim = graph.num_agents * graph.dim;

    graph.structure.add_nodes(3)
    graph.structure.add_edge(0, 1, True)
    graph.structure.add_edge(1, 2, True)

    goal_position_1 = np.array([0.30, 0.0, 0.3, 0.0, 0.0, 1.0, 0.0,
                                -0.30, 0.0, 0.3, 0.0, 0.0, 1.0, 0.0])
    phi0 = graph.add_robots_linear_eq(0, np.eye(joint_agent_dim), goal_position_1)
    # graph.add_grasp_change(phi1, "release", 0, 0);

    goal_position_2 = np.array([0.50, 0.0, 0.3, 0.0, 0.0, 1.0, 0.0,
                                -0.50, 0.0, 0.3, 0.0, 0.0, 1.0, 0.0])
    phi1 = graph.add_robots_linear_eq(1, np.eye(joint_agent_dim), goal_position_2)
    # graph.add_grasp_change(phi0, "grab", 0, 0);

    home_position_1 = np.array([0.40, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0,
                                -0.40, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0])
    # home_position_2 = np.array([0.40, 0.0, 0.4, 0.5, 0.5, -0.5, -0.5,
    #                             -0.40, 0.0, 0.4, 0.5, 0.5, -0.5, -0.5])
    # phi0 = graph.add_robots_linear_eq(0, np.eye(joint_agent_dim), home_position_1)
    phi2 = graph.add_robots_linear_eq(2, np.eye(joint_agent_dim), home_position_1)


def do_track_above(graph):
    joint_agent_dim = graph.num_agents * graph.dim;

    graph.structure.add_nodes(3)
    graph.structure.add_edge(0, 1, True)
    graph.structure.add_edge(0, 2, True)

    home_position_1 = np.array([0.30, -0.2, 0.5, 0.0, 0.0, 1.0, 0.0,
                                -0.30, -0.2, 0.5, 0.0, 0.0, 1.0, 0.0])
    phi0 = graph.add_robots_linear_eq(0, np.eye(joint_agent_dim), home_position_1)

    phi1 = graph.add_robot_above_cube_constraint(1, 0, 0, 0.2)
    phi2 = graph.add_robot_quat_linear_eq(1, 0, np.eye(4), np.array([0.0, 0.0, 1.0, 0.0]))
    graph.make_node_unpassable(1)

    phi3 = graph.add_robot_above_cube_constraint(2, 1, 1, 0.2)
    phi4 = graph.add_robot_quat_linear_eq(2, 1, np.eye(4), np.array([0.0, 0.0, 1.0, 0.0]))
    graph.make_node_unpassable(2)


def do_pick_and_pour(graph):
    joint_agent_dim = graph.num_agents * graph.dim;

    # RESET
    start_node = graph.structure.add_node()
    joint_agent_dim = graph.num_agents * graph.dim;
    home_position_1 = np.array([0.30, -0.2, 0.5, 0.0, -0.70701, -0.70701, 0.0,
                                -0.30, -0.2, 0.5, 0.0, 0.70701, -0.70701, 0.0])
    graph.add_robots_linear_eq(start_node, np.eye(joint_agent_dim), home_position_1)

    # PITCHER
    pitcher_approach, pitcher_pick_up = graph.structure.add_nodes(2)
    graph.structure.add_edge(start_node, pitcher_approach, True)
    graph.structure.add_edge(pitcher_approach, pitcher_pick_up, True)

    # graph.add_robot_to_point_alignment_cost(pitcher_approach,
    #                                         0, 0, np.array([0.0, 1.0, 1.0]),
    #                                         u_body_opt=np.array([1.0, 0.0, 0.0]),
    #                                         roll_ref_flat=True,
    #                                         w_flat=1.0)
    # graph.add_robot_relative_rotation_constraint(start_node, pitcher_approach, 0, RollPitchYaw(np.pi/2, 0.0, 0.0).ToQuaternion());
    graph.add_robot_relative_rotation_constraint(start_node, pitcher_approach, 0, RollPitchYaw(3*np.pi/8, 0.0, 0.0).ToQuaternion());
    graph.add_robot_to_point_displacement_constraint(pitcher_approach, 0, 0, np.array([-0.20, 0.00, -0.08]));

    # graph.make_node_unpassable(pitcher_approach)

    graph.add_robot_relative_rotation_constraint(pitcher_approach, pitcher_pick_up, 0, RollPitchYaw(0.0, 0.0, 0.0).ToQuaternion());
    phi1 = graph.add_robot_to_point_displacement_constraint(pitcher_pick_up, 0, 0, np.array([-0.14, 0.00, -0.06]));
    graph.add_grasp_change(phi1, "grab", 0, 0);

    # CUP
    cup_approach, cup_pick_up = graph.structure.add_nodes(2)
    graph.structure.add_edge(start_node, cup_approach, True)
    graph.structure.add_edge(cup_approach, cup_pick_up, True)

    # graph.add_robot_to_point_alignment_cost(cup_approach,
    #                                         1, 1, np.array([0.0, 1.0, 1.0]),
    #                                         u_body_opt=np.array([1.0, 0.0, 0.0]),
    #                                         roll_ref_flat=True,
    #                                         w_flat=1.0)
    graph.add_robot_relative_rotation_constraint(start_node, cup_approach, 1, RollPitchYaw(3*np.pi/8, 0.0, 0.0).ToQuaternion());
    graph.add_robot_to_point_displacement_cost(cup_approach, 1, 1, np.array([0.25, 0.0, -0.04]))

    # # graph.make_node_unpassable(cup_approach)

    graph.add_robot_relative_rotation_constraint(cup_approach, cup_pick_up, 1, RollPitchYaw(0.0, 0.0, 0.0).ToQuaternion());
    phi2 = graph.add_robot_to_point_displacement_constraint(cup_pick_up, 1, 1, np.array([0.11, 0.00, -0.06]));
    graph.add_grasp_change(phi2, "grab", 1, 1);

    # BRING PITCHER AND CUP CLOSE TO EACH OTHER
    bring_close = graph.structure.add_node()
    graph.structure.add_edge(pitcher_pick_up, bring_close, True)
    graph.structure.add_edge(cup_pick_up, bring_close, True)

    graph.add_robot_holding_cube_constraint(pitcher_pick_up, bring_close, 0, 0, 0.25, use_l2=True);
    graph.add_robot_holding_cube_constraint(cup_pick_up, bring_close, 1, 1, 0.25, use_l2=True);

    graph.add_robot_relative_rotation_constraint(pitcher_pick_up, bring_close, 0, RollPitchYaw(0.0, 0.0, 0.0).ToQuaternion());
    graph.add_robot_relative_rotation_constraint(cup_pick_up, bring_close, 1, RollPitchYaw(0.0, 0.0, 0.0).ToQuaternion());

    graph.add_point_to_point_displacement_cost(bring_close, 0, 1, np.array([-0.1, 0.0, -0.12]));
    graph.add_point_linear_eq(bring_close, 0, np.array([[0.0, 0.0, 0.0],
                                                        [0.0, 0.0, 0.0],
                                                        [0.0, 0.0, 1.0]]), np.array([0.0, 0.0, 0.25]))

    # OR OVER ANOTHER POINT IF WANTED:
    # graph.add_point_to_point_displacement_cost(bring_close, 1, 2, np.array([0.0, -0.08, -0.20]));

    # POUR
    pour = graph.structure.add_node()
    graph.structure.add_edge(bring_close, pour, True)
    graph.add_robot_holding_cube_constraint(bring_close, pour, 0, 0, 0.25, use_l2=True);
    graph.add_robot_holding_cube_constraint(bring_close, pour, 1, 1, 0.25, use_l2=True);
    graph.add_robot_relative_displacement_constraint(bring_close, pour, 1, np.array([0.0, 0.0, 0.0]));
    graph.add_point_to_point_displacement_cost(pour, 0, 1, np.array([-0.05, 0.0, -0.1]));
    graph.add_robot_relative_rotation_constraint(bring_close, pour, 0,
                                                 RollPitchYaw(-np.pi/3, 0.0, 0.0).ToQuaternion());
    graph.make_node_unpassable(pour)


def do_folding(graph):
    joint_agent_dim = graph.num_agents * graph.dim;

    # RESET
    start_node = graph.structure.add_node()
    joint_agent_dim = graph.num_agents * graph.dim;
    home_position_1 = np.array([0.30, -0.2, 0.5, 0.0, -0.70701, -0.70701, 0.0,
                                -0.30, -0.2, 0.5, 0.0, 0.70701, -0.70701, 0.0])
    graph.add_robots_linear_eq(0, np.eye(joint_agent_dim), home_position_1)

    # GRASP TWO CORNERS
    # corner_approach = graph.structure.add_node()
    corner_approach, corner_pick_up = graph.structure.add_nodes(2)
    graph.structure.add_edge(start_node, corner_approach, True)
    graph.structure.add_edge(corner_approach, corner_pick_up, True)

    graph.add_robot_relative_rotation_constraint(start_node, corner_approach, 0, RollPitchYaw(0.0, 1*np.pi/4, 0.0).ToQuaternion())
    phi1 = graph.add_robot_to_point_displacement_constraint(corner_approach, 0, 0, np.array([0.10, 0.15, -0.14]))

    graph.add_robot_relative_rotation_constraint(start_node, corner_approach, 1, RollPitchYaw(0.0, -1*np.pi/4, 0.0).ToQuaternion())
    phi2 = graph.add_robot_to_point_displacement_constraint(corner_approach, 1, 1, np.array([-0.10, 0.15, -0.14]))

    graph.add_robot_relative_displacement_constraint(corner_approach, corner_pick_up, 0, np.array([0.0, 0.02, 0.0]))
    graph.add_robot_relative_displacement_constraint(corner_approach, corner_pick_up, 1, np.array([0.0, 0.02, 0.0]))

    trivial_phi1 = graph.add_robots_linear_eq(corner_pick_up, np.zeros((1, joint_agent_dim)), np.zeros((1,)))
    trivial_phi2 = graph.add_robots_linear_eq(corner_pick_up, np.zeros((1, joint_agent_dim)), np.zeros((1,)))
    graph.add_grasp_change(trivial_phi1, "grab", 0, 0);
    graph.add_grasp_change(trivial_phi2, "grab", 1, 1);
    # graph.add_grasp_change(phi1, "grab", 0, 0);
    # graph.add_grasp_change(phi2, "grab", 1, 1);
    # corner_pick_up = corner_approach

    # PULL OVER
    pull_up, put_down = graph.structure.add_nodes(2)
    graph.structure.add_edge(corner_pick_up, pull_up, True)
    graph.structure.add_edge(pull_up, put_down, True)

    graph.add_robot_relative_rotation_constraint(corner_pick_up, pull_up, 0, RollPitchYaw(0.0, -3*np.pi/8, 0.0).ToQuaternion())
    graph.add_robot_relative_displacement_constraint(corner_pick_up, pull_up, 0, np.array([0.0, 0.05, 0.20]))
    # graph.add_robot_to_point_displacement_constraint(pull_up, 0, 2, np.array([0.10, 0.20, -0.50]))

    graph.add_robot_relative_rotation_constraint(corner_pick_up, pull_up, 1, RollPitchYaw(0.0, 3*np.pi/8, 0.0).ToQuaternion())
    graph.add_robot_relative_displacement_constraint(corner_pick_up, pull_up, 1, np.array([0.0, 0.05, 0.20]))
    # graph.add_robot_to_point_displacement_constraint(pull_up, 1, 3, np.array([-0.10, 0.20, -0.50]))

    graph.add_robots_linear_eq(pull_up, np.zeros((1, joint_agent_dim)), np.zeros((1,)))

    # AND PUT DOWN
    graph.add_robot_relative_displacement_constraint(pull_up, put_down, 0, np.array([0.0, 0.18, -0.12]))
    graph.add_robot_relative_displacement_constraint(pull_up, put_down, 1, np.array([0.0, 0.18, -0.12]))
    graph.add_robot_relative_rotation_constraint(start_node, put_down, 0, RollPitchYaw(0.0, -1*np.pi/4, 0.0).ToQuaternion())
    graph.add_robot_relative_rotation_constraint(start_node, put_down, 1, RollPitchYaw(0.0, 1*np.pi/4, 0.0).ToQuaternion())

    trivial_phi3 = graph.add_robots_linear_eq(put_down, np.zeros((1, joint_agent_dim)), np.zeros((1,)))
    trivial_phi4 = graph.add_robots_linear_eq(put_down, np.zeros((1, joint_agent_dim)), np.zeros((1,)))
    graph.add_grasp_change(trivial_phi3, "release", 0, 0);
    graph.add_grasp_change(trivial_phi4, "release", 1, 1);


# def do_stack_cubes(graph):
#     graph.structure.add_nodes(12)

#     graph.structure.add_edge(0, 1, True)
#     graph.structure.add_edge(0, 5, True)

#     graph.structure.add_edge(1, 2, True)
#     graph.structure.add_edge(2, 3, True)
#     graph.structure.add_edge(3, 4, True)

#     graph.structure.add_edge(4, 10, True)

#     graph.structure.add_edge(5, 6, True)
#     graph.structure.add_edge(6, 7, True)
#     graph.structure.add_edge(7, 8, True)
#     graph.structure.add_edge(8, 9, True)

#     graph.structure.add_edge(10, 7, True)

#     graph.structure.add_edge(9, 11, True)

#     left_safe_position = np.array([0.30, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0])
#     left_low_position = np.array([0.30, 0.0, 0.17, 0.0, 0.0, 1.0, 0.0])
#     right_safe_position = np.array([-0.30, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0])
#     right_low_position = np.array([-0.30, 0.0, 0.18, 0.0, 0.0, 1.0, 0.0])

#     home_position_1 = np.array([0.30, -0.2, 0.5, 0.0, 0.0, 1.0, 0.0,
#                                 -0.30, -0.2, 0.5, 0.0, 0.0, 1.0, 0.0])
#     phi0 = graph.add_robots_linear_eq(0, np.eye(joint_agent_dim), home_position_1)

#     # phi1 = graph.add_robot_above_cube_constraint(1, 0, 2, 0.20, y_offset=-0.02);
#     # phi2 = graph.add_robot_above_cube_constraint(2, 0, 2, 0.15, y_offset=-0.02);
#     # TODO: Either remove or keep this temporary offset
#     first_grasp_x_block_offset = -0.01
#     first_grasp_y_block_offset = 0
#     # WIP: Tweaking this initial left arm movement position
#     phi1 = graph.add_robot_above_cube_constraint(1, 0, 2, 0.20, x_offset=first_grasp_x_block_offset, y_offset=first_grasp_y_block_offset);
#     phi2 = graph.add_robot_above_cube_constraint(2, 0, 2, 0.15, x_offset=first_grasp_x_block_offset, y_offset=first_grasp_y_block_offset);
#     graph.add_grasp_change(phi2, "grab", 0, 2);
#     # graspPhi0 = graph.add_robot_holding_cube_constraint(0, 1, 0, 0, 0.1);

#     # phi3 = graph.add_robot_above_cube_constraint(3, 0, 1, 0.25, x_offset=-0.01, y_offset=-0.05);
#     # phi4 = graph.add_robot_above_cube_constraint(4, 0, 1, 0.18, x_offset=-0.01, y_offset=-0.05);
#     # WIP: '''
#     first_release_x_block_offset = -0.015
#     first_release_y_block_offset = -0.015
#     phi3 = graph.add_robot_above_cube_constraint(3, 0, 1, 0.25, x_offset=first_release_x_block_offset, y_offset=first_release_y_block_offset);
#     phi4 = graph.add_robot_above_cube_constraint(4, 0, 1, 0.18, x_offset=first_release_x_block_offset, y_offset=first_release_y_block_offset);
#     # phi3 = graph.add_robot_linear_eq(3, 0, np.eye(agent_dim), left_safe_position);
#     # phi4 = graph.add_robot_linear_eq(4, 0, np.eye(agent_dim), left_low_position);
#     graph.add_grasp_change(phi4, "release", 0, 2);

#     # phi5 = graph.add_robot_above_cube_constraint(5, 1, 0, 0.20, y_offset=-0.04);
#     # phi6 = graph.add_robot_above_cube_constraint(6, 1, 0, 0.15, y_offset=-0.04);

#     # WIP: Tweak initial right arm movement position
#     second_grasp_x_block_offset = 0
#     second_grasp_y_block_offset = 0
#     phi5 = graph.add_robot_above_cube_constraint(5, 1, 0, 0.23, x_offset=second_grasp_x_block_offset, y_offset=second_grasp_y_block_offset);
#     phi6 = graph.add_robot_above_cube_constraint(6, 1, 0, 0.16, x_offset=second_grasp_x_block_offset, y_offset=second_grasp_y_block_offset);
#     graph.add_grasp_change(phi6, "grab", 1, 0);

#     # graspPhi1 = graph.add_robot_holding_cube_constraint(2, 3, 1, 2, 0.1);

#     # phi7 = graph.add_robot_above_cube_constraint(7, 1, 2, 0.25, x_offset=0.02, y_offset=-0.05); # , x_offset=0.0,
#     # phi8 = graph.add_robot_above_cube_constraint(8, 1, 2, 0.19, x_offset=0.02, y_offset=-0.05); # , x_offset=0.0,
#     # WIP: '''
#     second_release_x_block_offset = 0.01
#     second_release_y_block_offset = -0.02
#     phi7 = graph.add_robot_above_cube_constraint(7, 1, 2, 0.23, x_offset=second_release_x_block_offset, y_offset=second_release_y_block_offset) # , x_offset=0.0,
#     phi8 = graph.add_robot_above_cube_constraint(8, 1, 2, 0.18, x_offset=second_release_x_block_offset, y_offset=second_release_y_block_offset) # , x_offset=0.0,
#     # phi7 = graph.add_robot_linear_eq(7, 1, np.eye(agent_dim), right_safe_position);
#     # phi8 = graph.add_robot_linear_eq(8, 1, np.eye(agent_dim), right_low_position);
#     graph.add_grasp_change(phi8, "release", 1, 0);

#     # graph.add_grasp_change(phi8, "release", 1, 2);

#     # Avoid knocking the stack over
#     phi9 = graph.add_robot_above_cube_constraint(9, 1, 2, 0.28, x_offset=0, y_offset=0)

#     phi10 = graph.add_robot_linear_eq(10, 0, np.eye(agent_dim), left_safe_position);
#     phi11 = graph.add_robot_linear_eq(11, 1, np.eye(agent_dim), right_safe_position);

def do_stack_cubes_better(graph):
    joint_agent_dim = graph.num_agents * graph.dim;

    # RESET
    start_node = graph.structure.add_node()
    home_position_1 = np.array([0.30, -0.2, 0.5, 0.0, 0.0, 1.0, 0.0,
                                -0.30, -0.2, 0.5, 0.0, 0.0, 1.0, 0.0])
    phi0 = graph.add_robots_linear_eq(start_node, np.eye(joint_agent_dim), home_position_1)

    # GRAB BLOCK 2
    grab1_approach, grab1 = graph.structure.add_nodes(2)
    graph.structure.add_edge(start_node, grab1_approach, True)
    graph.structure.add_edge(grab1_approach, grab1, True)

    graph.add_robot_relative_rotation_constraint(start_node, grab1_approach, 0, RollPitchYaw(-1*np.pi/4, 0.0, 0.0).ToQuaternion());

    graph.add_robot_to_point_displacement_constraint(grab1_approach, 0, 2, np.array([0.0, 0.15, -0.13]))
    phi2 = graph.add_robot_to_point_displacement_constraint(grab1, 0, 2, np.array([0.0, 0.09, -0.13]))

    # graph.make_node_unpassable(grab1)

    graph.add_grasp_change(phi2, "grab", 0, 2)

    # RELEASE BLOCK 2
    release1_approach = graph.structure.add_node()
    release1 = graph.structure.add_node()
    graph.structure.add_edge(grab1, release1_approach, True)
    graph.structure.add_edge(release1_approach, release1, True)

    # graph.add_robot_holding_cube_constraint(grab1, release1_approach, 0, 2, 0.75);
    # graph.add_robot_holding_cube_constraint(release1_approach, release1, 0, 2, 0.75);
    # phi3 = graph.add_point_to_point_displacement_cost(release1_approach, 2, 1, np.array([0.0, 0.0, -0.10]))
    # phi4 = graph.add_point_to_point_displacement_constraint(release1, 2, 1, np.array([0.02, 0.02, -0.05]))

    graph.add_robot_to_point_displacement_cost(release1_approach, 0, 1, np.array([0.04, 0.14, -0.30]))
    phi4 = graph.add_robot_to_point_displacement_constraint(release1, 0, 1, np.array([0.005, 0.095, -0.17]))
    graph.add_grasp_change(phi4, "release", 0, 2)

    # RETRACT FROM BLOCK AFTER RELEASE
    left_retract = graph.structure.add_node()
    graph.structure.add_edge(release1, left_retract, True)
    graph.add_robot_to_point_displacement_constraint(left_retract, 0, 1, np.array([0.02, 0.12, -0.25]))

    # GRAB BLOCK 0
    grab2_approach, grab2 = graph.structure.add_nodes(2)
    graph.structure.add_edge(start_node, grab2_approach, True)
    graph.structure.add_edge(grab2_approach, grab2, True)

    graph.add_robot_relative_rotation_constraint(start_node, grab2_approach, 1, RollPitchYaw(-1*np.pi/4, 0.0, 0.0).ToQuaternion());
    graph.add_robot_to_point_displacement_constraint(grab2_approach, 1, 0, np.array([-0.02, 0.15, -0.14]))
    phi6 = graph.add_robot_to_point_displacement_constraint(grab2, 1, 0, np.array([-0.02, 0.08, -0.14]))
    # graph.make_node_unpassable(grab2)
    graph.add_grasp_change(phi6, "grab", 1, 0)

    # RELEASE BLOCK 0
    release2_approach = graph.structure.add_node()
    release2 = graph.structure.add_node()
    graph.structure.add_edge(grab2, release2_approach, True)
    graph.structure.add_edge(release2_approach, release2, True)

    graph.add_robot_to_point_displacement_cost(release2_approach, 1, 2, np.array([-0.04, 0.14, -0.30]))
    phi8 = graph.add_robot_to_point_displacement_constraint(release2, 1, 2, np.array([-0.02, 0.11, -0.175]))
    graph.add_grasp_change(phi8, "release", 1, 0)

    # RETRACT FROM BLOCK AFTER RELEASE
    right_retract = graph.structure.add_node()
    graph.structure.add_edge(release2, right_retract, True)
    graph.add_robot_to_point_displacement_constraint(right_retract, 1, 2, np.array([0.02, 0.12, -0.25]))

    left_safe_pos_node = graph.structure.add_node()
    graph.structure.add_edge(left_retract, left_safe_pos_node, True)
    # also, don't go to release the second block until the left safe
    # position has been reached
    graph.structure.add_edge(left_safe_pos_node, release2_approach, True)

    left_safe_position = np.array([0.30, 0.0, 0.5])
    graph.add_robot_pos_linear_eq(left_safe_pos_node, 0, np.eye(3), left_safe_position);

    right_safe_pos_node = graph.structure.add_node()
    graph.structure.add_edge(right_retract, right_safe_pos_node, True)

    right_safe_position = np.array([-0.30, 0.0, 0.5])
    graph.add_robot_pos_linear_eq(right_safe_pos_node, 1, np.eye(3), right_safe_position);


def common_builder(n_points, graph_builder, phi_tolerance=0.03, close_phi_tolerance=0.02, time_delta_cutoff=0.01, close_time_delta_cutoff=0.0):
    env = SimpleDrakeGym(["free_body_0", "free_body_1"], [f"cube_{i}" for i in range(n_points)])

    state_lower_bound = -10.0
    state_upper_bound = 10.0
    symbolic_plant = env.plant.ToSymbolic()
    graph = GraphOfConstraints(symbolic_plant, ["free_body_0", "free_body_1"], [f"cube_{i}" for i in range(n_points)],
                               state_lower_bound, state_upper_bound)
    agent_dim = graph.dim;

    graph_builder(graph)

    # GoC-MPC
    spline_spec = [Block.R(3), Block.SO3()]
    goc_mpc = GraphOfConstraintsMPC(graph, spline_spec,
                                    time_delta_cutoff = time_delta_cutoff,
                                    close_time_delta_cutoff = close_time_delta_cutoff,
                                    short_path_time_per_step = 0.1,
                                    phi_tolerance = phi_tolerance,
                                    close_phi_tolerance = close_phi_tolerance)
                                    # max_vel = 0.05,  # maximum velocity for every joint
                                    # max_acc = 0.05,  # maximum acceleration for every joint
                                    # max_jerk = 0.05) # maximum jerk for every joint

    goc_mpc.reset()

    return env, graph, goc_mpc


def move_in_circles_builder():
    return common_builder(0, do_move_in_circles)

def track_above_builder():
    return common_builder(2, do_track_above)

def pick_and_pour_builder():
    return common_builder(2, do_pick_and_pour,
                          phi_tolerance=0.06,
                          close_phi_tolerance=0.03,
                          time_delta_cutoff=0.4,
                          close_time_delta_cutoff=0.0)
