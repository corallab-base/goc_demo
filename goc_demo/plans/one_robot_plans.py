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


TIME_DELTA_CUTOFF = 0.3
PHI_TOLERANCE = 0.05


def do_move_in_circles(graph):
    joint_agent_dim = graph.num_agents * graph.dim;

    graph.structure.add_nodes(3)
    graph.structure.add_edge(0, 1, True)
    graph.structure.add_edge(1, 2, True)

    triangle_origin = np.array([-0.5, 0.0, 0.5])

    goal_position_1 = triangle_origin + np.array([0.0, 0.1, 0.0])
    phi0 = graph.add_robots_linear_eq(0, np.eye(joint_agent_dim), goal_position_1)

    goal_position_2 = triangle_origin + np.array([0.0, -0.1, 0.0])
    phi1 = graph.add_robots_linear_eq(1, np.eye(joint_agent_dim), goal_position_2)

    home_position_1 = triangle_origin + np.array([0.0, 0.0, 0.1])
    phi2 = graph.add_robots_linear_eq(2, np.eye(joint_agent_dim), home_position_1)


def common_builder(n_points, graph_builder, phi_tolerance=PHI_TOLERANCE, time_delta_cutoff=TIME_DELTA_CUTOFF):
    env = SimpleDrakeGym(["point_mass"], [f"cube_{i}" for i in range(n_points)])

    state_lower_bound = -10.0
    state_upper_bound = 10.0
    graph = GraphOfConstraints(["point_mass"], [f"cube_{i}" for i in range(n_points)],
                               state_lower_bound, state_upper_bound)
    agent_dim = graph.dim;

    graph_builder(graph)

    # GoC-MPC
    spline_spec = [Block.R(3)]
    goc_mpc = GraphOfConstraintsMPC(graph, spline_spec,
                                    time_delta_cutoff = time_delta_cutoff,
                                    short_path_time_per_step = 0.1,
                                    phi_tolerance = phi_tolerance,
                                    # max_vel = 0.05,  # maximum velocity for every joint
                                    max_acc = 1.00,  # maximum acceleration for every joint
                                    # max_jerk = 0.05 # maximum jerk for every joint
                                    )

    goc_mpc.reset()

    return env, graph, goc_mpc

def move_in_circles_builder():
    return common_builder(0, do_move_in_circles)
