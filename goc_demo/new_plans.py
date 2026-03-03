import os
import numpy as np
from pydrake.math import RollPitchYaw

from goc_mpc.splines import Block
from goc_mpc.goc_mpc import GraphOfConstraints, GraphOfConstraintsMPC
from goc_mpc.simple_drake_env import SimpleDrakeGym
# this is where the graph of constrains is created

TIME_DELTA_CUTOFF = 0.3
PHI_TOLERANCE = 0.2

# the part where it selects the point in the spline to target, increase
def do_move_in_circles_single(graph):
    # set the dimension for a single agent
    joint_agent_dim = graph.num_agents * graph.dim

    # add five nodes to make a full single loop
    graph.structure.add_nodes(5)

    # link them in a linear chain to avoid a topological cycle
    graph.structure.add_edge(0, 1, True)
    graph.structure.add_edge(1, 2, True)
    graph.structure.add_edge(2, 3, True)
    graph.structure.add_edge(3, 4, True)

    # top of the circle
    pos_0 = np.array([0.0, 0.3, 0.5, 0.0, 0.0, 1.0, 0.0])
    graph.add_robots_linear_eq(0, np.eye(joint_agent_dim), pos_0)

    # right side
    pos_1 = np.array([0.10, 0.3, 0.4, 0.0, 0.0, 1.0, 0.0])
    graph.add_robots_linear_eq(1, np.eye(joint_agent_dim), pos_1)

    # bottom of the circle
    pos_2 = np.array([0.0, 0.3, 0.3, 0.0, 0.0, 1.0, 0.0])
    graph.add_robots_linear_eq(2, np.eye(joint_agent_dim), pos_2)

    # left side
    pos_3 = np.array([-0.10, 0.3, 0.4, 0.0, 0.0, 1.0, 0.0])
    graph.add_robots_linear_eq(3, np.eye(joint_agent_dim), pos_3)

    # back to the top to complete the shape
    pos_4 = np.array([0.0, 0.3, 0.5, 0.0, 0.0, 1.0, 0.0])
    graph.add_robots_linear_eq(4, np.eye(joint_agent_dim), pos_4)

def common_builder_single(n_points, graph_builder,
                          phi_tolerance=PHI_TOLERANCE,
                          time_delta_cutoff=TIME_DELTA_CUTOFF):

    env = SimpleDrakeGym(["free_body_0"], [f"cube_{i}" for i in range(n_points)])

    state_lower_bound = -10.0
    state_upper_bound = 10.0
    symbolic_plant = env.plant.ToSymbolic()
    graph = GraphOfConstraints(
        symbolic_plant,
        ["free_body_0"],
        [f"cube_{i}" for i in range(n_points)],
        state_lower_bound,
        state_upper_bound,
    )

    graph_builder(graph)

    spline_spec = [Block.R(3), Block.SO3()]
    goc_mpc = GraphOfConstraintsMPC(
        graph,
        spline_spec,
        time_delta_cutoff=time_delta_cutoff,
        short_path_time_per_step=0.1,
        phi_tolerance=phi_tolerance,
        max_acc=1.0,
    )

    goc_mpc.reset()

    return env, graph, goc_mpc

def move_in_circles_single_builder():
    return common_builder_single(0, do_move_in_circles_single)
