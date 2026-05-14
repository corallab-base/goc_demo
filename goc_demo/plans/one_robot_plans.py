import os
import numpy as np

from goc_mpc.splines import Block
from goc_mpc.goc_mpc import GraphOfConstraints, GraphOfConstraintsMPC


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


def do_pick_and_place(graph):
    joint_agent_dim = graph.num_agents * graph.dim;
    robot_id = 0

    def add_grasp(block):
        approach, pick_up = graph.structure.add_nodes(2)
        graph.structure.add_edge(approach, pick_up, True)

        graph.add_robot_to_point_displacement_constraint(approach, robot_id, block, np.array([0.0, 0.0, -0.35]));

        # aligned_phi = graph.add_edge_assignable_robot_to_point_displacement_constraint(
        #     u=approach, v=pick_up, var=robot, point_id=block,
        #     disp=np.array([0.0, 0.0, -0.25]),
        #     tol=np.array([0.15, 0.15, 0.3]))

        phi = graph.add_robot_to_point_displacement_constraint(pick_up, robot_id, block, np.array([0.02, 0.0, -0.17]));
        graph.add_grasp_change(phi, "grab", robot_id, block);

        return approach, pick_up

    def add_release(held_block, relative_to_block, displacement, randomize=False):
        approach, release, back_off = graph.structure.add_nodes(3)
        graph.structure.add_edge(approach, release, True)
        graph.structure.add_edge(release, back_off, True)

        if randomize:
            offset = np.random.standard_normal((3,)) * np.array([0.07, 0.07, 0.0])
            displacement = displacement + offset
            print(f"SAMPLED FOR INIT: {displacement}")
        
        graph.add_robot_to_point_displacement_constraint(approach, robot_id, relative_to_block, displacement + np.array([0.0, 0.0, -0.15]))
        # graph.add_point_to_point_displacement_constraint(approach, held_block, relative_to_block, np.array([-0.10, 0.0, -0.3]))

        # # keep holding between approach and putting down
        # graph.add_robot_holding_cube_constraint(approach, release, robot_id, held_block, 0.2)

        phi = graph.add_robot_to_point_displacement_constraint(release, robot_id, relative_to_block, displacement)
        # phi = graph.add_point_to_point_displacement_constraint(release, robot_id, relative_to_block, displacement)
        graph.add_grasp_change(phi, "release", robot_id, held_block)

        graph.add_robot_to_point_displacement_constraint(back_off, robot_id, relative_to_block, displacement + np.array([0.0, 0.0, -0.15]))

        return approach, release, back_off

    # grasp and release block 0
    approach_pick_up_0, pick_up_0 = add_grasp(block=0)
    approach_release_0, release_0, back_off_0 = add_release(held_block=0, relative_to_block=1, displacement=np.array([-0.10, 0.0, -0.24]))
    grasp_phi_0 = graph.add_robot_holding_cube_constraint(pick_up_0, approach_release_0, robot_id, 0, 0.30);

    graph.add_manual_backtrack_links(grasp_phi_0, [approach_pick_up_0, pick_up_0])

    graph.structure.add_edge(pick_up_0, approach_release_0, True)

    # go home
    go_home_0 = graph.structure.add_node()
    graph.structure.add_edge(back_off_0, go_home_0, True)
    graph.add_robot_pos_linear_eq(
        k=go_home_0, robot_id=robot_id, A=np.eye(3), b=np.array([-0.5, 0.0, 0.5]))

    # grasp and release block 0 again
    approach_pick_up_1, pick_up_1 = add_grasp(block=0)
    graph.structure.add_edge(go_home_0, approach_pick_up_1, True)
    approach_release_1, release_1, back_off_1 = add_release(held_block=0, relative_to_block=1, displacement=np.array([-0.30, 0.0, -0.23]), randomize=True)
    grasp_phi_1 = graph.add_robot_holding_cube_constraint(pick_up_1, release_1, robot_id, 0, 0.30);

    graph.add_manual_backtrack_links(grasp_phi_1, [approach_pick_up_1, pick_up_1])

    graph.structure.add_edge(pick_up_1, approach_release_1, True)

    # go home again
    go_home_1 = graph.structure.add_node()
    graph.structure.add_edge(back_off_1, go_home_1, True)
    graph.add_robot_pos_linear_eq(
        k=go_home_1, robot_id=robot_id, A=np.eye(3), b=np.array([-0.5, 0.0, 0.5]))

    # graph.make_node_unpassable(go_home_1)
    # graph.add_manual_backtrack_links(arrangedPhi0, [approach_pick_up_0, pick_up_0, release_0])


def common_builder(n_points, graph_builder, phi_tolerance=PHI_TOLERANCE, time_delta_cutoff=TIME_DELTA_CUTOFF):
    state_lower_bound = -10.0
    state_upper_bound = 10.0

    robot_spec = [Block.R(3)]
    object_spec = [Block.R(3)]

    graph = GraphOfConstraints([robot_spec], [object_spec for i in range(n_points)],
                               state_lower_bound, state_upper_bound,
                               robot_names=["point_mass"],
                               object_names=[f"cube_{i}" for i in range(n_points)])
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

    return graph, goc_mpc


def move_in_circles_builder():
    return common_builder(0, do_move_in_circles)


def pick_and_place_builder():
    return common_builder(2, do_pick_and_place)
