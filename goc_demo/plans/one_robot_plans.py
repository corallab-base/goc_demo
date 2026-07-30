import os
import numpy as np

from goc_mpc.splines import Block
from goc_mpc import (
    GraphOfConstraints, GraphOfConstraintsMPC,
    WaypointSolver, WaypointObjective
)

from pydrake.math import eq
from pydrake.symbolic import logical_and


TIME_DELTA_CUTOFF = 0.3
PHI_TOLERANCE = 0.05


def do_move_in_circles(graph):
    graph.structure.add_nodes(3)
    graph.structure.add_edge(0, 1, True)
    graph.structure.add_edge(1, 2, True)

    triangle_origin = np.array([-0.5, 0.0, 0.5])
    q0 = graph.agent_q(0)

    goal_position_1 = triangle_origin + np.array([0.0, 0.1, 0.0])
    phi0 = graph.add_constraint(0, eq(q0, goal_position_1))

    goal_position_2 = triangle_origin + np.array([0.0, -0.1, 0.0])
    phi1 = graph.add_constraint(1, eq(q0, goal_position_2))

    home_position_1 = triangle_origin + np.array([0.0, 0.0, 0.1])
    phi2 = graph.add_constraint(2, eq(q0, home_position_1))


def do_pick_and_place(graph):
    robot_id = 0
    q0 = graph.agent_q(robot_id)

    def add_holding_box(u, v, block, holding_distance_max):
        # Symbolic equivalent of add_robot_holding_cube_constraint(u, v, robot_id,
        # block, holding_distance_max): that helper compiles to an axis-aligned
        # |dx|,|dy|,|dz| <= holding_distance_max box applied independently at both
        # endpoints AND any interior node scheduled between them (see
        # AddBoxProximityConstraint in graph_of_constraints.cpp) -- exactly what a
        # plain (non-u_/v_) "along the edge" formula does here.
        obj = graph.object_q(block)
        dx, dy, dz = q0[0] - obj[0], q0[1] - obj[1], q0[2] - obj[2]
        d = holding_distance_max
        return graph.add_edge_constraint(
            u, v,
            logical_and(dx <= d, dx >= -d, dy <= d, dy >= -d, dz <= d, dz >= -d))

    def add_grasp(block):
        approach, pick_up = graph.structure.add_nodes(2)
        graph.structure.add_edge(approach, pick_up, True)

        graph.add_constraint(approach, eq(q0, graph.object_q(block) + np.array([0.0, 0.0, -0.35])))

        phi = graph.add_constraint(pick_up, eq(q0, graph.object_q(block) + np.array([0.02, 0.0, -0.17])))
        graph.add_grasp_change(phi, "grab", robot_id, block);

        return approach, pick_up

    def add_release(held_block, relative_to_block, displacement, randomize=False):
        approach, release, back_off = graph.structure.add_nodes(3)
        graph.structure.add_edge(approach, release, True)
        graph.structure.add_edge(release, back_off, True)

        if randomize:
            offset = np.random.uniform(low=-1.0, high=1.0, size=(3,)) * np.array([0.07, 0.07, 0.0])
            displacement = displacement + offset
            print(f"SAMPLED FOR INIT: {displacement}")

        obj = graph.object_q(relative_to_block)
        graph.add_constraint(approach, eq(q0, obj + displacement + np.array([0.0, 0.0, -0.15])))

        phi = graph.add_constraint(release, eq(q0, obj + displacement))
        graph.add_grasp_change(phi, "release", robot_id, held_block)

        graph.add_constraint(back_off, eq(q0, obj + displacement + np.array([0.0, 0.0, -0.15])))

        return approach, release, back_off

    # grasp and release block 0
    approach_pick_up_0, pick_up_0 = add_grasp(block=0)
    approach_release_0, release_0, back_off_0 = add_release(held_block=0, relative_to_block=1, displacement=np.array([-0.10, 0.0, -0.24]))
    grasp_phi_0 = add_holding_box(pick_up_0, approach_release_0, 0, 0.30)

    graph.add_manual_backtrack_links(grasp_phi_0, [approach_pick_up_0, pick_up_0])

    graph.structure.add_edge(pick_up_0, approach_release_0, True)

    # go home
    go_home_0 = graph.structure.add_node()
    graph.structure.add_edge(back_off_0, go_home_0, True)
    graph.add_constraint(go_home_0, eq(q0, np.array([-0.5, 0.0, 0.5])))

    # grasp and release block 0 again
    approach_pick_up_1, pick_up_1 = add_grasp(block=0)
    graph.structure.add_edge(go_home_0, approach_pick_up_1, True)
    approach_release_1, release_1, back_off_1 = add_release(held_block=0, relative_to_block=1, displacement=np.array([-0.30, 0.0, -0.23]), randomize=True)
    grasp_phi_1 = add_holding_box(pick_up_1, release_1, 0, 0.30)

    graph.add_manual_backtrack_links(grasp_phi_1, [approach_pick_up_1, pick_up_1])

    graph.structure.add_edge(pick_up_1, approach_release_1, True)

    # go home again
    go_home_1 = graph.structure.add_node()
    graph.structure.add_edge(back_off_1, go_home_1, True)

    home_offset = np.random.uniform(low=-1.0, high=1.0, size=(3,)) * np.array([0.1, 0.1, 0.07])
    home = np.array([-0.5, 0.0, 0.5]) + home_offset
    print(f"SAMPLED FOR HOME: {home}")

    graph.add_constraint(go_home_1, eq(q0, home))


    # grasp and release block 0
    approach_pick_up_0, pick_up_0 = add_grasp(block=0)
    approach_release_0, release_0, back_off_0 = add_release(held_block=0, relative_to_block=1, displacement=np.array([-0.10, 0.0, -0.24]))
    grasp_phi_0 = add_holding_box(pick_up_0, approach_release_0, 0, 0.30)

    graph.add_manual_backtrack_links(grasp_phi_0, [approach_pick_up_0, pick_up_0])

    graph.structure.add_edge(pick_up_0, approach_release_0, True)

    # go home
    go_home_0 = graph.structure.add_node()
    graph.structure.add_edge(back_off_0, go_home_0, True)
    graph.add_constraint(go_home_0, eq(q0, np.array([-0.5, 0.0, 0.5])))

    # grasp and release block 0 again
    approach_pick_up_1, pick_up_1 = add_grasp(block=0)
    graph.structure.add_edge(go_home_0, approach_pick_up_1, True)
    approach_release_1, release_1, back_off_1 = add_release(held_block=0, relative_to_block=1, displacement=np.array([-0.30, 0.0, -0.23]), randomize=True)
    grasp_phi_1 = add_holding_box(pick_up_1, release_1, 0, 0.30)

    graph.add_manual_backtrack_links(grasp_phi_1, [approach_pick_up_1, pick_up_1])

    graph.structure.add_edge(pick_up_1, approach_release_1, True)

    # go home again
    go_home_1 = graph.structure.add_node()
    graph.structure.add_edge(back_off_1, go_home_1, True)

    home_offset = np.random.uniform(low=-1.0, high=1.0, size=(3,)) * np.array([0.1, 0.1, 0.07])
    home = np.array([-0.5, 0.0, 0.5]) + home_offset
    print(f"SAMPLED FOR HOME: {home}")

    graph.add_constraint(go_home_1, eq(q0, home))

    # graph.make_node_unpassable(go_home_1)
    # graph.add_manual_backtrack_links(arrangedPhi0, [approach_pick_up_0, pick_up_0, release_0])


def do_test_yaw(graph):
    graph.structure.add_nodes(3)
    graph.structure.add_edge(0, 1, True)
    graph.structure.add_edge(1, 2, True)

    triangle_origin = np.array([-0.5, 0.0, 0.5, 0.0])
    q0 = graph.agent_q(0)

    goal_position_1 = triangle_origin + np.array([0.0, 0.1, 0.0, 0.0])
    phi0 = graph.add_constraint(0, eq(q0, goal_position_1))

    goal_position_2 = triangle_origin + np.array([0.0, -0.1, 0.0, 0.5])
    phi1 = graph.add_constraint(1, eq(q0, goal_position_2))

    home_position_1 = triangle_origin + np.array([0.0, 0.0, 0.1, -0.5])
    phi2 = graph.add_constraint(2, eq(q0, home_position_1))


def do_yaw_track_above(graph):
    joint_agent_dim = graph.num_agents * graph.dim;

    node = graph.structure.add_node()

    graph.add_constraint(
        node,
        eq(graph.agent_q(0),
           graph.object_q(0) + np.array([0.0, 0.0, 0.3, 0]))
    )
    graph.make_node_unpassable(node)


def do_move_spam(graph):
    robot_id = 0
    grasp_distance_limit = 0.26

    def add_holding_box(u, v, block, holding_distance_max):
        # Symbolic equivalent of add_robot_holding_cube_constraint -- see the
        # identical helper in do_pick_and_place for why. Only x, y, z are
        # boxed (not yaw): add_robot_holding_cube_constraint always compares
        # PoseFromRow/CubePosFromRow's leading 3 (position) components,
        # regardless of the robot/object block's full dimension.
        q = graph.agent_q(robot_id)
        obj = graph.object_q(block)
        dx, dy, dz = q[0] - obj[0], q[1] - obj[1], q[2] - obj[2]
        d = holding_distance_max
        return graph.add_edge_constraint(
            u, v,
            logical_and(dx <= d, dx >= -d, dy <= d, dy >= -d, dz <= d, dz >= -d))

    def add_grasp(block):
        approach, pick_up = graph.structure.add_nodes(2)
        graph.structure.add_edge(approach, pick_up, True)

        graph.add_constraint(
            approach,
            eq(graph.agent_q(robot_id),
               graph.object_q(block) + np.array([0.0, 0.0, 0.3, 0.0]))
        )

        phi = graph.add_constraint(
            pick_up,
            eq(graph.agent_q(robot_id),
               graph.object_q(block) + np.array([0.0, 0.0, 0.16, 0.0]))
        )
        graph.add_grasp_change(phi, "grab", robot_id, block);

        # graph.add_edge_constraint(
        #     approach,
        #     eq(graph.agent_q(robot_id),
        #        graph.object_q(block) + np.array([0.0, 0.0, 0.3, 0.0]))
        # )
        return approach, pick_up

    def add_release(held_block, position, randomize=False):
        approach, release, back_off = graph.structure.add_nodes(3)
        graph.structure.add_edge(approach, release, True)
        graph.structure.add_edge(release, back_off, True)

        if randomize:
            offset = np.random.uniform(low=-1.0, high=1.0, size=(4,)) * np.array([0.05, 0.05, 0.0, np.pi/2])
            position = position + offset
            print(f"SAMPLED FOR INIT: {position}")

        graph.add_constraint(
            approach,
            eq(graph.agent_q(robot_id),
               position + np.array([0.0, 0.0, 0.1, 0.0]))
        )

        phi = graph.add_constraint(
            release,
            eq(graph.agent_q(robot_id),
               position)
        )

        graph.add_grasp_change(phi, "release", robot_id, held_block)

        graph.add_constraint(
            back_off,
            eq(graph.agent_q(robot_id),
               position + np.array([0.0, 0.0, 0.1, 0.0]))
        )

        return approach, release, back_off

    # grasp and release block 0
    approach_pick_up_0, pick_up_0 = add_grasp(block=0)
    approach_release_0, release_0, back_off_0 = add_release(held_block=0, position=np.array([-0.7, 0.0, 0.24, 0.0]))
    graph.structure.add_edge(pick_up_0, approach_release_0, True)

    grasp_phi_0 = add_holding_box(pick_up_0, approach_release_0, 0, grasp_distance_limit)

    graph.add_manual_backtrack_links(grasp_phi_0, [approach_pick_up_0, pick_up_0])


    # go home
    go_home_0 = graph.structure.add_node()
    graph.structure.add_edge(back_off_0, go_home_0, True)

    phi = graph.add_constraint(
        go_home_0,
        eq(graph.agent_q(0), np.array([-0.5, 0.0, 0.5, 0.0]))
    )

    # grasp and release block 0 again
    approach_pick_up_1, pick_up_1 = add_grasp(block=0)
    graph.structure.add_edge(go_home_0, approach_pick_up_1, True)
    approach_release_1, release_1, back_off_1 = add_release(held_block=0, position=np.array([-0.42, 0.0, 0.24, 0.0]), randomize=True)
    graph.structure.add_edge(pick_up_1, approach_release_1, True)

    grasp_phi_1 = add_holding_box(pick_up_1, release_1, 0, grasp_distance_limit)

    graph.add_manual_backtrack_links(grasp_phi_1, [approach_pick_up_1, pick_up_1])

    # go home again
    go_home_1 = graph.structure.add_node()
    graph.structure.add_edge(back_off_1, go_home_1, True)

    home_offset = np.random.uniform(low=-1.0, high=1.0, size=(4,)) * np.array([0.1, 0.1, 0.04, np.pi/2])
    home = np.array([-0.5, 0.0, 0.5, 0.0]) + home_offset
    print(f"SAMPLED FOR HOME: {home}")

    graph.add_constraint(
        go_home_1,
        eq(graph.agent_q(0), home)
    )


def common_builder(n_points, graph_builder, phi_tolerance=PHI_TOLERANCE, time_delta_cutoff=TIME_DELTA_CUTOFF, needs_yaw=False):
    state_lower_bound = -10.0
    state_upper_bound = 10.0

    if needs_yaw:
        robot_spec = [Block.R(4)]
        object_spec = [Block.R(4)]
    else:
        robot_spec = [Block.R(3)]
        object_spec = [Block.R(3)]

    graph = GraphOfConstraints([robot_spec], [object_spec for i in range(n_points)],
                               state_lower_bound, state_upper_bound,
                               robot_names=["point_mass"],
                               object_names=[f"cube_{i}" for i in range(n_points)])
    agent_dim = graph.dim;

    graph_builder(graph)

    # GoC-MPC
    spline_spec = robot_spec
    goc_mpc = GraphOfConstraintsMPC(graph, spline_spec,
                                    # for waypoint solver:
                                    waypoint_solver = WaypointSolver.kGurobi,
                                    waypoint_objective = WaypointObjective.kAvgL1 if needs_yaw else WaypointObjective.kAvgL2,
                                    waypoint_enforce_rigidity = False,
                                    # for timing solver:
                                    time_delta_cutoff = time_delta_cutoff,
                                    short_path_time_per_step = 0.1,
                                    phi_tolerance = phi_tolerance,
                                    # max_vel = 0.05,  # maximum velocity for every joint
                                    max_acc = 1.00,  # maximum acceleration for every joint
                                    # max_jerk = 0.05 # maximum jerk for every joint
                                    linear_interpolation = True)

    goc_mpc.reset()

    return graph, goc_mpc


def move_in_circles_builder():
    return common_builder(0, do_move_in_circles)


def pick_and_place_builder():
    return common_builder(2, do_pick_and_place)

def test_yaw_builder():
    return common_builder(0, do_test_yaw, needs_yaw=True)

def yaw_track_above_builder():
    return common_builder(1, do_yaw_track_above, needs_yaw=True)

def move_spam_builder():
    return common_builder(1, do_move_spam, needs_yaw=True)
