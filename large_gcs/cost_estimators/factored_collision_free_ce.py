import ast
import logging
import re
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from large_gcs.contact.rigid_body import MobilityType
from large_gcs.cost_estimators.cost_estimator import CostEstimator
from large_gcs.graph.contact_cost_constraint_factory import (
    edge_constraint_position_continuity_factored,
)
from large_gcs.graph.contact_graph import ContactGraph
from large_gcs.graph.factored_collision_free_graph import FactoredCollisionFreeGraph
from large_gcs.graph.graph import Edge, Graph, ShortestPathSolution

logger = logging.getLogger(__name__)


class FactoredCollisionFreeCE(CostEstimator):
    def __init__(
        self,
        graph: ContactGraph,
        use_combined_gcs: bool = True,
        add_transition_cost: bool = True,
        obj_multiplier: float = 1.0,
    ):
        self._graph = graph
        self._should_add_transition_cost = add_transition_cost
        self._obj_multiplier = obj_multiplier
        self._use_combined_gcs = use_combined_gcs
        logger.info(
            f"creating {self._graph.n_objects + self._graph.n_robots} collision free graphs..."
        )
        self._cfree_graphs = [
            FactoredCollisionFreeGraph(
                body,
                self._graph.obstacles,
                self._graph.target_pos[i],
                cost_scaling=1.0
                if body.mobility_type == MobilityType.ACTUATED
                else self._obj_multiplier,
                workspace=self._graph.workspace,
            )
            for i, body in tqdm(enumerate(self._graph.objects + self._graph.robots))
        ]
        self._cfree_target_names = [g.target_name for g in self._cfree_graphs]

        # Look up table of cfree vertex name to cfree cost
        self._cfree_cost = {}
        self._cfree_init_pos = {}

    def estimate_cost(
        self,
        subgraph: Graph,
        edge: Edge,
        active_edges: List[Tuple[str, str]] = None,
        solve_convex_restriction: bool = False,
        use_convex_relaxation: bool = False,
    ) -> ShortestPathSolution:
        """Right now this function is unideally coupled because it returns a shortest path solution instead of just the cost."""

        neighbor = edge.v
        # Add neighbor and edge temporarily to the visited subgraph
        subgraph.add_vertex(self._graph.vertices[neighbor], neighbor)
        subgraph.add_edge(edge)
        # Check if this neighbor actually has an edge to the target
        # If so, add that edge instead of calculating the collision free cost
        neighbor_has_edge_to_target = (
            neighbor,
            self._graph.target_name,
        ) in self._graph.edges
        if neighbor_has_edge_to_target:
            # print(f"neighbor {neighbor} has edge to target")
            edge_to_target = self._graph.edges[(neighbor, self._graph.target_name)]
            subgraph.add_edge(edge_to_target)
            subgraph.set_target(self._graph.target_name)
            conv_res_active_edges = active_edges + [edge.key, edge_to_target.key]
        elif not self._use_combined_gcs:
            # set the neighbor as the target
            subgraph.set_target(neighbor)
            conv_res_active_edges = active_edges + [edge.key]

        if self._use_combined_gcs and not neighbor_has_edge_to_target:
            if not self._are_cfree_graphs_in_subgraph(subgraph):
                self._add_cfree_graphs_to_subgraph(subgraph)
            self._connect_vertex_to_cfree_subgraphs(subgraph, neighbor)
            if solve_convex_restriction:
                sol = subgraph.solve_factored_convex_restriction(
                    active_edges + [edge.key], neighbor, self._cfree_target_names
                )
            else:
                sol = subgraph.solve_factored_shortest_path(
                    neighbor,
                    self._cfree_target_names,
                    use_convex_relaxation=use_convex_relaxation,
                )
        else:
            if solve_convex_restriction:
                sol = subgraph.solve_convex_restriction(conv_res_active_edges)
            else:
                sol = subgraph.solve_shortest_path(
                    use_convex_relaxation=use_convex_relaxation
                )

        self._alg_metrics.update_after_gcs_solve(sol.time)
        # Clean up
        subgraph.remove_vertex(neighbor)

        if (
            not self._use_combined_gcs
            and sol.is_success
            and not neighbor_has_edge_to_target
        ):
            cfree_cost = self._get_cfree_cost_split(
                sol, neighbor, use_convex_relaxation
            )
            # Add heuristic cost to cost to come
            sol.cost += cfree_cost

        return sol

    def _are_cfree_graphs_in_subgraph(self, subgraph: Graph) -> bool:
        # Assume that if one vertex from the first cfree graph is in the subgraph, then all of them are
        cfree_vertex = next(iter(self._cfree_graphs[0].vertices))
        return cfree_vertex in subgraph.vertices

    def _add_cfree_graphs_to_subgraph(self, subgraph: Graph) -> None:
        # NOTE: This is very inefficient to be doing on exploration of each vertex.
        logger.info("adding cfree graphs to subgraph...")
        for cfree_graph in self._cfree_graphs:
            for vertex_name, vertex in cfree_graph.vertices.items():
                subgraph.add_vertex(vertex, vertex_name)
            for edge in cfree_graph.edges.values():
                subgraph.add_edge(edge)

    def _connect_vertex_to_cfree_subgraphs(
        self, subgraph: Graph, vertex_name: str
    ) -> None:
        # Add outgoing edges from transition vertex
        for i, cfree_vertex_name in enumerate(self.split_vertex_names(vertex_name)):
            continuity_con = edge_constraint_position_continuity_factored(
                body_index=i,
                u_vars=subgraph.vertices[vertex_name].convex_set.vars,
                v_vars=subgraph.vertices[cfree_vertex_name].convex_set.vars,
            )
            edge = Edge(vertex_name, cfree_vertex_name, constraints=[continuity_con])
            subgraph.add_edge(edge)

    def _get_cfree_cost_split(
        self,
        sol_s_to_neighbor: ShortestPathSolution,
        neighbor: str,
        use_convex_relaxation: bool,
    ) -> float:
        # Position within neighbor that the first solve ends at
        x_vars = self._graph.vertices[neighbor].convex_set.vars
        x_pos = x_vars.pos_from_all(sol_s_to_neighbor.ambient_path[-1])

        # Calculate or look up the collision free cost for each body
        cfree_cost = 0
        for i, cfree_vertex_name in enumerate(self.split_vertex_names(neighbor)):
            body_pos_end = x_pos[i].T[-1].flatten()
            g = self._cfree_graphs[i]
            if cfree_vertex_name in self._cfree_cost:
                logger.debug(
                    f"Using cached cfree cost for {cfree_vertex_name}, cost: {self._cfree_cost[cfree_vertex_name]}"
                )
                cfree_cost += self._cfree_cost[cfree_vertex_name]
                cfree_init_pos = self._cfree_init_pos[cfree_vertex_name]
            else:
                g.set_source(cfree_vertex_name)
                cfree_sol = g.solve_shortest_path(
                    use_convex_relaxation=use_convex_relaxation
                )
                if cfree_sol.is_success:
                    new_cfree_cost = (
                        self._obj_multiplier * cfree_sol.cost
                        if g.movable_body.mobility_type == MobilityType.UNACTUATED
                        else cfree_sol.cost
                    )
                    cfree_cost += new_cfree_cost
                    self._cfree_cost[cfree_vertex_name] = new_cfree_cost
                    x_cfree_vars = g.vertices[cfree_vertex_name].convex_set.vars
                    cfree_init_pos = (
                        x_cfree_vars.pos_from_all(cfree_sol.ambient_path[0])
                        .T[0]
                        .flatten()
                    )
                    self._cfree_init_pos[cfree_vertex_name] = cfree_init_pos
                    logger.debug(
                        f"Calculated cfree cost for {cfree_vertex_name}, cost: {cfree_sol.cost}"
                    )
                else:
                    cfree_cost += float("inf")
                    self._cfree_cost[cfree_vertex_name] = float("inf")
                    self._cfree_init_pos[cfree_vertex_name] = body_pos_end
                    cfree_init_pos = body_pos_end
                    logger.warn(
                        f"Could not find collision free path for {cfree_vertex_name}"
                    )

                self._cfree_cost[cfree_vertex_name] = cfree_cost
                self._alg_metrics.update_after_gcs_solve(cfree_sol.time)

            if self._should_add_transition_cost:
                transition_cost = np.linalg.norm(body_pos_end - cfree_init_pos)
                cfree_cost += (
                    self._obj_multiplier * transition_cost
                    if g.movable_body.mobility_type == MobilityType.UNACTUATED
                    else transition_cost
                )
        logger.debug(
            f"explored {neighbor} cost to come: {sol_s_to_neighbor.cost}, cfree cost: {cfree_cost}, total cost: {sol_s_to_neighbor.cost + cfree_cost}"
        )

        return cfree_cost

    @staticmethod
    def _find_obj_rob_numbers(s: str):
        match = re.search(r"(obj|rob)(\d+)", s)
        if match:
            return match.group(
                2
            )  # group 2 is the second capture group, which contains the digits
        else:
            return None

    @staticmethod
    def split_vertex_names(vertex_name: str):
        # Convert string representation of tuple to actual tuple
        tuple_vertex = ast.literal_eval(vertex_name)

        # Initialize dictionaries to store modes for each obj and rob
        obj_modes = {}
        rob_modes = {}

        for mode in tuple_vertex:
            # Check if mode contains both obj and rob
            if "obj" in mode and "rob" in mode:
                continue

            # Extract the entity number from the mode string
            entity_num = FactoredCollisionFreeCE._find_obj_rob_numbers(mode)
            # Add mode to appropriate dictionary
            if "obj" in mode:
                if entity_num not in obj_modes:
                    obj_modes[entity_num] = []
                obj_modes[entity_num].append(mode)
            elif "rob" in mode:
                if entity_num not in rob_modes:
                    rob_modes[entity_num] = []
                rob_modes[entity_num].append(mode)

        # Combine mode dictionaries into a list of tuples
        vertex_res = []
        for entity_num, modes in obj_modes.items():
            vertex_res.append(str(tuple(modes)))
        for entity_num, modes in rob_modes.items():
            vertex_res.append(str(tuple(modes)))
        return vertex_res

    @property
    def finger_print(self) -> str:
        return f"FactoredCollisionFreeCE-add_transition_cost-{self._should_add_transition_cost}"
