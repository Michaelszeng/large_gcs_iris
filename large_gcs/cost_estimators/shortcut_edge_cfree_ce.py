import logging
from typing import Optional

from large_gcs.algorithms.search_algorithm import SearchNode
from large_gcs.contact.contact_set import ContactPointSet, ContactSet
from large_gcs.cost_estimators.cost_estimator import CostEstimator
from large_gcs.graph.graph import Edge, Graph, ShortestPathSolution
from large_gcs.utils.hydra_utils import get_function_from_string

logger = logging.getLogger(__name__)


class ShortcutEdgeCfreeCE(CostEstimator):
    def __init__(
        self,
        graph: Graph,
        shortcut_edge_cost_factory=None,
        add_const_cost: bool = False,
        const_cost: float = 1e-1,
    ):
        print(shortcut_edge_cost_factory)
        print("================================================")
        # To allow function string path to be passed in from hydra config
        if type(shortcut_edge_cost_factory) == str:
            shortcut_edge_cost_factory = get_function_from_string(
                shortcut_edge_cost_factory
            )

        if (
            shortcut_edge_cost_factory is None
            and graph._default_costs_constraints.edge_costs is None
        ):
            raise ValueError(
                "If no shortcut_edge_cost_factory is specified, edge costs must be specified in the graph's default costs constraints."
            )
        self._graph = graph
        self._shortcut_edge_cost_factory = shortcut_edge_cost_factory
        self._add_const_cost = add_const_cost
        self._const_cost = const_cost
        
    def estimate_cost(
        self,
        graph: Graph,
        successor: str,
        node: SearchNode,
        heuristic_inflation_factor: float,
        override_skip_post_solve: Optional[bool] = None,
    ) -> ShortestPathSolution:
        """
        Computes \tilde{f}(\mathbf{v}).
        
        To estimate this cost, we add a shortcut edge from the successor to the 
        target with cost equal to the heuristic given in 
        _shortcut_edge_cost_factory (and no constraints).
        
        Then, we solve the convex restriction for the optimal path from source 
        to target using this modified graph with the shortcut edge.
        """       

        # Check if this neighbor is the target to see if shortcut edge is required
        add_shortcut_edge = successor != self._graph.target_name
        edge_to_successor = Edge.key_from_uv(node.vertex_name, successor)
        if add_shortcut_edge:
            # Add an edge from the neighbor to the target
            shortcut_edge_costs = None
            if self._shortcut_edge_cost_factory:
                
                # Compute shortcut edge costs
                shortcut_edge_costs = self._shortcut_edge_cost_factory(
                    u=successor,
                    base_dim=self._graph.base_dim,
                    num_knot_points=self._graph.num_knot_points,
                    heuristic_inflation_factor=heuristic_inflation_factor,
                    add_const_cost=self._add_const_cost,
                    const_cost=self._const_cost
                )               

            # Add shortcut edge to graph
            edge_to_target = Edge(
                u=successor,
                v=self._graph.target_name,
                key_suffix="shortcut",
                costs=shortcut_edge_costs,
            )
            graph.add_edge(edge_to_target)
            conv_res_active_edges = node.edge_path + [
                edge_to_successor,
                edge_to_target.key,
            ]
        else:  # successor is the target; no shortcut edge needed
            conv_res_active_edges = node.edge_path + [edge_to_successor]

        skip_post_solve = (
            add_shortcut_edge
            if override_skip_post_solve is None
            else override_skip_post_solve
        )
        # If used shortcut edge, do not parse the full result since we won't use the solution.
        sol = graph.solve_convex_restriction(
            conv_res_active_edges, skip_post_solve=skip_post_solve
        )
        print(f"{sol.vertex_path} sol.cost: {sol.cost}")

        self._alg_metrics.update_after_gcs_solve(sol.time)

        # Remove the extra added shortcut edge
        if add_shortcut_edge:
            logger.debug(f"Removing edge {edge_to_target.key}")
            graph.remove_edge(edge_to_target.key)

        return sol

    @property
    def finger_print(self) -> str:
        return f"ShortcutEdgeCE-{self._shortcut_edge_cost_factory.__name__}"
