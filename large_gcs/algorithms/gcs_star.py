import logging
import time
from collections import deque
from typing import Optional

import numpy as np
from tqdm import tqdm

from large_gcs.algorithms.search_algorithm import (
    AlgVisParams,
    SearchAlgorithm,
    SearchNode,
    TieBreak,
    profile_method,
)
from large_gcs.cost_estimators.cost_estimator import CostEstimator
from large_gcs.domination_checkers.domination_checker import DominationChecker
from large_gcs.domination_checkers.sampling_domination_checker import (
    SamplingDominationChecker,
)
from large_gcs.graph.graph import Graph, ShortestPathSolution
from large_gcs.graph.polyhedron_graph import PolyhedronGraph
from large_gcs.geometry.voxel import Voxel

logger = logging.getLogger(__name__)
# tracemalloc.start()


class GcsStar(SearchAlgorithm):
    """
    Note: this doesn't use a subgraph, but operates directly on the graph.
    """

    def __init__(
        self,
        graph: Graph,
        cost_estimator: CostEstimator,
        domination_checker: DominationChecker,
        tiebreak: TieBreak = TieBreak.FIFO,
        vis_params: Optional[AlgVisParams] = None,
        heuristic_inflation_factor: float = 1,
        terminate_early: bool = False,
        invert_S: bool = False,
        max_len_S_per_vertex: int = 0,  # 0 means no limit
        load_checkpoint_log_dir: Optional[str] = None,
        override_wall_clock_time: Optional[float] = None,
        save_expansion_order: bool = False,
        allow_cycles: bool = True,
        animate: bool = True,
    ):
        super().__init__(
            graph=graph,
            heuristic_inflation_factor=heuristic_inflation_factor,
            vis_params=vis_params,
            tiebreak=tiebreak,
        )
        self._cost_estimator = cost_estimator
        self._domination_checker = domination_checker
        self._terminate_early = terminate_early
        self._invert_S = invert_S
        self._max_len_S_per_vertex = max_len_S_per_vertex
        self._cost_estimator.set_alg_metrics(self._alg_metrics)
        self._domination_checker.set_alg_metrics(self._alg_metrics)

        if invert_S:
            self.add_node_to_S = self.add_node_to_S_left
            self.remove_node_from_S = self.remove_node_from_S_left

        self._load_checkpoint_log_dir = load_checkpoint_log_dir
        self._save_expansion_order = save_expansion_order
        self._allow_cycles = allow_cycles

        # For logging/metrics
        call_structure = {
            "_run_iteration": [
                "_explore_successor",
                "_generate_successors",
                "_save_metrics",
            ],
            "_explore_successor": [
                "_is_dominated",
            ],
        }
        self._alg_metrics.update_method_call_structure(call_structure)

        start_node = SearchNode.from_source(self._graph.source_name)
        start_node.sol = ShortestPathSolution(
            is_success=True,
            cost=0,
            time=0,
            vertex_path=[self._graph.source_name, self._graph.target_name],
            trajectory=[self._graph.vertices[self._graph.source_name].convex_set.center, self._graph.vertices[self._graph.target_name].convex_set.center],
        )
        self.push_node_on_Q(start_node)
        self.add_node_to_S(start_node)  # Visited dictionary
        self.update_visited(start_node)  # Purely for logging

        # For continuing search from a checkpoint
        if load_checkpoint_log_dir is not None:
            self.load_checkpoint(
                load_checkpoint_log_dir,
                override_wall_clock_time=override_wall_clock_time,
            )
            self._load_graph_from_checkpoint_data()

        # Override skip post solve if domination checker requires it
        self._override_skip_post_solve = None
        if (
            isinstance(self._domination_checker, SamplingDominationChecker)
            and self._domination_checker._should_use_candidate_sol_as_sample_as_sample == True
        ) or vis_params.animate:
            # If the domination checker is going to use the candidate solution,
            # Then the trajectory needs to be processed by the post solve.
            self._override_skip_post_solve = False

    def run(self) -> ShortestPathSolution:
        """Searches for a shortest path in the given graph."""
        logger.info(f"Running {self.__class__.__name__}")
        if self._load_checkpoint_log_dir is not None:
            start_time = time.time() - self._alg_metrics.time_wall_clock
        else:
            start_time = time.time()
        sol: Optional[ShortestPathSolution] = None
        while sol == None and len(self._Q) > 0:
            sol = self._run_iteration()
            self._alg_metrics.time_wall_clock = time.time() - start_time
            # self._graph.plt.pause(10)
        if sol is None:
            logger.warning(
                f"{self.__class__.__name__} failed to find a path to the target."
            )
            return
        logger.info(
            f"{self.__class__.__name__} complete! \ncost: {sol.cost}, time: {sol.time}"
            f"\nvertex path: {np.array(sol.vertex_path)}"
        )
        # Call post-solve again in case other solutions were found after this was first visited.
        self._graph._post_solve(sol)
        
        # Keep final solution plot open
        if self._vis_params.animate:
            self._graph.update_animation(block=True)
        
        return sol

    @profile_method
    def _run_iteration(self) -> Optional[ShortestPathSolution]:
        """Runs one iteration of the search algorithm."""
        n: SearchNode = self.pop_node_from_Q()
        
        # Janky way to skip covered voxels in the PolyhedronGraph "Best Voxel Inflation" algorithm
        if isinstance(self._graph, PolyhedronGraph):
            if isinstance(self._graph.vertices[n.vertex_name].convex_set, Voxel) and n.vertex_name not in self._graph.uncovered_voxels.map:  # i.e. node ends at a covered voxel
                print(f"Skipping node popped from Q because it ends at a covered voxel: {n.vertex_name}")
                return
        
        print(f"Popped node from Q with priority {n.priority}: {n.vertex_path}")
        
        if self._vis_params.animate:
            self._graph.update_animation(n.sol)
        
        if self._save_expansion_order:
            self._alg_metrics.expansion_order.append(n.vertex_path)

        # Check termination condition
        if n.vertex_name == self._graph.target_name:
            self._save_metrics(n, 0, override_save=True)
            return n.sol

        if n.vertex_name not in self._expanded:
            # Generate successors that you are about to explore
            self._generate_successors(n.vertex_name)

        self.update_expanded(n)  # Keep track of expanded vertices to prevent re-expansion.

        # Retrieve successors generated by generate_successors
        successors = self._graph.successors(n.vertex_name)

        self._save_metrics(n, len(successors))
        
        # FOR DEBUGGING
        # if n.vertex_name == "36":
        #     self._graph.plt.pause(100)
        
        # Iterate over successors, add their search nodes to Q (if not dominated)
        for v in successors:
            if not self._allow_cycles and v in n.vertex_path:
                continue
            
            early_terminate_sol = self._explore_successor(n, v)
            if early_terminate_sol is not None:
                return early_terminate_sol

    @profile_method
    def _generate_successors(self, vertex_name: str) -> None:
        """Generates neighbors for the given vertex.

        Wrapped to allow for profiling.
        """
        self._graph.generate_successors(vertex_name)

    @profile_method
    def _explore_successor(
        self, n: SearchNode, successor: str
    ) -> Optional[ShortestPathSolution]:
        """
        Decide that a candidate path is either dominated or not (in which case, 
        add it to Q and S).
        """
        
        # Create a path from s to successor
        n_next = SearchNode.from_parent(child_vertex_name=successor, parent=n)

        # We solve for the optimal path from s to successor to t for a few reasons
        # here. First, we can check if the path is feasible at all. Second, 
        # if it is feasible, we can use the resulting cost \tilde{f}(v) to order
        # this path in the queue. Third, if SamplingDominationChecker is used and
        # _should_use_candidate_sol_as_sample_as_sample is true, the optimal candidate
        # solution is used as a sample for domination checking.
        sol: ShortestPathSolution = self._cost_estimator.estimate_cost(
            self._graph,
            successor,
            n,
            heuristic_inflation_factor=self._heuristic_inflation_factor,
            override_skip_post_solve=self._override_skip_post_solve,
        )
        
        # Check feasibility of path before checking domination condition
        if not sol.is_success:
            logger.debug(f"Path not actually feasible")
            return  # Path invalid, do nothing, don't add to Q
        else:
            logger.debug(f"Path is feasible")

        n_next.sol = sol
        n_next.priority = sol.cost
        logger.debug(
            f"Exploring path (length {len(n_next.vertex_path)}) {n_next.vertex_path}"
        )

        # Check domination condition
        if (  # If going to target, do not need to check domination condition
            successor != self._target_name
            # and n.vertex_name != self._graph.source_name
            and self._is_dominated(n_next)
        ):
            # Path does not reach new areas, do not add to Q or S
            logger.debug(f"Not added to Q: Path to is dominated")
            self.update_pruned(n_next)
            return
        logger.debug(f"Added to Q: Path not dominated")
        if (
            self._max_len_S_per_vertex != 0
            and len(self._S[n_next.vertex_name]) >= self._max_len_S_per_vertex
        ):
            self.remove_node_from_S(n_next.vertex_name)
        self.add_node_to_S(n_next)
        self.push_node_on_Q(n_next)
        self.update_visited(n_next)  # Purely for logging

        # Early Termination
        if self._terminate_early and successor == self._target_name:
            logger.info(f"EARLY TERMINATION: Visited path to target.")
            self._save_metrics(n_next, 0, override_save=True)
            return n_next.sol

    @profile_method
    def _is_dominated(self, n: SearchNode) -> bool:
        """Checks if the given node is dominated by any other node in the
        visited set."""
        # Check for trivial domination case
        if n.vertex_name not in self._S:
            return False

        return self._domination_checker.is_dominated(n, self._S[n.vertex_name])

    def load_checkpoint(self, checkpoint_log_dir, override_wall_clock_time=None):
        super().load_checkpoint(checkpoint_log_dir, override_wall_clock_time)
        self._cost_estimator.set_alg_metrics(self._alg_metrics)
        self._domination_checker.set_alg_metrics(self._alg_metrics)

    def _load_graph_from_checkpoint_data(self):
        # Create a FIFO queue of nodes to expand
        vertices_to_expand = deque()
        vertices_to_expand.append(self._graph.source_name)
        newly_expanded = set()
        total_vertices = len(self._expanded)

        # Use tqdm with leave=False to prevent it from leaving a progress bar after completion
        with tqdm(
            total=total_vertices, desc="Loading graph from checkpoint data", leave=False
        ) as pbar:
            while len(vertices_to_expand) > 0:
                vertex_name = vertices_to_expand.popleft()
                self._graph.generate_successors(vertex_name)
                newly_expanded.add(vertex_name)
                edges = self._graph.outgoing_edges(vertex_name)
                for edge in edges:
                    neighbor = edge.v
                    if (
                        (neighbor in self._expanded)
                        and (neighbor not in newly_expanded)
                        and (neighbor not in vertices_to_expand)
                    ):
                        vertices_to_expand.append(neighbor)
                pbar.update(1)

        logger.info(f"Graph loaded with {len(newly_expanded)} vertices expanded.")
