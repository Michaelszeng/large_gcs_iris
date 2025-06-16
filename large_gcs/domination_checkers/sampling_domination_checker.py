import logging
from dataclasses import dataclass
from typing import Set, Tuple, List, Optional
from itertools import chain

import numpy as np
from pydrake.all import MathematicalProgram, Solve, SolverOptions

from large_gcs.algorithms.search_algorithm import AlgMetrics, SearchNode, profile_method
from large_gcs.domination_checkers.domination_checker import DominationChecker
from large_gcs.geometry.point import Point
from large_gcs.graph.graph import Edge, Graph, ShortestPathSolution, Vertex
from large_gcs.graph.cfree_cost_constraint_factory import shortcut_edge_l2norm_cost_factory, vertex_constraint_last_pos_equality_cfree
logger = logging.getLogger(__name__)


@dataclass
class SetSamples:
    """
    Cache samples for a vertex.
    Note: samples are cached vertex-wise, not path-wise.
    """
    vertex_name: str
    samples: np.ndarray

    @classmethod
    def from_vertex(cls, vertex_name: str, vertex: Vertex, num_samples: int):
        if num_samples == 0:
            samples = np.array([])
        elif isinstance(vertex.convex_set, Point):
            # Do not sample from them, just use the point.
            samples = np.array([vertex.convex_set.center])
        else:
            # np.random.seed(0)
            samples = vertex.convex_set.get_samples(n_samples=num_samples)
            # Round the samples to the nearest 1e-6
            # samples = np.round(samples, 6)
        return cls(
            vertex_name=vertex_name,
            samples=samples,
        )

    def project_single(
        self, graph: Graph, node: SearchNode, sample: np.ndarray
    ) -> np.ndarray:
        """
        Takes a sample point and projects it into the feasible subspace of the path.
        i.e. find the closest point such that the path to that point satisfies 
        the vertex constraints and edge constraints of the path.
        
        Returns the projected sample point.
        """
        vertex_names = node.vertex_path
        edge_names = node.edge_path
        
        # vertices and edges in the path of the search node
        vertices = [graph.vertices[name] for name in vertex_names]
        edges = [graph.edges[edge] for edge in edge_names]

        prog = MathematicalProgram()
        
        # Name the vertices by index since cycles are allowed otherwise might get duplicate names.
        vertex_vars = [
            prog.NewContinuousVariables(v.convex_set.set.ambient_dimension(), name=f"v{v_idx}_vars")
            for v_idx, v in enumerate(vertices)
        ]
        sample_vars = vertex_vars[-1][-graph.base_dim:]  # i.e. the last knot point

        # Add cost to minimize the distance of the sample variables to the actual sample
        prog.AddCost((sample_vars - sample).dot(sample_vars - sample))
        
        # Vertex Constraints
        for (v, x) in zip(vertices, vertex_vars):
            v.convex_set.set.AddPointInSetConstraints(prog, x)  # containment in convex set
            for constraint in v.constraints:  # other constraints on the vertex
                prog.AddConstraint(constraint, x)

        # Edge Constraints
        for idx, (e, e_name) in enumerate(zip(edges, edge_names)):
            for constraint in e.constraints:
                u_idx, v_idx = idx, idx + 1
                variables = np.hstack((vertex_vars[u_idx], vertex_vars[v_idx]))
                prog.AddConstraint(constraint, variables)

        solver_options = SolverOptions()
        result = Solve(prog, solver_options=solver_options)
        if not result.is_success():
            logger.error(
                f"Failed to project sample for vertex {node.vertex_name}"
                f"\nnum total samples for this vertex: {len(self.samples)}"
                f"\nsample: {sample}"
                f"\nvertex_path: {node.vertex_path}"
            )
            return None
        return result.GetSolution(sample_vars)


class SamplingDominationChecker(DominationChecker):
    def __init__(
        self,
        graph: Graph,
        num_samples_per_vertex: int,
        should_use_candidate_sol_as_sample: bool = False,
    ):
        super().__init__(graph)

        self._num_samples_per_vertex = num_samples_per_vertex
        
        # This option is not well-investigated. Basically, if true, the 
        # optimal candidate solution is used as a sample for domination checking.
        # Unclear if this is a good idea.
        self._should_use_candidate_sol_as_sample_as_sample = should_use_candidate_sol_as_sample
        
        # Keeps track of samples for each vertex(set) in the graph.
        # These samples are not used directly but first projected into the feasible subspace of a particular path.
        self._set_samples: dict[str, SetSamples] = {}

    def set_alg_metrics(self, alg_metrics: AlgMetrics):
        self._alg_metrics = alg_metrics
        call_structure = {
            "_is_dominated": [
                "_maybe_add_set_samples",
                "project_single",
            ],
        }
        alg_metrics.update_method_call_structure(call_structure)
        
    def is_dominated(
        self, candidate_node: SearchNode, alternate_nodes: list[SearchNode]
    ) -> bool:
        """
        Return False if candidate is not dominated by union of alternate paths
        (i.e. there exists a sample where the candidate is cheaper).
        
        Else, return True.
        """
        last_vertex_name = candidate_node.vertex_name
        
        # Generate samples if samples don't already exist for the last vertex in the path 
        # (and cache them in self._set_samples)
        self._maybe_add_set_samples(last_vertex_name)
        samples = []
        if self._should_use_candidate_sol_as_sample_as_sample:
            # The last vertex in the trajectory will be the target,
            # The second last would be the candidate vertex
            samples.append(candidate_node.sol.trajectory[-2])
        samples += list(self._set_samples[last_vertex_name].samples)
        
        # Cache for path costs
        node_sols: dict[SearchNode, ShortestPathSolution] = {}
        
        # Return values
        candidate_is_dominated = False
        alt_n_to_prune_from_S = []  # List of nodes to prune from S
        
        # For each node, check if it's non-dominated by any other node
        for node in chain([candidate_node], alternate_nodes):  # use itertools.chain to avoid copying the list
            node_is_dominated = True
            # Compare node to other nodes on each sample
            for idx, sample in enumerate(samples):
                # Project samples into the feasible subspace of the path
                # Candidate sol does not need to be projected
                if self._should_use_candidate_sol_as_sample_as_sample and idx == 0:
                    logger.debug(f"Using candidate sol as sample")
                    proj_sample = sample
                else:
                    proj_sample = self._set_samples[node.vertex_name].project_single(self._graph, node, sample)

                if proj_sample is None:
                    # If the projection failed assume that the candidate is not feasible, and reject the path
                    return True
                
                # Add a new vertex to replace the last vertex of the path.
                # This is necessary because costs and constraints are added by vertex name.
                # We need to apply a unique constraint on last vertex to ensure
                # equality of the last knot point with the sample. If the last 
                # vertex is visited multiple times throughout the path, the constraint 
                # will be applied multiple times and the solver will fail. Therefore, 
                # we add a new vertex with a new name.
                sample_vertex_name = f"{last_vertex_name}_sample"
                self._graph.add_vertex(
                    vertex=Vertex(
                        convex_set=self._graph.vertices[last_vertex_name].convex_set,
                        costs=self._graph.vertices[last_vertex_name].costs,  # Copy cost from original vertex
                        constraints=[vertex_constraint_last_pos_equality_cfree(self._graph.base_dim, self._graph.num_knot_points, sample_vertex_name, proj_sample)],
                    ),
                    name=sample_vertex_name,
                )
                
                if node not in node_sols:
                    # Solve the convex restriction for this path to the current sample.
                    # _compute_candidate_sol is just a wrapper around _solve_conv_res_to_sample
                    # except for REACHESNEW domination checks the candidate solution
                    # isn't needed, so this wrapper just returns (None, True). 
                    node_sol, suceeded = self._compute_candidate_sol(
                        node, sample_vertex_name, sample
                    )
                    node_sols[node] = node_sol
                
                if not suceeded:
                    self._graph.remove_vertex(sample_vertex_name)
                    continue
                
                # Check if the node is dominated by any other nodes on this sample
                any_single_domination = False
                for other_node in chain([candidate_node], alternate_nodes):
                    # Don't compare node to itself
                    # if other_node == node:
                    #     continue
                    
                    if other_node not in node_sols:
                        alt_sol, suceeded = self._compute_candidate_sol(
                            other_node, sample_vertex_name, sample
                        )
                        node_sols[other_node] = alt_sol
                    
                    if self._is_single_dominated(node_sols[node], node_sols[other_node]):
                        self._graph.remove_vertex(sample_vertex_name)
                        any_single_domination = True
                        break
                    
                # Must check all samples for this node before pruning
                if any_single_domination:
                    continue
            
                # node was not dominated by any other node on this sample
                # no need to check other samples
                node_is_dominated = False
                break
            
            if node_is_dominated:
                if node == candidate_node:
                    candidate_is_dominated = True
                else:
                    alt_n_to_prune_from_S.append(node)
            
        if sample_vertex_name in self._graph.vertices:
            self._graph.remove_vertex(sample_vertex_name)
        self._graph.set_target(self._target)
        print(f"Path {candidate_node.vertex_path} is_dominated: {candidate_is_dominated}")
        print(f"Pruning {len(alt_n_to_prune_from_S)} nodes from S.")
        return candidate_is_dominated, alt_n_to_prune_from_S

    def _is_single_dominated(
        self, candidate_sol: ShortestPathSolution, alt_sol: ShortestPathSolution
    ) -> bool:
        raise NotImplementedError

    def _compute_candidate_sol(
        self, candidate_node: SearchNode, sample_name: str, sample: np.ndarray
    ) -> Optional[ShortestPathSolution]:
        raise NotImplementedError

    @profile_method
    def _maybe_add_set_samples(self, vertex_name: str) -> None:
        """
        Generate samples for a vertex if they don't already exist.
        """
        # Subtract 1 from the number of samples needed if we should use the provided sample is provided
        n_samples_needed = (
            self._num_samples_per_vertex - 1
            if self._should_use_candidate_sol_as_sample_as_sample
            else self._num_samples_per_vertex
        )

        # Generate samples if samples don't already exist (and cache them in self._set_samples)
        if vertex_name not in self._set_samples:
            logger.debug(f"Adding samples for {vertex_name}")
            # Generate sample within convex set of vertex
            self._set_samples[vertex_name] = SetSamples.from_vertex(
                vertex_name,
                self._graph.vertices[vertex_name],
                n_samples_needed,
            )

    def _solve_conv_res_to_sample(
        self, node: SearchNode, sample_vertex_name: str, sample: np.ndarray
    ) -> ShortestPathSolution:
        """Solve convex restriction along node's path, but with the last vertex
        replaced by the sample vertex (which is identical, just with the added 
        constraint that the last knot point must equal the sample)."""
        
        # Edge case: If the path is only a single vertex, return a trivial solution
        if len(node.edge_path) == 0:
            return ShortestPathSolution(
                is_success=True,
                cost=0,
                time=0,
                vertex_path=[node.vertex_name],
                trajectory=[self._graph.vertices[node.vertex_name].convex_set.center, sample],
            )
            
        # Add edge between the sample and the second to last vertex in the path
        # effectively replacing the original last edge in the path that connected
        # the second last vertex to the original last vertex.
        e = self._graph.edges[node.edge_path[-1]]
        edge_to_sample = Edge(
            u=e.u,
            v=sample_vertex_name,
            costs=e.costs,
            constraints=e.constraints,
        )
        self._graph.add_edge(edge_to_sample)
        self._graph.set_target(sample_vertex_name)
        active_edges = node.edge_path.copy()
        active_edges[-1] = edge_to_sample.key

        # sol = self._graph.solve_convex_restriction(active_edges, skip_post_solve=True)
        sol = self._graph.solve_convex_restriction(active_edges, skip_post_solve=False)
        self._alg_metrics.update_after_gcs_solve(sol.time)
        # Clean up edge, but leave the sample vertex (which may be used by other alternate paths)
        self._graph.remove_edge(edge_to_sample.key)
        return sol

    def plot_set_samples(self, vertex_name: str):
        self._maybe_add_set_samples(vertex_name)
        samples = self._set_samples[vertex_name].samples
        self._graph.plot_points(samples, edgecolor="black")
        self._graph.vertices[vertex_name].convex_set.plot()

    def plot_projected_samples(self, node: SearchNode):
        self._maybe_add_set_samples(node.vertex_name)
        projected_samples = self._set_samples[node.vertex_name].project_all_gcs(
            self._graph, node, AlgMetrics()
        )
        self._graph.plot_points(projected_samples, edgecolor="blue")
