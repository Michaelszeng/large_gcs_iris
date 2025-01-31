import logging
from dataclasses import dataclass
from typing import Optional

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
            samples = vertex.convex_set.get_samples(num_samples)
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
        active_edges = node.edge_path
        
        # gcs vertices and edges
        vertices = [graph.vertices[name].gcs_vertex for name in vertex_names]
        edges = [graph.edges[edge].gcs_edge for edge in active_edges]

        prog = MathematicalProgram()
        
        # Name the vertices by index since cycles are allowed otherwise might get duplicate names.
        vertex_vars = [
            prog.NewContinuousVariables(v.ambient_dimension(), name=f"v{v_idx}_vars")
            for v_idx, v in enumerate(vertices)
        ]
        sample_vars = vertex_vars[-1][-graph.base_dim:]  # i.e. the last knot point

        # Add cost to minimize the distance of the sample variables to the actual sample
        prog.AddCost((sample_vars - sample).dot(sample_vars - sample))
        
        # Vertex Constraints
        for i, (v, x) in enumerate(zip(vertices, vertex_vars)):

            v.set().AddPointInSetConstraints(prog, x)  # containment in convex set            
            for binding in v.GetConstraints():  # other constraints on the vertex
                constraint = binding.evaluator()
                prog.AddConstraint(constraint, x)

        # Edge Constraints
        for idx, (e, e_name) in enumerate(zip(edges, active_edges)):
            for binding in e.GetConstraints():
                constraint = binding.evaluator()
                u_idx, v_idx = idx, idx + 1
                variables = np.hstack((vertex_vars[u_idx], vertex_vars[v_idx]))
                prog.AddConstraint(constraint, variables)

        solver_options = SolverOptions()
        result = Solve(prog, solver_options=solver_options)
        if not result.is_success():
            logger.error(
                f"Failed to project sample for vertex {node.vertex_name}"
                f"\nnum total samples for this vertex: {len(self.samples)}"
                f"sample: {sample}"
                f"vertex_path: {node.vertex_path}"
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
        Functional overview:
        
        1. Generate samples if samples in the last vertex of the candidate path if 
           they don't already exist (and cache them in self._set_samples)
        2. For each sample:
            a. Project sample into the feasible subspace of the candidate path by
               by solving a convex program that enforces vertex and edge constraints
               along the path to ensure path feasibility but minimizes distance 
               of the final knot point to the sample.
            b. Calculate cost of the candidate path to the project sample by solving 
               the convex restriction with the final vertex of the candidate path
               as the target and with the final knot point of the path being 
               constrained to equal the projected sample.
            b. For each alternate path:
                i. Solve convex restriction for the alternate path to that sample in a similar manner
                ii. Check domination conditions
        """
        
        is_dominated = True
        
        # Generate samples if samples don't already exist (and cache them in self._set_samples)
        self._maybe_add_set_samples(candidate_node.vertex_name)

        samples = []
        if self._should_use_candidate_sol_as_sample_as_sample:
            # The last vertex in the trajectory will be the target,
            # The second last would be the candidate vertex
            samples.append(candidate_node.sol.trajectory[-2])
        
        # Compile cached samples
        samples += list(self._set_samples[candidate_node.vertex_name].samples)

        # Project samples into the feasible subspace of the path
        for idx, sample in enumerate(
            self._set_samples[candidate_node.vertex_name].samples
        ):
            if self._should_use_candidate_sol_as_sample_as_sample and idx == 0:
                logger.debug(f"Using candidate sol as sample")
                # Candidate sol does not need to be projected
                proj_sample = sample
            else:
                proj_sample = self._set_samples[
                    candidate_node.vertex_name
                ].project_single(self._graph, candidate_node, sample)

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
            sample_vertex_name = f"{candidate_node.vertex_name}_sample"
            self._graph.add_vertex(
                vertex=Vertex(
                    convex_set=self._graph.vertices[candidate_node.vertex_name].convex_set,
                    costs=self._graph.vertices[candidate_node.vertex_name].costs,  # Copy cost from original vertex
                    constraints=[vertex_constraint_last_pos_equality_cfree(self._graph.base_dim, self._graph.num_knot_points, sample_vertex_name, sample)],
                ),
                name=sample_vertex_name,
            )

            # Solve the convex restriction for the candidate path to that sample.
            # _compute_candidate_sol is just a wrapper around _solve_conv_res_to_sample
            # except for REACHESNEW domination checks the candidate solution
            # isn't needed, so this wrapper just does nothing. 
            candidate_sol, suceeded = self._compute_candidate_sol(
                candidate_node, sample_vertex_name, sample
            )
            
            if not suceeded:
                self._graph.remove_vertex(sample_vertex_name)
                continue
            
            # Check if the candidate path is dominated by any alternates to the sample
            any_single_domination = False
            for alt_i, alt_n in enumerate(alternate_nodes):
                alt_sol = self._solve_conv_res_to_sample(alt_n, sample_vertex_name, sample)
                if self._is_single_dominated(candidate_sol, alt_sol):
                    self._graph.remove_vertex(sample_vertex_name)
                    any_single_domination = True
                    break

            # If, for the current sample, the candidate path is dominated by some 
            # alternate path, search the next sample to see if the candidate 
            # path is cheaper for a different sample.
            if any_single_domination:
                continue

            # If, for the current sample, the candidate path is not dominated by any alternate path, do not need to check more samples. Return non-dominated.
            is_dominated = False
            if sample_vertex_name in self._graph.vertices:
                self._graph.remove_vertex(sample_vertex_name)
            logger.debug(f"Sample {idx} reached new/cheaper by candidate path")
            break  # and return non-dominated

        self._graph.set_target(self._target)
        return is_dominated

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
        # Subtract 1 from the number of samples needed if we should use the provided sample is provided
        n_samples_needed = (
            self._num_samples_per_vertex - 1
            if self._should_use_candidate_sol_as_sample_as_sample
            else self._num_samples_per_vertex
        )

        # Generate samples if samples don't already exist (and cache them in self._set_samples)
        if vertex_name not in self._set_samples:
            logger.debug(f"Adding samples for {vertex_name}")
            # Generate sample wtihin convex set of vertex
            self._set_samples[vertex_name] = SetSamples.from_vertex(
                vertex_name,
                self._graph.vertices[vertex_name],
                n_samples_needed,
            )

    def _solve_conv_res_to_sample(
        self, node: SearchNode, sample_vertex_name: str, sample: np.ndarray
    ) -> ShortestPathSolution:
        
        # Edge case: If the path is only a single vertex, return a trivial solution
        if len(node.edge_path) == 0:
            return ShortestPathSolution(
                is_success=True,
                cost=0,
                time=0,
                vertex_path=[node.vertex_name],
                trajectory=[self._graph.vertices[node.vertex_name].convex_set.center, sample],
            )
            
        # Add edge between the sample and the second last vertex in the path
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
