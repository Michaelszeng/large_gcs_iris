import logging
import numpy as np
from typing import Tuple

from large_gcs.algorithms.search_algorithm import SearchNode
from large_gcs.domination_checkers.sampling_domination_checker import (
    SamplingDominationChecker,
)
from large_gcs.graph.graph import ShortestPathSolution

logger = logging.getLogger(__name__)


class ReachesCheaperSampling(SamplingDominationChecker):
    """Checks samples to see if this path reaches any projected sample cheaper
    than any previous path.

    Assumes that this path is feasible.
    """

    def _is_single_dominated(
        self, candidate_sol: ShortestPathSolution, alt_sol: ShortestPathSolution
    ) -> bool:
        # Assumes candidate_sol is feasible
        
        if candidate_sol.vertex_path is not None and candidate_sol.vertex_path == ["source", "0_0_", "-1_0_", "-1_-1_"]:
            print("\n\n\n\n\n\n\n")
            print(f"candidate_sol.vertex_path: {candidate_sol.vertex_path}")
            print(f"alt_sol.vertex_path: {alt_sol.vertex_path}")
            print("alt_sol.cost: ", alt_sol.cost)
            print("candidate_sol.cost: ", candidate_sol.cost)
            print("alt_sol.is_success: ", alt_sol.is_success)
            print("\n\n\n\n\n\n\n")
            
        return alt_sol.is_success and alt_sol.cost <= candidate_sol.cost

    def _compute_candidate_sol(
        self, candidate_node: SearchNode, sample: np.ndarray
    ) -> Tuple[ShortestPathSolution | None, bool]:
        candidate_sol = self._solve_conv_res_to_sample(
            candidate_node, sample
        )
        if not candidate_sol.is_success:
            logger.error(
                f"Candidate path was not feasible to reach sample {sample}"
                f"\nvertex_path: {candidate_node.vertex_path}"
                f"\n Skipping to next sample"
            )
            # assert sol.is_success, "Candidate path should be feasible"
            
        print(f"{candidate_sol.vertex_path} candidate_sol.cost_: {candidate_sol.cost}")

        return candidate_sol, candidate_sol.is_success
