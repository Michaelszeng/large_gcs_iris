import logging
from typing import List

import numpy as np

from large_gcs.algorithms.search_algorithm import AlgMetrics, SearchNode
from large_gcs.domination_checkers.ah_containment_domination_checker import (
    AHContainmentDominationChecker,
)
from large_gcs.domination_checkers.ah_containment_last_pos import (
    AHContainmentLastPos,
    ReachesCheaperLastPosContainment,
    ReachesNewLastPosContainment,
)
from large_gcs.domination_checkers.domination_checker import DominationChecker
from large_gcs.domination_checkers.reaches_cheaper_containment import (
    ReachesCheaperContainment,
)
from large_gcs.domination_checkers.reaches_cheaper_sampling import (
    ReachesCheaperSampling,
)
from large_gcs.domination_checkers.reaches_new_containment import ReachesNewContainment
from large_gcs.domination_checkers.reaches_new_sampling import ReachesNewSampling
from large_gcs.domination_checkers.sampling_containment_domination_checker import (
    SamplingContainmentDominationChecker,
)
from large_gcs.domination_checkers.sampling_domination_checker import (
    SamplingDominationChecker,
    SetSamples,
)
from large_gcs.domination_checkers.sampling_last_pos import (
    ReachesCheaperLastPosSampling,
    ReachesNewLastPosSampling,
    SamplingLastPos,
)
from large_gcs.graph.graph import Graph

logger = logging.getLogger(__name__)


class SamplingContainmentWPruneSDominationChecker(SamplingContainmentDominationChecker):

    def is_dominated(
        self, candidate_node: SearchNode, alternate_nodes: List[SearchNode]
    ) -> bool:

        # Get single sample
        self._maybe_add_set_samples(candidate_node.vertex_name)
        sample = self._set_samples[candidate_node.vertex_name].samples[0]
        # Before we project the first sample we need to init the graph
        self._set_samples[candidate_node.vertex_name].init_graph_for_projection(
            self._graph, candidate_node, self._alg_metrics
        )
        proj_sample = self._set_samples[candidate_node.vertex_name].project_single_gcs(
            self._graph, candidate_node, sample
        )
        # Check whether the candidate is dominated by each of the alternate nodes for that sample

        # Create a new vertex for the sample and add it to the graph
        sample_vertex_name = f"{candidate_node.vertex_name}_sample_{0}"

        self._add_sample_to_graph(
            sample=proj_sample,
            sample_vertex_name=sample_vertex_name,
            candidate_node=candidate_node,
        )

        candidate_sol, suceeded = self._compute_candidate_sol(
            candidate_node, sample_vertex_name
        )
        if not suceeded:
            self._graph.remove_vertex(sample_vertex_name)
            # This should never happen
            self._graph.set_target(self._target)
            return False

        sample_is_dominated = np.full(len(alternate_nodes), False)

        for alt_i, alt_n in enumerate(alternate_nodes):
            # logger.debug(f"Checking alternate path {alt_i} of {len(alternate_nodes)} for sample {idx}")
            alt_sol = self._solve_conv_res_to_sample(alt_n, sample_vertex_name)
            sample_is_dominated[alt_i] = self._is_single_dominated(
                candidate_sol, alt_sol
            )

        # Clean up sample vertex
        self._graph.remove_vertex(sample_vertex_name)
        self._graph.set_target(self._target)

        AH_n = self._maybe_create_path_AH_polytope(candidate_node)
        logger.debug(
            f"Checking domination of candidate node terminating at vertex {candidate_node.vertex_name}"
            f"\n via path: {candidate_node.vertex_path}"
        )
        are_keeping = False
        alt_paths_to_prune = []
        for alt_i, alt_n in enumerate(alternate_nodes):
            AH_alt = self._maybe_create_path_AH_polytope(alt_n)
            if sample_is_dominated[alt_i] and not are_keeping:
                # Check whether candidate is contained in alternate
                logger.debug(
                    f"Checking if candidate node is dominated by alternate node with path:"
                    f"{alt_n.vertex_path}"
                )

                if self.is_contained_in(AH_n, AH_alt):
                    return True
            elif not sample_is_dominated[alt_i]:
                if self.is_contained_in(AH_alt, AH_n):
                    alt_paths_to_prune.append(alt_n)
                    are_keeping = True
        # Prune alternate paths
        for alt_n in alt_paths_to_prune:
            alternate_nodes.remove(alt_n)
        self._alg_metrics._S_pruned_counts[candidate_node.vertex_name] += len(
            alt_paths_to_prune
        )

        return False


class ReachesNewSamplingContainmentWPruneS(
    SamplingContainmentWPruneSDominationChecker,
    ReachesNewContainment,
    ReachesNewSampling,
):
    pass


class ReachesCheaperSamplingContainmentWPruneS(
    SamplingContainmentWPruneSDominationChecker,
    ReachesCheaperContainment,
    ReachesCheaperSampling,
):
    pass


class SamplingContainmentLastPosDominationCheckerWPruneS(
    SamplingContainmentWPruneSDominationChecker, AHContainmentLastPos, SamplingLastPos
):
    pass


class ReachesNewLastPosSamplingContainmentWPruneS(
    SamplingContainmentWPruneSDominationChecker,
    ReachesNewLastPosContainment,
    ReachesNewLastPosSampling,
):
    pass


class ReachesCheaperLastPosSamplingContainmentWPruneS(
    SamplingContainmentWPruneSDominationChecker,
    ReachesCheaperLastPosContainment,
    ReachesCheaperLastPosSampling,
):
    pass
