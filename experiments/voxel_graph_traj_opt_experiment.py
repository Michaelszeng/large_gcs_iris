import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from hydra.utils import get_original_cwd, instantiate
from omegaconf import OmegaConf, open_dict

from large_gcs.algorithms.search_algorithm import AlgVisParams, SearchAlgorithm
from large_gcs.cost_estimators.cost_estimator import CostEstimator
from large_gcs.domination_checkers.domination_checker import DominationChecker
from large_gcs.graph.voxel_graph import VoxelGraph
from large_gcs.graph.graph import ShortestPathSolution
from large_gcs.graph.incremental_contact_graph import IncrementalContactGraph
from large_gcs.graph_generators.contact_graph_generator import (
    ContactGraphGeneratorParams,
)
from large_gcs.geometry.polyhedron import Polyhedron
from large_gcs.utils.hydra_utils import get_cfg_from_folder

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../config", config_name="voxel_graph_traj_opt")
def main(cfg: OmegaConf) -> None:
    # Add log dir to config
    hydra_config = HydraConfig.get()
    full_log_dir = hydra_config.runtime.output_dir
    with open_dict(cfg):
        cfg.log_dir = os.path.relpath(full_log_dir, get_original_cwd() + "/outputs")

    # Save the configuration to the log directory
    run_folder = Path(full_log_dir)
    config_file = run_folder / "config.yaml"
    with open(config_file, "w") as f:
        OmegaConf.save(cfg, f)

    logger.info(cfg)
    
    g = VoxelGraph(
        [Polyhedron.from_vertices([[0.9,0.1],[0.1,0.9],[0.9,0.9]])],  # obstacles
        np.array([0, 0]),  # source
        np.array([5, 5]),  # target
        np.array([[-6,  6],    # workspace
                  [-6,  6]]),
        default_voxel_size = 1,
        should_add_gcs = True,
        should_add_const_edge_cost = True,
    )
    
    cost_estimator: CostEstimator = instantiate(
        cfg.cost_estimator, graph=g, add_const_cost=cfg.should_add_const_edge_cost
    )
    domination_checker: DominationChecker = instantiate(
        cfg.domination_checker, graph=g
    )
    alg: SearchAlgorithm = instantiate(
        cfg.algorithm,
        graph=g,
        cost_estimator=cost_estimator,
        domination_checker=domination_checker,
        vis_params=AlgVisParams(log_dir=full_log_dir),
    )

    sol: ShortestPathSolution = alg.run()
    logger.info(f"Solution Trajectory: {sol.trajectory}")

    g.plot_solution(sol)
    plt.show()
    
    return
    
    

    if "load_checkpoint_log_dir" in cfg.algorithm:
        # Make sure checkpoint graph is the same as current graph
        checkpoint_cfg = get_cfg_from_folder(
            Path(cfg.algorithm.load_checkpoint_log_dir)
        )
        if cfg.graph_name != checkpoint_cfg.graph_name:
            raise ValueError("Checkpoint graph name does not match current graph name.")


    save_outputs = cfg.save_metrics or cfg.save_visualization or cfg.save_solution
    if save_outputs:
        output_base = (
            f"{alg.__class__.__name__}_"
            + f"{cost_estimator.finger_print}_{cfg.graph_name}"
        )

    if cfg.save_metrics:
        metrics_path = Path(full_log_dir) / f"{output_base}_metrics.json"
        alg.save_alg_metrics_to_file(metrics_path)

    if sol is not None and cfg.save_solution:
        sol_path = Path(full_log_dir) / f"{output_base}_solution.pkl"
        sol.save(sol_path)

    if sol is not None and cfg.save_visualization:
        vid_file = os.path.join(full_log_dir, f"{output_base}.mp4")

        anim = cg.animate_solution()
        anim.save(vid_file)

        # Generate both a png and a pdf
        traj_figure_file = Path(full_log_dir) / f"{output_base}_trajectory.pdf"
        traj_figure_image = Path(full_log_dir) / f"{output_base}_trajectory.jpg"
        cg.plot_current_solution(traj_figure_file)
        cg.plot_current_solution(traj_figure_image)

    logger.info(f"hydra log dir: {full_log_dir}")


if __name__ == "__main__":
    main()
