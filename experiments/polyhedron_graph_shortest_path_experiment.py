"""
Experiment finding shortest path in a 2D or 3D voxel graph with user-defined 
obstacles.
"""

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
from large_gcs.graph.polyhedron_graph import PolyhedronGraph
from large_gcs.geometry.voxel_collision_checker import VoxelCollisionCheckerConvexObstacles
from large_gcs.graph.graph import ShortestPathSolution
from large_gcs.geometry.polyhedron import Polyhedron
from large_gcs.utils.hydra_utils import get_cfg_from_folder

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../config", config_name="shortest_piecewise_linear_path")
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

    # MANUALLY SET
    d = 2
    
    if d == 2:
        # 2D Test
        # obstacles = [Polyhedron.from_vertices([[2,0],[0,2],[2,2]]), Polyhedron.from_vertices([[-2,-0.3],[-0.3,-2],[-2,-2]])]
        # obstacles = [Polyhedron.from_vertices([[0,0],[2,0],[0,2],[2,2]])]
        obstacles = [Polyhedron.from_vertices([[0,0],[3.1,0],[0,3.1],[3.1,3.1]])]  # NO SOLUTION
        workspace = np.array([[-4, 4],    # workspace x-lim
                              [-4, 4]])   # workspace y-lim
        g = PolyhedronGraph(
            s = np.array([-3, -3]),
            t = np.array([3.5, 3.5]),
            workspace = workspace,
            default_voxel_size = 1,
            const_edge_cost=cfg.const_edge_cost,
            voxel_collision_checker=VoxelCollisionCheckerConvexObstacles(obstacles, workspace),
        )
    else:
        # 3D Test
        obstacles = [Polyhedron.from_vertices([[1,0,-1],[0,1,-1],[1,1,-1],[0,0,1],[1,0,1],[0,1,1]])]
        # obstacles = [Polyhedron.from_vertices([[0.9,0.1,-1],[0.1,0.9,-1],[0.9,0.9,-1],[0.9,0.1,1],[0.1,0.9,1],[0.9,0.9,1]])]
        workspace = np.array([[-2.5, 2.5],    # workspace x-lim
                              [-2.5, 2.5],    # workspace y-lim
                              [-2.5, 2.5]])   # workspace z-lim
        g = PolyhedronGraph(
            s = np.array([-1, -1, -1]),
            t = np.array([1.9, 1.9, 1.9]),
            workspace = workspace,
            default_voxel_size = 1,
            const_edge_cost=cfg.const_edge_cost,
            voxel_collision_checker=VoxelCollisionCheckerConvexObstacles(obstacles, workspace),
        )
    
    cost_estimator: CostEstimator = instantiate(
        cfg.cost_estimator, graph=g, add_const_cost=cfg.should_add_const_edge_cost, const_cost=cfg.const_edge_cost
    )
    domination_checker: DominationChecker = instantiate(
        cfg.domination_checker, graph=g
    )
    alg: SearchAlgorithm = instantiate(
        cfg.algorithm,
        graph=g,
        cost_estimator=cost_estimator,
        heuristic_inflation_factor=cfg.heuristic_inflation_factor,
        domination_checker=domination_checker,
        vis_params=AlgVisParams(log_dir=full_log_dir),
    )
    
    g.init_animation()

    sol: ShortestPathSolution = alg.run()
    assert sol is not None, "No solution found."
    logger.info(f"Solution Trajectory: {sol.trajectory}")

    # g.plot(vertices=g.vertices, sol=sol)
    # plt.show()
    
    if cfg.save_metrics:
        output_base = (
            f"{alg.__class__.__name__}_"
            + f"{cost_estimator.finger_print}_{cfg.graph_name}"
        )
        metrics_path = Path(full_log_dir) / f"{output_base}_metrics.json"
        alg.save_alg_metrics_to_file(metrics_path)
    
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
