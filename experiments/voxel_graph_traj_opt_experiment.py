"""
Experiment finding shortest path in a 2D or 3D voxel graph with user-defined 
obstacles.
"""

import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from pydrake.all import (
    StartMeshcat,
    AddDefaultVisualization,
    Simulator,
    RobotDiagramBuilder,
    RandomGenerator,
)

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from hydra.utils import get_original_cwd, instantiate
from omegaconf import OmegaConf, open_dict

from large_gcs.algorithms.search_algorithm import AlgVisParams, SearchAlgorithm
from large_gcs.cost_estimators.cost_estimator import CostEstimator
from large_gcs.domination_checkers.domination_checker import DominationChecker
from large_gcs.graph.voxel_graph import VoxelGraph
from large_gcs.geometry.voxel_collision_checker import VoxelSceneGraphCollisionChecker
from large_gcs.graph.graph import ShortestPathSolution
from large_gcs.geometry.polyhedron import Polyhedron
from large_gcs.utils.hydra_utils import get_cfg_from_folder

TEST_SCENE = "2DOFFLIPPER"
# TEST_SCENE = "3DOFFLIPPER"
# TEST_SCENE = "5DOFUR3"
# TEST_SCENE = "6DOFUR3"
# TEST_SCENE = "7DOFIIWA"
# TEST_SCENE = "7DOFBINS"
# TEST_SCENE = "7DOF4SHELVES"
# TEST_SCENE = "14DOFIIWAS"
# TEST_SCENE = "15DOFALLEGRO"

rng = RandomGenerator(1234)
np.random.seed(1234)

logger = logging.getLogger(__name__)

src_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(src_directory)
data_directory = os.path.join(parent_directory)
scene_yaml_file = os.path.join(data_directory, "iris_benchmarks_scenes_urdf", "yamls", TEST_SCENE + ".dmd.yaml")

meshcat = StartMeshcat()

robot_diagram_builder = RobotDiagramBuilder()
parser = robot_diagram_builder.parser()
iris_environement_assets = os.path.join(data_directory, "iris_benchmarks_scenes_urdf", "iris_environments", "assets")
parser.package_map().Add("iris_environments", iris_environement_assets)
robot_model_instances = parser.AddModels(scene_yaml_file)
plant = robot_diagram_builder.plant()
plant.Finalize()
AddDefaultVisualization(robot_diagram_builder.builder(), meshcat=meshcat)
diagram = robot_diagram_builder.Build()

# Roll forward sim a bit to show the visualization
simulator = Simulator(diagram)
simulator.AdvanceTo(0.001)

plant_context = plant.CreateDefaultContext()

num_robot_positions = plant.num_positions()

collision_checker_params = {}
collision_checker_params["robot_model_instances"] = robot_model_instances
collision_checker_params["model"] = diagram
collision_checker_params["edge_step_size"] = 0.125

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
    
    workspace = np.hstack([plant.GetPositionLowerLimits().reshape(-1, 1), plant.GetPositionUpperLimits().reshape(-1, 1)])
    print(f"Workspace: {workspace}")  
    
    # 2D Test 
    g = VoxelGraph(
        s = np.array([0, 0]),
        t = np.array([2, 2]),
        workspace = workspace,
        default_voxel_size = 0.25,
        should_add_gcs = True,
        const_edge_cost=cfg.const_edge_cost,
        voxel_collision_checker=VoxelSceneGraphCollisionChecker(collision_checker_params),
    )
    
    # 3D Test
    # g = VoxelGraph(
    #     s = np.array([0, 0, 0]),
    #     t = np.array([2, 2, 2]),
    #     workspace = np.array([[-4,  4],    # workspace x-lim
    #                           [-4,  4],    # workspace y-lim
    #                           [-4,  4]]),  # workspace z-lim
    #     default_voxel_size = 1,
    #     should_add_gcs = True,
    #     const_edge_cost=cfg.const_edge_cost,
    #     voxel_collision_checker=VoxelSceneGraphCollisionChecker(collision_checker_params),
    # )
    
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
    logger.info(f"Solution Trajectory: {sol.trajectory}")
    
    if cfg.save_metrics:
        output_base = (
            f"{alg.__class__.__name__}_"
            + f"{cost_estimator.finger_print}_{cfg.graph_name}"
        )
        metrics_path = Path(full_log_dir) / f"{output_base}_metrics.json"
        alg.save_alg_metrics_to_file(metrics_path)
    
    return

if __name__ == "__main__":
    main()