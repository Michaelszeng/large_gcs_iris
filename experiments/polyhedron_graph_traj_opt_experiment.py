"""
Experiment finding shortest path in a robot configuration space.
"""

import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

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
from large_gcs.graph.polyhedron_graph import PolyhedronGraph
from large_gcs.geometry.voxel_collision_checker import VoxelSceneGraphCollisionChecker
from large_gcs.graph.graph import ShortestPathSolution
from large_gcs.geometry.polyhedron import Polyhedron
from large_gcs.utils.hydra_utils import get_cfg_from_folder

# TEST_SCENE = "2DOFFLIPPER"
TEST_SCENE = "3DOFFLIPPER"
# TEST_SCENE = "5DOFUR3"
# TEST_SCENE = "6DOFUR3"
# TEST_SCENE = "7DOFIIWA"
# TEST_SCENE = "7DOFBINS"
# TEST_SCENE = "7DOF4SHELVES"
# TEST_SCENE = "14DOFIIWAS"
# TEST_SCENE = "15DOFALLEGRO"

if TEST_SCENE == "2DOFFLIPPER":
    s = np.array([0, 0])
    t = np.array([-1, 1.7])
    voxel_size = 0.25
elif TEST_SCENE == "3DOFFLIPPER":
    s = np.array([0.18, -0.1, -0.78])
    t = np.array([-1.7, 1.0, 1.5])
    voxel_size = 0.5
else:
    raise ValueError(f"TEST_SCENE {TEST_SCENE} not supported yet.")
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

simulator = Simulator(diagram)
meshcat.StartRecording()
simulator.AdvanceTo(0.001)  # Roll forward sim a bit to show the visualization

simulator_context = simulator.get_mutable_context()
plant_context = plant.GetMyMutableContextFromRoot(simulator_context)

num_robot_positions = plant.num_positions()

collision_checker_params = {}
collision_checker_params["robot_model_instances"] = robot_model_instances
collision_checker_params["model"] = diagram
collision_checker_params["edge_step_size"] = 0.125

def animate_sol_meshcat(sol):
    """ Constant-speed stepthrough animation of the solution trajectory."""
    # Get last knot point from each trajectory point
    last_knot_points = []
    for traj_point in sol.trajectory:
        # Reshape into knot points and take the last one
        num_knot_points = len(traj_point) // num_robot_positions
        knot_points = traj_point.reshape(num_knot_points, num_robot_positions)
        last_knot_points.append(knot_points[-1])
    
    # Calculate total path length to normalize speed
    path_length = 0
    for i in range(len(sol.vertex_path)-1):
        diff = last_knot_points[i+1] - last_knot_points[i]
        path_length += np.linalg.norm(diff)
    
    # Set desired animation duration in seconds
    duration = 2.0
    steps_per_segment = 100
    
    for i in range(len(sol.vertex_path)-1):
        start_pos = last_knot_points[i]
        end_pos = last_knot_points[i+1]
        
        # Calculate segment length for speed normalization
        segment_length = np.linalg.norm(end_pos - start_pos)
        
        # Interpolate between points
        for t in range(steps_per_segment):
            alpha = t / steps_per_segment
            current_pos = start_pos + alpha * (end_pos - start_pos)
            
            # Set the position in the plant
            plant.SetPositions(plant_context, current_pos)
            simulator.AdvanceTo(simulator_context.get_time() + 0.001)
            
            # Calculate sleep duration based on segment length relative to total path
            sleep_duration = (duration * segment_length / path_length) / steps_per_segment
            time.sleep(sleep_duration)
            
    meshcat.PublishRecording()


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
    
    workspace = np.hstack([plant.GetPositionLowerLimits().reshape(-1, 1), plant.GetPositionUpperLimits().reshape(-1, 1)])
    print(f"Workspace: {workspace}")  
    
    g = PolyhedronGraph(
        s = s,
        t = t,
        workspace = workspace,
        default_voxel_size = voxel_size,
        const_edge_cost=cfg.const_edge_cost,
        voxel_collision_checker=VoxelSceneGraphCollisionChecker(collision_checker_params),
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
        terminate_early=cfg.terminate_early,
    )
    
    g.init_animation()

    sol: ShortestPathSolution = alg.run()
    assert sol is not None, "No solution found."
    logger.info(f"Solution Trajectory: {sol.trajectory}")
    
    if cfg.save_metrics:
        output_base = (
            f"{alg.__class__.__name__}_"
            + f"{cost_estimator.finger_print}_{cfg.graph_name}"
        )
        metrics_path = Path(full_log_dir) / f"{output_base}_metrics.json"
        alg.save_alg_metrics_to_file(metrics_path)
        
    animate_sol_meshcat(sol)
    
    return

if __name__ == "__main__":
    main()