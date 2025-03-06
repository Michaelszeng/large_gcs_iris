import numpy as np
from typing import List, Dict, Any, Optional
from pydrake.all import (
    CollisionChecker,
    SceneGraphCollisionChecker,
    ConfigurationSpaceObstacleCollisionChecker,
    RobotDiagramBuilder,
    RandomGenerator,
)

from large_gcs.geometry.voxel import Voxel
from large_gcs.geometry.convex_set import ConvexSet


class VoxelCollisionChecker():
    """Abstract base class for wrappers of VoxelCollisionChecker."""
    def __init__(self):
        pass
          
    def check_voxel_collision_free(self, voxel: Voxel) -> bool:
        pass

class VoxelCollisionCheckerConvexObstacles(VoxelCollisionChecker):
    def __init__(self, obstacles: List[ConvexSet], workspace: np.ndarray, collision_checker_params: Optional[Dict[str, Any]] = None):
        assert len(set([obstacle.dim for obstacle in obstacles])) == 1, "All obstacles must have the same ambient dimension."
        
        self.obstacles = obstacles
        self.ambient_dim = obstacles[0].dim
        
        if collision_checker_params is None:
            collision_checker_params = {}
            
            # Create a urdf of a free-moving point robot in a box
            # Only handle 2D and 3D for now
            assert self.ambient_dim in [2, 3], "Only 2D and 3D are supported for now."
            
            print(workspace)
            print(workspace[0,0])
            
            if self.ambient_dim == 2:
                urdf = f"""<robot name="robot">
                    <link name="movable">
                        <collision name="sphere">
                            <geometry><sphere radius="0.001"/></geometry>
                        </collision>
                    </link>
                    <link name="for_joint"/>
                    <joint name="x" type="prismatic">
                        <axis xyz="1 0 0"/>
                        <limit lower="{workspace[0, 0]}" upper="{workspace[0, 1]}"/>
                        <parent link="world"/>
                        <child link="for_joint"/>
                    </joint>
                    <joint name="y" type="prismatic">
                        <axis xyz="0 1 0"/>
                        <limit lower="{workspace[1, 0]}" upper="{workspace[1, 1]}"/>
                        <parent link="for_joint"/>
                        <child link="movable"/>
                    </joint>
                </robot>"""
            elif self.ambient_dim == 3:
                urdf = f"""<robot name="robot">
                    <link name="movable">
                        <collision name="sphere">
                            <geometry><sphere radius="0.001"/></geometry>
                        </collision>
                    </link>
                    <link name="for_joint"/>
                    <joint name="x" type="prismatic">
                        <axis xyz="1 0 0"/>
                        <limit lower="{workspace[0, 0]}" upper="{workspace[0, 1]}"/>
                        <parent link="world"/>
                        <child link="for_joint"/>
                    </joint>
                    <joint name="y" type="prismatic">
                        <axis xyz="0 1 0"/>
                        <limit lower="{workspace[1, 0]}" upper="{workspace[1, 1]}"/>
                        <parent link="for_joint"/>
                        <child link="movable"/>
                    </joint>
                    <joint name="z" type="prismatic">
                        <axis xyz="0 0 1"/>
                        <limit lower="{workspace[2, 0]}" upper="{workspace[2, 1]}"/>
                        <parent link="for_joint"/>
                        <child link="movable"/>
                    </joint>
                </robot>"""
            
        builder = RobotDiagramBuilder(0.0)
        collision_checker_params["robot_model_instances"] = builder.parser().AddModelsFromString(urdf, "urdf")
        collision_checker_params["model"] = builder.Build()
        collision_checker_params["edge_step_size"] = 0.01
        scene_graph_collision_checker = SceneGraphCollisionChecker(**collision_checker_params)
        self.collision_checker = ConfigurationSpaceObstacleCollisionChecker(scene_graph_collision_checker, [obstacle.set for obstacle in self.obstacles])
        
        self.rng = RandomGenerator(1234)
        
    def check_voxel_collision_free(self, voxel: Voxel, use_intersection: bool = False, num_samples: int = 20) -> bool:
        """
        Check if a voxel is collision free.
        
        If use_intersection is True, we use take the intersection between the polyhedrons to 
        determine whether a collision is present.
        """
        if use_intersection:
            for obstacle in self.obstacles:
                if not obstacle.set.Intersection(voxel.set_in_space.MakeHPolyhedron()).IsEmpty():
                    return False
        else:
            for _ in range(num_samples):
                sample = voxel.set_in_space.UniformSample(self.rng)
                for obstacle in self.obstacles:
                    if obstacle.set.PointInSet(sample):
                        return False  # Collision detected
        return True
    
    @property
    def checker(self):
        return self.collision_checker

class VoxelSceneGraphCollisionChecker(VoxelCollisionChecker):
    def __init__(self, collision_checker_params: Dict[str, Any], num_samples_per_voxel: int = 100):
        self.collision_checker = SceneGraphCollisionChecker(**collision_checker_params)
        self.num_samples_per_voxel = num_samples_per_voxel
          
    def check_voxel_collision_free(self, voxel: Voxel) -> bool:
        """
        Check if a voxel is collision free. We consider a voxel collision-free
        if all samples are collision-free (i.e. no part of the voxel is in collision).
        """
        samples = voxel.get_samples(self.num_samples_per_voxel)
        if np.all(self.collision_checker.CheckConfigsCollisionFree(samples)):
            return True
        else:
            return False
        
    def check_configs_collision_free(self, configs: np.ndarray) -> bool:
        """
        Check if an array of configurations are collision free.
        
        A simple wrapper around Drake's SceneGraphCollisionChecker.CheckConfigsCollisionFree method.
        """
        return self.collision_checker.CheckConfigsCollisionFree(configs)
    
    @property
    def checker(self):
        return self.collision_checker
