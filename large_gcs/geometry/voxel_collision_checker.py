import numpy as np
from typing import List, Dict, Any
from pydrake.all import (
    CollisionChecker,
    SceneGraphCollisionChecker,
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
    def __init__(self, obstacles: List[ConvexSet]):
        self.obstacles = obstacles
        
    def check_voxel_collision_free(self, voxel: Voxel) -> bool:
        """
        Check if a voxel is collision free.
        """
        for obstacle in self.obstacles:
            if not obstacle.set.Intersection(voxel.set_in_space.MakeHPolyhedron()).IsEmpty():
                return False
        return True

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
        Check if a list of configurations are collision free.
        """
        return np.all(self.collision_checker.CheckConfigsCollisionFree(configs))
