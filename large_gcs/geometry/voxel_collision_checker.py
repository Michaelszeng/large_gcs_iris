import numpy as np
from typing import List, Dict, Any
from pydrake.all import (
    CollisionChecker,
)

from large_gcs.geometry.voxel import Voxel
from large_gcs.geometry.convex_set import ConvexSet


class VoxelCollisionChecker():
    def __init__(self):
        pass
          
    def check_voxel_collision_free(self, voxel: Voxel) -> bool:
        """
        Unimplemented
        """
        return True



class VoxelCollisionCheckerConvexObstacles(VoxelCollisionChecker):
    def __init__(self, obstacles: List[ConvexSet]):
        pass
        
    def check_voxel_collision_free(self, voxel: Voxel) -> bool:
        """
        Check if a voxel is collision free.
        """
        return True


class VoxelSceneGraphCollisionChecker(VoxelCollisionChecker):
    def __init__(self, collision_checker_params: Dict[str, Any]):
        pass
          
    def check_voxel_collision_free(self, voxel: Voxel) -> bool:
        """
        Check if a voxel is collision free.
        """
        return True
