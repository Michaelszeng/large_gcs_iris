import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from pydrake.all import Hyperrectangle as DrakeHyperrectangle

from large_gcs.geometry.convex_set import ConvexSet


class Voxel(ConvexSet):
    """
    Voxel convex set defined by a center and a voxel size (side length).
    """

    def __init__(self, center, voxel_size, num_knot_points):
        """
        The voxel will have dimension equal to the dimension of `center`.
        
        However, the Hyperrectangle representation of the voxel will have 
        dimension equal to `center * num_knot_points`, because GCS wants to
        express the vertex in the form x ∈ K, where x contains all the knot 
        points and K is the voxel's underlying convex set.
        """
        lb = np.hstack([center - voxel_size / 2] * num_knot_points)  # Repeat the lower bound for each knot point
        ub = np.hstack([center + voxel_size / 2] * num_knot_points)  # Repeat the upper bound for each knot point
        self._set = DrakeHyperrectangle(lb, ub)
        
        lb_in_space = np.hstack([center - voxel_size / 2])
        ub_in_space = np.hstack([center + voxel_size / 2])
        self._set_in_space = DrakeHyperrectangle(lb_in_space, ub_in_space)
        
        self._voxel_size = voxel_size
        self._num_knot_points = num_knot_points
        
    def _plot(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()

        if self.dim == 2:
            lb = self.lb
            ub = self.ub
            vertices = np.array([
                [lb[0], lb[1]],
                [ub[0], lb[1]],
                [ub[0], ub[1]],
                [lb[0], ub[1]],
            ])
            
            polygon = Polygon(vertices, closed=True, **kwargs)
            ax.add_patch(polygon)
        else:
            raise NotImplementedError("Plotting is only supported for 2D Voxels.")
    
    @property
    def lb(self):
        """Lower bound of voxel in space; NOT of the underlying convex set."""
        return self._set.lb()[:self.dim]
    
    @property
    def ub(self):
        """Upper bound of voxel in space; NOT of the underlying convex set."""
        return self._set.ub()[:self.dim]

    @property
    def dim(self):
        """Dimension of space; NOT of the underlying convex set."""
        return self._set.lb().shape[0] // self._num_knot_points

    @property
    def set(self):
        """Hyperrectangle representation of the voxel in knot space."""
        return self._set
    
    @property
    def set_in_space(self):
        """Hyperrectangle representation of the voxel in space."""
        return self._set_in_space

    @property
    def center(self):
        """Center of the voxel in space; NOT of the underlying convex set."""
        return ((self._set.lb() + self._set.ub()) / 2)[:self.dim]
    
    @property
    def size(self):
        """Side length of voxel"""
        return self._voxel_size