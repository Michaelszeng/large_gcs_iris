import logging
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from itertools import product

from pydrake.all import Hyperrectangle as DrakeHyperrectangle
from pydrake.all import RandomGenerator

from large_gcs.geometry.convex_set import ConvexSet

logger = logging.getLogger(__name__)


class Voxel(ConvexSet):
    """
    Voxel convex set defined by a center and a voxel size (side length).
    """

    def __init__(self, center, voxel_size, num_knot_points, parent_region_name=None):
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
        
        # For PolyhedronGraph, the parent region is the IRIS region whose boundary the voxel was generated at
        self._parent_region_name = parent_region_name
        
    def get_samples(self, sample_in_space=True, n_samples=100) -> np.ndarray:
        """
        This needs to be overridden in a voxel graph because the voxel's 
        _set representation has dimension `self.dim * self._num_knot_points`
        while we want the samples to have dimension `self.dim`. Instead of
        sampling from _set, we sample from _set_in_space.
        """
        if sample_in_space:
            samples = []
            generator = RandomGenerator()
            try:
                samples.append(self.set_in_space.UniformSample(generator))
                logger.debug(f"Sampled 1 points from convex set")
                for i in range(n_samples - 1):
                    # Hyperrectangle doesn't need previous sample or mixing steps
                    samples.append(self.set_in_space.UniformSample(generator))
                    logger.debug(f"Sampled {i+2} points from convex set")
            except (RuntimeError, ValueError) as e:
                chebyshev_center = self.set.ChebyshevCenter()
                logger.warn("Failed to sample convex set" f"\n{e}")
                return np.array([chebyshev_center])
            return np.array(samples)
        else:
            return super().get_samples(n_samples)
    
    def get_vertices(self):
        """
        Get the vertices of the voxel in space. Return then as a n x k array,
        where n is the dimension of space and k is the number of vertices.
        """
        lb = self.set_in_space.lb()
        ub = self.set_in_space.ub()
        vertices = np.array(list(product(*[[lb[i], ub[i]] for i in range(self.dim)]))).T
        return vertices
        
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
    
    @property
    def parent_region_name(self):
        """Parent region of the voxel"""
        return self._parent_region_name
    
    def __eq__(self, other):
        """
        Two voxels are considered equal if they have the same center (within numerical precision).
        """
        if not isinstance(other, Voxel):
            return False
        return np.all(np.isclose(self.center, other.center))