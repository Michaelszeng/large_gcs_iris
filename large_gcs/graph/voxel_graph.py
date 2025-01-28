import logging
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations, product
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from scipy.spatial import ConvexHull
import numpy as np
from pydrake.all import (
    Constraint, 
    Cost, 
)
from tqdm import tqdm

from large_gcs.geometry.convex_set import ConvexSet
from large_gcs.geometry.point import Point
from large_gcs.geometry.voxel import Voxel

from large_gcs.graph.cfree_cost_constraint_factory import (
    create_cfree_l2norm_vertex_cost,
    create_cfree_constant_edge_cost,
    create_cfree_continuity_edge_constraint,
)

from large_gcs.graph.graph import Graph, ShortestPathSolution, Edge, Vertex


logger = logging.getLogger(__name__)


def generate_all_offsets(dim):
    """
    Generate all possible offsets in {-1,0,1}^dim (excluding the all-zero offset),
    using a base-3 trick to avoid itertools.product.
    
    Faster replacement for itertools.product([-1,0,1], repeat=dim).
    """
    n = 3 ** dim  # total combinations (3^dim)
    # Fill an (n, dim) array, each row a distinct combination of -1,0,1
    offsets = np.empty((n, dim), dtype=np.int8)
    
    # Fill offsets by counting in base 3, mapping {0->-1, 1->0, 2->1}
    for i in range(n):
        x = i
        for j in range(dim - 1, -1, -1):
            offsets[i, j] = (x % 3) - 1
            x //= 3
    
    # Mask out the row of all zeros
    mask = np.any(offsets != 0, axis=1)
    return offsets[mask]


class VoxelGraph(Graph):
    def __init__(
        self,
        obstacles: List[ConvexSet],
        s: np.ndarray,
        t: np.ndarray,
        workspace: np.ndarray = None,
        default_voxel_size: float = 0.2,
        num_knot_points_per_voxel: int = 2,
        should_add_gcs: bool = True,
        const_edge_cost: float = 1e-4,
    ):
        Graph.__init__(self, workspace=workspace)
        assert self.workspace is not None, "Must specify workspace"
        
        self.obstacles = obstacles
        self.s = s
        self.t = t
        self.workspace = workspace
        self.default_voxel_size = default_voxel_size
        self.num_knot_points = num_knot_points_per_voxel
        self._should_add_gcs = should_add_gcs
        self._const_edge_cost = const_edge_cost
        
        self.base_dim = np.shape(s)[0]  # dimension of space
        
        self.offsets = generate_all_offsets(self.base_dim)  # convenience variable containing all possible offsets in {-1,0,1}^dim
        
        sets = []
        set_ids = []

        sets += [Point(s), Point(t)]      
        set_ids += ["source", "target"]

        # Add convex sets to graph (Need to do this before generating edges)
        self.add_vertices_from_sets(
            sets,
            costs=self._create_vertex_costs(sets),
            constraints=self._create_vertex_constraints(sets),
            names=set_ids,
        )  # This function is defined in graph.py
        
        self.set_source("source")
        self.set_target("target") 
        
    def generate_successors(self, vertex_name: str) -> None:
        """Generates neighbors and adds them to the graph, also adds edges from
        vertex to neighbors."""
        print("generating successors")
        
        neighbors = []
        
        if vertex_name == self.source_name:
            # Find the voxel that the source is in
            # It's fine if s is on the boundary between multiple voxels; we only
            # add one of the voxels it is in but those boundary voxels will be
            # added as successors of the first voxel
            s_center = self.vertices[self.source_name].convex_set.center           
            neighbor_voxel_center = self.default_voxel_size * (np.round(s_center / self.default_voxel_size))
            neighbors.append((self.source_name, "0_"*self.base_dim, False, Voxel(neighbor_voxel_center, self.default_voxel_size, self.num_knot_points)))  # neighbor is named "0_0_..."
        elif vertex_name == self.target_name:
            raise ValueError("Should not need to generate neighbors for target vertex")
        else:
            voxel_center = self.vertices[vertex_name].convex_set.center
            vertex_indices = np.array([int(x) for x in vertex_name.strip('_').split('_')])
            
            # Vectorized computation of neighbor centers
            neighbor_centers = voxel_center + self.offsets * self.default_voxel_size
            
            # Vectorized computation of neighbor names
            neighbor_indices = vertex_indices + self.offsets
            neighbor_names = [''.join(f"{idx}_" for idx in indices) for indices in neighbor_indices]
            
            # Filter existing vertices
            valid_indices = [i for i, name in enumerate(neighbor_names) 
                           if name not in self.vertices]
            
            # Create voxels and check collisions only for new vertices
            for idx in valid_indices:
                neighbor_voxel = Voxel(
                    neighbor_centers[idx], 
                    self.default_voxel_size, 
                    self.num_knot_points
                )
                
                # Skip voxel if it is in collision
                if not self._is_voxel_collision_free(neighbor_voxel):
                    continue
                
                neighbors.append((
                    vertex_name,
                    neighbor_names[idx],
                    False,
                    neighbor_voxel
                ))
                
        for neighbor_data in neighbors:
            self._generate_neighbor(*neighbor_data)
            
            neighbor_voxel_name = neighbor_data[1]
            
            # Draw edges from source vertex if it is in the neighbor voxel
            if neighbor_data[0] == self.source_name:
                self.add_undirected_edge(
                    Edge(
                        u=self.source_name,
                        v=neighbor_voxel_name,
                        costs=self._create_single_edge_costs(self.source_name, neighbor_voxel_name),
                        constraints=self._create_single_edge_constraints(self.source_name, neighbor_voxel_name),
                    ),
                    should_add_to_gcs=self._should_add_gcs,
                )
            
            # Draw edges to the target vertex if it is in the neighbor voxel
            if self._does_vertex_have_possible_edge_to_target(neighbor_voxel_name):
                # Directed edge to target
                self.add_edge(
                    Edge(
                        u=neighbor_voxel_name,
                        v=self.target_name,
                        costs=self._create_single_edge_costs(neighbor_voxel_name, self.target_name),
                        constraints=self._create_single_edge_constraints(neighbor_voxel_name, self.target_name),
                    ),
                    should_add_to_gcs=self._should_add_gcs,
                )
                
    def _generate_neighbor(
        self, u: str, v: str, is_v_in_vertices: bool, v_set: Voxel = None
    ) -> None:
        """
        Generates a neighbor of u (called v) and adds it to the graph.
        
        Also adds edges between the v and all intersecting voxels with v.
        """       
        if not is_v_in_vertices:
            vertex = Vertex(
                v_set,
                costs=self._create_single_vertex_costs(v_set),
                constraints=self._create_single_vertex_constraints(v_set),
            )
            self.add_vertex(vertex, v, should_add_to_gcs=self._should_add_gcs)
        
        # Add edges between the generated neighbor (v) and all adjacent voxels
        v_indices = v.split("_")
        
        # Iterate through all possible neighbors of v (call them w)
        for offset in self.offsets:
            # Skip the current voxel (all zeros offset)
            if all(o == 0 for o in offset):
                continue
            
            # Generate w's name
            neighbor_voxel_name = ""
            for d in range(self.base_dim):
                neighbor_voxel_name += f"{int(v_indices[d]) + offset[d]}_"
            
            if neighbor_voxel_name in self.vertices:                           
                # Add edge between v and w
                self.add_undirected_edge(
                    Edge(
                        u=v, 
                        v=neighbor_voxel_name, 
                        costs=self._create_single_edge_costs(v, neighbor_voxel_name),
                        constraints=self._create_single_edge_constraints(v, neighbor_voxel_name),
                    ),
                    should_add_to_gcs=self._should_add_gcs,
                )
            
    def _is_voxel_collision_free(self, voxel: Voxel) -> bool:
        """Only handles Polyhedron obstacles for now."""
        for obstacle in self.obstacles:
            if not obstacle.set.Intersection(voxel.set_in_space.MakeHPolyhedron()).IsEmpty():
                return False
        return True
    
    def _does_vertex_have_possible_edge_to_target(self, vertex_name: str) -> bool:
        """Determine if we can add an edge to the target vertex."""
        return self.vertices[vertex_name].convex_set.set_in_space.PointInSet(self.t)
        
    ############################################################################
    ### VERTEX AND EDGE COSTS AND CONSTRAINTS ###
    ############################################################################
    def _create_vertex_costs(self, sets: List[ConvexSet]) -> List[List[Cost]]:
        # Create a list of costs for each vertex
        logger.info("Creating vertex costs...")
        costs = [
            # Add shortest-path cost to Voxel sets
            # Source and Target are Point sets, so we don't add costs for them
            self._create_single_vertex_costs(set) if isinstance(set, Voxel) else []
            for set in tqdm(sets)
        ]
        return costs

    def _create_single_vertex_costs(self, set: ConvexSet) -> List[Cost]:
        # Path length penalty
        return [create_cfree_l2norm_vertex_cost(self.base_dim)]

    def _create_vertex_constraints(self, sets: List[ConvexSet]) -> List[List[Constraint]]:
        # Create a list of constraints for each vertex
        return [self._create_single_vertex_constraints(set) for set in sets]

    def _create_single_vertex_constraints(self, set: ConvexSet) -> List[Constraint]:
        # Note: set-containment constraints are automatically handled by the Vertex object
        # and don't need to be explicitly added here
        return []

    def _create_edge_costs(self, edges: List[Tuple[str, str]]) -> List[List[Cost]]:
        logger.info("Creating edge costs...")
        return [self._create_single_edge_costs(u, v) for u, v in tqdm(edges)]

    def _create_single_edge_costs(self, u: str, v: str) -> List[Cost]:
        # Penalizes each active edge a constant value
        return [create_cfree_constant_edge_cost(self.base_dim, u, v, self.num_knot_points, constant_cost=self._const_edge_cost)]

    def _create_edge_constraints(
        self, edges: List[Tuple[str, str]]
    ) -> List[List[Constraint]]:
        # Create a list of constraints for each edge
        logger.info("Creating edge constraints...")
        return [self._create_single_edge_constraints(u, v) for u, v in tqdm(edges)]

    def _create_single_edge_constraints(self, u: str, v: str) -> List[Constraint]:
        # Path continuity constraint
        return [create_cfree_continuity_edge_constraint(self.base_dim, u, v,self.num_knot_points)]

    ############################################################################
    ### PLOTTING ###
    ############################################################################
    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        sol: ShortestPathSolution = None,
        show_edges: bool = False,
    ):
        """
        Plot the voxel graph in 2D or 3D.
        """
        if self.base_dim not in [2, 3]:
            raise ValueError("Can only plot 2D or 3D voxel graphs")
            
        # If this is the first time plotting, create a new figure and plot the 
        # source, target, and obstacles (which are static)
        fig = None
        if ax is None:
            fig = plt.figure(figsize=(8, 8))
            if self.base_dim == 3:
                ax = fig.add_subplot(111, projection='3d')
            else:
                ax = fig.add_subplot(111)
                
            # Plot source and target
            if self.base_dim == 2:
                ax.plot(
                    self.s[0], self.s[1],
                    marker='*', color='green', markersize=15, label='Source'
                )
                ax.plot(
                    self.t[0], self.t[1],
                    marker='x', color='red', markersize=15, label='Target'
                )
            elif self.base_dim == 3:
                ax.scatter(
                    self.s[0], self.s[1], self.s[2],
                    marker='*', color='green', s=100, label='Source'
                )
                ax.scatter(
                    self.t[0], self.t[1], self.t[2],
                    marker='x', color='red', s=100, label='Target'
                )
        
            # Plot obstacles
            for obstacle in self.obstacles:
                if hasattr(obstacle, 'vertices'):
                    obstacle_vertices = obstacle.vertices
                    if self.base_dim == 2:
                        # Close the polygon by appending the first vertex
                        obstacle_vertices = np.vstack((obstacle_vertices, obstacle_vertices[0]))
                        ax.fill(
                            obstacle_vertices[:, 0], obstacle_vertices[:, 1],
                            color='black', alpha=0.5, edgecolor='black', linewidth=1
                        )
                    elif self.base_dim == 3:
                        hull = ConvexHull(obstacle_vertices)
                        
                        # Each "simplex" in the hull is a triangle (defined by 3 vertices)
                        faces = [obstacle_vertices[s] for s in hull.simplices]
                        
                        # Create a Poly3DCollection (triangular faces) for the hull
                        hull_collection = Poly3DCollection(
                            faces,
                            alpha=0.5,          # Transparency
                            facecolors='black', # Fill color
                            edgecolors='black', # Edge color
                            linewidths=1
                        )
                        
                        # Add this 3D collection of triangles to the Axes
                        ax.add_collection3d(hull_collection)
            
            # Create a line object for the path that we'll update
            if self.base_dim == 2:
                self.path_line, = ax.plot(
                    [], [], 'bo-',
                    linewidth=2, markersize=5,
                    label='Current Path',
                    zorder=10
                )
            else:
                self.path_line, = ax.plot3D(
                    [], [], [], 'bo-',
                    linewidth=2, markersize=5,
                    label='Current Path'
                )
                
            # Initialize empty collections for voxels
            self.voxel_patches = []
        
        # Plot edges
        if show_edges:
            for edge in self.edges.values():
                u_center = self.vertices[edge.u].convex_set.center
                v_center = self.vertices[edge.v].convex_set.center
                if self.base_dim == 2:
                    ax.arrow(
                        u_center[0], u_center[1],
                        v_center[0] - u_center[0], v_center[1] - u_center[1],
                        color='grey', width=0.01, head_width=0.05,
                        alpha=0.5, length_includes_head=True
                    )
                elif self.base_dim == 3:
                    pass  # for visibility, don't plot edges
                    # ax.plot3D(
                    #     [u_center[0], v_center[0]],
                    #     [u_center[1], v_center[1]],
                    #     [u_center[2], v_center[2]],
                    #     color='grey', alpha=0.5
                    # )
                
        # Remove old voxel patches
        for patch in self.voxel_patches:
            patch.remove()
        self.voxel_patches = []
        
        # Add new voxel patches
        for vertex_name, vertex in self.vertices.items():
            if isinstance(vertex.convex_set, Voxel):
                center = vertex.convex_set.center
                size = vertex.convex_set.size
                if self.base_dim == 2:
                    rect = plt.Rectangle(
                        center - size / 2, size, size,
                        fill=False, edgecolor='black', linewidth=1
                    )
                    self.animation_ax.add_patch(rect)
                    self.voxel_patches.append(rect)
        
        # Update trajectory if provided
        if sol is not None and sol.trajectory is not None:
            points = []
            for point in sol.trajectory:
                if len(point) == self.base_dim:
                    points.append(point)
                else:
                    points.append(point[:self.base_dim])
                    points.append(point[self.base_dim:])
            
            trajectory = np.vstack(points)
            self.path_line.set_data(trajectory[:, 0], trajectory[:, 1])
        
        if self.workspace is not None:
            ax.set_xlim(self.workspace[0])
            ax.set_ylim(self.workspace[1])
            if self.base_dim == 3:    
                ax.set_zlim(self.workspace[2])
        
        if self.base_dim == 2:
            ax.set_aspect('equal')
        elif self.base_dim == 3:
            ax.set_box_aspect([1, 1, 1])
            
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title('Voxel Graph')
        
        return fig, ax
    
    def init_animation(self):
        """Initialize a persistent figure for animation."""
        self.animation_fig, self.animation_ax = self.plot()

        plt.ion()  # Turn on interactive mode
        self.animation_fig.canvas.draw()
        plt.show(block=False)  # Show the figure but don't block
        plt.pause(0.1)  # Small pause to ensure window appears
        
    def update_animation(self, sol: Optional[ShortestPathSolution] = None):
        """Update the animation with new voxels and optionally a new solution."""
        if not hasattr(self, 'animation_fig'):
            self.init_animation()
            
        self.plot(ax=self.animation_ax, sol=sol)
        
        # Redraw
        self.animation_fig.canvas.draw()
        self.animation_fig.canvas.flush_events()
        
    def plot_current_solution(self, loc: Optional[Path] = None):
        pass

    ### SERIALIZATION METHODS ###

    def save_to_file(self, path: str):
        pass

    @classmethod
    def load_from_file(
        cls,
        path: str,
        vertex_inclusion: List[str] = None,
        vertex_exclusion: List[str] = None,
        should_use_l1_norm_vertex_cost: bool = False,
    ):
        pass
    