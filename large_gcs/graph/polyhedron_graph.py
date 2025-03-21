import logging
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations, product
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from scipy.spatial import ConvexHull
import numpy as np
from pydrake.all import (
    Constraint, 
    Cost,
    LinearConstraint,
    MathematicalProgram,
    Solve,
    SolveInParallel,
    Meshcat,
    MultibodyPlant,
    Context,
    IrisZo,
    IrisZoOptions,
    FastIris,
    FastIrisOptions,
    Hyperellipsoid,
    FastCliqueInflation,
    FastCliqueInflationOptions,
    HPolyhedron,
    SceneGraphCollisionChecker,
)
from manipulation.meshcat_utils import AddMeshcatTriad
from tqdm import tqdm

from large_gcs.geometry.convex_set import ConvexSet
from large_gcs.geometry.polyhedron import Polyhedron
from large_gcs.geometry.point import Point
from large_gcs.geometry.voxel import Voxel
from large_gcs.geometry.voxel_collision_checker import (
    VoxelCollisionChecker,
    VoxelSceneGraphCollisionChecker, 
    VoxelCollisionCheckerConvexObstacles
)
from large_gcs.geometry.utils import SuppressOutput
from large_gcs.graph.cfree_cost_constraint_factory import (
    create_cfree_l2norm_vertex_cost,
    create_cfree_constant_edge_cost,
    create_cfree_continuity_edge_constraint,
)
from large_gcs.graph.graph import Graph, ShortestPathSolution, Edge, Vertex
from large_gcs.geometry.utils import ik


logger = logging.getLogger(__name__)


class PolyhedronGraph(Graph):
    def __init__(
        self,
        # meshcat: Meshcat,
        # plant: MultibodyPlant,
        # plant_context: Context,
        s: np.ndarray,
        t: np.ndarray,
        workspace: np.ndarray = None,
        default_voxel_size: float = 0.2,
        num_knot_points_per_set: int = 2,
        should_add_gcs: bool = True,
        const_edge_cost: float = 1e-4,
        voxel_collision_checker: VoxelCollisionChecker = None,
    ):
        Graph.__init__(self, workspace=workspace)
        assert self.workspace is not None, "Must specify workspace"
        
        # self.meshcat = meshcat
        # self.plant = plant
        # self.plant_context = plant_context
        self.s = s
        self.t = t
        self.workspace = workspace
        self.domain = HPolyhedron.MakeBox(workspace[:, 0], workspace[:, 1])  # translate workspace to HPolyhedron
        self.default_voxel_size = default_voxel_size
        self.num_knot_points = num_knot_points_per_set
        self._should_add_gcs = should_add_gcs
        self._const_edge_cost = const_edge_cost
        self.voxel_collision_checker = voxel_collision_checker
        
        self.base_dim = np.shape(s)[0]  # dimension of space
        
        # IRIS Setup
        self.iris_options = IrisZoOptions()
        self.iris_options.random_seed = 2
        self.iris_options.mixing_steps = 51
        self.iris_options.epsilon = 1e-3
        # self.iris_options.verbose = True
        self.iris_options.require_sample_point_is_contained = True
        self.kEpsilonEllipsoid = 1e-5
        
        self.clique_inflation_options = FastCliqueInflationOptions()
        # self.clique_inflation_options.verbose = True
        
        self.num_vertices = 0  # this var should only be modified by calls to get_new_vertex_name; it is only used for naming new reigons
        
        self.uncovered_voxels = set()  # Set of Voxel names (str); keep track of voxels that are not fully contained in a region
    
        sets = [Point(s), Point(t)]
        # Add convex sets to graph (Need to do this before generating edges)
        self.add_vertices_from_sets(
            sets,
            costs=self._create_vertex_costs(sets),
            constraints=self._create_vertex_constraints(sets),
            names=["source", "target"],
        )  # This function is defined in graph.py
        
        self.set_source("source")
        self.set_target("target") 
        
    def get_new_vertex_name(self):
        n = self.num_vertices
        self.num_vertices += 1
        return str(n)
        
    def generate_successors(self, vertex_name: str) -> None:
        """Generates neighbors and adds them to the graph, also adds edges from
        vertex to neighbors."""
        neighbors = []  # List of (u_name, v_name, is_v_in_vertices, ConvexSet, v_neighbors) tuples
        
        if vertex_name == self.source_name:
            # Grow a region around the source
            starting_ellipse = Hyperellipsoid.MakeHypersphere(self.kEpsilonEllipsoid, self.s)
            region = IrisZo(self.voxel_collision_checker.checker, starting_ellipse, self.domain, self.iris_options)
            polyhedron = Polyhedron.from_drake_hpoly(region, should_compute_vertices=True if self.base_dim in [2, 3] else False, num_knot_points=self.num_knot_points)  # Compute vertices for 2D/3D visualization
            neighbors.append((self.source_name, self.get_new_vertex_name(), False, polyhedron, [self.source_name]))
            
        elif vertex_name == self.target_name:
            raise ValueError("Should not need to generate neighbors for target vertex")
        
        else:
            if isinstance(self.vertices[vertex_name].convex_set, Voxel):
                """
                1. Check if voxel is in `self.uncovered_voxels`; if not, continue
                2. Inflate a new region around the voxel
                3. Switch the vertices[vertex_name] from containing the voxel to containing the new region
                4. For `other_voxel` that is in `self.uncovered_voxels`:
                    if `other_voxel` is fully-contained in new region: remove from `self.uncovered_voxels`
                5. Continue in the instance(self.vertices[vertex_name].convex_set, Polyhedron) case below (now that the vertex contains a region)
                """
                voxel = self.vertices[vertex_name].convex_set
                if vertex_name not in self.uncovered_voxels:  # i.e. voxel is covered by a region
                    return
                
                # Inflate a new region around the voxel
                # Seed region using clique formed by voxel vertices
                try:
                    # Errors may occur if collision sampling does not detect a collision
                    region = FastCliqueInflation(self.voxel_collision_checker.checker, voxel.get_vertices(), self.domain, self.clique_inflation_options)
                except:
                    print(f"Error inflating region for voxel: {voxel.center}.")
                    return
                polyhedron = Polyhedron.from_drake_hpoly(region, should_compute_vertices=True if self.base_dim in [2, 3] else False, num_knot_points=self.num_knot_points)  # Compute vertices for 2D/3D visualization
                
                # Swap the voxel in the graph with the new region
                self.vertices[vertex_name].convex_set = polyhedron  # Replace voxel with region
                
                # Remove the voxel from `self.uncovered_voxels`
                self.uncovered_voxels.remove(vertex_name)
                
                # Check if new region covered any voxels; if so:
                # 1. remove those voxels from `self.uncovered_voxels`
                # 2. add edges between new region and that covered voxel's parent region
                covered_voxels = []
                for other_voxel_name in self.uncovered_voxels:
                    # Check if all corners of other_voxel are in region
                    voxel_covered = True
                    for vtx in self.vertices[other_voxel_name].convex_set.get_vertices().T:
                        if not region.PointInSet(vtx):
                            voxel_covered = False
                            break
                    if not voxel_covered:
                        continue
                    
                    print(f"voxel: {other_voxel_name} is covered.")
                    covered_voxels.append(other_voxel_name)
                
                for voxel_name in covered_voxels:
                    self.uncovered_voxels.remove(voxel_name)
                    voxel_parent_region_name = self.vertices[voxel_name].convex_set.parent_region_name
                    self.add_undirected_edge(
                        Edge(
                            u=vertex_name,
                            v=voxel_parent_region_name,
                            costs=self._create_single_edge_costs(vertex_name, voxel_parent_region_name),
                            constraints=self._create_single_edge_constraints(vertex_name, voxel_parent_region_name),
                        ),
                        should_add_to_gcs=self._should_add_gcs, 
                    )

            """
            Now, handle generation of voxel successors of polyhedron
            
            1. Discretize boundary of polytope into partially-contained voxels (that are also not fully contained in any other region)
            2. Add each voxel to the graph
            3. Add a path (and solve its convex restriction) ending at each of those voxels to queue
            """
            # Generate voxels on boundary of current vertex
            # First, find axis-aligned bounding box of current region
            region = self.vertices[vertex_name].convex_set  # Polyhedron
            min_coords, max_coords = region.axis_aligned_bounding_box()
            # Inflate bounding box by 0.1*voxel_size in each dimension (to ensure voxels are not on the boundary of the region)
            min_coords -= [0.1 * self.default_voxel_size] * self.base_dim
            max_coords += [0.1 * self.default_voxel_size] * self.base_dim
            
            # Generate voxels within bounding box
            voxels = []
            # Calculate number of voxels in each dimension
            steps = np.ceil((max_coords - min_coords) / self.default_voxel_size).astype(int)
            # Generate center points for voxels
            for idx in np.ndindex(*steps):
                center = min_coords + (np.array(idx) + 0.5) * self.default_voxel_size
                if center.shape != min_coords.shape:  # Ensure correct dimensionality
                    center = center[:len(min_coords)]
                voxels.append(Voxel(center, self.default_voxel_size, self.num_knot_points, parent_region_name=vertex_name))
                
            # Check partial containment of voxels
            start = time.time()
            non_contained_voxels = []
            voxel_progs = []
            for voxel in voxels:
                # Check if vertices of voxels are all contained in region (NOT full containment)
                vtxs = voxel.get_vertices()  # base_dim x num_vertices
                num_vtxs = vtxs.shape[1]
                if np.all(region.H @ vtxs <= np.tile(region.h, (num_vtxs, 1)).T):
                    continue
                
                # Check if voxel intersects with obstacle
                if not self.voxel_collision_checker.check_voxel_collision_free(voxel):
                    continue
                
                non_contained_voxels.append(voxel)
                
                # Create MathematicalProgram to check that voxel intersects with region (partial containment)
                prog = MathematicalProgram()
                # Find x subject to: x is a convex combination of vertices, x is in region
                x = prog.NewContinuousVariables(voxel.dim, "x")  # (base_dim,)
                lam = prog.NewContinuousVariables(num_vtxs, "lam")  # (num_vertices,) - One lambda per vertex
                # lam >= 0
                for i in range(num_vtxs):
                    prog.AddConstraint(lam[i] >= 0)
                #1^T lam = 1  (i.e. components of lambda must sum to 1)
                prog.AddLinearEqualityConstraint(np.ones((1, num_vtxs)), [1], lam)
                # x is a convex combination of vertices
                for d in range(voxel.dim):
                    # x[d] = sum_{i=1}^{num_vtxs} lam[i] * vtxs[d, i]
                    prog.AddConstraint(x[d] == vtxs[d, :] @ lam)
                # x ∈ region
                prog.AddConstraint(LinearConstraint(region.H, np.full((region.H.shape[0],), -np.inf), region.h), x)
                voxel_progs.append(prog)
                
            # Solve all programs
            partially_contained_voxels = []
            voxel_solve_start = time.time()
            results = SolveInParallel(voxel_progs) 
            print(f"Time for voxel SolveInParallel: {time.time() - voxel_solve_start}")
            for i, result in enumerate(results):
                if result.is_success():  # Found a point in the intersection
                    partially_contained_voxels.append(non_contained_voxels[i])
                else:
                    pass
                    # print(f"No intersection found for voxel with center: {voxel.center}")
            print(f"Time taken to solve voxels: {time.time() - start}")
            
            # Add voxel successors to neighbors
            for voxel in partially_contained_voxels:
                neighbors.append((
                    vertex_name,
                    self.get_new_vertex_name(),
                    False,
                    voxel
                ))
            
            # # Just for plotting
            # for voxel in partially_contained_voxels:
            #     vertex = Vertex(voxel, costs=[], constraints=[])
            #     self.add_vertex(vertex, f"test_voxel_{self.get_new_region_name()}", should_add_to_gcs=self._should_add_gcs)
            # self.update_animation(None)
            # time.sleep(5)
                
            # return

                
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
        self, u: str, v: str, is_v_in_vertices: bool, v_set: ConvexSet = None,
        v_neighbors: List[str] = None
    ) -> None:
        """
        Generates a neighbor of u (called v) and adds it to the graph.
        
        If v is a voxel, add edge between v and its parent region.
        If v is a polyhedron, add edge between v and all vertices in `v_neighbors` (which
        should contain the names of intersecting regions).
        """
        if not is_v_in_vertices:
            vertex = Vertex(
                v_set,
                costs=self._create_single_vertex_costs(v_set),
                constraints=self._create_single_vertex_constraints(v_set),
            )
            self.add_vertex(vertex, v, should_add_to_gcs=self._should_add_gcs)

        # If v is a voxel, add edge from v to its parent region
        if isinstance(v_set, Voxel):
            self.add_undirected_edge(
                Edge(
                    u=u, 
                    v=v, 
                    costs=self._create_single_edge_costs(u, v),
                    constraints=self._create_single_edge_constraints(u, v),
                ),
                should_add_to_gcs=self._should_add_gcs,
            )
            self.uncovered_voxels.add(v)
        
        
        # If v is a polyhedron, add edge from v to intersecting neighbors given in v_neighbors
        elif isinstance(v_set, Polyhedron):
            assert v_neighbors is not None, "Must provide neighbors for polyhedron"
            for neighbor_name in v_neighbors:
                self.add_undirected_edge(
                    Edge(
                        u=neighbor_name, 
                        v=v, 
                        costs=self._create_single_edge_costs(neighbor_name, v),
                        constraints=self._create_single_edge_constraints(neighbor_name, v),
                    ),
                    should_add_to_gcs=self._should_add_gcs,
                )
    
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
        Plot the polyhedron graph in 2D or 3D.
        """
        if self.base_dim not in [2, 3]:
            raise ValueError("Can only plot 2D or 3D graphs")
        
        # If this is the first time plotting, create a new figure and plot the 
        # source, target, and obstacles (which are static)
        fig = None
        if ax is None:
            fig = plt.figure(figsize=(11, 8))
            if self.base_dim == 3:
                ax = fig.add_subplot(111, projection='3d')
            else:
                ax = fig.add_subplot(111)
                
            ####################################################################
            ### PLOTTED ONCE AT FIGURE CREATION ###
            ####################################################################
                
            # Draw workspace outline
            if self.workspace is not None:
                if self.base_dim == 2:
                    # Draw a rectangle for the workspace outline
                    workspace_outline = plt.Rectangle(
                        (self.workspace[0][0], self.workspace[1][0]),
                        self.workspace[0][1] - self.workspace[0][0],
                        self.workspace[1][1] - self.workspace[1][0],
                        fill=False, edgecolor='black', linewidth=2
                    )
                    ax.add_patch(workspace_outline)
                elif self.base_dim == 3:
                    # Draw a box for the workspace outline
                    x = [self.workspace[0][0], self.workspace[0][1]]
                    y = [self.workspace[1][0], self.workspace[1][1]]
                    z = [self.workspace[2][0], self.workspace[2][1]]
                    # Create the vertices of the box
                    vertices = np.array(list(product(x, y, z)))
                    edges = [
                        [vertices[i], vertices[j]] for i, j in combinations(range(8), 2)
                        if np.sum(np.abs(vertices[i] - vertices[j])) == (x[1] - x[0]) + (y[1] - y[0]) + (z[1] - z[0])
                    ]
                    for edge in edges:
                        ax.plot3D(*zip(*edge), color='black', linewidth=2)
                        
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
            # Convex Polyhedral Obstacles
            if isinstance(self.voxel_collision_checker, VoxelCollisionCheckerConvexObstacles):
                for obstacle in self.voxel_collision_checker.obstacles:
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
            
            # Scene Graph Obstacles
            elif isinstance(self.voxel_collision_checker, VoxelSceneGraphCollisionChecker):
                # Rejection sample and plot points where collisions are sampled
                # Sample points in workspace
                num_samples = 10000
                
                # Sample points uniformly from workspace
                samples = np.random.uniform(
                    low=[bounds[0] for bounds in self.workspace],
                    high=[bounds[1] for bounds in self.workspace],
                    size=(num_samples, self.base_dim)
                )
                
                # Check collisions for all samples
                collision_free = self.voxel_collision_checker.check_configs_collision_free(samples)
                collision_points = samples[~np.array(collision_free, dtype=bool)]
                # print(f"collision_points: {collision_points}")
                
                # Plot the points
                if self.base_dim == 2:
                    scatter = ax.scatter(
                        collision_points[:, 0], 
                        collision_points[:, 1],
                        color='black', 
                        s=10, 
                        label='In Collision',
                    )
                elif self.base_dim == 3:
                    ax.scatter(
                        collision_points[:, 0], 
                        collision_points[:, 1], 
                        collision_points[:, 2],
                        color='black', 
                        s=10, 
                        label='In Collision'
                    )
            
            # Create line objects for the path that we'll update
            # One line object for the main path, one for the shortcut edge (which will be red instead of blue)
            if self.base_dim == 2:
                self.path_line, = ax.plot(
                    [], [], 'bo-',
                    linewidth=2, markersize=5,
                    label='Path',
                    zorder=10
                )
                self.path_line_final_segment, = ax.plot(
                    [], [], 'ro-',  # Red color for final segment
                    linewidth=2, markersize=5,
                    label='Shortcut Edge',
                    zorder=11  # Higher zorder to draw on top
                )
            else:
                self.path_line, = ax.plot3D(
                    [], [], [], 'bo-',
                    linewidth=2, markersize=5,
                    label='Path'
                )
                self.path_line_final_segment, = ax.plot3D(
                    [], [], [], 'ro-',  # Red color for final segment
                    linewidth=2, markersize=5,
                    label='Shortcut Edge'
                )
                
            # Initialize empty collections for voxels
            self.voxel_patches = []
        
        ########################################################################
        ### PLOTTED AT EACH FRAME ###
        ########################################################################
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
        
        # Add polyhedron and voxel patches
        for vertex_name, vertex in self.vertices.items():
            if self.base_dim == 2:
                if isinstance(vertex.convex_set, Voxel):
                    if vertex_name not in self.uncovered_voxels:
                        continue
                    center = vertex.convex_set.center
                    size = vertex.convex_set.size
                    if self.base_dim == 2:
                        rect = plt.Rectangle(
                            center - size / 2, size, size,
                            fill=False, edgecolor='magenta', linewidth=1
                        )
                        self.animation_ax.add_patch(rect)
                        self.voxel_patches.append(rect)
                elif isinstance(vertex.convex_set, Polyhedron):
                    vertices = vertex.convex_set.vertices
                    polygon_patch = plt.Polygon(vertices, closed=True, facecolor='red', alpha=0.2, edgecolor='magenta', linewidth=1)
                    self.animation_ax.add_patch(polygon_patch)
                    self.voxel_patches.append(polygon_patch)
            elif self.base_dim == 3:
                if isinstance(vertex.convex_set, Voxel):
                    if vertex_name not in self.uncovered_voxels:
                        continue
                    center = vertex.convex_set.center
                    size = vertex.convex_set.size
                    # Create the 8 vertices of the cube
                    r = size / 2
                    cube_vertices = np.array(list(product([center[0]-r, center[0]+r],
                                                            [center[1]-r, center[1]+r],
                                                            [center[2]-r, center[2]+r])))
                
                    # Define the 12 edges of the cube
                    edges = []
                    for i, j in combinations(range(8), 2):
                        if np.sum(np.abs(cube_vertices[i] - cube_vertices[j])) == size:
                            line, = self.animation_ax.plot3D(
                                [cube_vertices[i,0], cube_vertices[j,0]],
                                [cube_vertices[i,1], cube_vertices[j,1]],
                                [cube_vertices[i,2], cube_vertices[j,2]],
                                color='magenta', linewidth=1
                            )
                            edges.append(line)
                    self.voxel_patches.extend(edges)
                elif isinstance(vertex.convex_set, Polyhedron):
                    vertices = vertex.convex_set.vertices
                    vertices_poly3d = [[vertices[j] for j in range(len(vertices))]]
                    polygon_patch_3d = Poly3DCollection(vertices_poly3d, facecolor='red', alpha=0.2, edgecolor='magenta', linewidths=1)
                    self.animation_ax.add_collection3d(polygon_patch_3d)
                    self.voxel_patches.append(polygon_patch_3d)
        
        # Update trajectory if provided
        if sol is not None and sol.trajectory is not None:
            # Extract trajectory points from vertices, handling both source and target 
            # (which contain only 1 point) and other vertices (which have multiple knot points)
            points = []
            for point in sol.trajectory:
                if len(point) == self.base_dim:
                    points.append(point)
                else:
                    points.append(point[:self.base_dim])
                    points.append(point[self.base_dim:])
            
            trajectory = np.vstack(points)
            
            if self.base_dim == 2:
                # Set trajectory points
                # Main path (all but last segment)
                self.path_line.set_data(trajectory[:-1, 0], trajectory[:-1, 1])
                # Final segment (last two points)
                self.path_line_final_segment.set_data(trajectory[-2:, 0], trajectory[-2:, 1])
                
                # Highlight voxels in the vertex path
                for vertex_name in sol.vertex_path:
                    if vertex_name not in ["source", "target"]:
                        if isinstance(self.vertices[vertex_name].convex_set, Voxel):
                            vertex = self.vertices[vertex_name]
                            center = vertex.convex_set.center
                            size = vertex.convex_set.size
                            rect = plt.Rectangle(
                                center - size / 2, size, size,
                                fill=True, facecolor='yellow', alpha=0.3,
                                edgecolor='magenta', linewidth=1
                            )
                            self.animation_ax.add_patch(rect)
                            self.voxel_patches.append(rect)
                        elif isinstance(self.vertices[vertex_name].convex_set, Polyhedron):
                            vertices = self.vertices[vertex_name].convex_set.vertices
                            polygon_patch = plt.Polygon(vertices, closed=True, facecolor='yellow', alpha=0.3, edgecolor='magenta', linewidth=1)
                            self.animation_ax.add_patch(polygon_patch)
                            self.voxel_patches.append(polygon_patch)
                
                # Add vertex path text
                # Remove old text annotation if it exists
                if hasattr(self, 'path_text') and self.path_text in self.animation_ax.texts:
                    self.path_text.remove()
                
                # Format the vertex path for display
                path_str = ' → '.join(sol.vertex_path[:-1])
                self.path_text = self.animation_ax.text(
                    0.05, 0.95,  # Position in axes coordinates (5% from left, 95% from bottom)
                    f'Path: {path_str}',
                    transform=self.animation_ax.transAxes,  # Use axes coordinates
                    verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                    fontsize=8,
                    wrap=True
                )
                self.voxel_patches.append(self.path_text)  # Add to patches so it gets cleaned up
            elif self.base_dim == 3:
                # Set trajectory points
                # Main path (all but last segment)
                self.path_line.set_data(trajectory[:-1, 0], trajectory[:-1, 1])
                self.path_line.set_3d_properties(trajectory[:-1, 2])
                # Final segment
                self.path_line_final_segment.set_data(trajectory[-2:, 0], trajectory[-2:, 1])
                self.path_line_final_segment.set_3d_properties(trajectory[-2:, 2])
                
                # Highlight in the vertex path
                for vertex_name in sol.vertex_path:
                    if vertex_name not in ["source", "target"]:
                        if isinstance(self.vertices[vertex_name].convex_set, Voxel):
                            vertex = self.vertices[vertex_name]
                            center = vertex.convex_set.center
                            size = vertex.convex_set.size
                            r = size / 2
                            
                            # Create vertices for the 6 faces of the cube
                            faces = []
                            # Front and back faces
                            for z in [center[2]-r, center[2]+r]:
                                face = np.array([
                                    [center[0]-r, center[1]-r, z],
                                    [center[0]+r, center[1]-r, z],
                                    [center[0]+r, center[1]+r, z],
                                    [center[0]-r, center[1]+r, z]
                                ])
                                faces.append(face)
                            # Left and right faces
                            for x in [center[0]-r, center[0]+r]:
                                face = np.array([
                                    [x, center[1]-r, center[2]-r],
                                    [x, center[1]+r, center[2]-r],
                                    [x, center[1]+r, center[2]+r],
                                    [x, center[1]-r, center[2]+r]
                                ])
                                faces.append(face)
                            # Top and bottom faces
                            for y in [center[1]-r, center[1]+r]:
                                face = np.array([
                                    [center[0]-r, y, center[2]-r],
                                    [center[0]+r, y, center[2]-r],
                                    [center[0]+r, y, center[2]+r],
                                    [center[0]-r, y, center[2]+r]
                                ])
                                faces.append(face)
                                
                            # Create the collection of faces
                            cube = Poly3DCollection(
                                faces,
                                facecolors='yellow',
                                alpha=0.3,
                                edgecolors='black'
                            )
                            self.animation_ax.add_collection3d(cube)
                            self.voxel_patches.append(cube)
                        elif isinstance(self.vertices[vertex_name].convex_set, Polyhedron):
                            vertices = vertices[vertex_name].convex_set.vertices
                            vertices_poly3d = [[vertices[j] for j in range(len(vertices))]]
                            polygon_patch_3d = Poly3DCollection(vertices_poly3d, facecolor='yellow', alpha=0.3, edgecolor='black', linewidths=1)
                            self.animation_ax.add_collection3d(polygon_patch_3d)
                            self.voxel_patches.append(polygon_patch_3d)
            
        
        # Set the limits of the axes to be 10% larger than the workspace
        if self.workspace is not None:
            x_range = self.workspace[0][1] - self.workspace[0][0]
            y_range = self.workspace[1][1] - self.workspace[1][0]
            x_margin = 0.1 * x_range
            y_margin = 0.1 * y_range
            ax.set_xlim(self.workspace[0][0] - x_margin, self.workspace[0][1] + x_margin)
            ax.set_ylim(self.workspace[1][0] - y_margin, self.workspace[1][1] + y_margin)
            if self.base_dim == 3:    
                z_range = self.workspace[2][1] - self.workspace[2][0]
                z_margin = 0.1 * z_range
                ax.set_zlim(self.workspace[2][0] - z_margin, self.workspace[2][1] + z_margin)
            
        if self.base_dim == 2:
            ax.set_aspect('equal')
        elif self.base_dim == 3:
            ax.set_box_aspect([1, 1, 1])
            
        ax.grid(True, alpha=0.3)
        # ax.legend()
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
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