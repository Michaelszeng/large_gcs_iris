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
from typing import Set
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

class VoxelSet:
    """
    "Set-like" datastructure to keep track of Voxel names (str) and Voxel 
    objects. Allows for quick lookup of whether a voxel is in the set (by 
    using the voxel's center's byte string as a hashable key).
    """
    map: Dict[str, Voxel]
    voxel_centers: Set[bytes]
    
    def __init__(self):
        self.map = {}
        self.voxel_centers = set()
        self._precision = 6
    
    def remove(self, voxel_name: str):
        voxel = self.map.pop(voxel_name)
        # Round to a reasonable precision before converting to bytes
        rounded_center = np.round(voxel.center, decimals=self._precision)
        self.voxel_centers.remove(rounded_center.tobytes())
        
    def add(self, voxel_name: str, voxel: Voxel):
        self.map[voxel_name] = voxel
        # Round to a reasonable precision before converting to bytes
        rounded_center = np.round(voxel.center, decimals=self._precision)
        self.voxel_centers.add(rounded_center.tobytes())
        
    def __contains__(self, voxel: Voxel) -> bool:
        assert isinstance(voxel, Voxel), f"is type {type(voxel)}, should be Voxel"
        # Round to a reasonable precision before converting to bytes
        rounded_center = np.round(voxel.center, decimals=self._precision)
        return rounded_center.tobytes() in self.voxel_centers
    
    def __str__(self) -> str:
        if not self.map:
            return "VoxelSet(empty)"
        voxel_info = []
        for name, voxel in self.map.items():
            center_str = np.array2string(voxel.center, precision=self._precision, separator=', ')
            voxel_info.append(f"{name}: {center_str}")
        return f"VoxelSet(count={len(self.map)}, voxels=[{', '.join(voxel_info)}])"

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
        const_edge_cost: float = 1e-4,
        voxel_collision_checker: VoxelCollisionChecker = None,
    ):
        Graph.__init__(self, workspace=workspace)
        assert self.workspace is not None, "Must specify workspace"
        
        self.plt = plt  # make accessible to gcs_star.py
        
        self.s = s
        self.t = t
        self.workspace = workspace
        self.domain = HPolyhedron.MakeBox(workspace[:, 0], workspace[:, 1])  # translate workspace to HPolyhedron
        self.default_voxel_size = default_voxel_size
        self.num_knot_points = num_knot_points_per_set
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
        
        self.containment_tol = 1e-2  # This is actually very important; allowable tolerance to determine if a voxel is fully contained in a region
                                     # If too low, voxel may not be considered contained, thus resulting in an infinite loop of inflating this voxel, then regenerating the voxel...
        
        self.num_vertices = 0  # this var should only be modified by calls to get_new_vertex_name; it is only used for naming new reigons
        
        self.uncovered_voxels = VoxelSet()  # Keep track of Voxel names (str) and Voxel objects that are not fully contained in a region
        self.inflated_voxels = VoxelSet()  # Keep track of Voxel names (str) and Voxel objects that have been inflated into a region
                                           # This is used to prevent inflating the same voxel multiple times in the edge case where a voxel is mistakenly determined not-in-collision, 
                                           # resulting in the inflated region not fully covering the voxel, resulting in the voxel failing future "covered voxel" checks.
    
        sets = [Point(s), Point(t)]
        # Add convex sets to graph (Need to do this before generating edges)
        self.add_vertices_from_sets(
            sets,
            costs=self._create_vertex_costs(sets),
            constraints=self._create_vertex_constraints(sets),
            names=["source", "target"]
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
            neighbors.append((self.source_name, self.get_new_vertex_name(), polyhedron, [self.source_name]))
            
        elif vertex_name == self.target_name:
            raise ValueError("Should not need to generate neighbors for target vertex")
        
        else:
            if isinstance(self.vertices[vertex_name].convex_set, Voxel):
                """
                1. Inflate a new region around the voxel
                2. Switch the vertices[vertex_name] from containing the voxel to containing the new region
                3. For `other_voxel` that is in `self.uncovered_voxels`:
                    if `other_voxel` is fully-contained in new region: remove from `self.uncovered_voxels`
                    then add edge between that voxel's parent region and the new region 
                    (this guarantees at least 1 edge is drawn to any newly generated region, since the new region is guaranteed to cover the voxel it was seeded with)
                4. Continue in the instance(self.vertices[vertex_name].convex_set, Polyhedron) case below (now that the vertex contains a region)
                """
                voxel = self.vertices[vertex_name].convex_set
                
                # Inflated voxels are no longer uncovered
                self.uncovered_voxels.remove(vertex_name)
                self.inflated_voxels.add(vertex_name, voxel)
                
                # Inflate a new region around the voxel
                # Seed region using clique formed by voxel vertices
                try:
                    # Errors may occur if collision sampling does not detect a collision
                    print(f"Inflating region around voxel {vertex_name} with center {voxel.center}.")
                    # NOTE: FastCliqueInflation guarantees resulting region contains the clique (unless the clique's convex hull contains a collision)
                    region = FastCliqueInflation(self.voxel_collision_checker.checker, voxel.get_vertices(), self.domain, self.clique_inflation_options)
                except Exception as e:
                    logger.error(f"{self.__class__.__name__} Failed to inflate region around voxel {vertex_name} with center {voxel.center}.")
                    print(f"Error: {e}")
                    return
                polyhedron = Polyhedron.from_drake_hpoly(region, should_compute_vertices=True if self.base_dim in [2, 3] else False, num_knot_points=self.num_knot_points)  # Compute vertices for 2D/3D visualization
                
                # Swap the voxel in the graph with the new region
                self.vertices[vertex_name].convex_set = polyhedron  # Replace voxel with region

                # Check if new region covered any voxels; if so:
                # 1. remove those voxels from `self.uncovered_voxels`
                # 2. add edges between new region and that covered voxel's parent region
                covered_voxels = []
                for other_voxel_name, other_voxel in self.uncovered_voxels.map.items():
                    # Check if all corners of other_voxel are in region
                    voxel_covered = True
                    for vtx in other_voxel.get_vertices().T:
                        if not region.PointInSet(vtx):
                            voxel_covered = False
                            break
                    if not voxel_covered:
                        continue
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
                    )

            """
            Now, handle generation of voxel successors of polyhedron
            
            1. Discretize boundary of polytope into partially-contained voxels (that are also not fully contained in any other region, and that don't already exist (in `self.uncovered_voxels`))
            2. Add each voxel to the graph
            3. Add a path (and solve its convex restriction) ending at each of those voxels to queue
            """
            # Generate voxels on boundary of current vertex
            # First, find axis-aligned bounding box of current region
            region = self.vertices[vertex_name].convex_set  # Polyhedron
            min_coords, max_coords = region.axis_aligned_bounding_box()
            # Round min_coords down to the nearest voxel edge and max_coords up to the nearest voxel edge
            min_coords = (np.floor((min_coords - 0.5 * self.default_voxel_size) / self.default_voxel_size) * self.default_voxel_size) + 0.5 * self.default_voxel_size
            max_coords = (np.ceil((max_coords - 0.5 * self.default_voxel_size) / self.default_voxel_size) * self.default_voxel_size) + 0.5 * self.default_voxel_size
            
            # Generate voxels within bounding box
            voxels = []
            # Calculate number of voxels in each dimension
            steps = np.ceil((max_coords - min_coords) / self.default_voxel_size).astype(int)
            # Generate center points for voxels
            for idx in np.ndindex(*steps):
                center = min_coords + (np.array(idx) + 0.5) * self.default_voxel_size
                if center.shape != min_coords.shape:  # Ensure correct dimensionality
                    center = center[:len(min_coords)]
                new_voxel = Voxel(center, self.default_voxel_size, self.num_knot_points, parent_region_name=vertex_name)
                if new_voxel in self.uncovered_voxels or new_voxel in self.inflated_voxels:  # Voxels are considered equal if they have the same center
                    continue
                voxels.append(new_voxel)
                
            # Check partial containment of voxels
            start = time.time()
            non_contained_voxels = []
            voxel_progs = []
            for voxel in voxels:
                vtxs = voxel.get_vertices()  # base_dim x num_vertices
                num_vtxs = vtxs.shape[1]
                
                # Ensure not all vertices of voxel are contained in current region
                if np.all(region.H @ vtxs <= np.tile(region.h, (num_vtxs, 1)).T + self.containment_tol):
                    # print(f"Voxel {voxel.center} is fully contained in current region: {vertex_name}.")
                    continue
                
                # Ensure not all vertices of voxel are contained in any other region
                for other_region in self.vertices.values():
                    if isinstance(other_region.convex_set, Polyhedron):
                        fully_contained_in_other_region = False
                        if np.all(other_region.convex_set.H @ vtxs <= np.tile(other_region.convex_set.h, (num_vtxs, 1)).T + self.containment_tol):
                            fully_contained_in_other_region = True  
                            # print(f"Voxel {voxel.center} is fully contained in other region.")
                            break
                if fully_contained_in_other_region:
                    continue
                
                # Ensure all vertices of voxel are in workspace
                if not np.all(self.domain.A() @ vtxs <= np.tile(self.domain.b(), (num_vtxs, 1)).T):
                    # print(f"Voxel {voxel.center} is not in workspace.")
                    continue
                
                # Ensure voxel doesn't intersect with obstacle
                if not self.voxel_collision_checker.check_voxel_collision_free(voxel):
                    # print(f"Voxel {voxel.center} intersects with obstacle.")
                    continue
                
                non_contained_voxels.append(voxel)
                
                # Create MathematicalProgram that ensures voxel has non-empty intersection with region
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
            
            print(f"partially_contained_voxels: {list(map(lambda x: x.center, partially_contained_voxels))}")
            
            # Add voxel successors to neighbors
            for voxel in partially_contained_voxels:
                neighbors.append((
                    vertex_name,
                    self.get_new_vertex_name(),
                    voxel
                ))
            
            # # Just for plotting
            # for voxel in partially_contained_voxels:
            #     vertex = Vertex(voxel, costs=[], constraints=[])
            #     self.add_vertex(vertex, f"test_voxel_{self.get_new_region_name()}")
            # self.update_animation(None)
            # time.sleep(5)
                
            # return

                
        for neighbor_data in neighbors:
            self._generate_neighbor(*neighbor_data)
            
            neighbor_name = neighbor_data[1]
            
            # Draw edges from source vertex if it is in the neighbor
            if neighbor_data[0] == self.source_name:
                self.add_undirected_edge(
                    Edge(
                        u=self.source_name,
                        v=neighbor_name,
                        costs=self._create_single_edge_costs(self.source_name, neighbor_name),
                        constraints=self._create_single_edge_constraints(self.source_name, neighbor_name),
                    ),
                )
        
        # Draw edge from newly generated region to the target vertex if the newly generated region contains the target
        if vertex_name != "source" and self._does_vertex_have_possible_edge_to_target(vertex_name):
            print(f"Adding edge from {vertex_name} to {self.target_name}")
            # Directed edge to target
            self.add_edge(
                Edge(
                    u=vertex_name,
                    v=self.target_name,
                    costs=self._create_single_edge_costs(vertex_name, self.target_name),
                    constraints=self._create_single_edge_constraints(vertex_name, self.target_name),
                )
            )
                
    def _generate_neighbor(
        self, u: str, v: str, v_set: ConvexSet = None,
        v_neighbors: List[str] = None
    ) -> None:
        """
        Generates a neighbor of u (called v) and adds it to the graph.
        
        If v is a voxel, add edge between v and its parent region.
        If v is a polyhedron, add edge between v and all vertices in `v_neighbors` (which
        should contain the names of intersecting regions).
        """
        vertex = Vertex(
            v_set,
            costs=self._create_single_vertex_costs(v_set),
            constraints=self._create_single_vertex_constraints(v_set),
        )
        self.add_vertex(vertex, v)

        # If v is a voxel, add edge from v to its parent region
        if isinstance(v_set, Voxel):
            self.add_undirected_edge(
                Edge(
                    u=u, 
                    v=v, 
                    costs=self._create_single_edge_costs(u, v),
                    constraints=self._create_single_edge_constraints(u, v),
                )
            )
            self.uncovered_voxels.add(v, v_set)
        
        
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
    def plot_voxel(self, vertex, fill=False, facecolor='yellow', alpha=0.3):
        if self.base_dim == 2:
            center = vertex.convex_set.center
            size = vertex.convex_set.size
            if fill:
                rect = plt.Rectangle(
                    center - size / 2, size, size,
                    fill=fill, facecolor=facecolor, alpha=alpha, edgecolor='magenta', linewidth=1
                )
            else:
                rect = plt.Rectangle(
                    center - size / 2, size, size,
                    fill=fill, edgecolor='magenta', linewidth=1
                )
            return rect
        elif self.base_dim == 3:
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
            if not fill:
                alpha = 0
                
            cube = Poly3DCollection(
                faces,
                facecolors=facecolor,
                alpha=alpha,
                edgecolors='black'
            )
            return cube
            
        
    def plot_polyhedron(self, vertex, facecolor='red', alpha=0.2):
        if self.base_dim == 2:
            vertices = vertex.convex_set.vertices
            polygon_patch = plt.Polygon(vertices, closed=True, facecolor=facecolor, alpha=alpha, edgecolor='magenta', linewidth=1)
            return polygon_patch
        elif self.base_dim == 3:
            vertices = vertex.convex_set.vertices
            hull = ConvexHull(vertices)
            faces = [vertices[s] for s in hull.simplices]
            polygon_patch_3d = Poly3DCollection(
                faces,
                alpha=alpha,           # Transparency
                facecolors=facecolor,  # Fill color
                linewidths=1
            )
            return polygon_patch_3d
        
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
                    
                    # Define the 12 edges of a cube by their vertex indices
                    # Each edge connects two vertices that differ in exactly one coordinate
                    edge_indices = [
                        # Bottom face edges (z=min)
                        (0, 1), (1, 3), (3, 2), (2, 0),
                        # Top face edges (z=max)
                        (4, 5), (5, 7), (7, 6), (6, 4),
                        # Vertical edges connecting top and bottom faces
                        (0, 4), (1, 5), (2, 6), (3, 7)
                    ]
                    
                    # Plot each edge
                    for i, j in edge_indices:
                        ax.plot3D(
                            [vertices[i][0], vertices[j][0]],
                            [vertices[i][1], vertices[j][1]],
                            [vertices[i][2], vertices[j][2]],
                            color='black', linewidth=2
                        )
                        
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
                            faces = [obstacle_vertices[s] for s in hull.simplices]
                            hull_collection = Poly3DCollection(
                                faces,
                                alpha=0.5,          # Transparency
                                facecolors='black', # Fill color
                                linewidths=1
                            )
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
                    pass  # plotting obstacle points makes it impossible to see anything else
                    # ax.scatter(
                    #     collision_points[:, 0], 
                    #     collision_points[:, 1], 
                    #     collision_points[:, 2],
                    #     color='black', 
                    #     s=10, 
                    #     label='In Collision'
                    # )
            
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
                    if vertex_name not in self.uncovered_voxels.map or vertex_name in self.inflated_voxels.map:
                        continue
                    rect = self.plot_voxel(vertex, fill=False)
                    self.animation_ax.add_patch(rect)
                    self.voxel_patches.append(rect)
                    
                    # Add voxel name as text in the center of the voxel
                    if vertex_name not in ["source", "target"]:
                        center = vertex.convex_set.center
                        text = self.animation_ax.text(
                            center[0], center[1], 
                            vertex_name,
                            ha='center', va='center',
                            fontsize=8, color='black'
                        )
                        self.voxel_patches.append(text)
                    
                elif isinstance(vertex.convex_set, Polyhedron):
                    polygon_patch = self.plot_polyhedron(vertex, facecolor='red', alpha=0.2)
                    self.animation_ax.add_patch(polygon_patch)
                    self.voxel_patches.append(polygon_patch)
            elif self.base_dim == 3:
                if isinstance(vertex.convex_set, Voxel):
                    if vertex_name not in self.uncovered_voxels.map or vertex_name in self.inflated_voxels.map:
                        continue
                    cube = self.plot_voxel(vertex, fill=False)
                    self.animation_ax.add_collection3d(cube)
                    self.voxel_patches.append(cube)
                    
                    # Add voxel name as text in the center of the voxel
                    if vertex_name not in ["source", "target"]:
                        center = vertex.convex_set.center
                        text = self.animation_ax.text3D(
                            center[0], center[1], center[2],
                            vertex_name,
                            ha='center', va='center',
                            fontsize=8, color='black'
                        )
                        self.voxel_patches.append(text)
                    
                elif isinstance(vertex.convex_set, Polyhedron):
                    polygon_patch_3d = self.plot_polyhedron(vertex, facecolor='red', alpha=0.2)
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
            
            # Generate vertex path text
            # Remove old text annotation if it exists
            if hasattr(self, 'path_text') and self.path_text in self.animation_ax.texts:
                self.path_text.remove()
            
            # Format the vertex path for display
            formatted_vertices = []
            for vertex_name in sol.vertex_path[:-1]:  # Exclude the last vertex (which is the target)
                if vertex_name == "source":
                    formatted_vertices.append(vertex_name)
                elif isinstance(self.vertices[vertex_name].convex_set, Voxel):
                    formatted_vertices.append(f"v{vertex_name}")  # Prefix with 'v' for voxels
                else:
                    formatted_vertices.append(f"r{vertex_name}")  # Prefix with 'r' for regions
            path_str = ' → '.join(formatted_vertices)
            
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
                            rect = self.plot_voxel(self.vertices[vertex_name], fill=True, facecolor='green', alpha=0.5)
                            self.animation_ax.add_patch(rect)
                            self.voxel_patches.append(rect)
                        elif isinstance(self.vertices[vertex_name].convex_set, Polyhedron):
                            polygon_patch = self.plot_polyhedron(self.vertices[vertex_name], facecolor='yellow', alpha=0.3)
                            self.animation_ax.add_patch(polygon_patch)
                            self.voxel_patches.append(polygon_patch)
                
                # Add vertex path text
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
                            cube = self.plot_voxel(self.vertices[vertex_name], fill=True, facecolor='green', alpha=0.5)
                            self.animation_ax.add_collection3d(cube)
                            self.voxel_patches.append(cube)
                        elif isinstance(self.vertices[vertex_name].convex_set, Polyhedron):
                            polygon_patch_3d = self.plot_polyhedron(self.vertices[vertex_name], facecolor='yellow', alpha=0.3)
                            self.animation_ax.add_collection3d(polygon_patch_3d)
                            self.voxel_patches.append(polygon_patch_3d)
                
                # Add vertex path text
                self.path_text = self.animation_ax.text2D(
                    0.05, 0.95,
                    f'Path: {path_str}',
                    transform=self.animation_ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                    fontsize=8,
                    wrap=True
                )
                self.voxel_patches.append(self.path_text)  # Add to patches so it gets cleaned up
            
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
        
        # Set the limits of the axes to be 10% larger than the workspace
        if self.workspace is not None:
            x_range = self.workspace[0][1] - self.workspace[0][0]
            y_range = self.workspace[1][1] - self.workspace[1][0]
            x_margin = 0.1 * x_range
            y_margin = 0.1 * y_range
            self.animation_ax.set_xlim(self.workspace[0][0] - x_margin, self.workspace[0][1] + x_margin)
            self.animation_ax.set_ylim(self.workspace[1][0] - y_margin, self.workspace[1][1] + y_margin)
            if self.base_dim == 3:    
                z_range = self.workspace[2][1] - self.workspace[2][0]
                z_margin = 0.1 * z_range
                self.animation_ax.set_zlim(self.workspace[2][0] - z_margin, self.workspace[2][1] + z_margin)
        
    def update_animation(self, sol: Optional[ShortestPathSolution] = None, block: bool = False):
        """Update the animation with new voxels and optionally a new solution."""
        if not hasattr(self, 'animation_fig'):
            self.init_animation()
            
        self.plot(ax=self.animation_ax, sol=sol)
        
        # Redraw
        self.animation_fig.canvas.draw()
        self.animation_fig.canvas.flush_events()
        
        if block:
            plt.show(block=True)
        
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