import logging
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations, product
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import time
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from scipy.spatial import ConvexHull
from typing import Set, Callable
import numpy as np
from collections import deque
from itertools import product
from treelib import Node, Tree
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
from large_gcs.geometry.voxel import Voxel, VoxelStatus
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
        voxel_tree_max_depth: int = 4,
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
        self.voxel_tree_max_depth = voxel_tree_max_depth
        self.num_knot_points = num_knot_points_per_set
        self._const_edge_cost = const_edge_cost
        self.voxel_collision_checker = voxel_collision_checker
        
        self.base_dim = np.shape(s)[0]  # dimension of space
        
        # IRIS Setup
        self.iris_options = IrisZoOptions()
        self.iris_options.random_seed = 2
        self.iris_options.mixing_steps = 51
        self.iris_options.epsilon = 5e-3  # Admissible fraction allowed to be in collision
        # self.iris_options.verbose = True
        self.iris_options.require_sample_point_is_contained = True
        self.kEpsilonEllipsoid = 1e-5
        
        self.clique_inflation_options = FastCliqueInflationOptions()
        self.clique_inflation_options.admissible_proportion_in_collision = 1e-2
        # self.clique_inflation_options.verbose = True
        
        self.containment_tol = 5e-2  # This is actually very important; allowable tolerance to determine if a voxel is fully contained in a region
                                     # If too low, voxel may not be considered contained, thus resulting in an infinite loop of inflating this voxel, then regenerating the voxel...
        
        self.num_vertices = 0  # this var should only be modified by calls to get_new_vertex_name; it is only used for naming new reigons
        
        # Initalize voxel tree with a single voxel that covers the entire workspace
        assert np.all(self.workspace[:, 0] == self.workspace[0, 0]) and np.all(self.workspace[:, 1] == self.workspace[0, 1]), "For now, workspace must be equal sizes in all dimensions."
        assert np.all(self.workspace[:, 0] + self.workspace[:, 1] == 0), "For now, workspace must be centered at origin."
        self.voxel_tree_root = Voxel(np.zeros(self.base_dim), self.workspace[0, 1] - self.workspace[0, 0], self.num_knot_points)
        self.voxel_tree = Tree()
        self.voxel_tree.create_node(self.voxel_tree_root.key, self.voxel_tree_root.key, data=self.voxel_tree_root)

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
        self.first_region_name = "0"
        
    def get_new_vertex_name(self):
        n = self.num_vertices
        self.num_vertices += 1
        return str(n)
    
    def successors(self, vertex_name: str) -> List[str]:
        """Override the default implementation of successors to return only
        successors that are voxels."""
        # self.first_region_name is an exception -- it is generated directly from the source vertex, so we do need to pop it off the queue to explicitely generate its voxel neighbors
        return [v for v in self._adjacency_list[vertex_name] if v == self.first_region_name or v == self.target_name or isinstance(self.vertices[v].convex_set, Voxel)]
    
    def check_voxel_fully_contained_in_region(self, voxel: Voxel, region: Polyhedron, check_other_regions: bool = True) -> bool:
        """
        Check if all vertices of a voxel are contained in a region.
        
        TODO: parallelize for multiple voxels (i.e. in a layer of the tree)?
        """
        vtxs = voxel.get_vertices()  # base_dim x num_vertices
        num_vtxs = vtxs.shape[1]
        
        # Ensure not all vertices of voxel are contained in current region
        if np.all(region.H @ vtxs <= np.tile(region.h, (num_vtxs, 1)).T + self.containment_tol):
            return True
        
        # Ensure not all vertices of voxel are contained in any other region
        if check_other_regions:
            for other_region in self.vertices.values():
                if isinstance(other_region.convex_set, Polyhedron):
                    if np.all(other_region.convex_set.H @ vtxs <= np.tile(other_region.convex_set.h, (num_vtxs, 1)).T + self.containment_tol):
                        return True

        return False
    
    def first_active_termination_condition(self, boundary_voxels: List[Voxel]) -> bool:
        """
        Termination condition for voxel tree search.
        
        Terminates after the first layer where a voxel is found that is
        partially-contained and collision-free.
        """
        if boundary_voxels:
            return True
        return False
    
    def max_depth_termination_condition(self, boundary_voxels: List[Voxel]) -> bool:
        """
        Termination condition for voxel tree search.
        
        Terminates if all relevant voxels have been maximially subdivided.
        Note that `find_polyhedron_boundary_voxels` will naturally terminate
        when all relevant voxels have been maximally subdivided; thus this 
        function doesn't need to do anything.
        """
        return False
    
    def find_polyhedron_boundary_voxels(self, region: Polyhedron, termination_condition: Callable[[List[Voxel]], bool]) -> List[Voxel]:
        """
        Search the voxel tree to find all voxels that are on the boundary of the 
        given region and are not in collision.
        
        These voxels will be added to the graph as successors to the current region.
        """
        boundary_voxels = []  # List of voxels on the boundary of the region to be returned
        
        q = deque([self.voxel_tree.get_node(self.voxel_tree.root)])
        while q:
            layer_size = len(q)  # Number of voxels at current layer
            
            ####################################################################
            # Process voxels at current layer (i.e. check partial containment, collision-free, etc.)
            ####################################################################
            # Iterate through non-closed voxels at current layer
            layer_voxels = []  # Build list of voxels at current layer that are not fully-contained in a region
            voxel_progs = []
            for _ in range(layer_size):
                voxel_node = q.popleft()
                voxel = voxel_node.data
                voxel_id = voxel.key

                vtxs = voxel.get_vertices()  # base_dim x num_vertices
                num_vtxs = vtxs.shape[1]
                
                # # Ensure voxel did not get fully contained in newly-generated region
                # if self.check_voxel_fully_contained_in_region(voxel, region, check_other_regions=False):
                #     voxel.status = VoxelStatus.CLOSED
                #     continue
                
                layer_voxels.append(voxel)
                
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
            # Then build list of voxels that are partially-contained by current region (and not fully-contained by any other region)
            partially_contained_voxels = []  # partially_contained_voxels will be a subset of non_contained_voxels
            voxel_solve_start = time.time()
            results = SolveInParallel(voxel_progs) 
            print(f"Time for voxel SolveInParallel: {time.time() - voxel_solve_start}")
            for i, result in enumerate(results):
                if result.is_success():  # Found a point in the intersection
                    partially_contained_voxels.append(layer_voxels[i])
                else:
                    pass
                    # print(f"No intersection found for voxel with center: {voxel.center}"
            
            # Check if partially-contained voxels are collision-free
            for voxel in partially_contained_voxels:
                if self.voxel_collision_checker.check_voxel_collision_free(voxel):
                    # print(f"voxel {voxel.center} with size {voxel.size} is collision-free")
                    voxel.status = VoxelStatus.ACTIVE
                    boundary_voxels.append(voxel)  # If so, this voxel should become ACTIVE and added to the graph as a successor to the current region
                    
            if termination_condition(boundary_voxels):
                # print(f"boundary_voxels: {list(map(lambda x: x.center, boundary_voxels))}")
                return boundary_voxels
            
            ####################################################################
            # If we did not find any partially-contained voxels not in collision, 
            # then search the next layer of the voxel tree, generating voxels
            # at that layer if needed.
            ####################################################################
            for voxel in partially_contained_voxels:
                voxel_id = voxel.key
                # If voxel has no children and voxel tree depth is less than max depth, then subdivide the voxel further (generating new children)
                if voxel.status != VoxelStatus.CLOSED and not self.voxel_tree.children(voxel_id) and self.voxel_tree.depth(voxel_id) < self.voxel_tree_max_depth-1:
                    # Genenerate voxel's 2^self.base_dim children
                    for direction in product([-1, 1], repeat=self.base_dim):
                        direction_array = np.array(direction, dtype=float)
                        offset = (voxel.size / 4) * direction_array  # The offset is (voxel.size/4) in each dimension multiplied by the direction factor.
                        child_center = voxel.center + offset
                        # print(f"Generating child voxel with center {child_center}.")
                        child_voxel = Voxel(child_center, voxel.size / 2, self.num_knot_points)
                        child = self.voxel_tree.create_node(child_voxel.key, child_voxel.key, data=child_voxel, parent=voxel_id)
                        
                        # Before adding child to voxel tree, check if it is fully-contained in current region or any other region
                        if self.check_voxel_fully_contained_in_region(child_voxel, region):
                            child_voxel.status = VoxelStatus.CLOSED
                            continue
                
                        # Only append child to queue if it is not fully-contained in any region
                        q.append(child)
                
                # Append voxels in next layer to queue
                # self.voxel_tree.show()  # Print voxel tree
                for child in self.voxel_tree.children(voxel_id):
                    child_voxel = child.data
                    if child_voxel.status != VoxelStatus.CLOSED:
                        q.append(child)

        return boundary_voxels

    def draw_edges_to_new_region(self, vertex_name: str, region: Polyhedron) -> None:
        """
        Detect intersections between newly generated region and any other regions
        and add respective edges to graph.
        """
        region_progs = []
        for other_region_name, other_region in self.vertices.items():
            if isinstance(other_region.convex_set, Polyhedron):
                # Create MathematicalProgram that searches a point in the intersection of region and other_region
                prog = MathematicalProgram()
                # Find x subject to: x is in region and x is in other_region
                x = prog.NewContinuousVariables(region.dim, "x")  # (base_dim,)
                # x ∈ region
                prog.AddConstraint(LinearConstraint(region.H, np.full((region.H.shape[0],), -np.inf), region.h), x)
                # x ∈ other_region
                prog.AddConstraint(LinearConstraint(other_region.convex_set.H, np.full((other_region.convex_set.H.shape[0],), -np.inf), other_region.convex_set.h), x)
                region_progs.append(prog)
                
        results = SolveInParallel(region_progs)
        for i, result in enumerate(results):
            if result.is_success():  # Found a point in the intersection
                self.add_undirected_edge(
                    Edge(
                        u=vertex_name,
                        v=other_region_name,
                        costs=self._create_single_edge_costs(vertex_name, other_region_name),
                        constraints=self._create_single_edge_constraints(vertex_name, other_region_name),
                    ),
                )

    def generate_successors(self, vertex_name: str) -> None:
        """Generates neighbors and adds them to the graph, also adds edges from
        vertex to neighbors."""
        neighbors = []  # List of (u_name, v_name, is_v_in_vertices, ConvexSet, v_neighbors) tuples
        
        if vertex_name == self.source_name:
            # Grow a region around the source
            starting_ellipse = Hyperellipsoid.MakeHypersphere(self.kEpsilonEllipsoid, self.s)
            region = IrisZo(self.voxel_collision_checker.checker, starting_ellipse, self.domain, self.iris_options)
            polyhedron = Polyhedron.from_drake_hpoly(region, should_compute_vertices=True if self.base_dim in [2, 3] else False, num_knot_points=self.num_knot_points)  # Compute vertices for 2D/3D visualization
            neighbors.append((self.source_name, self.first_region_name, polyhedron, [self.source_name]))
            
        elif vertex_name == self.target_name:
            # Should not generate neighbors for target vertex
            return
 
        else:
            if isinstance(self.vertices[vertex_name].convex_set, Voxel):
                """
                Inflate a new region around the voxel, updating statuses of any
                voxels that got covered by the new region.
                
                1. Inflate a new region around the voxel
                2. Switch the vertices[vertex_name] from containing the voxel to containing the new region
                3. Check if any other regions intersect with the new region, and add respective edges to graph
                """
                voxel = self.vertices[vertex_name].convex_set
                voxel.status = VoxelStatus.CLOSED  # Inflated voxels get closed (because they get covered by the new region)
                
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
                self.vertices[vertex_name].convex_set = polyhedron
                
                # CLOSE the inflated voxel
                voxel.status = VoxelStatus.CLOSED
                
                # Draw edges to newly generated region
                self.draw_edges_to_new_region(vertex_name, polyhedron)
                
                # Check if any non-CLOSED voxels were covered by the new region (if so, update their status to CLOSED)
                for voxel_node in self.voxel_tree.leaves():
                    voxel = voxel_node.data
                    if voxel.status != VoxelStatus.CLOSED and self.check_voxel_fully_contained_in_region(voxel, polyhedron, check_other_regions=False):
                        voxel.status = VoxelStatus.CLOSED
            
            else:  # For safety
                assert vertex_name == self.first_region_name, "generate_successors received a non-voxel vertex that is not the first region -- this is unexpected behavior. Please investigate."

                
            """
            Now, handle search for voxel successors of polyhedron
            
            1. Find voxels on the boundary (i.e. partially-contained) of the polyhedron that are not in collision using adaptive-size voxel tree search
               Simultaneously, update the voxel tree to reflect the status of voxels that got covered by the new region
            2. Add each voxel to the graph
            3. In gcs_star.py: Add a path (and solve its convex restriction) ending at each of those voxels to queue
            """
            polyhedron_boundary_voxels = self.find_polyhedron_boundary_voxels(self.vertices[vertex_name].convex_set, self.first_active_termination_condition)
            # polyhedron_boundary_voxels = self.find_polyhedron_boundary_voxels(self.vertices[vertex_name].convex_set, self.max_depth_termination_condition)
            
            # Add voxel successors to neighbors
            for voxel in polyhedron_boundary_voxels:
                neighbors.append((
                    vertex_name,
                    self.get_new_vertex_name(),
                    voxel
                ))
            
        # Add neighbors to graph
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
    def plot_voxel(self, voxel, fill=False, facecolor='yellow', edgecolor='magenta', alpha=0.3, hatch='/'):
        if self.base_dim == 2:
            center = voxel.center
            size = voxel.size
            if fill:
                rect = plt.Rectangle(
                    center - size / 2, size, size,
                    fill=fill, facecolor=facecolor, alpha=alpha, edgecolor=edgecolor, linewidth=1, hatch=hatch
                )
            else:
                rect = plt.Rectangle(
                    center - size / 2, size, size,
                    fill=fill, edgecolor=edgecolor, linewidth=1, hatch=hatch
                )
            return rect
        elif self.base_dim == 3:
            center = voxel.center
            size = voxel.size
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
                edgecolors='black',
                hatch=hatch
            )
            return cube
            
        
    def plot_polyhedron(self, polyhedron, facecolor='red', alpha=0.2):
        if self.base_dim == 2:
            vertices = polyhedron.vertices
            polygon_patch = plt.Polygon(vertices, closed=True, facecolor=facecolor, alpha=alpha, edgecolor='magenta', linewidth=1)
            return polygon_patch
        elif self.base_dim == 3:
            vertices = polyhedron.vertices
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
            self.patches = []
        
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
                
        # Remove old polyhedron and voxel patches
        for patch in self.patches:
            patch.remove()
        self.patches = []
        
        # Add voxel patches for all voxels in the voxel tree
        for voxel_node in self.voxel_tree.all_nodes_itr():
            voxel = voxel_node.data
            if voxel.status == VoxelStatus.CLOSED:
                fill = True
                face_color = 'gray'
                hatch = '/'
            elif voxel.status == VoxelStatus.ACTIVE:
                fill = True
                face_color = 'magenta'
                hatch = None
            else:  # voxel.status == VoxelStatus.OPEN
                fill = False
                face_color = None
                hatch = None
                    
            if self.base_dim == 2:
                rect = self.plot_voxel(voxel, fill=fill, facecolor=face_color, hatch=hatch)
                ax.add_patch(rect)
                self.patches.append(rect)
                
                # # Add voxel name as text in the center of the voxel
                # center = voxel.center
                # text = self.animation_ax.text(
                #     center[0], center[1], 
                #     voxel_node.key,
                #     ha='center', va='center',
                #     fontsize=8, color='black'
                # )
                # self.patches.append(text)
            
            elif self.base_dim == 3:
                # Only plot active voxels for visibility
                if voxel.status == VoxelStatus.ACTIVE:
                    cube = self.plot_voxel(voxel, fill=fill, facecolor=face_color, hatch=hatch)
                    ax.add_collection3d(cube)
                    self.patches.append(cube)
                
                # # Add voxel name as text in the center of the voxel
                # if vertex_name not in ["source", "target"]:
                #     center = vertex.convex_set.center
                #     text = ax.text3D(
                #         center[0], center[1], center[2],
                #         vertex_name,
                #         ha='center', va='center',
                #         fontsize=8, color='black'
                #     )
                #     self.patches.append(text)
        
        # Add polyhedron patches
        for vertex_name, vertex in self.vertices.items():
            if self.base_dim == 2:
                if isinstance(vertex.convex_set, Polyhedron):
                    polygon_patch = self.plot_polyhedron(vertex.convex_set, facecolor='red', alpha=0.1)
                    ax.add_patch(polygon_patch)
                    self.patches.append(polygon_patch)
                    
                    # Add region name at the center of the polyhedron
                    if vertex_name not in ["source", "target"]:
                        # Calculate the center of the polyhedron (average of vertices)
                        if hasattr(vertex.convex_set, 'vertices') and len(vertex.convex_set.vertices) > 0:
                            center = np.mean(vertex.convex_set.vertices, axis=0)
                            text = ax.text(
                                center[0], center[1], 
                                vertex_name,
                                ha='center', va='center',
                                fontsize=10, color='red', weight='bold'
                            )
                            self.patches.append(text)
            
            elif self.base_dim == 3:
                if isinstance(vertex.convex_set, Polyhedron):
                    polygon_patch_3d = self.plot_polyhedron(vertex.convex_set, facecolor='red', alpha=0.1)
                    ax.add_collection3d(polygon_patch_3d)
                    self.patches.append(polygon_patch_3d)

                    # Add region name at the center of the polyhedron
                    if vertex_name not in ["source", "target"]:
                        # Calculate the center of the polyhedron (average of vertices)
                        if hasattr(vertex.convex_set, 'vertices') and len(vertex.convex_set.vertices) > 0:
                            center = np.mean(vertex.convex_set.vertices, axis=0)
                            text = ax.text3D(
                                center[0], center[1], center[2],
                                vertex_name,
                                ha='center', va='center',
                                fontsize=10, color='red', weight='bold'
                            )
                            self.patches.append(text)
                    
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
                            rect = self.plot_voxel(self.vertices[vertex_name].convex_set, fill=True, facecolor='green', alpha=0.75, hatch='*')
                            ax.add_patch(rect)
                            self.patches.append(rect)
                        elif isinstance(self.vertices[vertex_name].convex_set, Polyhedron):
                            polygon_patch = self.plot_polyhedron(self.vertices[vertex_name].convex_set, facecolor='yellow', alpha=0.3)
                            ax.add_patch(polygon_patch)
                            self.patches.append(polygon_patch)
                
                # Add vertex path text
                self.path_text = ax.text(
                    0.05, 0.95,  # Position in axes coordinates (5% from left, 95% from bottom)
                    f'Path: {path_str}',
                    transform=ax.transAxes,  # Use axes coordinates
                    verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                    fontsize=8,
                    wrap=True
                )
                self.patches.append(self.path_text)  # Add to patches so it gets cleaned up
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
                            cube = self.plot_voxel(self.vertices[vertex_name].convex_set, fill=True, facecolor='green', alpha=0.75, hatch='*')
                            ax.add_collection3d(cube)
                            self.patches.append(cube)
                        elif isinstance(self.vertices[vertex_name].convex_set, Polyhedron):
                            polygon_patch_3d = self.plot_polyhedron(self.vertices[vertex_name].convex_set, facecolor='yellow', alpha=0.3)
                            ax.add_collection3d(polygon_patch_3d)
                            self.patches.append(polygon_patch_3d)
                
                # Add vertex path text
                self.path_text = ax.text2D(
                    0.05, 0.95,
                    f'Path: {path_str}',
                    transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                    fontsize=8,
                    wrap=True
                )
                self.patches.append(self.path_text)  # Add to patches so it gets cleaned up
            
        if self.base_dim == 2:
            ax.set_aspect('equal')
        elif self.base_dim == 3:
            ax.set_box_aspect([1, 1, 1])
            
        ax.grid(False)
        # ax.legend()
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        return fig, ax    
    
    def init_animation(self, save_animation: bool = False, animation_output_path: str = None):
        """Initialize a persistent figure for animation."""
        if not self.base_dim in [2, 3]:
            logger.warning(
                f"{self.__class__.__name__} Animation is not supported for base_dim {self.base_dim}."
            )
            return
        
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
                
        # Setup directory to save animation frames to compile animation video
        self.save_animation = save_animation
        if self.save_animation:
            # Create directory for frames if it doesn't exist
            print(f"animation_output_path: {animation_output_path}")
            self.frames_dir = os.path.join(os.path.dirname(animation_output_path), "animation_frames")
            os.makedirs(self.frames_dir, exist_ok=True)
            self.animation_output_path = animation_output_path
            self.frame_count = 0  # Initialize frame counter
        
    def update_animation(self, sol: Optional[ShortestPathSolution] = None, block: bool = False):
        """Update the animation with new voxels and optionally a new solution."""
        if not self.base_dim in [2, 3]:
            return
        
        if not hasattr(self, 'animation_fig'):
            raise ValueError("Animation not initialized. Call init_animation() first.")
            
        self.plot(ax=self.animation_ax, sol=sol)
        
        # Redraw
        self.animation_fig.canvas.draw()
        self.animation_fig.canvas.flush_events()
        
        if self.save_animation and os.path.exists(self.frames_dir):  # frames_dir may have been deleted already if we're visualizing post-algorithm termination (so animation video has already been saved)           
            # Save animation frame
            if not hasattr(self, 'frames_dir'):
                logger.warning("Animation recording not initialized. Call init_animation_recording first.")
                return
            
            # Save the current figure as a frame
            frame_path = os.path.join(self.frames_dir, f"frame_{self.frame_count:06d}.png")
            self.animation_fig.savefig(frame_path, dpi=200)
            
            # Increment frame counter
            self.frame_count += 1
        
        if block:
            plt.show(block=True)
            
    def compile_animation(self, fps: int = 2):
        """
        Compile the captured frames into a video file.
        Call this method after the algorithm has finished.
        
        Args:
            fps: Frames per second for the video
        """
        if not hasattr(self, 'frames_dir') or not hasattr(self, 'animation_output_path'):
            logger.warning("Animation recording not initialized. Call init_animation_recording first.")
            return
        
        import subprocess
        
        # Check if we have any frames
        if self.frame_count == 0:
            logger.warning("No frames captured. Cannot compile animation.")
            return
        
        # Use ffmpeg to compile frames into a video
        try:
            cmd = [
                "ffmpeg", "-y",  # Overwrite output file if it exists
                "-framerate", str(fps),
                "-i", os.path.join(self.frames_dir, "frame_%06d.png"),
                "-c:v", "libx264",
                "-profile:v", "high",
                "-crf", "20",  # Quality (lower is better)
                "-pix_fmt", "yuv420p",  # Required for compatibility
                self.animation_output_path
            ]
            
            logger.info(f"Compiling animation with command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            logger.info(f"Animation saved to {self.animation_output_path}")
            
            # Optionally clean up frames
            import shutil
            shutil.rmtree(self.frames_dir)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to compile animation: {e}")
            logger.info(f"Frames are saved in {self.frames_dir}")

        
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