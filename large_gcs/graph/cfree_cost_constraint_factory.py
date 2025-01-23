import numpy as np
import scipy
from pydrake.all import (
    L2NormCost, 
    LinearCost,
    LinearConstraint,
    LinearEqualityConstraint,
)


def create_cfree_l2norm_vertex_cost(base_dim: int) -> L2NormCost:
    """
    Creates an L2 norm cost for distance between two knot points.
    
    Assumes that the vertex has exactly 2 knot points.

    A = [I, -I]
    b = 0
    x = [x0, x1]
    """
    A = np.hstack((np.eye(base_dim), -np.eye(base_dim)))
    b = np.zeros((base_dim,1))
    return L2NormCost(A, b)


def create_cfree_constant_edge_cost(base_dim: int, u: str, v: str, num_knot_points: int, constant_cost: float = 1) -> LinearCost:
    """
    Creates a cost that penalizes each active edge a constant value.
    
    Linear cost of the form: a'x + b, where a is a vector of coefficients and b is a constant.
    """
    # Depending on whether vertex is source or target, we need to adjust the dimension of the cost vector
    if u == "source" or v == "target":
        if u == "source" and v == "target":
            total_dim = base_dim + base_dim  # source and target only have one knot point
        else:
            total_dim = num_knot_points * base_dim + base_dim  
    else:
        total_dim = 2 * num_knot_points * base_dim
    
    a = np.zeros((total_dim, 1))
    b = constant_cost
    return LinearCost(a, b)


def create_cfree_continuity_edge_constraint(base_dim: int, u: str, v: str, num_knot_points: int) -> LinearEqualityConstraint:
    """
    Ensures path continuity between two vertices by enforcing equality between
    the last knot point of vertex u and the first knot point of vertex v.
    
    A = [0,...,0, I, -I, 0,...,0]
                 ^    ^
                 |    |
            u's last  v's first
    b = 0
    x = [u_x0, ..., u_x{n-1}, v_x0, ..., v_x{n-1}]
    """
    if u == "source":
        u_part = np.eye(base_dim)  # source is a point and so has only base_dim variables
    else:
        u_part = np.zeros((base_dim, num_knot_points * base_dim))
        u_part[:, -base_dim:] = np.eye(base_dim)  # select last knot point
    
    if v == "target":
        v_part = -np.eye(base_dim)  # target is a point and so has only base_dim variables
    else:
        v_part = np.zeros((base_dim, num_knot_points * base_dim))
        v_part[:, :base_dim] = -np.eye(base_dim)  # select first knot point

    A = np.hstack((u_part, v_part))
    b = np.zeros((base_dim,))
    
    return LinearEqualityConstraint(A, b)


def create_source_region_edge_constraint(base_dim: int) -> LinearEqualityConstraint:
    """
    Equates source region point to vertex' first knot point.
    
    A = [I, -I, 0]
    b = 0
    x = [s_x, v_x0, v_x1]
    """
    A = np.hstack((np.eye(base_dim), -np.eye(base_dim), np.zeros((base_dim, base_dim))))
    b = np.zeros((base_dim,))
    return LinearEqualityConstraint(A, b)


def create_region_target_edge_constraint(base_dim: int) -> LinearEqualityConstraint:
    """
    Equates target region point to vertex' last knot point.
    
    A = [0, I, -I]
    b = 0
    x = [v_x0, v_x1, t_x]
    """
    A = np.hstack((np.zeros((base_dim, base_dim)), np.eye(base_dim), -np.eye(base_dim)))
    b = np.zeros((base_dim,))
    return LinearEqualityConstraint(A, b)


def vertex_constraint_last_pos_equality_cfree(
    sample: np.ndarray,
) -> LinearEqualityConstraint:
    """
    Creates a constraint to ensure the last position of a vertex matches a sampled point.

    Assumes:
        - The convex set has 2 knot points.
        - The last position is the second knot point (i.e., the second half of the variables).

    A = [0, I]
    b = sample_last_position
    x = [vertex_pos0, vertex_pos1]
    """
    base_dim = sample.shape[0] // 2
    A = np.hstack((np.zeros((base_dim, base_dim)), np.eye(base_dim)))
    b = sample[base_dim:]
    return LinearEqualityConstraint(A, b)
