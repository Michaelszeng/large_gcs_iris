import numpy as np
from typing import List
from pydrake.all import (
    Cost,
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
    if "source" in u or v == "target":
        if "source" in u and v == "target":
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
    if "source" in u:
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
    b = np.zeros((base_dim,1))
    
    return LinearEqualityConstraint(A, b)


def vertex_constraint_last_pos_equality_cfree(base_dim: int, num_knot_points: int, u: str, sample: np.ndarray) -> LinearEqualityConstraint:
    """
    Creates a constraint to ensure the last knot point of a vertex matches a sampled point.

    A = [0, I]
    b = sample_last_position
    x = [x0, ..., x{n-1}]
    """
    if "source" in u:
        A = np.eye(base_dim)  # source is a point and so has only base_dim variables
    else:
        A = np.zeros((base_dim, num_knot_points * base_dim))
        A[:, -base_dim:] = np.eye(base_dim)  # select last knot point
    
    # b = sample
    # return LinearEqualityConstraint(A, b)
    
    tol = 1e-1
    lb = sample - tol
    ub = sample + tol
    return LinearConstraint(A, lb, ub)


def shortcut_edge_l2norm_cost_factory(
    u: str,
    base_dim: int,
    num_knot_points: int,
    heuristic_inflation_factor: float = 1,
    add_const_cost: bool = False,
    const_cost: float = 1e-1,
) -> List[Cost]:
    """
    Cost function defining the heuristic. Specifically, the heuristic is the 
    L2 distance from the last knot point of the successor set u to some point,
    potentially inflated by a scaling factor.
    
    This is used in two places:
    1. L2 norm heuristic to the target
    2. Sampling domination checker: Adding a cost for the final edge to a sample 
    point when evaluating a path to the sample point
    
    Linear cost of the form: a'x + b, where a is a vector of coefficients and b is a constant.
    
    Heuristic_inflation_factor simply scales the outputted cost, which may be
    useful when using this function to generate the heuristic cost, to prune
    more aggressively.
    
    A = [0,...,0, I, -I]
                  ^   ^
                  |   |
            u's last  target
    b = 0
    x = [u_x0, ..., u_x{n-1}, target_x]
    """
    # L2 distance cost
    if u == "source":
        u_part = np.eye(base_dim)  # source is a point and so has only base_dim variables
    else:
        u_part = np.zeros((base_dim, num_knot_points * base_dim))
        u_part[:, -base_dim:] = np.eye(base_dim)  # select last knot point
    
    target_part = -np.eye(base_dim)  # target is a point and so has only base_dim variables

    A = heuristic_inflation_factor * np.hstack((u_part, target_part))
    b = np.zeros((base_dim, 1))
    
    costs = [L2NormCost(A, b)]
    
    # Constant cost for edge activation
    if add_const_cost:
        if u == "source":
            total_dim = base_dim + base_dim  # source and target only have one knot point
        else:
            total_dim = num_knot_points * base_dim + base_dim
        # Constant cost for the edge
        a = np.zeros((total_dim, 1))
        # We add 2 because if a shortcut is used it minimally replaces 2 edges
        constant_cost = 2 * const_cost
        costs.append(LinearCost(a, constant_cost))
    
    return costs