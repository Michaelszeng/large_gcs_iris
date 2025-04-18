import numpy as np
import pytest

from large_gcs.algorithms.gcs_naive_astar import GcsNaiveAstar
from large_gcs.algorithms.search_algorithm import ReexploreLevel
from large_gcs.cost_estimators.shortcut_edge_ce import ShortcutEdgeCE
from large_gcs.graph.contact_cost_constraint_factory import (
    contact_shortcut_edge_l2norm_cost_factory_obj_weighted,
)
from large_gcs.graph.contact_graph import ContactGraph
from large_gcs.graph.graph import ShortestPathSolution
from large_gcs.graph.incremental_contact_graph import IncrementalContactGraph
from large_gcs.graph_generators.contact_graph_generator import (
    ContactGraphGeneratorParams,
)

tol = 1e-3

pytestmark = pytest.mark.skip(reason="These tests need to be fixed, skipping tests.")


def test_shortcut_edge_conv_res_cg_simple_2():
    graph_file = ContactGraphGeneratorParams.graph_file_path_from_name("cg_simple_2")
    cg = ContactGraph.load_from_file(graph_file)
    cost_estimator = ShortcutEdgeCE(
        cg,
        shortcut_edge_cost_factory=contact_shortcut_edge_l2norm_cost_factory_obj_weighted,
    )
    gcs_astar = GcsNaiveAstar(
        cg,
        cost_estimator=cost_estimator,
        reexplore_level=ReexploreLevel.NONE,
    )
    sol: ShortestPathSolution = gcs_astar.run()
    # fmt: off
    ambient_path = [[ 0.,  0., -2., -2.],[ 0.        , -0.        ,  0.        ,  0.        , -2.        , -0.2333346 , -2.        , -1.00000724, -0.        , -0.        ,  1.7666654 ,  0.99999276,  1.7666654 ,  0.99999276],[-0.        ,  0.22293125,  0.        ,  0.22293125, -0.2333346 , -0.51040336, -1.00000724, -0.27707599,  0.22293125,  0.22293125, -0.27706875,  0.72293125, -0.27706875,  0.72293125,  0.        ,  0.31527239],[ 2.22931245e-01,  1.44267189e+00,  2.22931245e-01,  2.22931245e-01, -5.10403358e-01,  7.09337283e-01, -2.77075995e-01,  7.22924005e-01,  1.21974064e+00, -0.00000000e+00,  1.21974064e+00,  1.00000000e+00,  7.85425201e+02,  1.00000000e+00,  7.84205461e+02,  1.21974064e+00],[ 1.44267189,  2.        ,  0.22293125,  0.        ,  0.70933728,  1.4666654 ,  0.72292401,  0.99999276,  0.55732811, -0.22293125,  0.75732811,  0.27706875,  0.75732811,  0.27706875,  0.        ,  0.60026075],[ 2.        ,  2.        , -0.        ,  0.        ,  1.4666654 ,  2.5       ,  0.99999276,  2.        , -0.        , -0.        ,  1.0333346 ,  1.00000724,  1.0333346 ,  1.00000724],[2. , 0. , 2.5, 2. ],]
    # fmt: on
    vertex_path = np.array(
        [
            "source",
            "('NC|obj0_f0-rob0_v1',)",
            "('IC|obj0_v0-rob0_f0',)",
            "('IC|obj0_f3-rob0_v0',)",
            "('IC|obj0_v3-rob0_f2',)",
            "('NC|obj0_f2-rob0_v2',)",
            "target",
        ]
    )
    assert np.isclose(sol.cost, 13.303138059978277, atol=tol)
    for v, v_sol in zip(sol.trajectory, ambient_path):
        if not np.allclose(v, v_sol, atol=tol):
            print(f"v: {v}\nv_sol: {v_sol}")
        assert np.allclose(v, v_sol, atol=tol)

    assert np.array_equal(sol.vertex_path, vertex_path)


def test_shortcut_edge_conv_res_cg_simple_2_inc():
    graph_file = ContactGraphGeneratorParams.graph_file_path_from_name("cg_simple_2")
    cg = IncrementalContactGraph.load_from_file(
        graph_file,
        should_incl_simul_mode_switches=False,
        should_add_const_edge_cost=True,
        should_add_gcs=True,
    )
    cost_estimator = ShortcutEdgeCE(
        cg,
        shortcut_edge_cost_factory=contact_shortcut_edge_l2norm_cost_factory_obj_weighted,
    )
    gcs_astar = GcsNaiveAstar(
        cg,
        cost_estimator=cost_estimator,
        reexplore_level=ReexploreLevel.NONE,
    )
    sol: ShortestPathSolution = gcs_astar.run()
    # fmt: off
    ambient_path = [[ 0.,  0., -2., -2.],[ 0.        , -0.        ,  0.        ,  0.        , -2.        , -0.2333346 , -2.        , -1.00000724, -0.        , -0.        ,  1.7666654 ,  0.99999276,  1.7666654 ,  0.99999276],[-0.        ,  0.22293125,  0.        ,  0.22293125, -0.2333346 , -0.51040336, -1.00000724, -0.27707599,  0.22293125,  0.22293125, -0.27706875,  0.72293125, -0.27706875,  0.72293125,  0.        ,  0.31527239],[ 2.22931245e-01,  1.44267189e+00,  2.22931245e-01,  2.22931245e-01, -5.10403358e-01,  7.09337283e-01, -2.77075995e-01,  7.22924005e-01,  1.21974064e+00, -0.00000000e+00,  1.21974064e+00,  1.00000000e+00,  7.85425201e+02,  1.00000000e+00,  7.84205461e+02,  1.21974064e+00],[ 1.44267189,  2.        ,  0.22293125,  0.        ,  0.70933728,  1.4666654 ,  0.72292401,  0.99999276,  0.55732811, -0.22293125,  0.75732811,  0.27706875,  0.75732811,  0.27706875,  0.        ,  0.60026075],[ 2.        ,  2.        , -0.        ,  0.        ,  1.4666654 ,  2.5       ,  0.99999276,  2.        , -0.        , -0.        ,  1.0333346 ,  1.00000724,  1.0333346 ,  1.00000724],[2. , 0. , 2.5, 2. ],]
    # fmt: on
    vertex_path = np.array(
        [
            "source",
            "('NC|obj0_f0-rob0_v1',)",
            "('IC|obj0_v0-rob0_f0',)",
            "('IC|obj0_f3-rob0_v0',)",
            "('IC|obj0_v3-rob0_f2',)",
            "('NC|obj0_f2-rob0_v2',)",
            "target",
        ]
    )
    assert np.isclose(sol.cost, 13.303138059978277, atol=tol)
    assert all(
        np.allclose(v, v_sol, atol=tol)
        for v, v_sol in zip(sol.trajectory, ambient_path)
    )
    assert np.array_equal(sol.vertex_path, vertex_path)


def test_shortcut_edge_conv_res_cg_simple_3_inc():
    """Test the incremental graph for cg_simple_3 with shortcut edge cost
    estimator.

    What's different from cg_simple_2 is that this graph has a target
    region instead of a target position.
    """
    graph_file = ContactGraphGeneratorParams.graph_file_path_from_name("cg_simple_3")
    cg = IncrementalContactGraph.load_from_file(
        graph_file,
        should_incl_simul_mode_switches=False,
        should_add_const_edge_cost=True,
        should_add_gcs=True,
    )
    cost_estimator = ShortcutEdgeCE(
        cg,
        shortcut_edge_cost_factory=contact_shortcut_edge_l2norm_cost_factory_obj_weighted,
    )
    gcs_astar = GcsNaiveAstar(
        cg,
        cost_estimator=cost_estimator,
        reexplore_level=ReexploreLevel.NONE,
    )
    sol: ShortestPathSolution = gcs_astar.run()
    ambient_path = [
        [0.000, 0.000, -2.000, -2.000],
        [
            0.000,
            -0.000,
            0.000,
            0.000,
            -2.000,
            -0.233,
            -2.000,
            -1.000,
            -0.000,
            -0.000,
            1.767,
            1.000,
            1.767,
            1.000,
        ],
        [
            -0.000,
            0.354,
            0.000,
            0.354,
            -0.233,
            -0.380,
            -1.000,
            -0.146,
            0.354,
            0.354,
            -0.146,
            0.854,
            217.147,
            218.147,
            307.299,
            0.500,
        ],
        [
            0.354,
            2.000,
            0.354,
            0.354,
            -0.380,
            1.267,
            -0.146,
            -0.146,
            1.646,
            -0.000,
            1.646,
            0.000,
            395.393,
            0.000,
            393.747,
            1.646,
        ],
        [2.000, 0.354, 1.267, -0.146],
    ]
    vertex_path = np.array(
        [
            "source",
            "('NC|obj0_f0-rob0_v1',)",
            "('IC|obj0_v0-rob0_f0',)",
            "('IC|obj0_f3-rob0_v0',)",
            "target",
        ]
    )
    assert np.isclose(sol.cost, 9.35848932653506, atol=tol)
    assert all(
        np.allclose(v, v_sol, atol=tol)
        for v, v_sol in zip(sol.trajectory, ambient_path)
    )
    assert np.array_equal(sol.vertex_path, vertex_path)


@pytest.mark.slow_test
def test_shortcut_edge_conv_res_cg_trichal2():
    graph_file = ContactGraphGeneratorParams.graph_file_path_from_name("cg_trichal2")
    cg = ContactGraph.load_from_file(graph_file)
    cost_estimator = ShortcutEdgeCE(
        cg,
        shortcut_edge_cost_factory=contact_shortcut_edge_l2norm_cost_factory_obj_weighted,
    )
    gcs_astar = GcsNaiveAstar(
        cg,
        cost_estimator=cost_estimator,
        reexplore_level=ReexploreLevel.NONE,
    )
    sol: ShortestPathSolution = gcs_astar.run()
    ambient_path = [
        [3.250, 0.000, 1.500, 0.500],
        [
            3.250,
            3.250,
            0.000,
            0.000,
            1.500,
            2.417,
            0.500,
            -1.167,
            -0.000,
            -0.000,
            0.917,
            -1.667,
            0.917,
            -1.667,
        ],
        [
            3.250,
            3.250,
            0.000,
            0.000,
            2.417,
            3.917,
            -1.167,
            -1.167,
            -0.000,
            -0.000,
            1.500,
            0.000,
            1.500,
            0.000,
        ],
        [
            3.250,
            1.000,
            0.000,
            -0.000,
            3.917,
            1.667,
            -1.167,
            -1.167,
            -2.250,
            -0.000,
            -2.250,
            0.000,
            -2.250,
            0.000,
            0.000,
            2.250,
        ],
        [
            1.000,
            -0.500,
            -0.000,
            -1.500,
            1.667,
            0.167,
            -1.167,
            -2.138,
            -1.500,
            -1.500,
            -1.500,
            -0.971,
            -1.500,
            -0.971,
            2.121,
            0.000,
            2.121,
            3.000,
        ],
        [
            -0.500,
            -1.500,
            -1.500,
            -1.500,
            0.167,
            -0.833,
            -2.138,
            -2.667,
            -1.000,
            -0.000,
            -1.000,
            -0.529,
            -1.000,
            -0.529,
            0.000,
            0.000,
            0.000,
            1.000,
        ],
        [
            -1.500,
            -1.500,
            -1.500,
            -1.500,
            -0.833,
            -1.833,
            -2.667,
            -2.667,
            -0.000,
            -0.000,
            -1.000,
            -0.000,
            -1.000,
            -0.000,
            0.000,
            0.000,
            0.000,
            -0.000,
        ],
        [
            -1.500,
            -1.500,
            -1.500,
            0.000,
            -1.833,
            -2.333,
            -2.667,
            -0.167,
            -0.000,
            1.500,
            -0.500,
            2.500,
            -0.500,
            2.500,
            3.000,
            0.000,
            3.000,
            3.354,
        ],
        [
            -1.500,
            -1.500,
            0.000,
            0.000,
            -2.333,
            -3.000,
            -0.167,
            0.000,
            -0.000,
            -0.000,
            -0.667,
            0.167,
            -0.667,
            0.167,
            0.000,
            0.000,
        ],
        [-1.500, 0.000, -3.000, 0.000],
    ]
    vertex_path = np.array(
        [
            "source",
            "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v1', 'NC|obj0_f3-rob0_v0')",
            "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v1', 'NC|obj0_f0-rob0_v1')",
            "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v1', 'IC|obj0_f1-rob0_f1')",
            "('IC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v1', 'IC|obj0_f1-rob0_f1')",
            "('IC|obs0_v0-obj0_f2', 'NC|obs0_f0-rob0_v1', 'IC|obj0_f1-rob0_f1')",
            "('IC|obs0_f2-obj0_f1', 'NC|obs0_f0-rob0_v1', 'IC|obj0_f0-rob0_v1')",
            "('IC|obs0_f2-obj0_f1', 'NC|obs0_f2-rob0_v0', 'IC|obj0_v0-rob0_f0')",
            "('IC|obs0_f2-obj0_f1', 'NC|obs0_f2-rob0_v0', 'NC|obj0_f3-rob0_v0')",
            "target",
        ]
    )
    assert np.isclose(sol.cost, 24.51282601210196, atol=tol)
    assert all(
        np.allclose(v, v_sol, atol=tol)
        for v, v_sol in zip(sol.trajectory, ambient_path)
    )
    assert np.array_equal(sol.vertex_path, vertex_path)
