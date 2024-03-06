import numpy as np

from large_gcs.geometry.geometry_utils import unique_rows_with_tolerance_ignore_nan


def test_unique_rows_with_tolerance_ignore_nan():
    arr = np.array(
        [
            [1.0, np.nan, 3.0],
            [1.0, 2.001, 3.0],
            [1.0, 2.00, 3.0],
            [4.0, 5.0, 6.0],
            [4.002, 5.002, 6.002],
            [1.00001, np.nan, 3.00001],
            [7.0, 8.0, 9.0],
            [7.011, 8.0, 9.0],
            [np.nan, np.nan, np.nan],  # All NaN row for demonstration
            [1.0, np.nan, 3.002],
        ]
    )

    tol = 0.01  # Tolerance level
    unique_arr = unique_rows_with_tolerance_ignore_nan(arr, tol)
    sol = np.array(
        [
            [1.0, 2.001, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [7.011, 8.0, 9.0],
        ]
    )
    assert np.all(unique_arr == sol)
