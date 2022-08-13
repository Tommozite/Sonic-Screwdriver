import numpy as np
import math
import collections
import matplotlib.pyplot as plt


def check_inside(points, vertices, include_boundary=True):
    """Winding Algorithm, check if points lie within a polygon, including boundaries
    Input
    points: Mx2 Array of points to check against (cast to numpy array)
    vertices: Nx2 Array of vertices of polygon, in order (cast to numpy array)
    Output
    Mx1 Boolean array, value for each point in points"""

    points = np.array(points)
    vertices = np.array(vertices)

    # Algorithm fails for points on boundary,
    # So check for these separate, and stop work on them if no longer in consideration

    # test for point on vertices
    points_on_boundary = np.array([True if i in vertices else False for i in points])

    num_lines = len(vertices)  # number of lines in polygon

    def is_left(p, a, b):
        return np.sign(
            (b[0] - a[0]) * (p[..., 1] - a[1]) - (p[..., 0] - a[0]) * (b[1] - a[1])
        )

    winding_number = np.zeros(len(points))
    # Loop over line segments in polygon
    for l in range(num_lines):
        a, b = vertices[l], vertices[l + 1 % num_lines]
        # Array of indices of points still to be considered
        working_indices = np.nonzero(~points_on_boundary)
        working_points = points[working_indices]
        # Case 1: horizontal line
        if a[1] == b[1]:
            # Check if points lie on line, add to boundary array
            mask = np.logical_and(
                (working_points[:, 1] == a[1]),
                (working_points[:, 0] > a[0]),
                (working_points[:, 0] < b[0]),
            )
            new_boundary_indices = working_indices[np.nonzero(mask)]
            new_boundary_bool = np.zeros(len(points), dtype=bool)
            new_boundary_bool[new_boundary_indices] = True
            points_on_boundary = np.logical_or(points_on_boundary, new_boundary_bool)

        # Case 2: non-horizontal
        else:
            # Step 1: if point is horizontally inline with vertex, add epsilon
            in_line_a = (working_points[..., 1] == a[1]).astype(float)
            in_line_b = (working_points[..., 1] == b[1]).astype(float)
            eps_arr = in_line_a * np.spacing(a[1]) + in_line_b * np.spacing(b[1])
            print(working_points[..., 1].shape)
            print(eps_arr.shape)
            working_points[:, 1] += eps_arr
            # Step 2: check if point lies between ay and by
            crossing_mask = np.nonzero(
                np.logical_and(
                    working_points[..., 1] > min(a[1], b[1]),
                    working_points[..., 1] < max(a[1], b[1]),
                )
            )
            crossing_indices = working_indices[crossing_mask]
            crossing_points = working_points[crossing_mask]
            # Calc contribution to winding number from line segment
            winding_number_line = is_left(crossing_points, a, b)
            # Any points that lie on line added to boundary array
            new_boundary_indices = working_indices[winding_number_line == 0]
            new_boundary_bool = np.zeros(len(points), dtype=bool)
            new_boundary_bool[new_boundary_indices] = True
            points_on_boundary = np.logical_or(points_on_boundary, new_boundary_bool)
            winding_number[crossing_indices] += winding_number_line

    points_inside = np.nonzero(winding_number)

    if include_boundary:
        result = np.logical_or(points_inside, points_on_boundary)
    else:
        result = np.logical_and(points_inside, ~points_on_boundary)
    return result


def determine_epsilon(array_in):
    """Determines a small value epsilon, 1/100 of the smallest
    difference between any two non-equal elements of array"""
    array_sorted = np.sort(array_in)
    difference = np.diff(array_sorted)
    # Looking for a small non-zero number
    epsilon = np.min(difference[difference != 0]) / 100
    return epsilon


def rec_dd():
    return collections.defaultdict(rec_dd)


def wherein(y, x):
    """For every element in y, find the index of where it first appears in x. Returns
    a numpy mask array (np.ma) with the corresponding indices.
    Code from https://stackoverflow.com/questions/8251541/numpy-for-every-element-in-one-array-find-the-index-in-another-array"""
    y = np.array(y)
    x = np.array(x)
    index = np.argsort(x)
    sorted_x = x[index]
    sorted_index = np.searchsorted(sorted_x, y)

    yindex = np.take(index, sorted_index, mode="clip")
    mask = x[yindex] != y

    result = np.ma.array(yindex, mask=mask)
    return result


def get_colour(ax):
    return next(ax._get_lines.prop_cycler)["color"]
