"""RDP selection with literal preparation/gather and owned segment arithmetic."""
import numpy as np
from . import _core

_INTEGERS = (int, np.int8, np.int16, np.int32, np.int64,
             np.uint8, np.uint16, np.uint32, np.uint64)


def _scalar_trig(dr, dc):
    angle = -np.arctan2(np.float64(dr), np.float64(dc))
    return np.sin(angle), np.cos(angle)


def approximate_polygon(coords, tolerance):
    # scikit-image0.26 _polygon.py, BSD-3-Clause; see the retained license
    # in third_party/scikit-image-LICENSE.txt. This protocol runs exactly once.
    if tolerance <= 0:
        return coords
    native_result = False
    scalar = (type(tolerance) in (bool, float, np.float16, np.float32, np.float64)
              or type(tolerance) in _INTEGERS and -(1 << 53) <= tolerance <= (1 << 53))
    if (scalar and type(coords) is np.ndarray and coords.dtype == np.dtype('float64')
            and coords.ndim == 2 and coords.shape[1] == 2 and coords.shape[0] >= 1):
        prepared = np.require(coords, requirements=('C', 'A'))
        keep = np.empty(coords.shape[0], dtype=np.bool_)
        distances = np.empty(coords.shape[0], dtype=np.float64)
        native_result = _core.approximate_polygon_apply(
            prepared, float(tolerance), keep, distances, _core.get_num_threads(), _scalar_trig)
        if native_result:
            return coords[keep, :]
    # Keep the original positive-tolerance body as the explicit provider route.
    chain = np.zeros(coords.shape[0], 'bool')
    # pre-allocate distance array for all points
    dists = np.zeros(coords.shape[0])
    chain[0] = True
    chain[-1] = True
    pos_stack = [(0, chain.shape[0] - 1)]
    end_of_chain = False

    while not end_of_chain:
        start, end = pos_stack.pop()
        # determine properties of current line segment
        r0, c0 = coords[start, :]
        r1, c1 = coords[end, :]
        dr = r1 - r0
        dc = c1 - c0
        segment_angle = -np.arctan2(dr, dc)
        segment_dist = c0 * np.sin(segment_angle) + r0 * np.cos(segment_angle)

        # select points in-between line segment
        segment_coords = coords[start + 1 : end, :]
        segment_dists = dists[start + 1 : end]

        # check whether to take perpendicular or euclidean distance with
        # inner product of vectors

        # vectors from points -> start and end
        dr0 = segment_coords[:, 0] - r0
        dc0 = segment_coords[:, 1] - c0
        dr1 = segment_coords[:, 0] - r1
        dc1 = segment_coords[:, 1] - c1
        # vectors points -> start and end projected on start -> end vector
        projected_lengths0 = dr0 * dr + dc0 * dc
        projected_lengths1 = -dr1 * dr - dc1 * dc
        perp = np.logical_and(projected_lengths0 > 0, projected_lengths1 > 0)
        eucl = np.logical_not(perp)
        segment_dists[perp] = np.abs(
            segment_coords[perp, 0] * np.cos(segment_angle)
            + segment_coords[perp, 1] * np.sin(segment_angle)
            - segment_dist
        )
        segment_dists[eucl] = np.minimum(
            # distance to start point
            np.sqrt(dc0[eucl] ** 2 + dr0[eucl] ** 2),
            # distance to end point
            np.sqrt(dc1[eucl] ** 2 + dr1[eucl] ** 2),
        )

        if np.any(segment_dists > tolerance):
            # select point with maximum distance to line
            new_end = start + np.argmax(segment_dists) + 1
            pos_stack.append((new_end, end))
            pos_stack.append((start, new_end))
            chain[new_end] = True

        if len(pos_stack) == 0:
            end_of_chain = True

    return coords[chain, :]
