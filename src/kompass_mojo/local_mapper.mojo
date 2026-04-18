# Local occupancy-map GPU kernel (Mojo port).
#
# Ported from kompass-core's SYCL implementation at:
#   src/kompass_cpp/kompass_cpp/src/mapping/local_mapper_gpu.cpp
#
# Converts a laserscan (angles + ranges) to
# a 2D occupancy grid using per-ray super-cover Bresenham line drawing.
#
# Launch: grid_dim = scan_size  (one work-group per ray)
#         block_dim = max_points_per_line  (one thread per pixel along the ray)
#
# Thread 0 of each group computes the ray endpoint in grid coordinates and
# writes the deltas and steps into shared memory; the barrier lets the rest
# of the group read them and each thread walks one step along the ray. Writes
# use atomic fetch_max against OccupancyType codes, so a later EMPTY cannot
# downgrade an earlier OCCUPIED.
#
# Grid memory is column-major (Eigen's default): cell (x, y) lives
# at flat index `x + y * rows`, where `rows == grid_height`

from std.math import cos, sin, ceil, round
from std.memory import UnsafePointer, stack_allocation
from std.os.atomic import Atomic
from std.gpu import (
    barrier,
    block_idx,
    thread_idx,
)
from std.gpu.memory import AddressSpace


comptime OCC_UNEXPLORED: Int32 = -1
comptime OCC_EMPTY: Int32 = 0
comptime OCC_OCCUPIED: Int32 = 100


# ---------------------------------------------------------------------------
# scan_to_grid_kernel
#
# SYCL source: local_mapper_gpu.cpp:63-152 (scanToGridKernel). Transliterated
# line-by-line for parity — do not "simplify" without matching the SYCL path.
# ---------------------------------------------------------------------------


def scan_to_grid_kernel(
    ranges: UnsafePointer[Float64, MutAnyOrigin],      # [scan_size]
    angles: UnsafePointer[Float64, MutAnyOrigin],      # [scan_size]
    grid: UnsafePointer[Int32, MutAnyOrigin],          # [rows*cols], col-major
    distances: UnsafePointer[Float32, MutAnyOrigin],   # [rows*cols], col-major
    rows: Int,
    cols: Int,
    resolution: Float32,
    laserscan_orientation: Float32,
    start_x_f: Float32,     # laserscan_position.x (local frame, metres)
    start_y_f: Float32,     # laserscan_position.y
    central_x: Int32,
    central_y: Int32,
    start_x: Int32,         # start_point in grid coords
    start_y: Int32,
):
    var group_id = Int(block_idx.x)
    var local_id = Int(thread_idx.x)

    # Shared: endpoint + deltas + steps. Thread 0 fills them, rest read.
    var to_point = stack_allocation[
        2, Int32, address_space=AddressSpace.SHARED
    ]()
    var deltas = stack_allocation[
        2, Int32, address_space=AddressSpace.SHARED
    ]()
    var steps = stack_allocation[
        2, Int32, address_space=AddressSpace.SHARED
    ]()

    var range_val = ranges[group_id]        # Float64
    var angle = angles[group_id]            # Float64

    if local_id == 0:
        # Downcast to float32 first
        var theta = laserscan_orientation + Float32(angle)
        var cos_a = cos(theta)
        var sin_a = sin(theta)
        var range_f = Float32(range_val)

        var to_local_x = start_x_f + range_f * cos_a
        var to_local_y = start_y_f + range_f * sin_a

        to_point[0] = central_x + Int32(ceil(to_local_x / resolution))
        to_point[1] = central_y + Int32(ceil(to_local_y / resolution))
        deltas[0] = to_point[0] - start_x
        deltas[1] = to_point[1] - start_y
        steps[0] = Int32(1) if deltas[0] >= Int32(0) else Int32(-1)
        steps[1] = Int32(1) if deltas[1] >= Int32(0) else Int32(-1)

    barrier()

    var delta_x_f = Float32(deltas[0])
    var delta_y_f = Float32(deltas[1])
    var abs_dx = delta_x_f if delta_x_f >= Float32(0.0) else -delta_x_f
    var abs_dy = delta_y_f if delta_y_f >= Float32(0.0) else -delta_y_f

    var x_float: Float32
    var y_float: Float32

    if abs_dx >= abs_dy:
        # Major axis = x: step along x, compute y from slope.
        var g = delta_y_f / delta_x_f if delta_x_f != Float32(0.0) else Float32(0.0)
        var sign_x: Float32
        if delta_x_f > Float32(0.0):
            sign_x = Float32(1.0)
        elif delta_x_f < Float32(0.0):
            sign_x = Float32(-1.0)
        else:
            sign_x = Float32(0.0)
        x_float = Float32(start_x) + sign_x * Float32(local_id)
        y_float = Float32(start_y) + g * (x_float - Float32(start_x))
    else:
        # Major axis = y: step along y, compute x from slope.
        var g = delta_x_f / delta_y_f if delta_y_f != Float32(0.0) else Float32(0.0)
        var sign_y: Float32
        if delta_y_f > Float32(0.0):
            sign_y = Float32(1.0)
        elif delta_y_f < Float32(0.0):
            sign_y = Float32(-1.0)
        else:
            sign_y = Float32(0.0)
        y_float = Float32(start_y) + sign_y * Float32(local_id)
        x_float = Float32(start_x) + g * (y_float - Float32(start_y))

    var x = Int32(round(x_float))
    var y = Int32(round(y_float))

    # Only stamp if (x, y) is inside the grid
    if x >= Int32(0) and x < Int32(rows) and y >= Int32(0) and y < Int32(cols):
        var main_idx = Int(x) + Int(y) * rows
        var xstep_idx = Int(x - steps[0]) + Int(y) * rows
        var ystep_idx = Int(x) + Int(y - steps[1]) * rows

        if x == to_point[0] and y == to_point[1]:
            # Endpoint cell: obstacle. Neighbours get EMPTY to handle inside corners
            Atomic.max(grid + main_idx, OCC_OCCUPIED)
            Atomic.max(grid + xstep_idx, OCC_EMPTY)
            Atomic.max(grid + ystep_idx, OCC_EMPTY)
        else:
            # Super-cover line interior: only mark EMPTY if this cell is
            # genuinely closer to the sensor than the measured range
            if distances[main_idx] < Float32(range_val):
                Atomic.max(grid + main_idx, OCC_EMPTY)
                Atomic.max(grid + xstep_idx, OCC_EMPTY)
                Atomic.max(grid + ystep_idx, OCC_EMPTY)
