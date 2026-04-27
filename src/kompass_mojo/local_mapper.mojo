# Local occupancy-map GPU kernels (Mojo port).
#
# Ported from kompass-core's SYCL implementation at:
#   src/kompass_cpp/kompass_cpp/src/mapping/local_mapper_gpu.cpp
#
# Two kernels: a pointcloud→laserscan converter and a laserscan→grid
# raycast. The pointcloud path is `cloud → laserscan_to_grid_kernel ⇒
# scan_to_grid_kernel`; the laserscan-input path skips the first kernel.
#
# 1. pointcloud_to_laserscan_kernel — one thread per point.
#    Reads (x, y, z) from raw PointCloud2 bytes, applies a z-range filter
#    and an origin filter, computes the angular bin via a PTX-safe atan2
#    polynomial, and atomic-fetch-min's the point's range into
#    ranges[bin]. Caller pre-fills ranges with `max_range` so empty bins
#    read back as max_range. Output is float32 to keep atomic_min cheap.
#    Launch: grid_dim = ceildiv(num_points, WG)
#            block_dim = WG (e.g. max_points_per_line, 256 typical)
#
# 2. scan_to_grid_kernel — one work-group per laserscan ray.
#    Thread 0 of each group computes the ray endpoint in grid coordinates
#    and writes deltas and steps into shared memory; the barrier lets the
#    rest of the group read them and each thread walks one step along the
#    ray, drawing a super-cover Bresenham line. Writes use atomic fetch_max
#    against OccupancyType codes, so a later EMPTY cannot downgrade an
#    earlier OCCUPIED.
#    Launch: grid_dim = scan_size            (one work-group per ray)
#            block_dim = max_points_per_line (one thread per pixel)
#
# Grid memory is column-major (Eigen's default): cell (x, y) lives at
# flat index `x + y * rows`, where `rows == grid_height`. OccupancyType
# codes: UNEXPLORED = -1, EMPTY = 0, OCCUPIED = 100.
#
# Note on _fast_atan2: `std.math.atan2` lowers to libm's atan2f, which
# does not link in PTX. The Hastings 5-term polynomial below is well
# below bin precision at all relevant scan_size values (max error ~1e-5
# rad vs smallest bin width 0.001745 rad at 3600 bins).

from std.math import cos, sin, ceil, round, sqrt, pi
from std.memory import UnsafePointer, stack_allocation
from std.os.atomic import Atomic
from std.gpu import (
    barrier,
    block_dim,
    block_idx,
    thread_idx,
)
from std.gpu.memory import AddressSpace


comptime OCC_UNEXPLORED: Int32 = -1
comptime OCC_EMPTY: Int32 = 0
comptime OCC_OCCUPIED: Int32 = 100

comptime TWO_PI: Float32 = Float32(2.0) * Float32(pi)
comptime HALF_PI: Float32 = Float32(0.5) * Float32(pi)
comptime PI_F32: Float32 = Float32(pi)


def _fast_atan2(y: Float32, x: Float32) -> Float32:
    # PTX-safe atan2. `std.math.atan2` lowers to libm's atan2f which doesn't
    # link in PTX. Hastings' 5-term odd polynomial for atan(z) on |z|<=1 has
    # max error ~1e-5 rad — orders of magnitude below the smallest bin width
    # we use (3600 bins → 0.001745 rad/bin, 63 bins → 0.0997 rad/bin), so
    # binning is exact at all relevant angular resolutions.
    var ax: Float32 = x if x >= Float32(0.0) else -x
    var ay: Float32 = y if y >= Float32(0.0) else -y

    var num = ay if ax >= ay else ax
    var den = ax if ax >= ay else ay
    var z: Float32 = num / den if den > Float32(0.0) else Float32(0.0)

    # Hastings: atan(z) ≈ z * (a1 + a3·z² + a5·z⁴ + a7·z⁶ + a9·z⁸)
    # with alternating signs folded into the coefficients.
    var z2 = z * z
    var atan_z = z * (
        Float32(0.9998660)
        - z2 * (
            Float32(0.3302995)
            - z2 * (
                Float32(0.1801410)
                - z2 * (Float32(0.0851330) - z2 * Float32(0.0208351))
            )
        )
    )

    var result: Float32 = atan_z if ax >= ay else (HALF_PI - atan_z)
    if x < Float32(0.0):
        result = PI_F32 - result
    if y < Float32(0.0):
        result = -result
    return result


# ---------------------------------------------------------------------------
# scan_to_grid_kernel
#
# SYCL source: local_mapper_gpu.cpp:63-152 (scanToGridKernel). Transliterated
# line-by-line for parity.
# ---------------------------------------------------------------------------


def scan_to_grid_kernel(
    ranges: UnsafePointer[Float32, MutAnyOrigin],      # [scan_size]
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

    var range_val = ranges[group_id]        # Float32
    var angle = angles[group_id]            # Float64

    if local_id == 0:
        # Downcast angle to float32 to match kompass-core's body math.
        var theta = laserscan_orientation + Float32(angle)
        var cos_a = cos(theta)
        var sin_a = sin(theta)

        var to_local_x = start_x_f + range_val * cos_a
        var to_local_y = start_y_f + range_val * sin_a

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
            if distances[main_idx] < range_val:
                Atomic.max(grid + main_idx, OCC_EMPTY)
                Atomic.max(grid + xstep_idx, OCC_EMPTY)
                Atomic.max(grid + ystep_idx, OCC_EMPTY)


# ---------------------------------------------------------------------------
# pointcloud_to_laserscan_kernel
#
# SYCL source: local_mapper_gpu.cpp:55-164 (pointcloudToLaserScanKernel).
# One thread per point: load (x, y, z) from raw PointCloud2 bytes, apply
# a z-range filter and an origin filter, compute the angular bin via
# atan2, and atomic-fetch-min the point's range into ranges[bin]. The
# caller pre-fills ranges with `max_range` so empty bins read back as
# max_range. Transliterated line-by-line for parity.
# ---------------------------------------------------------------------------


def pointcloud_to_laserscan_kernel(
    raw_bytes: UnsafePointer[Int8, MutAnyOrigin],
    total_bytes: Int,
    num_points: Int,
    point_step: Int,
    row_step: Int,
    width: Int,
    is_contiguous_i: Int32,    # 1 if row_step == width * point_step
    x_offset: Int,
    y_offset: Int,
    z_offset: Int,
    min_z: Float32,
    max_z: Float32,
    max_z_enabled_i: Int32,    # 1 if max_z >= 0
    num_bins: Int,
    inv_two_pi_times_bins: Float32,   # num_bins / (2*pi)
    ranges_out: UnsafePointer[Float32, MutAnyOrigin],   # [num_bins]
):
    var tid = Int(thread_idx.x)
    var gid = Int(block_idx.x) * Int(block_dim.x) + tid

    if gid >= num_points:
        return

    var is_contiguous = is_contiguous_i != Int32(0)
    var max_z_enabled = max_z_enabled_i != Int32(0)

    var byte_offset: Int
    if is_contiguous:
        byte_offset = gid * point_step
    else:
        var row = gid // width
        var col = gid % width
        byte_offset = row * row_step + col * point_step

    # Bounds guard: furthest field + 4 bytes (float32) must fit.
    var x_off_bytes = byte_offset + x_offset
    var y_off_bytes = byte_offset + y_offset
    var z_off_bytes = byte_offset + z_offset
    var max_end = x_off_bytes
    if y_off_bytes > max_end:
        max_end = y_off_bytes
    if z_off_bytes > max_end:
        max_end = z_off_bytes
    max_end = max_end + 4
    if max_end > total_bytes:
        return

    var z_ptr = (raw_bytes + z_off_bytes).bitcast[Float32]()
    var z = z_ptr[]
    if z < min_z:
        return
    if max_z_enabled and z > max_z:
        return

    var x_ptr = (raw_bytes + x_off_bytes).bitcast[Float32]()
    var y_ptr = (raw_bytes + y_off_bytes).bitcast[Float32]()
    var x = x_ptr[]
    var y = y_ptr[]

    var r2 = x * x + y * y
    if r2 < Float32(1.0e-6):
        return

    # atan2 normalized to [0, 2π), then bin via clamp.
    var angle = _fast_atan2(y, x)
    if angle < Float32(0.0):
        angle = angle + TWO_PI
    var bin = Int(angle * inv_two_pi_times_bins)
    if bin >= num_bins:
        bin = num_bins - 1

    var dist = sqrt(r2)
    _ = Atomic.min(ranges_out + bin, dist)
