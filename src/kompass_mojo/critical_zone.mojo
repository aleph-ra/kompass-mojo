# Critical-zone safety checker GPU kernels (Mojo port).
#
# Ported from kompass-core's SYCL implementation at:
#   src/kompass_cpp/kompass_cpp/src/utils/critical_zone_check_gpu.cpp
#
# Two kernels, each producing a single float safety factor in [0, 1]:
#   1.0 → safe (robot can proceed at full speed)
#   0.0 → stop (an obstacle sits inside the critical zone)
#   intermediate → slowdown factor (linear interpolation inside the
#                  slowdown band between crit_dist and slow_dist)
#
# Both kernels take a `forward` bool (passed via per-kernel args) that
# selects which angular cone to evaluate.

# - critical_zone_laserscan_kernel: ONE block of WG_SIZE threads,
#   stride over the sparse critical_indices buffer, tree-reduce the
#   partials in shared memory, thread 0 writes the result. No atomic
#   needed because there's only one block.
# - critical_zone_pointcloud_kernel: MANY blocks of WG_SIZE threads,
#   stride over the full num_points. Each block tree-reduces its own
#   partials, thread 0 atomic_min's into the global result.

from std.math import sqrt
from std.memory import UnsafePointer, stack_allocation
from std.atomic import Atomic
from std.gpu import (
    barrier,
    block_dim,
    block_idx,
    thread_idx,
)
from std.gpu.memory import AddressSpace

from kompass_mojo.cost_evaluator import F32, WG_SIZE, LOG2_WG


# ---------------------------------------------------------------------------
# Laserscan kernel
#
# SYCL source: critical_zone_check_gpu.cpp:199-290.
# Launch: grid_dim=1, block_dim=WG_SIZE. The whole reduction fits in one
# block because num_work_items is the sparse forward/backward cone, which
# is typically O(100) for a 360-bin scan.
# ---------------------------------------------------------------------------


def critical_zone_laserscan_kernel(
    ranges: UnsafePointer[F32, MutAnyOrigin],           # [scan_size]
    cos_angles: UnsafePointer[F32, MutAnyOrigin],       # [scan_size]
    sin_angles: UnsafePointer[F32, MutAnyOrigin],       # [scan_size]
    critical_indices: UnsafePointer[Int32, MutAnyOrigin],  # [num_work_items]
    num_work_items: Int,
    tf00: F32, tf01: F32, tf03: F32,
    tf10: F32, tf11: F32, tf13: F32,
    robot_radius: F32,
    crit_dist: F32,
    dist_denom: F32,           # slow_dist - crit_dist
    safe_threshold_sq: F32,    # (slow_dist + robot_radius)^2
    out_factor: UnsafePointer[F32, MutAnyOrigin], # 1.0
):
    var tid = Int(thread_idx.x)

    var partials = stack_allocation[
        WG_SIZE, F32, address_space=AddressSpace.SHARED
    ]()

    # Each thread walks its slice of critical_indices in stride WG_SIZE.
    # Track the running minimum safety factor; default to 1.0 (safe).
    var local_min: F32 = 1.0
    var i = tid
    while i < num_work_items:
        var global_idx = Int(critical_indices[i])
        var r = ranges[global_idx]
        var c = cos_angles[global_idx]
        var s = sin_angles[global_idx]

        # Transform sensor-frame point (r*cos, r*sin, 0) to body frame
        # via the 2D submatrix
        var x = tf00 * r * c + tf01 * r * s + tf03
        var y = tf10 * r * c + tf11 * r * s + tf13

        var dist_sq = x * x + y * y

        # Early-out: points farther than slow_dist + radius can't affect
        # the minimum (factor stays at 1.0 for them).
        if dist_sq < safe_threshold_sq:
            var dist = sqrt(dist_sq)
            var surface = dist - robot_radius

            if surface <= crit_dist:
                local_min = F32(0.0)
            else:
                var factor = (surface - crit_dist) / dist_denom
                if factor < local_min:
                    local_min = factor
        i = i + WG_SIZE

    partials[tid] = local_min
    barrier()

    # Tree reduction across the block.
    comptime for step in range(LOG2_WG):
        if tid < (WG_SIZE >> (step + 1)):
            var a = partials[tid]
            var b = partials[tid + (WG_SIZE >> (step + 1))]
            partials[tid] = a if a < b else b
        barrier()

    if tid == 0:
        out_factor[0] = partials[0]


# ---------------------------------------------------------------------------
# Pointcloud kernel
#
# SYCL source: critical_zone_check_gpu.cpp:78-186 (CheckRawCloudSafety).
# Launch: grid_dim=ceil(num_points / WG_SIZE), block_dim=WG_SIZE. Each
# block reduces its own partials; thread 0 of each block does a global
# Atomic.min into out_factor.
# ---------------------------------------------------------------------------


def critical_zone_pointcloud_kernel(
    raw_bytes: UnsafePointer[Int8, MutAnyOrigin],
    total_bytes: Int,
    num_points: Int,
    point_step: Int,
    row_step: Int,
    width: Int,
    is_contiguous_i: Int32,    # 1 if contiguous, 0 otherwise
    x_offset: Int,
    y_offset: Int,
    z_offset: Int,
    min_z: F32,
    max_z: F32,
    tf00: F32, tf01: F32, tf03: F32,
    tf10: F32, tf11: F32, tf13: F32,
    cos_sq_crit_angle: F32,    # cos^2(half-angle). Cone check uses this
                               # instead of atan2(y, x).
    robot_radius: F32,
    crit_dist: F32,
    inv_dist_range: F32,       # 1 / (slow_dist - crit_dist)
    slow_dist_sq_limit: F32,   # (slow_dist + robot_radius)^2
    check_forward_i: Int32,    # 1 = forward, 0 = backward
    out_factor: UnsafePointer[F32, MutAnyOrigin],
):
    var tid = Int(thread_idx.x)
    var gid = Int(block_idx.x) * Int(block_dim.x) + tid

    var partials = stack_allocation[
        WG_SIZE, F32, address_space=AddressSpace.SHARED
    ]()

    var is_contiguous = is_contiguous_i != Int32(0)
    var check_forward = check_forward_i != Int32(0)

    var local_min: F32 = 1.0

    # Single pass: one thread per point (with out-of-range guard). The
    # launcher sizes grid_dim so gid covers all points.
    if gid < num_points:
        var byte_offset: Int
        if is_contiguous:
            byte_offset = gid * point_step
        else:
            var row = gid // width
            var col = gid % width
            byte_offset = row * row_step + col * point_step

        # Bounds guard: point header + largest offset + 4 bytes (float) must
        # fit inside the buffer.
        var x_off_bytes = byte_offset + x_offset
        var y_off_bytes = byte_offset + y_offset
        var z_off_bytes = byte_offset + z_offset
        var max_end = x_off_bytes
        if y_off_bytes > max_end:
            max_end = y_off_bytes
        if z_off_bytes > max_end:
            max_end = z_off_bytes
        max_end = max_end + 4  # sizeof(float32)

        if max_end <= total_bytes:
            # Load x, y, z as float32. PointCloud2 fields are 4-byte floats;
            # upstream also supports int8/16/32 variants but the benchmark
            # and tests use float32
            var x_ptr = (raw_bytes + x_off_bytes).bitcast[F32]()
            var y_ptr = (raw_bytes + y_off_bytes).bitcast[F32]()
            var z_ptr = (raw_bytes + z_off_bytes).bitcast[F32]()
            var x_sens = x_ptr[]
            var y_sens = y_ptr[]
            var z = z_ptr[]

            var valid = True
            # Z-range filter.
            if z < min_z or z > max_z:
                valid = False

            # Origin filter: reject (0, 0, *) in sensor frame.
            if valid:
                var r2_sens = x_sens * x_sens + y_sens * y_sens
                if r2_sens < F32(1.0e-6):
                    valid = False

            if valid:
                # Transform to body frame (2D submatrix only; z dropped).
                var x_body = tf00 * x_sens + tf01 * y_sens + tf03
                var y_body = tf10 * x_sens + tf11 * y_sens + tf13

                var dist_sq = x_body * x_body + y_body * y_body

                # Angular cone filter, expressed without atan2. Forward cone
                # `|atan2(y, x)| ≤ θ` is equivalent to `x > 0 AND x^2 ≥
                # cos^2(θ) * (x^2+y^2)`. Backward cone `|atan2(y, x)| ≥ π-θ` is
                # equivalent to `x < 0 AND x^2 ≥ cos^2(θ) * (x^2+y^2)`. The
                # sign of x_body differs for each mode.
                var in_zone: Bool
                var x_body_sq = x_body * x_body
                var cone_sq = cos_sq_crit_angle * dist_sq
                if check_forward:
                    in_zone = (x_body > F32(0.0)) and (x_body_sq >= cone_sq)
                else:
                    in_zone = (x_body < F32(0.0)) and (x_body_sq >= cone_sq)

                if in_zone:
                    if dist_sq < slow_dist_sq_limit:
                        var dist = sqrt(dist_sq)
                        var surface = dist - robot_radius
                        if surface <= crit_dist:
                            local_min = F32(0.0)
                        else:
                            var factor = (surface - crit_dist) * inv_dist_range
                            if factor < local_min:
                                local_min = factor

    partials[tid] = local_min
    barrier()

    # Tree reduction within the block.
    comptime for step in range(LOG2_WG):
        if tid < (WG_SIZE >> (step + 1)):
            var a = partials[tid]
            var b = partials[tid + (WG_SIZE >> (step + 1))]
            partials[tid] = a if a < b else b
        barrier()

    # Thread 0 of each block combines the block's min into the global
    # result. Atomic because multiple blocks race for the same scalar.
    if tid == 0:
        Atomic.min(out_factor, partials[0])
