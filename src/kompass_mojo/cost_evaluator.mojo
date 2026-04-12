# Trajectory cost evaluator GPU kernels (Mojo port).
#
# Ported from kompass-core's SYCL implementation at:
#   src/kompass_cpp/kompass_cpp/src/utils/cost_evaluator_gpu.cpp
#
# All six kernels accumulate trajectory costs via atomic add into a per-
# trajectory `costs` buffer. The final `min_cost_block_reduce` +
# `min_cost_final_reduce` pair picks the lowest-cost trajectory and its
# index (two-pass because Mojo kernels cannot synchronize across blocks).

from std.math import sqrt
from std.memory import UnsafePointer, stack_allocation
from std.os.atomic import Atomic
from std.gpu import (
    barrier,
    block_dim,
    block_idx,
    global_idx,
    thread_idx,
)
from std.gpu.memory import AddressSpace


comptime DTYPE = DType.float32
comptime F32 = Scalar[DTYPE]

# TODO: Block size for all kernels. Must be a power of two so the tree reduction
# unrolls cleanly. 256 is a conservative setup. This is different from dynamic
# size allocation at kernel launch time. Need to either expose this as a
# compile time param or explore dynamic allocation in mojo.
comptime WG_SIZE: Int = 256
comptime LOG2_WG: Int = 8  # log2(WG_SIZE)

comptime F32_MAX: F32 = 3.4028235e38  # std::numeric_limits<float>::max()
comptime F32_INF: F32 = F32_MAX       # padding sentinel (large enough)


# ---------------------------------------------------------------------------
# 1. goal_cost_kernel
#
# SYCL: cost_evaluator_gpu.cpp:546-573 (goalCostKernel)
# One thread per trajectory, distance from last trajectory point to the goal,
# normalized by reference-path length, atomically added to the trajectory
# cost.
# ---------------------------------------------------------------------------


fn goal_cost_kernel(
    paths_x: UnsafePointer[F32, MutAnyOrigin],
    paths_y: UnsafePointer[F32, MutAnyOrigin],
    costs: UnsafePointer[F32, MutAnyOrigin],
    trajs_size: Int,
    path_size: Int,
    goal_x: F32,
    goal_y: F32,
    inv_path_length: F32,
    cost_weight: F32,
):
    var tid = Int(global_idx.x)
    if tid >= trajs_size:
        return

    var last_idx = tid * path_size + (path_size - 1)
    var dx = paths_x[last_idx] - goal_x
    var dy = paths_y[last_idx] - goal_y
    var distance = sqrt(dx * dx + dy * dy) * inv_path_length

    _ = Atomic.fetch_add(costs + tid, cost_weight * distance)


# ---------------------------------------------------------------------------
# 2. smoothness_cost_kernel
#
# SYCL: cost_evaluator_gpu.cpp:610-667 (smoothnessCostKernel)
# One block per trajectory. Each thread computes partial sum of squared
# first-derivative velocity deltas, then block tree-reduces and thread 0
# atomically adds the normalized cost.
# ---------------------------------------------------------------------------


fn smoothness_cost_kernel(
    vel_vx: UnsafePointer[F32, MutAnyOrigin],
    vel_vy: UnsafePointer[F32, MutAnyOrigin],
    vel_omega: UnsafePointer[F32, MutAnyOrigin],
    costs: UnsafePointer[F32, MutAnyOrigin],
    velocities_count: Int,    # = points_per_traj - 1
    inv_lim_x: F32,
    inv_lim_y: F32,
    inv_lim_omega: F32,
    cost_weight: F32,
):
    var traj_idx = Int(block_idx.x)
    var tid = Int(thread_idx.x)

    var partials = stack_allocation[
        WG_SIZE, F32, address_space=AddressSpace.SHARED
    ]()

    var local_contrib: F32 = 0.0
    var i = tid
    while i < velocities_count:
        if i >= 1:
            var curr = traj_idx * velocities_count + i
            var prev = curr - 1

            var dvx = vel_vx[curr] - vel_vx[prev]
            local_contrib = local_contrib + (dvx * dvx) * inv_lim_x

            var dvy = vel_vy[curr] - vel_vy[prev]
            local_contrib = local_contrib + (dvy * dvy) * inv_lim_y

            var dvw = vel_omega[curr] - vel_omega[prev]
            local_contrib = local_contrib + (dvw * dvw) * inv_lim_omega

        i = i + WG_SIZE

    partials[tid] = local_contrib
    barrier()

    @parameter
    for step in range(LOG2_WG):
        if tid < (WG_SIZE >> (step + 1)):
            partials[tid] = partials[tid] + partials[tid + (WG_SIZE >> (step + 1))]
        barrier()

    if tid == 0:
        var total = partials[0]
        var norm = F32(3.0) * F32(velocities_count)
        var final_val = cost_weight * (total / norm)
        _ = Atomic.fetch_add(costs + traj_idx, final_val)


# ---------------------------------------------------------------------------
# 3. jerk_cost_kernel
#
# SYCL: cost_evaluator_gpu.cpp:703-765 (jerkCostKernel)
# Same launch shape as smoothness, but per-index contribution is
# (v[i] - 2*v[i-1] + v[i-2])^2 — the discrete second derivative.
# ---------------------------------------------------------------------------


fn jerk_cost_kernel(
    vel_vx: UnsafePointer[F32, MutAnyOrigin],
    vel_vy: UnsafePointer[F32, MutAnyOrigin],
    vel_omega: UnsafePointer[F32, MutAnyOrigin],
    costs: UnsafePointer[F32, MutAnyOrigin],
    velocities_count: Int,
    inv_lim_x: F32,
    inv_lim_y: F32,
    inv_lim_omega: F32,
    cost_weight: F32,
):
    var traj_idx = Int(block_idx.x)
    var tid = Int(thread_idx.x)

    var partials = stack_allocation[
        WG_SIZE, F32, address_space=AddressSpace.SHARED
    ]()

    var local_contrib: F32 = 0.0
    var i = tid
    while i < velocities_count:
        if i >= 2:
            var curr = traj_idx * velocities_count + i
            var prev1 = curr - 1
            var prev2 = curr - 2

            var vx_term = vel_vx[curr] - F32(2.0) * vel_vx[prev1] + vel_vx[prev2]
            local_contrib = local_contrib + (vx_term * vx_term) * inv_lim_x

            var vy_term = vel_vy[curr] - F32(2.0) * vel_vy[prev1] + vel_vy[prev2]
            local_contrib = local_contrib + (vy_term * vy_term) * inv_lim_y

            var w_term = vel_omega[curr] - F32(2.0) * vel_omega[prev1] + vel_omega[prev2]
            local_contrib = local_contrib + (w_term * w_term) * inv_lim_omega

        i = i + WG_SIZE

    partials[tid] = local_contrib
    barrier()

    @parameter
    for step in range(LOG2_WG):
        if tid < (WG_SIZE >> (step + 1)):
            partials[tid] = partials[tid] + partials[tid + (WG_SIZE >> (step + 1))]
        barrier()

    if tid == 0:
        var total = partials[0]
        var norm = F32(3.0) * F32(velocities_count)
        var final_val = cost_weight * (total / norm)
        _ = Atomic.fetch_add(costs + traj_idx, final_val)


# ---------------------------------------------------------------------------
# 4. ref_path_cost_kernel
#
# SYCL: cost_evaluator_gpu.cpp:405-507 (refPathCostKernel)
# One block per trajectory. Each thread handles one (or more, strided)
# reference-path points; for each ref point it finds the closest trajectory
# point using a shared-memory tile of trajectory positions, then the block
# tree-reduces the sum of min-distances and thread 0 adds the end-point
# distance and atomically writes the normalized cost.
# ---------------------------------------------------------------------------


fn ref_path_cost_kernel(
    paths_x: UnsafePointer[F32, MutAnyOrigin],
    paths_y: UnsafePointer[F32, MutAnyOrigin],
    tracked_x: UnsafePointer[F32, MutAnyOrigin],
    tracked_y: UnsafePointer[F32, MutAnyOrigin],
    costs: UnsafePointer[F32, MutAnyOrigin],
    path_size: Int,            # numPointsPerTrajectory_
    ref_size: Int,             # tracked_segment_size
    inv_ref_length: F32,       # 1 / tracked_segment_length
    inv_ref_size_count: F32,   # 1 / refSize
    cost_weight: F32,
):
    var traj_idx = Int(block_idx.x)
    var tid = Int(thread_idx.x)

    var tile_tx = stack_allocation[
        WG_SIZE, F32, address_space=AddressSpace.SHARED
    ]()
    var tile_ty = stack_allocation[
        WG_SIZE, F32, address_space=AddressSpace.SHARED
    ]()
    var partials = stack_allocation[
        WG_SIZE, F32, address_space=AddressSpace.SHARED
    ]()

    var local_total_dist: F32 = 0.0

    # Align outer loop bound up to a multiple of WG_SIZE so all threads
    # participate in the tile loads/barriers each iter.
    var aligned_ref_size = ((ref_size + WG_SIZE - 1) // WG_SIZE) * WG_SIZE

    var k = tid
    while k < aligned_ref_size:
        var valid_ref_point = k < ref_size
        var rx: F32 = 0.0
        var ry: F32 = 0.0
        var r_min_dist_sq: F32 = F32_MAX

        if valid_ref_point:
            rx = tracked_x[k]
            ry = tracked_y[k]

        var tile_base = 0
        while tile_base < path_size:
            var traj_pt_idx = tile_base + tid
            if traj_pt_idx < path_size:
                var global_traj_idx = traj_idx * path_size + traj_pt_idx
                tile_tx[tid] = paths_x[global_traj_idx]
                tile_ty[tid] = paths_y[global_traj_idx]
            else:
                tile_tx[tid] = F32_INF
                tile_ty[tid] = F32_INF

            barrier()

            if valid_ref_point:
                @parameter
                for t in range(WG_SIZE):
                    var dx = rx - tile_tx[t]
                    var dy = ry - tile_ty[t]
                    var d2 = dx * dx + dy * dy
                    if d2 < r_min_dist_sq:
                        r_min_dist_sq = d2

            barrier()
            tile_base = tile_base + WG_SIZE

        if valid_ref_point:
            local_total_dist = local_total_dist + sqrt(r_min_dist_sq)

        k = k + WG_SIZE

    partials[tid] = local_total_dist
    barrier()

    @parameter
    for step in range(LOG2_WG):
        if tid < (WG_SIZE >> (step + 1)):
            partials[tid] = partials[tid] + partials[tid + (WG_SIZE >> (step + 1))]
        barrier()

    if tid == 0:
        var total_dist_sum = partials[0]
        var avg_path_dist = total_dist_sum * inv_ref_size_count

        var last_traj_idx = traj_idx * path_size + (path_size - 1)
        var last_ref_idx = ref_size - 1
        var end_dx = paths_x[last_traj_idx] - tracked_x[last_ref_idx]
        var end_dy = paths_y[last_traj_idx] - tracked_y[last_ref_idx]
        var end_dist = sqrt(end_dx * end_dx + end_dy * end_dy) * inv_ref_length

        var final_val = cost_weight * ((avg_path_dist + end_dist) * F32(0.5))
        _ = Atomic.fetch_add(costs + traj_idx, final_val)


# ---------------------------------------------------------------------------
# 5. obstacles_dist_cost_kernel
#
# SYCL: cost_evaluator_gpu.cpp:806-891 (obstaclesDistCostKernel)
# One block per trajectory. Threads tile over obstacles (WG_SIZE obstacles
# into shared memory per outer iter), then stride over trajectory points,
# computing all-pairs squared distance and keeping the running minimum.
# Block min-reduces across threads, then thread 0 converts to normalized
# cost and atomically adds.
# ---------------------------------------------------------------------------


fn obstacles_dist_cost_kernel(
    paths_x: UnsafePointer[F32, MutAnyOrigin],
    paths_y: UnsafePointer[F32, MutAnyOrigin],
    obs_x: UnsafePointer[F32, MutAnyOrigin],
    obs_y: UnsafePointer[F32, MutAnyOrigin],
    costs: UnsafePointer[F32, MutAnyOrigin],
    path_size: Int,
    obs_size: Int,
    max_obstacle_distance: F32,
    cost_weight: F32,
):
    var traj_idx = Int(block_idx.x)
    var tid = Int(thread_idx.x)

    var tile_ox = stack_allocation[
        WG_SIZE, F32, address_space=AddressSpace.SHARED
    ]()
    var tile_oy = stack_allocation[
        WG_SIZE, F32, address_space=AddressSpace.SHARED
    ]()
    var partials = stack_allocation[
        WG_SIZE, F32, address_space=AddressSpace.SHARED
    ]()

    var min_dist_sq_for_point: F32 = F32_MAX

    var tile_base = 0
    while tile_base < obs_size:
        var obs_idx = tile_base + tid
        if obs_idx < obs_size:
            tile_ox[tid] = obs_x[obs_idx]
            tile_oy[tid] = obs_y[obs_idx]
        else:
            tile_ox[tid] = F32_INF
            tile_oy[tid] = F32_INF

        barrier()

        var k = tid
        while k < path_size:
            var px = paths_x[traj_idx * path_size + k]
            var py = paths_y[traj_idx * path_size + k]

            @parameter
            for j in range(WG_SIZE):
                var dx = px - tile_ox[j]
                var dy = py - tile_oy[j]
                var d2 = dx * dx + dy * dy
                if d2 < min_dist_sq_for_point:
                    min_dist_sq_for_point = d2

            k = k + WG_SIZE

        barrier()
        tile_base = tile_base + WG_SIZE

    partials[tid] = min_dist_sq_for_point
    barrier()

    @parameter
    for step in range(LOG2_WG):
        if tid < (WG_SIZE >> (step + 1)):
            var a = partials[tid]
            var b = partials[tid + (WG_SIZE >> (step + 1))]
            partials[tid] = a if a < b else b
        barrier()

    if tid == 0:
        var traj_min_dist_sq = partials[0]
        var traj_min_dist = sqrt(traj_min_dist_sq)
        var clamped = max_obstacle_distance - traj_min_dist
        if clamped < F32(0.0):
            clamped = F32(0.0)
        var normalized_cost = clamped / max_obstacle_distance
        _ = Atomic.fetch_add(costs + traj_idx, cost_weight * normalized_cost)


# ---------------------------------------------------------------------------
# 6. min_cost reduction (two-pass: per-block + final single-block)
#
# SYCL: cost_evaluator_gpu.cpp:338-349 (minimumCostReduction, LowestCost
# combine operator from datatypes/trajectory.h:610)
#
# Mojo kernels can't synchronize across blocks, so we emit one (cost, idx)
# pair per block in the first pass and reduce those into a single pair in
# the second pass. Tie-break on lower index matches LowestCost::combine.
# ---------------------------------------------------------------------------


fn min_cost_block_reduce(
    costs: UnsafePointer[F32, MutAnyOrigin],
    block_min_cost: UnsafePointer[F32, MutAnyOrigin],
    block_min_idx: UnsafePointer[Int32, MutAnyOrigin],
    n: Int,
):
    var tid = Int(thread_idx.x)
    var gid = Int(block_idx.x) * Int(block_dim.x) + tid

    var s_cost = stack_allocation[
        WG_SIZE, F32, address_space=AddressSpace.SHARED
    ]()
    var s_idx = stack_allocation[
        WG_SIZE, Int32, address_space=AddressSpace.SHARED
    ]()

    if gid < n:
        s_cost[tid] = costs[gid]
        s_idx[tid] = Int32(gid)
    else:
        s_cost[tid] = F32_MAX
        s_idx[tid] = Int32(0)

    barrier()

    @parameter
    for step in range(LOG2_WG):
        if tid < (WG_SIZE >> (step + 1)):
            var other = tid + (WG_SIZE >> (step + 1))
            var a_cost = s_cost[tid]
            var a_idx = s_idx[tid]
            var b_cost = s_cost[other]
            var b_idx = s_idx[other]
            var take_b = (b_cost < a_cost) or (b_cost == a_cost and b_idx < a_idx)
            if take_b:
                s_cost[tid] = b_cost
                s_idx[tid] = b_idx
        barrier()

    if tid == 0:
        block_min_cost[Int(block_idx.x)] = s_cost[0]
        block_min_idx[Int(block_idx.x)] = s_idx[0]


fn min_cost_final_reduce(
    block_min_cost: UnsafePointer[F32, MutAnyOrigin],
    block_min_idx: UnsafePointer[Int32, MutAnyOrigin],
    out_cost: UnsafePointer[F32, MutAnyOrigin],
    out_idx: UnsafePointer[Int32, MutAnyOrigin],
    num_blocks: Int,
):
    var tid = Int(thread_idx.x)

    var s_cost = stack_allocation[
        WG_SIZE, F32, address_space=AddressSpace.SHARED
    ]()
    var s_idx = stack_allocation[
        WG_SIZE, Int32, address_space=AddressSpace.SHARED
    ]()

    # Fold num_blocks into the WG_SIZE shared slots by striding.
    var my_cost: F32 = F32_MAX
    var my_idx: Int32 = 0
    var i = tid
    while i < num_blocks:
        var c = block_min_cost[i]
        var ix = block_min_idx[i]
        var take_b = (c < my_cost) or (c == my_cost and ix < my_idx)
        if take_b:
            my_cost = c
            my_idx = ix
        i = i + WG_SIZE

    s_cost[tid] = my_cost
    s_idx[tid] = my_idx
    barrier()

    @parameter
    for step in range(LOG2_WG):
        if tid < (WG_SIZE >> (step + 1)):
            var other = tid + (WG_SIZE >> (step + 1))
            var a_cost = s_cost[tid]
            var a_idx = s_idx[tid]
            var b_cost = s_cost[other]
            var b_idx = s_idx[other]
            var take_b = (b_cost < a_cost) or (b_cost == a_cost and b_idx < a_idx)
            if take_b:
                s_cost[tid] = b_cost
                s_idx[tid] = b_idx
        barrier()

    if tid == 0:
        out_cost[0] = s_cost[0]
        out_idx[0] = s_idx[0]
