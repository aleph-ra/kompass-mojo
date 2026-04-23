# C FFI entry points for the Mojo cost evaluator kernels.
#
# Exposes create/run/destroy to C via @export. The C contract is declared
# in integration/kompass_mojo.h — keep that file in sync with the @export
# signatures below.

from std.math import ceildiv, sqrt, cos, sin
from std.memory import UnsafePointer, alloc
from std.gpu.host import DeviceContext, DeviceBuffer

from kompass_mojo.cost_evaluator import (
    DTYPE,
    F32,
    WG_SIZE,
    goal_cost_kernel,
    smoothness_cost_kernel,
    jerk_cost_kernel,
    ref_path_cost_kernel,
    obstacles_dist_cost_kernel,
    min_cost_block_reduce,
    min_cost_final_reduce,
)
from kompass_mojo.local_mapper import (
    OCC_UNEXPLORED,
    scan_to_grid_kernel,
)
from kompass_mojo.critical_zone import (
    critical_zone_laserscan_kernel,
    critical_zone_pointcloud_kernel,
)


# ---------------------------------------------------------------------------
# C-compatible config POD (kept in byte-order sync with kompass_mojo.h).
# ---------------------------------------------------------------------------


struct MojoCostEvalConfig(TrivialRegisterPassable):
    var acc_lim_x: F32
    var acc_lim_y: F32
    var acc_lim_omega: F32
    var ref_path_weight: F32
    var goal_weight: F32
    var smoothness_weight: F32
    var jerk_weight: F32
    var obstacles_weight: F32
    var max_obstacles_distance: F32


# ---------------------------------------------------------------------------
# Opaque handle owning the DeviceContext and all device buffers. Allocated
# at create() with max capacities so run() never reallocates.
# ---------------------------------------------------------------------------


struct CostEvalHandle(Movable):
    var ctx: DeviceContext

    var paths_x: DeviceBuffer[DTYPE]
    var paths_y: DeviceBuffer[DTYPE]
    var vel_vx: DeviceBuffer[DTYPE]
    var vel_vy: DeviceBuffer[DTYPE]
    var vel_omega: DeviceBuffer[DTYPE]
    var tracked_x: DeviceBuffer[DTYPE]
    var tracked_y: DeviceBuffer[DTYPE]
    var obstacles_x: DeviceBuffer[DTYPE]
    var obstacles_y: DeviceBuffer[DTYPE]
    var costs: DeviceBuffer[DTYPE]
    var block_min_cost: DeviceBuffer[DTYPE]
    var block_min_idx: DeviceBuffer[DType.int32]
    var final_min_cost: DeviceBuffer[DTYPE]
    var final_min_idx: DeviceBuffer[DType.int32]

    var max_trajs: Int
    var points_per_traj: Int
    var max_obstacles: Int
    var cfg: MojoCostEvalConfig

    def __init__(
        out self,
        max_trajs: Int,
        points_per_traj: Int,
        max_obstacles: Int,
        cfg: MojoCostEvalConfig,
    ) raises:
        self.ctx = DeviceContext()
        self.max_trajs = max_trajs
        self.points_per_traj = points_per_traj
        self.max_obstacles = max_obstacles
        self.cfg = cfg

        var points_n = max_trajs * points_per_traj
        var vel_n = max_trajs * (points_per_traj - 1)
        # SYCL ref path interpolation produces (points_per_traj + 1) points
        # for kompass-core's waypoint layout; size generously.
        var max_ref = points_per_traj + 2
        var max_obs = max_obstacles if max_obstacles > 0 else 1
        var max_blocks = ceildiv(max_trajs, WG_SIZE)

        self.paths_x = self.ctx.enqueue_create_buffer[DTYPE](points_n)
        self.paths_y = self.ctx.enqueue_create_buffer[DTYPE](points_n)
        self.vel_vx = self.ctx.enqueue_create_buffer[DTYPE](vel_n)
        self.vel_vy = self.ctx.enqueue_create_buffer[DTYPE](vel_n)
        self.vel_omega = self.ctx.enqueue_create_buffer[DTYPE](vel_n)
        self.tracked_x = self.ctx.enqueue_create_buffer[DTYPE](max_ref)
        self.tracked_y = self.ctx.enqueue_create_buffer[DTYPE](max_ref)
        self.obstacles_x = self.ctx.enqueue_create_buffer[DTYPE](max_obs)
        self.obstacles_y = self.ctx.enqueue_create_buffer[DTYPE](max_obs)
        self.costs = self.ctx.enqueue_create_buffer[DTYPE](max_trajs)
        self.block_min_cost = self.ctx.enqueue_create_buffer[DTYPE](max_blocks)
        self.block_min_idx = self.ctx.enqueue_create_buffer[DType.int32](max_blocks)
        self.final_min_cost = self.ctx.enqueue_create_buffer[DTYPE](1)
        self.final_min_idx = self.ctx.enqueue_create_buffer[DType.int32](1)


# ---------------------------------------------------------------------------
# Exported C entry points
# ---------------------------------------------------------------------------


@export
def mojo_cost_eval_create(
    max_trajs: Int32,
    points_per_traj: Int32,
    max_obstacles: Int32,
    cfg: UnsafePointer[MojoCostEvalConfig, MutExternalOrigin],
) -> UnsafePointer[CostEvalHandle, MutExternalOrigin]:
    try:
        var p = alloc[CostEvalHandle](1)
        p.init_pointee_move(
            CostEvalHandle(
                Int(max_trajs),
                Int(points_per_traj),
                Int(max_obstacles),
                cfg[0],
            )
        )
        return p
    except:
        return UnsafePointer[CostEvalHandle, MutExternalOrigin]()


@export
def mojo_cost_eval_destroy(handle: UnsafePointer[CostEvalHandle, MutExternalOrigin]):
    if handle:
        handle.destroy_pointee()
        handle.free()


@export
def mojo_cost_eval_run(
    handle: UnsafePointer[CostEvalHandle, MutExternalOrigin],
    host_paths_x: UnsafePointer[F32, MutExternalOrigin],
    host_paths_y: UnsafePointer[F32, MutExternalOrigin],
    host_vel_vx: UnsafePointer[F32, MutExternalOrigin],
    host_vel_vy: UnsafePointer[F32, MutExternalOrigin],
    host_vel_omega: UnsafePointer[F32, MutExternalOrigin],
    trajs_size: Int32,
    host_tracked_x: UnsafePointer[F32, MutExternalOrigin],
    host_tracked_y: UnsafePointer[F32, MutExternalOrigin],
    ref_path_size: Int32,
    goal_x: F32,
    goal_y: F32,
    goal_path_length: F32,
    host_obs_x: UnsafePointer[F32, MutExternalOrigin],
    host_obs_y: UnsafePointer[F32, MutExternalOrigin],
    obs_size: Int32,
    out_min_cost: UnsafePointer[F32, MutExternalOrigin],
    out_min_idx: UnsafePointer[Int32, MutExternalOrigin],
) -> Int32:
    if not handle:
        return Int32(-1)
    try:
        _run_impl(
            handle,
            host_paths_x, host_paths_y,
            host_vel_vx, host_vel_vy, host_vel_omega,
            Int(trajs_size),
            host_tracked_x, host_tracked_y,
            Int(ref_path_size),
            goal_x, goal_y, goal_path_length,
            host_obs_x, host_obs_y, Int(obs_size),
            out_min_cost, out_min_idx,
        )
    except e:
        print("mojo_cost_eval_run error:", e)
        return Int32(-2)
    return Int32(0)


def _run_impl(
    handle: UnsafePointer[CostEvalHandle, MutExternalOrigin],
    host_paths_x: UnsafePointer[F32, MutExternalOrigin],
    host_paths_y: UnsafePointer[F32, MutExternalOrigin],
    host_vel_vx: UnsafePointer[F32, MutExternalOrigin],
    host_vel_vy: UnsafePointer[F32, MutExternalOrigin],
    host_vel_omega: UnsafePointer[F32, MutExternalOrigin],
    trajs_size: Int,
    host_tracked_x: UnsafePointer[F32, MutExternalOrigin],
    host_tracked_y: UnsafePointer[F32, MutExternalOrigin],
    ref_path_size: Int,
    goal_x: F32,
    goal_y: F32,
    goal_path_length: F32,
    host_obs_x: UnsafePointer[F32, MutExternalOrigin],
    host_obs_y: UnsafePointer[F32, MutExternalOrigin],
    obs_size: Int,
    out_min_cost: UnsafePointer[F32, MutExternalOrigin],
    out_min_idx: UnsafePointer[Int32, MutExternalOrigin],
) raises:
    ref h = handle[]
    var path_size = h.points_per_traj
    var velocities_count = path_size - 1
    var points_n = trajs_size * path_size
    var vel_n = trajs_size * velocities_count

    # 1. Reset costs buffer.
    h.ctx.enqueue_memset(h.costs, F32(0.0))

    # 2. Host -> device copies.
    _memcpy_h2d(h.ctx, h.paths_x, host_paths_x, points_n)
    _memcpy_h2d(h.ctx, h.paths_y, host_paths_y, points_n)
    _memcpy_h2d(h.ctx, h.vel_vx, host_vel_vx, vel_n)
    _memcpy_h2d(h.ctx, h.vel_vy, host_vel_vy, vel_n)
    _memcpy_h2d(h.ctx, h.vel_omega, host_vel_omega, vel_n)
    _memcpy_h2d(h.ctx, h.tracked_x, host_tracked_x, ref_path_size)
    _memcpy_h2d(h.ctx, h.tracked_y, host_tracked_y, ref_path_size)
    if obs_size > 0:
        _memcpy_h2d(h.ctx, h.obstacles_x, host_obs_x, obs_size)
        _memcpy_h2d(h.ctx, h.obstacles_y, host_obs_y, obs_size)

    # 3. Raw device pointers for kernel args.
    var paths_x_ptr = h.paths_x.unsafe_ptr()
    var paths_y_ptr = h.paths_y.unsafe_ptr()
    var vel_vx_ptr = h.vel_vx.unsafe_ptr()
    var vel_vy_ptr = h.vel_vy.unsafe_ptr()
    var vel_omega_ptr = h.vel_omega.unsafe_ptr()
    var tracked_x_ptr = h.tracked_x.unsafe_ptr()
    var tracked_y_ptr = h.tracked_y.unsafe_ptr()
    var obs_x_ptr = h.obstacles_x.unsafe_ptr()
    var obs_y_ptr = h.obstacles_y.unsafe_ptr()
    var costs_ptr = h.costs.unsafe_ptr()

    var inv_lim_x = F32(1.0) / h.cfg.acc_lim_x if h.cfg.acc_lim_x > 0.0 else F32(0.0)
    var inv_lim_y = F32(1.0) / h.cfg.acc_lim_y if h.cfg.acc_lim_y > 0.0 else F32(0.0)
    var inv_lim_w = F32(1.0) / h.cfg.acc_lim_omega if h.cfg.acc_lim_omega > 0.0 else F32(0.0)

    # 4. Reference path cost.
    if h.cfg.ref_path_weight > 0.0:
        var inv_traj_size_count = F32(1.0) / F32(path_size) if path_size > 0 else F32(0.0)
        h.ctx.enqueue_function[ref_path_cost_kernel, ref_path_cost_kernel](
            paths_x_ptr, paths_y_ptr,
            tracked_x_ptr, tracked_y_ptr,
            costs_ptr,
            path_size, ref_path_size,
            inv_traj_size_count,
            h.cfg.ref_path_weight,
            grid_dim=trajs_size,
            block_dim=WG_SIZE,
        )

    # 5. Goal distance cost. Uses the explicit goal passed in — SYCL uses the
    # FULL reference path's end point, not the tracked segment's end point.
    if h.cfg.goal_weight > 0.0:
        var inv_goal_length = F32(1.0) / goal_path_length if goal_path_length > 0.0 else F32(0.0)
        var padded = ceildiv(trajs_size, WG_SIZE) * WG_SIZE
        h.ctx.enqueue_function[goal_cost_kernel, goal_cost_kernel](
            paths_x_ptr, paths_y_ptr, costs_ptr,
            trajs_size, path_size,
            goal_x, goal_y, inv_goal_length, h.cfg.goal_weight,
            grid_dim=padded // WG_SIZE,
            block_dim=WG_SIZE,
        )

    # 6. Smoothness.
    if h.cfg.smoothness_weight > 0.0:
        h.ctx.enqueue_function[smoothness_cost_kernel, smoothness_cost_kernel](
            vel_vx_ptr, vel_vy_ptr, vel_omega_ptr,
            costs_ptr, velocities_count,
            inv_lim_x, inv_lim_y, inv_lim_w, h.cfg.smoothness_weight,
            grid_dim=trajs_size,
            block_dim=WG_SIZE,
        )

    # 7. Jerk.
    if h.cfg.jerk_weight > 0.0:
        h.ctx.enqueue_function[jerk_cost_kernel, jerk_cost_kernel](
            vel_vx_ptr, vel_vy_ptr, vel_omega_ptr,
            costs_ptr, velocities_count,
            inv_lim_x, inv_lim_y, inv_lim_w, h.cfg.jerk_weight,
            grid_dim=trajs_size,
            block_dim=WG_SIZE,
        )

    # 8. Obstacles.
    if h.cfg.obstacles_weight > 0.0 and obs_size > 0:
        h.ctx.enqueue_function[obstacles_dist_cost_kernel, obstacles_dist_cost_kernel](
            paths_x_ptr, paths_y_ptr,
            obs_x_ptr, obs_y_ptr,
            costs_ptr, path_size, obs_size,
            h.cfg.max_obstacles_distance, h.cfg.obstacles_weight,
            grid_dim=trajs_size,
            block_dim=WG_SIZE,
        )

    # 9. Two-pass min reduction.
    var num_blocks = ceildiv(trajs_size, WG_SIZE)
    h.ctx.enqueue_function[min_cost_block_reduce, min_cost_block_reduce](
        costs_ptr,
        h.block_min_cost.unsafe_ptr(),
        h.block_min_idx.unsafe_ptr(),
        trajs_size,
        grid_dim=num_blocks,
        block_dim=WG_SIZE,
    )
    h.ctx.enqueue_function[min_cost_final_reduce, min_cost_final_reduce](
        h.block_min_cost.unsafe_ptr(),
        h.block_min_idx.unsafe_ptr(),
        h.final_min_cost.unsafe_ptr(),
        h.final_min_idx.unsafe_ptr(),
        num_blocks,
        grid_dim=1,
        block_dim=WG_SIZE,
    )

    # 10. Result device -> host.
    _memcpy_d2h_one_f32(h.ctx, out_min_cost, h.final_min_cost)
    _memcpy_d2h_one_i32(h.ctx, out_min_idx, h.final_min_idx)
    h.ctx.synchronize()


# ---------------------------------------------------------------------------
# Memcpy helpers. Wrap raw host pointers in non-owning DeviceBuffers so
# enqueue_copy can accept them uniformly.
# ---------------------------------------------------------------------------


def _memcpy_h2d(
    ctx: DeviceContext,
    dst: DeviceBuffer[DTYPE],
    host_src: UnsafePointer[F32, MutExternalOrigin],
    count: Int,
) raises:
    ctx.enqueue_copy(dst_buf=dst, src_ptr=host_src)


def _memcpy_d2h_one_f32(
    ctx: DeviceContext,
    host_dst: UnsafePointer[F32, MutExternalOrigin],
    src: DeviceBuffer[DTYPE],
) raises:
    ctx.enqueue_copy(dst_ptr=host_dst, src_buf=src)


def _memcpy_d2h_one_i32(
    ctx: DeviceContext,
    host_dst: UnsafePointer[Int32, MutExternalOrigin],
    src: DeviceBuffer[DType.int32],
) raises:
    ctx.enqueue_copy(dst_ptr=host_dst, src_buf=src)


# ===========================================================================
# Local mapper FFI
#
# Exposes create/run/destroy over a LocalMapperGPU interface.
# See integration/kompass_mojo.h for the C contract.
# ===========================================================================


struct MojoLocalMapperConfig(TrivialRegisterPassable):
    var resolution: F32
    var laserscan_orientation: F32
    var laserscan_pos_x: F32
    var laserscan_pos_y: F32
    var laserscan_pos_z: F32   # unused by kernel, kept for parity with SYCL


struct MapperHandle(Movable):
    var ctx: DeviceContext
    var grid: DeviceBuffer[DType.int32]
    var distances: DeviceBuffer[DType.float32]
    var ranges: DeviceBuffer[DType.float64]
    var angles: DeviceBuffer[DType.float64]

    var rows: Int
    var cols: Int
    var scan_size: Int
    var max_points_per_line: Int
    var cfg: MojoLocalMapperConfig

    # Geometry precomputed once
    var central_x: Int32
    var central_y: Int32
    var start_x: Int32
    var start_y: Int32

    def __init__(
        out self,
        rows: Int,
        cols: Int,
        scan_size: Int,
        max_points_per_line: Int,
        cfg: MojoLocalMapperConfig,
    ) raises:
        self.ctx = DeviceContext()
        self.rows = rows
        self.cols = cols
        self.scan_size = scan_size
        self.max_points_per_line = max_points_per_line
        self.cfg = cfg

        # Match std::round(int/2) - 1.
        # For int dividends this is just integer division minus one.
        self.central_x = Int32(rows // 2 - 1)
        self.central_y = Int32(cols // 2 - 1)
        self.start_x = self.central_x + Int32(
            Int(cfg.laserscan_pos_x / cfg.resolution)
        )
        self.start_y = self.central_y + Int32(
            Int(cfg.laserscan_pos_y / cfg.resolution)
        )

        var cells = rows * cols
        self.grid = self.ctx.enqueue_create_buffer[DType.int32](cells)
        self.distances = self.ctx.enqueue_create_buffer[DType.float32](cells)
        self.ranges = self.ctx.enqueue_create_buffer[DType.float64](scan_size)
        self.angles = self.ctx.enqueue_create_buffer[DType.float64](scan_size)

        # Precompute per-cell distance from laserscan origin
        var host_dist = self.ctx.enqueue_create_host_buffer[DType.float32](cells)
        var dst = host_dist.unsafe_ptr().value()
        for j in range(cols):
            for i in range(rows):
                var dest_x = F32(self.central_x - Int32(i)) * cfg.resolution
                var dest_y = F32(self.central_y - Int32(j)) * cfg.resolution
                var diff_x = dest_x - cfg.laserscan_pos_x
                var diff_y = dest_y - cfg.laserscan_pos_y
                var diff_z = F32(0.0) - cfg.laserscan_pos_z
                var d2 = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z
                (dst + (i + j * rows))[] = sqrt(d2)
        self.ctx.enqueue_copy(dst_buf=self.distances, src_buf=host_dist)
        self.ctx.synchronize()


@export
def mojo_local_mapper_create(
    rows: Int32,
    cols: Int32,
    scan_size: Int32,
    max_points_per_line: Int32,
    cfg: UnsafePointer[MojoLocalMapperConfig, MutExternalOrigin],
) -> UnsafePointer[MapperHandle, MutExternalOrigin]:
    try:
        var p = alloc[MapperHandle](1)
        p.init_pointee_move(
            MapperHandle(
                Int(rows),
                Int(cols),
                Int(scan_size),
                Int(max_points_per_line),
                cfg[0],
            )
        )
        return p
    except:
        return UnsafePointer[MapperHandle, MutExternalOrigin]()


@export
def mojo_local_mapper_destroy(
    handle: UnsafePointer[MapperHandle, MutExternalOrigin],
):
    if handle:
        handle.destroy_pointee()
        handle.free()


@export
def mojo_local_mapper_run(
    handle: UnsafePointer[MapperHandle, MutExternalOrigin],
    host_angles: UnsafePointer[Float64, MutExternalOrigin],
    host_ranges: UnsafePointer[Float64, MutExternalOrigin],
    host_grid_out: UnsafePointer[Int32, MutExternalOrigin],
) -> Int32:
    if not handle:
        return Int32(-1)
    try:
        _mapper_run_impl(handle, host_angles, host_ranges, host_grid_out)
    except e:
        print("mojo_local_mapper_run error:", e)
        return Int32(-2)
    return Int32(0)


def _mapper_run_impl(
    handle: UnsafePointer[MapperHandle, MutExternalOrigin],
    host_angles: UnsafePointer[Float64, MutExternalOrigin],
    host_ranges: UnsafePointer[Float64, MutExternalOrigin],
    host_grid_out: UnsafePointer[Int32, MutExternalOrigin],
) raises:
    ref h = handle[]

    # 1. Reset grid to UNEXPLORED (matches m_q.fill in scanToGrid).
    h.ctx.enqueue_memset(h.grid, OCC_UNEXPLORED)

    # 2. H->D: angles and ranges.
    h.ctx.enqueue_copy(dst_buf=h.angles, src_ptr=host_angles)
    h.ctx.enqueue_copy(dst_buf=h.ranges, src_ptr=host_ranges)

    # 3. Launch.
    h.ctx.enqueue_function[scan_to_grid_kernel, scan_to_grid_kernel](
        h.ranges.unsafe_ptr(),
        h.angles.unsafe_ptr(),
        h.grid.unsafe_ptr(),
        h.distances.unsafe_ptr(),
        h.rows, h.cols,
        h.cfg.resolution,
        h.cfg.laserscan_orientation,
        h.cfg.laserscan_pos_x,
        h.cfg.laserscan_pos_y,
        h.central_x, h.central_y,
        h.start_x, h.start_y,
        grid_dim=h.scan_size,
        block_dim=h.max_points_per_line,
    )

    # 4. D->H grid.
    h.ctx.enqueue_copy(dst_ptr=host_grid_out, src_buf=h.grid)
    h.ctx.synchronize()


# ===========================================================================
# Critical-zone checker FFI
#
# Third kernel group. Exposes create/destroy + two run methods (laserscan
# and pointcloud) over a CriticalZoneCheckerGPU-like interface. See
# integration/kompass_mojo.h for the C contract.
# ===========================================================================


struct MojoCritZoneConfig(TrivialRegisterPassable):
    # 2D submatrix of the sensor→body Isometry3f. The kernels drop z so
    # only six scalars are needed.
    var tf00: F32
    var tf01: F32
    var tf03: F32
    var tf10: F32
    var tf11: F32
    var tf13: F32
    # Safety geometry.
    var robot_radius: F32
    var critical_angle: F32     # half-angle in radians (must be <= pi/2)
    var critical_distance: F32
    var slowdown_distance: F32
    # Z-filter bounds for the pointcloud path.
    var min_height: F32
    var max_height: F32


struct CritZoneHandle(Movable):
    var ctx: DeviceContext
    # Laserscan buffers (allocated when scan_size > 0, otherwise 1-element
    # placeholders).
    var ranges: DeviceBuffer[DTYPE]
    var cos_angles: DeviceBuffer[DTYPE]
    var sin_angles: DeviceBuffer[DTYPE]
    var fwd_indices: DeviceBuffer[DType.int32]
    var bwd_indices: DeviceBuffer[DType.int32]
    # Pointcloud buffer, grown on demand in run_pointcloud.
    var raw_bytes: DeviceBuffer[DType.int8]
    var raw_capacity: Int
    # Shared output.
    var out_factor: DeviceBuffer[DTYPE]

    var scan_size: Int
    var n_fwd: Int
    var n_bwd: Int
    var cfg: MojoCritZoneConfig

    # Derived kernel-friendly constants.
    var cos_sq_crit_angle: F32
    var dist_denom: F32
    var safe_threshold_sq: F32
    var inv_dist_range: F32

    def __init__(
        out self,
        scan_angles: UnsafePointer[Float64, MutExternalOrigin],
        scan_size: Int,
        max_cloud_bytes: Int,
        cfg: MojoCritZoneConfig,
    ) raises:
        self.ctx = DeviceContext()
        self.cfg = cfg
        self.scan_size = scan_size
        self.n_fwd = 0
        self.n_bwd = 0
        self.raw_capacity = 0

        # Derived constants computed once on host.
        var c = cos(cfg.critical_angle)
        self.cos_sq_crit_angle = F32(c * c)
        self.dist_denom = cfg.slowdown_distance - cfg.critical_distance
        var safe = cfg.slowdown_distance + cfg.robot_radius
        self.safe_threshold_sq = safe * safe
        self.inv_dist_range = F32(1.0) / self.dist_denom

        self.out_factor = self.ctx.enqueue_create_buffer[DTYPE](1)

        # Laserscan-side buffers, only meaningful when scan_size > 0.
        var scan_alloc = scan_size if scan_size > 0 else 1
        self.ranges = self.ctx.enqueue_create_buffer[DTYPE](scan_alloc)
        self.cos_angles = self.ctx.enqueue_create_buffer[DTYPE](scan_alloc)
        self.sin_angles = self.ctx.enqueue_create_buffer[DTYPE](scan_alloc)

        if scan_size > 0:
            # Compute presets, cos_i, sin_i host-side. Classify each angle into
            # fwd/bwd cones using the same test as the kernel, applied against
            # the body-frame point (tf00*c + tf01*s + tf03, tf10*c + tf11*s + tf13).
            var host_cos = self.ctx.enqueue_create_host_buffer[DTYPE](scan_size)
            var host_sin = self.ctx.enqueue_create_host_buffer[DTYPE](scan_size)
            var cos_ptr = host_cos.unsafe_ptr().value()
            var sin_ptr = host_sin.unsafe_ptr().value()
            var host_fwd = List[Int32]()
            var host_bwd = List[Int32]()

            for i in range(scan_size):
                var theta = (scan_angles + i)[]
                var c_t = F32(cos(theta))
                var s_t = F32(sin(theta))
                (cos_ptr + i)[] = c_t
                (sin_ptr + i)[] = s_t

                # Body-frame transform of direction (c, s, 0).
                var x_body = cfg.tf00 * c_t + cfg.tf01 * s_t + cfg.tf03
                var y_body = cfg.tf10 * c_t + cfg.tf11 * s_t + cfg.tf13
                var dist_sq = x_body * x_body + y_body * y_body
                var x_sq = x_body * x_body
                var cone_sq = self.cos_sq_crit_angle * dist_sq
                if x_body > F32(0.0) and x_sq >= cone_sq:
                    host_fwd.append(Int32(i))
                if x_body < F32(0.0) and x_sq >= cone_sq:
                    host_bwd.append(Int32(i))

            self.ctx.enqueue_copy(dst_buf=self.cos_angles, src_buf=host_cos)
            self.ctx.enqueue_copy(dst_buf=self.sin_angles, src_buf=host_sin)

            self.n_fwd = len(host_fwd)
            self.n_bwd = len(host_bwd)

            var fwd_alloc = self.n_fwd if self.n_fwd > 0 else 1
            var bwd_alloc = self.n_bwd if self.n_bwd > 0 else 1
            self.fwd_indices = self.ctx.enqueue_create_buffer[DType.int32](fwd_alloc)
            self.bwd_indices = self.ctx.enqueue_create_buffer[DType.int32](bwd_alloc)

            if self.n_fwd > 0:
                var host_fwd_buf = self.ctx.enqueue_create_host_buffer[
                    DType.int32
                ](self.n_fwd)
                var fwd_ptr = host_fwd_buf.unsafe_ptr().value()
                for i in range(self.n_fwd):
                    (fwd_ptr + i)[] = host_fwd[i]
                self.ctx.enqueue_copy(
                    dst_buf=self.fwd_indices, src_buf=host_fwd_buf,
                )
            if self.n_bwd > 0:
                var host_bwd_buf = self.ctx.enqueue_create_host_buffer[
                    DType.int32
                ](self.n_bwd)
                var bwd_ptr = host_bwd_buf.unsafe_ptr().value()
                for i in range(self.n_bwd):
                    (bwd_ptr + i)[] = host_bwd[i]
                self.ctx.enqueue_copy(
                    dst_buf=self.bwd_indices, src_buf=host_bwd_buf,
                )
        else:
            # Pointcloud-only usage: allocate 1-element placeholders.
            self.fwd_indices = self.ctx.enqueue_create_buffer[DType.int32](1)
            self.bwd_indices = self.ctx.enqueue_create_buffer[DType.int32](1)

        # Pointcloud raw-bytes buffer. Grown on demand
        if max_cloud_bytes > 0:
            self.raw_bytes = self.ctx.enqueue_create_buffer[DType.int8](
                max_cloud_bytes,
            )
            self.raw_capacity = max_cloud_bytes
        else:
            self.raw_bytes = self.ctx.enqueue_create_buffer[DType.int8](1)
            self.raw_capacity = 0

        self.ctx.synchronize()


@export
def mojo_crit_zone_create(
    scan_angles: UnsafePointer[Float64, MutExternalOrigin],
    scan_size: Int32,
    max_cloud_bytes: Int32,
    cfg: UnsafePointer[MojoCritZoneConfig, MutExternalOrigin],
) -> UnsafePointer[CritZoneHandle, MutExternalOrigin]:
    try:
        var p = alloc[CritZoneHandle](1)
        p.init_pointee_move(
            CritZoneHandle(
                scan_angles, Int(scan_size), Int(max_cloud_bytes), cfg[0],
            )
        )
        return p
    except:
        return UnsafePointer[CritZoneHandle, MutExternalOrigin]()


@export
def mojo_crit_zone_destroy(
    handle: UnsafePointer[CritZoneHandle, MutExternalOrigin],
):
    if handle:
        handle.destroy_pointee()
        handle.free()


@export
def mojo_crit_zone_run_laserscan(
    handle: UnsafePointer[CritZoneHandle, MutExternalOrigin],
    host_ranges: UnsafePointer[F32, MutExternalOrigin],
    forward: Int32,
    out_factor: UnsafePointer[F32, MutExternalOrigin],
) -> Int32:
    if not handle:
        return Int32(-1)
    try:
        _cz_run_laserscan_impl(handle, host_ranges, forward, out_factor)
    except e:
        print("mojo_crit_zone_run_laserscan error:", e)
        return Int32(-2)
    return Int32(0)


def _cz_run_laserscan_impl(
    handle: UnsafePointer[CritZoneHandle, MutExternalOrigin],
    host_ranges: UnsafePointer[F32, MutExternalOrigin],
    forward: Int32,
    out_factor: UnsafePointer[F32, MutExternalOrigin],
) raises:
    ref h = handle[]
    if h.scan_size <= 0:
        raise Error("crit_zone run_laserscan: handle has no scan data")

    # Reset out to min-identity.
    h.out_factor.enqueue_fill(F32(1.0))

    # Upload ranges (caller already converted to float).
    h.ctx.enqueue_copy(dst_buf=h.ranges, src_ptr=host_ranges)

    var use_forward = forward != Int32(0)
    var n_items = h.n_fwd if use_forward else h.n_bwd
    if n_items <= 0:
        # No scan bins fall inside the requested cone; result stays 1.0.
        h.ctx.enqueue_copy(dst_ptr=out_factor, src_buf=h.out_factor)
        h.ctx.synchronize()
        return

    var idx_ptr = (
        h.fwd_indices.unsafe_ptr() if use_forward else h.bwd_indices.unsafe_ptr()
    )

    h.ctx.enqueue_function[
        critical_zone_laserscan_kernel, critical_zone_laserscan_kernel,
    ](
        h.ranges.unsafe_ptr(),
        h.cos_angles.unsafe_ptr(),
        h.sin_angles.unsafe_ptr(),
        idx_ptr, n_items,
        h.cfg.tf00, h.cfg.tf01, h.cfg.tf03,
        h.cfg.tf10, h.cfg.tf11, h.cfg.tf13,
        h.cfg.robot_radius, h.cfg.critical_distance,
        h.dist_denom, h.safe_threshold_sq,
        h.out_factor.unsafe_ptr(),
        grid_dim=1, block_dim=WG_SIZE,
    )

    h.ctx.enqueue_copy(dst_ptr=out_factor, src_buf=h.out_factor)
    h.ctx.synchronize()


@export
def mojo_crit_zone_run_pointcloud(
    handle: UnsafePointer[CritZoneHandle, MutExternalOrigin],
    host_bytes: UnsafePointer[Int8, MutExternalOrigin],
    total_bytes: Int32,
    point_step: Int32,
    row_step: Int32,
    height: Int32,
    width: Int32,
    x_offset: Int32,
    y_offset: Int32,
    z_offset: Int32,
    forward: Int32,
    out_factor: UnsafePointer[F32, MutExternalOrigin],
) -> Int32:
    if not handle:
        return Int32(-1)
    try:
        _cz_run_pointcloud_impl(
            handle, host_bytes, total_bytes,
            point_step, row_step, height, width,
            x_offset, y_offset, z_offset,
            forward, out_factor,
        )
    except e:
        print("mojo_crit_zone_run_pointcloud error:", e)
        return Int32(-2)
    return Int32(0)


def _cz_run_pointcloud_impl(
    handle: UnsafePointer[CritZoneHandle, MutExternalOrigin],
    host_bytes: UnsafePointer[Int8, MutExternalOrigin],
    total_bytes: Int32,
    point_step: Int32,
    row_step: Int32,
    height: Int32,
    width: Int32,
    x_offset: Int32,
    y_offset: Int32,
    z_offset: Int32,
    forward: Int32,
    out_factor: UnsafePointer[F32, MutExternalOrigin],
) raises:
    ref h = handle[]
    var tb = Int(total_bytes)
    var h_i = Int(height)
    var w_i = Int(width)
    var num_points = h_i * w_i

    h.out_factor.enqueue_fill(F32(1.0))

    if tb == 0 or num_points == 0:
        # Empty cloud
        h.ctx.enqueue_copy(dst_ptr=out_factor, src_buf=h.out_factor)
        h.ctx.synchronize()
        return

    # Grow raw-bytes buffer if the incoming scan is larger than current capacity
    if h.raw_capacity < tb:
        h.raw_bytes = h.ctx.enqueue_create_buffer[DType.int8](tb)
        h.raw_capacity = tb

    h.ctx.enqueue_copy(dst_buf=h.raw_bytes, src_ptr=host_bytes)

    var is_contiguous = Int32(1) if Int(row_step) == w_i * Int(point_step) else Int32(0)
    var num_blocks = ceildiv(num_points, WG_SIZE)

    h.ctx.enqueue_function[
        critical_zone_pointcloud_kernel, critical_zone_pointcloud_kernel,
    ](
        h.raw_bytes.unsafe_ptr(), tb, num_points,
        Int(point_step), Int(row_step), w_i,
        is_contiguous,
        Int(x_offset), Int(y_offset), Int(z_offset),
        h.cfg.min_height, h.cfg.max_height,
        h.cfg.tf00, h.cfg.tf01, h.cfg.tf03,
        h.cfg.tf10, h.cfg.tf11, h.cfg.tf13,
        h.cos_sq_crit_angle,
        h.cfg.robot_radius, h.cfg.critical_distance,
        h.inv_dist_range, h.safe_threshold_sq,
        forward,
        h.out_factor.unsafe_ptr(),
        grid_dim=num_blocks, block_dim=WG_SIZE,
    )

    h.ctx.enqueue_copy(dst_ptr=out_factor, src_buf=h.out_factor)
    h.ctx.synchronize()
