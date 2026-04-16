# C FFI entry points for the Mojo cost evaluator kernels.
#
# Exposes create/run/destroy to C via @export. The C contract is declared
# in integration/kompass_mojo.h — keep that file in sync with the @export
# signatures below.

from std.math import ceildiv
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
    tracked_segment_length: F32,
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
            Int(ref_path_size), tracked_segment_length,
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
    tracked_segment_length: F32,
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
        var inv_ref_length = F32(1.0) / tracked_segment_length if tracked_segment_length > 0.0 else F32(0.0)
        var inv_ref_size_count = F32(1.0) / F32(ref_path_size) if ref_path_size > 0 else F32(0.0)
        h.ctx.enqueue_function[ref_path_cost_kernel, ref_path_cost_kernel](
            paths_x_ptr, paths_y_ptr,
            tracked_x_ptr, tracked_y_ptr,
            costs_ptr,
            path_size, ref_path_size,
            inv_ref_length, inv_ref_size_count,
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
