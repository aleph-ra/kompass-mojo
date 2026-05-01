# Unit tests for trajectory cost evaluator GPU kernels.
#
# Each test allocates small synthetic inputs on the GPU, launches one kernel,
# reads the result back, and compares against a hand-computed expected value.
# Tests run on the GPU — a Mojo-supported accelerator is required.

from std.testing import assert_true, assert_almost_equal
from std.math import ceildiv, sqrt
from std.memory import UnsafePointer
from std.gpu.host import DeviceContext, DeviceBuffer

from kompass_mojo.cost_evaluator import (
    DTYPE, F32, WG_SIZE, LOG2_WG,
    goal_cost_kernel,
    smoothness_cost_kernel,
    jerk_cost_kernel,
    ref_path_cost_kernel,
    obstacles_dist_cost_kernel,
    min_cost_block_reduce,
    min_cost_final_reduce,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fill_buffer(
    ctx: DeviceContext,
    buf: DeviceBuffer[DTYPE],
    values: List[Float32],
) raises:
    """Fill a device buffer from a list of floats."""
    var host = ctx.enqueue_create_host_buffer[DTYPE](len(values))
    var ptr = host.unsafe_ptr()
    for i in range(len(values)):
        (ptr + i)[] = values[i]
    ctx.enqueue_copy(dst_buf=buf, src_buf=host)
    ctx.synchronize()


def _read_buffer(
    ctx: DeviceContext,
    buf: DeviceBuffer[DTYPE],
    n: Int,
) raises -> List[Float32]:
    """Read n floats from a device buffer."""
    var result = List[Float32]()
    with buf.map_to_host() as mapped:
        var ptr = mapped.unsafe_ptr()
        for i in range(n):
            result.append(Float32((ptr + i)[]))
    return result^


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_goal_cost_kernel(ctx: DeviceContext) raises:
    """Distance-along-path goal cost.
    Tracked segment is a 5-point straight line at x=[0,1,2,3,4], y=0; the
    full ref path extends to x=10, length 10.
    Traj 0 endpoint (4, 0): closest seg point = idx 4, acc = 4.
        arc_remaining_normalized = (10 - 4) / 10 = 0.6
        tie_breaker = 0 / 10 = 0
        cost = 1.0 * (0.6 + 0) = 0.6
    Traj 1 endpoint (1, 0): closest = idx 1, acc = 1.
        arc_remaining_normalized = 0.9, tie_breaker = 0. cost = 0.9.
    """
    comptime TRAJS = 2
    comptime PTS = 4
    comptime SEG = 5
    comptime N = TRAJS * PTS

    var paths_x = ctx.enqueue_create_buffer[DTYPE](N)
    var paths_y = ctx.enqueue_create_buffer[DTYPE](N)
    var tracked_x = ctx.enqueue_create_buffer[DTYPE](SEG)
    var tracked_y = ctx.enqueue_create_buffer[DTYPE](SEG)
    var tracked_acc = ctx.enqueue_create_buffer[DTYPE](SEG)
    var costs = ctx.enqueue_create_buffer[DTYPE](TRAJS)

    # Traj 0 endpoint = (4, 0)  Traj 1 endpoint = (1, 0)
    _fill_buffer(ctx, paths_x, [0.0, 1.0, 2.0, 4.0, 0.0, 0.5, 0.8, 1.0])
    _fill_buffer(ctx, paths_y, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    _fill_buffer(ctx, tracked_x, [0.0, 1.0, 2.0, 3.0, 4.0])
    _fill_buffer(ctx, tracked_y, [0.0, 0.0, 0.0, 0.0, 0.0])
    _fill_buffer(ctx, tracked_acc, [0.0, 1.0, 2.0, 3.0, 4.0])
    costs.enqueue_fill(F32(0.0))
    ctx.synchronize()

    var ref_path_length = Float32(10.0)
    ctx.enqueue_function[goal_cost_kernel, goal_cost_kernel](
        paths_x.unsafe_ptr(), paths_y.unsafe_ptr(),
        tracked_x.unsafe_ptr(), tracked_y.unsafe_ptr(), tracked_acc.unsafe_ptr(),
        costs.unsafe_ptr(),
        PTS, SEG,
        F32(ref_path_length), F32(1.0 / ref_path_length),
        F32(1.0),
        grid_dim=TRAJS,
        block_dim=WG_SIZE,
    )
    ctx.synchronize()

    var result = _read_buffer(ctx, costs, TRAJS)
    assert_almost_equal(result[0], 0.6, atol=1e-5, msg="goal: traj 0")
    assert_almost_equal(result[1], 0.9, atol=1e-5, msg="goal: traj 1")
    print("  PASS: goal_cost_kernel (traj0=", result[0], "traj1=", result[1], ")")


def test_goal_cost_tie_breaker(ctx: DeviceContext) raises:
    """Two trajectories with the same closest-segment index but different
    lateral offsets: arc_remaining_normalized is identical; only the
    tie-breaker (sqrt(min_dist_sq) / ref_path_length) differs.
    Both endpoints at x=4 -> closest acc = 4, arc_remaining_normalized = 0.6.
    A: y=0.1 -> tie_breaker = 0.1 / 10 = 0.01 -> cost = 0.61.
    B: y=0.5 -> tie_breaker = 0.5 / 10 = 0.05 -> cost = 0.65.
    """
    comptime TRAJS = 2
    comptime PTS = 4
    comptime SEG = 5

    var paths_x = ctx.enqueue_create_buffer[DTYPE](TRAJS * PTS)
    var paths_y = ctx.enqueue_create_buffer[DTYPE](TRAJS * PTS)
    var tracked_x = ctx.enqueue_create_buffer[DTYPE](SEG)
    var tracked_y = ctx.enqueue_create_buffer[DTYPE](SEG)
    var tracked_acc = ctx.enqueue_create_buffer[DTYPE](SEG)
    var costs = ctx.enqueue_create_buffer[DTYPE](TRAJS)

    _fill_buffer(ctx, paths_x, [0.0, 1.0, 2.0, 4.0, 0.0, 1.0, 2.0, 4.0])
    _fill_buffer(ctx, paths_y, [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.5])
    _fill_buffer(ctx, tracked_x, [0.0, 1.0, 2.0, 3.0, 4.0])
    _fill_buffer(ctx, tracked_y, [0.0, 0.0, 0.0, 0.0, 0.0])
    _fill_buffer(ctx, tracked_acc, [0.0, 1.0, 2.0, 3.0, 4.0])
    costs.enqueue_fill(F32(0.0))
    ctx.synchronize()

    var ref_path_length = Float32(10.0)
    ctx.enqueue_function[goal_cost_kernel, goal_cost_kernel](
        paths_x.unsafe_ptr(), paths_y.unsafe_ptr(),
        tracked_x.unsafe_ptr(), tracked_y.unsafe_ptr(), tracked_acc.unsafe_ptr(),
        costs.unsafe_ptr(),
        PTS, SEG,
        F32(ref_path_length), F32(1.0 / ref_path_length),
        F32(1.0),
        grid_dim=TRAJS,
        block_dim=WG_SIZE,
    )
    ctx.synchronize()

    var result = _read_buffer(ctx, costs, TRAJS)
    assert_almost_equal(result[0], 0.61, atol=1e-5, msg="goal tie: A")
    assert_almost_equal(result[1], 0.65, atol=1e-5, msg="goal tie: B")
    assert_true(result[0] < result[1], "tie: A should be < B")
    print("  PASS: goal_cost_tie_breaker (A=", result[0], "B=", result[1], ")")


def test_smoothness_cost_kernel(ctx: DeviceContext) raises:
    """One trajectory, 6 velocity points (velocities_count = 6).
    vx = [1, 2, 4, 7, 11, 16], vy = omega = 0.
    Deltas (i>=1): [1, 2, 3, 4, 5]. Squared: [1, 4, 9, 16, 25]. Sum = 55.
    With inv_lim_x = 1.0: total = 55.
    Normalized: weight * total / (3 * velocities_count) = 1.0 * 55 / 18 ≈ 3.0556
    """
    comptime TRAJS = 1
    comptime VEL_COUNT = 6
    comptime N = TRAJS * VEL_COUNT

    var vel_vx = ctx.enqueue_create_buffer[DTYPE](N)
    var vel_vy = ctx.enqueue_create_buffer[DTYPE](N)
    var vel_omega = ctx.enqueue_create_buffer[DTYPE](N)
    var costs = ctx.enqueue_create_buffer[DTYPE](TRAJS)

    _fill_buffer(ctx, vel_vx, [1.0, 2.0, 4.0, 7.0, 11.0, 16.0])
    vel_vy.enqueue_fill(F32(0.0))
    vel_omega.enqueue_fill(F32(0.0))
    costs.enqueue_fill(F32(0.0))
    ctx.synchronize()

    ctx.enqueue_function[smoothness_cost_kernel, smoothness_cost_kernel](
        vel_vx.unsafe_ptr(), vel_vy.unsafe_ptr(), vel_omega.unsafe_ptr(),
        costs.unsafe_ptr(), VEL_COUNT,
        F32(1.0), F32(1.0), F32(1.0),  # inv_lim_x/y/omega
        F32(1.0),                        # cost_weight
        grid_dim=TRAJS,
        block_dim=WG_SIZE,
    )
    ctx.synchronize()

    var result = _read_buffer(ctx, costs, TRAJS)
    var expected = Float32(55.0 / 18.0)  # ≈ 3.0556
    assert_almost_equal(result[0], expected, atol=1e-4, msg="smoothness: traj 0")
    print("  PASS: smoothness_cost_kernel")


def test_jerk_cost_kernel(ctx: DeviceContext) raises:
    """One trajectory, 6 velocity points.
    vx = [1, 2, 4, 7, 11, 16], vy = omega = 0.
    Second derivatives (i>=2): v[i] - 2*v[i-1] + v[i-2]
      i=2: 4 - 4 + 1  = 1
      i=3: 7 - 8 + 4  = 3  wait that's wrong
      i=3: 7 - 2*4 + 2  = 7 - 8 + 2 = 1
      i=4: 11 - 2*7 + 4 = 11 - 14 + 4 = 1
      i=5: 16 - 2*11 + 7 = 16 - 22 + 7 = 1
    All second derivatives = 1. Squared sum = 4.
    Normalized: 1.0 * 4 / (3 * 6) = 4/18 ≈ 0.2222
    """
    comptime TRAJS = 1
    comptime VEL_COUNT = 6
    comptime N = TRAJS * VEL_COUNT

    var vel_vx = ctx.enqueue_create_buffer[DTYPE](N)
    var vel_vy = ctx.enqueue_create_buffer[DTYPE](N)
    var vel_omega = ctx.enqueue_create_buffer[DTYPE](N)
    var costs = ctx.enqueue_create_buffer[DTYPE](TRAJS)

    _fill_buffer(ctx, vel_vx, [1.0, 2.0, 4.0, 7.0, 11.0, 16.0])
    vel_vy.enqueue_fill(F32(0.0))
    vel_omega.enqueue_fill(F32(0.0))
    costs.enqueue_fill(F32(0.0))
    ctx.synchronize()

    ctx.enqueue_function[jerk_cost_kernel, jerk_cost_kernel](
        vel_vx.unsafe_ptr(), vel_vy.unsafe_ptr(), vel_omega.unsafe_ptr(),
        costs.unsafe_ptr(), VEL_COUNT,
        F32(1.0), F32(1.0), F32(1.0),
        F32(1.0),
        grid_dim=TRAJS,
        block_dim=WG_SIZE,
    )
    ctx.synchronize()

    var result = _read_buffer(ctx, costs, TRAJS)
    var expected = Float32(4.0 / 18.0)  # ≈ 0.2222
    assert_almost_equal(result[0], expected, atol=1e-4, msg="jerk: traj 0")
    print("  PASS: jerk_cost_kernel")


def test_ref_path_cost_kernel(ctx: DeviceContext) raises:
    """Two trajectories, 4 points. Ref path = 3 points along x = [0, 1, 2].
    Traj 0: x=[0, 1, 2, 3], y=0 (follows ref closely).
    Traj 1: x=[0, 1, 2, 3], y=1 (offset by 1 in y).
    Cost = weight * (avg_cross_track + end_dist/segment_length) * 0.5.
    Traj 0: cross-track dists = [0, 0, 0, 1]; avg = 0.25.
            end_dist = dist((3,0),(2,0)) = 1.0; normalized = 1.0/2 = 0.5.
            final = (0.25 + 0.5) * 0.5 = 0.375.
    Traj 1: cross-track dists = [1, 1, 1, sqrt(2)]; avg ~ 1.1036.
            end_dist = dist((3,1),(2,0)) = sqrt(2); normalized ~ 0.7071.
            final = (1.1036 + 0.7071) * 0.5 ~ 0.9054.
    """
    comptime TRAJS = 2
    comptime PTS = 4
    comptime REF = 3
    comptime N = TRAJS * PTS

    var paths_x = ctx.enqueue_create_buffer[DTYPE](N)
    var paths_y = ctx.enqueue_create_buffer[DTYPE](N)
    var tracked_x = ctx.enqueue_create_buffer[DTYPE](REF)
    var tracked_y = ctx.enqueue_create_buffer[DTYPE](REF)
    var costs = ctx.enqueue_create_buffer[DTYPE](TRAJS)

    # Traj 0: (0,0),(1,0),(2,0),(3,0)  Traj 1: (0,1),(1,1),(2,1),(3,1)
    _fill_buffer(ctx, paths_x, [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0])
    _fill_buffer(ctx, paths_y, [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    _fill_buffer(ctx, tracked_x, [0.0, 1.0, 2.0])
    _fill_buffer(ctx, tracked_y, [0.0, 0.0, 0.0])
    costs.enqueue_fill(F32(0.0))
    ctx.synchronize()

    var segment_length = Float32(2.0)  # (0,0)->(1,0)->(2,0)
    ctx.enqueue_function[ref_path_cost_kernel, ref_path_cost_kernel](
        paths_x.unsafe_ptr(), paths_y.unsafe_ptr(),
        tracked_x.unsafe_ptr(), tracked_y.unsafe_ptr(),
        costs.unsafe_ptr(),
        PTS, REF,
        F32(1.0 / Float32(PTS)),  # inv_traj_size_count
        F32(1.0 / segment_length),  # inv_segment_length
        F32(1.0),  # cost_weight
        grid_dim=TRAJS,
        block_dim=WG_SIZE,
    )
    ctx.synchronize()

    var result = _read_buffer(ctx, costs, TRAJS)
    assert_almost_equal(result[0], 0.375, atol=0.01, msg="ref_path: traj 0")
    assert_almost_equal(result[1], 0.9054, atol=0.02, msg="ref_path: traj 1")
    assert_true(result[0] < result[1], "ref_path: traj 0 should be < traj 1")
    print("  PASS: ref_path_cost_kernel (traj0=", result[0], "traj1=", result[1], ")")


def test_obstacles_dist_cost_kernel(ctx: DeviceContext) raises:
    """One trajectory, 4 points along x=[0, 1, 2, 3], y=0.
    One obstacle at (1.5, 0). Min distance across all traj points = 0.5.
    max_obstacle_distance = 5.0.
    normalized_cost = (5.0 - 0.5) / 5.0 = 0.9.
    cost = weight * 0.9 = 0.9
    """
    comptime TRAJS = 1
    comptime PTS = 4
    comptime OBS = 1

    var paths_x = ctx.enqueue_create_buffer[DTYPE](PTS)
    var paths_y = ctx.enqueue_create_buffer[DTYPE](PTS)
    var obs_x = ctx.enqueue_create_buffer[DTYPE](OBS)
    var obs_y = ctx.enqueue_create_buffer[DTYPE](OBS)
    var costs = ctx.enqueue_create_buffer[DTYPE](TRAJS)

    _fill_buffer(ctx, paths_x, [0.0, 1.0, 2.0, 3.0])
    _fill_buffer(ctx, paths_y, [0.0, 0.0, 0.0, 0.0])
    _fill_buffer(ctx, obs_x, [1.5])
    _fill_buffer(ctx, obs_y, [0.0])
    costs.enqueue_fill(F32(0.0))
    ctx.synchronize()

    ctx.enqueue_function[obstacles_dist_cost_kernel, obstacles_dist_cost_kernel](
        paths_x.unsafe_ptr(), paths_y.unsafe_ptr(),
        obs_x.unsafe_ptr(), obs_y.unsafe_ptr(),
        costs.unsafe_ptr(),
        PTS, OBS,
        F32(5.0),  # max_obstacle_distance
        F32(1.0),  # cost_weight
        grid_dim=TRAJS,
        block_dim=WG_SIZE,
    )
    ctx.synchronize()

    var result = _read_buffer(ctx, costs, TRAJS)
    var expected = Float32(0.9)  # (5.0 - 0.5) / 5.0
    assert_almost_equal(result[0], expected, atol=1e-4, msg="obstacles: traj 0")
    print("  PASS: obstacles_dist_cost_kernel")


def test_min_cost_reduction(ctx: DeviceContext) raises:
    """8 costs: [5.0, 3.0, 7.0, 1.0, 9.0, 2.0, 8.0, 4.0].
    Minimum is 1.0 at index 3.
    """
    comptime N = 8

    var costs = ctx.enqueue_create_buffer[DTYPE](N)
    var block_min_cost = ctx.enqueue_create_buffer[DTYPE](1)
    var block_min_idx = ctx.enqueue_create_buffer[DType.int32](1)
    var final_cost = ctx.enqueue_create_buffer[DTYPE](1)
    var final_idx = ctx.enqueue_create_buffer[DType.int32](1)

    _fill_buffer(ctx, costs, [5.0, 3.0, 7.0, 1.0, 9.0, 2.0, 8.0, 4.0])
    ctx.synchronize()

    # Pass 1: one block covers all 8 elements
    ctx.enqueue_function[min_cost_block_reduce, min_cost_block_reduce](
        costs.unsafe_ptr(),
        block_min_cost.unsafe_ptr(),
        block_min_idx.unsafe_ptr(),
        N,
        grid_dim=1,
        block_dim=WG_SIZE,
    )
    # Pass 2: reduce the 1 block result to final
    ctx.enqueue_function[min_cost_final_reduce, min_cost_final_reduce](
        block_min_cost.unsafe_ptr(),
        block_min_idx.unsafe_ptr(),
        final_cost.unsafe_ptr(),
        final_idx.unsafe_ptr(),
        1,  # num_blocks
        grid_dim=1,
        block_dim=WG_SIZE,
    )
    ctx.synchronize()

    var cost_result = _read_buffer(ctx, final_cost, 1)
    assert_almost_equal(cost_result[0], 1.0, atol=1e-5, msg="reduction: min cost")

    with final_idx.map_to_host() as mapped:
        var idx = Int(mapped[0])
        assert_true(idx == 3, "reduction: min idx should be 3, got " + String(idx))

    print("  PASS: min_cost_reduction")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main() raises:
    var ctx = DeviceContext()
    print("Running cost evaluator tests on:", ctx.name())

    test_goal_cost_kernel(ctx)
    test_goal_cost_tie_breaker(ctx)
    test_smoothness_cost_kernel(ctx)
    test_jerk_cost_kernel(ctx)
    test_ref_path_cost_kernel(ctx)
    test_obstacles_dist_cost_kernel(ctx)
    test_min_cost_reduction(ctx)

    print("All tests passed.")
