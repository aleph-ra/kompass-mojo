# Unit tests for the CriticalZoneChecker GPU kernels.
#
# Each test builds a small deterministic input, launches the relevant
# kernel directly (bypassing the FFI), reads the resulting safety factor
# back, and asserts against a hand-computed expected value.
#
# For simplicity every test uses an IDENTITY sensor-to-body transform
# (sensor at body origin, no rotation). That keeps test math trivial.

from std.testing import assert_true, assert_almost_equal
from std.math import sqrt, cos, sin, atan2, pi, cos as mcos
from std.memory import UnsafePointer
from std.gpu.host import DeviceContext, DeviceBuffer

from kompass_mojo.cost_evaluator import WG_SIZE
from kompass_mojo.critical_zone import (
    critical_zone_laserscan_kernel,
    critical_zone_pointcloud_kernel,
)


# ---------------------------------------------------------------------------
# Constants shared across tests
# ---------------------------------------------------------------------------


comptime ROBOT_RADIUS: Float32 = 0.51
comptime CRIT_DIST: Float32 = 0.3
comptime SLOW_DIST: Float32 = 0.6
comptime CRIT_ANGLE_DEG: Float32 = 160.0
# Upstream halves the input angle (deg) then converts to radians.
comptime CRIT_ANGLE: Float32 = (CRIT_ANGLE_DEG * 3.14159265358979323846 / 180.0) / 2.0
# cos^2(crit_angle) — precomputed here so the kernel can check the cone
# without calling atan2
comptime COS_CRIT: Float32 = 0.17364817766693041  # cos(80 degree)
comptime COS_SQ_CRIT_ANGLE: Float32 = COS_CRIT * COS_CRIT

comptime MIN_Z: Float32 = 0.1
comptime MAX_Z: Float32 = 2.0
comptime RANGE_MAX: Float32 = 20.0

# Derived thresholds that the kernel receives pre-computed.
comptime DIST_DENOM: Float32 = SLOW_DIST - CRIT_DIST
comptime SAFE_THRESHOLD: Float32 = SLOW_DIST + ROBOT_RADIUS
comptime SAFE_THRESHOLD_SQ: Float32 = SAFE_THRESHOLD * SAFE_THRESHOLD
comptime INV_DIST_RANGE: Float32 = 1.0 / DIST_DENOM


# 16B per point, float32 x/y/z at offsets 0/4/8, 4B padding.
comptime PC_STRIDE: Int = 16
comptime PC_X_OFFSET: Int = 0
comptime PC_Y_OFFSET: Int = 4
comptime PC_Z_OFFSET: Int = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fill_f32(
    ctx: DeviceContext, buf: DeviceBuffer[DType.float32], values: List[Float32],
) raises:
    var host = ctx.enqueue_create_host_buffer[DType.float32](len(values))
    var ptr = host.unsafe_ptr()
    for i in range(len(values)):
        (ptr + i)[] = values[i]
    ctx.enqueue_copy(dst_buf=buf, src_buf=host)
    ctx.synchronize()


def _fill_i32(
    ctx: DeviceContext, buf: DeviceBuffer[DType.int32], values: List[Int32],
) raises:
    var host = ctx.enqueue_create_host_buffer[DType.int32](len(values))
    var ptr = host.unsafe_ptr()
    for i in range(len(values)):
        (ptr + i)[] = values[i]
    ctx.enqueue_copy(dst_buf=buf, src_buf=host)
    ctx.synchronize()


def _fill_i8(
    ctx: DeviceContext, buf: DeviceBuffer[DType.int8], values: List[Int8],
) raises:
    var host = ctx.enqueue_create_host_buffer[DType.int8](len(values))
    var ptr = host.unsafe_ptr()
    for i in range(len(values)):
        (ptr + i)[] = values[i]
    ctx.enqueue_copy(dst_buf=buf, src_buf=host)
    ctx.synchronize()


def _read_f32(ctx: DeviceContext, buf: DeviceBuffer[DType.float32], n: Int) raises -> List[Float32]:
    var result = List[Float32]()
    with buf.map_to_host() as mapped:
        var ptr = mapped.unsafe_ptr()
        for i in range(n):
            result.append(Float32((ptr + i)[]))
    return result^


# Pack an (x, y, z) float32 triple into 16 raw bytes (PointCloud2 layout).
def _pack_point(x: Float32, y: Float32, z: Float32) -> List[Int8]:
    """Pack (x, y, z) float32 + 4B padding = 16B raw point."""
    var result = List[Int8]()
    var vals: List[Float32] = [x, y, z, Float32(0.0)]
    for v_i in range(4):
        var v = vals[v_i]
        var p = UnsafePointer(to=v).bitcast[Int8]()
        for b in range(4):
            result.append(p[b])
    return result^


def _pack_points(points: List[Tuple[Float32, Float32, Float32]]) -> List[Int8]:
    var result = List[Int8]()
    for k in range(len(points)):
        var xyz = points[k]
        var bytes = _pack_point(xyz[0], xyz[1], xyz[2])
        for b in range(len(bytes)):
            result.append(bytes[b])
    return result^


# Identity-transform helper: compute indicies_forward/backward for an
# identity sensor-to-body transform
def _compute_cone_indices_identity(
    angles: List[Float64], crit_angle: Float32,
) -> Tuple[List[Int32], List[Int32]]:
    var fwd = List[Int32]()
    var bwd = List[Int32]()
    var pi_f = Float32(pi)
    for i in range(len(angles)):
        var theta = Float32(angles[i])
        # Normalise to [-π, π] via atan2(sin, cos) to match
        # abs(atan2(y_body, x_body)) which inherently wraps.
        var abs_theta = atan2(sin(theta), cos(theta))
        if abs_theta < Float32(0.0):
            abs_theta = -abs_theta
        if abs_theta <= crit_angle:
            fwd.append(Int32(i))
        if abs_theta >= pi_f - crit_angle:
            bwd.append(Int32(i))
    return (fwd^, bwd^)


# ---------------------------------------------------------------------------
# Laserscan tests
# ---------------------------------------------------------------------------


def test_laserscan_stop_front(ctx: DeviceContext) raises:
    """Range at angle 0 = 0.2m. With identity transform, distance from
    body = 0.2m < robot_radius + crit_dist. Expected factor = 0.0."""
    # Scan: 360 bins, uniformly spaced over [0, 2π).
    comptime N = 360
    var angles = List[Float64]()
    for i in range(N):
        angles.append(Float64(2.0) * pi * Float64(i) / Float64(N))

    # Build ranges: default 10m, set angles 0 and near-0 to 0.2m.
    var ranges_host = List[Float32]()
    for i in range(N):
        ranges_host.append(Float32(10.0))
    ranges_host[0] = Float32(0.2)
    ranges_host[1] = Float32(0.2)
    ranges_host[N - 1] = Float32(0.2)

    var cos_host = List[Float32]()
    var sin_host = List[Float32]()
    for i in range(N):
        cos_host.append(Float32(cos(angles[i])))
        sin_host.append(Float32(sin(angles[i])))

    var cones = _compute_cone_indices_identity(angles, CRIT_ANGLE)
    var fwd = cones[0].copy()

    var ranges_dev = ctx.enqueue_create_buffer[DType.float32](N)
    var cos_dev = ctx.enqueue_create_buffer[DType.float32](N)
    var sin_dev = ctx.enqueue_create_buffer[DType.float32](N)
    var fwd_dev = ctx.enqueue_create_buffer[DType.int32](len(fwd))
    var out_dev = ctx.enqueue_create_buffer[DType.float32](1)

    _fill_f32(ctx, ranges_dev, ranges_host)
    _fill_f32(ctx, cos_dev, cos_host)
    _fill_f32(ctx, sin_dev, sin_host)
    _fill_i32(ctx, fwd_dev, fwd)
    out_dev.enqueue_fill(Float32(1.0))
    ctx.synchronize()

    ctx.enqueue_function[
        critical_zone_laserscan_kernel, critical_zone_laserscan_kernel,
    ](
        ranges_dev.unsafe_ptr(), cos_dev.unsafe_ptr(), sin_dev.unsafe_ptr(),
        fwd_dev.unsafe_ptr(), len(fwd),
        Float32(1.0), Float32(0.0), Float32(0.0),
        Float32(0.0), Float32(1.0), Float32(0.0),
        ROBOT_RADIUS, CRIT_DIST, DIST_DENOM, SAFE_THRESHOLD_SQ,
        out_dev.unsafe_ptr(),
        grid_dim=1, block_dim=WG_SIZE,
    )
    ctx.synchronize()

    var result = _read_f32(ctx, out_dev, 1)
    assert_almost_equal(
        result[0], Float32(0.0), atol=1e-5,
        msg="laserscan_stop_front: expected 0.0, got " + String(result[0]),
    )
    print("  PASS: test_laserscan_stop_front (factor=", result[0], ")")


def test_laserscan_slowdown_front(ctx: DeviceContext) raises:
    """Range at angle 0 = 0.87m. Identity transform → distance from body
    = 0.87m. surface = 0.87 - 0.51 = 0.36m, in slowdown band [0.3, 0.6].
    Expected factor = (0.36 - 0.3) / (0.6 - 0.3) = 0.2."""
    comptime N = 360
    var angles = List[Float64]()
    for i in range(N):
        angles.append(Float64(2.0) * pi * Float64(i) / Float64(N))

    var ranges_host = List[Float32]()
    for i in range(N):
        ranges_host.append(Float32(10.0))
    ranges_host[0] = Float32(0.87)

    var cos_host = List[Float32]()
    var sin_host = List[Float32]()
    for i in range(N):
        cos_host.append(Float32(cos(angles[i])))
        sin_host.append(Float32(sin(angles[i])))

    var cones = _compute_cone_indices_identity(angles, CRIT_ANGLE)
    var fwd = cones[0].copy()

    var ranges_dev = ctx.enqueue_create_buffer[DType.float32](N)
    var cos_dev = ctx.enqueue_create_buffer[DType.float32](N)
    var sin_dev = ctx.enqueue_create_buffer[DType.float32](N)
    var fwd_dev = ctx.enqueue_create_buffer[DType.int32](len(fwd))
    var out_dev = ctx.enqueue_create_buffer[DType.float32](1)

    _fill_f32(ctx, ranges_dev, ranges_host)
    _fill_f32(ctx, cos_dev, cos_host)
    _fill_f32(ctx, sin_dev, sin_host)
    _fill_i32(ctx, fwd_dev, fwd)
    out_dev.enqueue_fill(Float32(1.0))
    ctx.synchronize()

    ctx.enqueue_function[
        critical_zone_laserscan_kernel, critical_zone_laserscan_kernel,
    ](
        ranges_dev.unsafe_ptr(), cos_dev.unsafe_ptr(), sin_dev.unsafe_ptr(),
        fwd_dev.unsafe_ptr(), len(fwd),
        Float32(1.0), Float32(0.0), Float32(0.0),
        Float32(0.0), Float32(1.0), Float32(0.0),
        ROBOT_RADIUS, CRIT_DIST, DIST_DENOM, SAFE_THRESHOLD_SQ,
        out_dev.unsafe_ptr(),
        grid_dim=1, block_dim=WG_SIZE,
    )
    ctx.synchronize()

    var result = _read_f32(ctx, out_dev, 1)
    assert_almost_equal(
        result[0], Float32(0.2), atol=1e-3,
        msg="laserscan_slowdown_front: expected ~0.2, got " + String(result[0]),
    )
    print("  PASS: test_laserscan_slowdown_front (factor=", result[0], ")")


# ---------------------------------------------------------------------------
# Pointcloud tests
# ---------------------------------------------------------------------------


def _launch_pointcloud(
    ctx: DeviceContext,
    points: List[Tuple[Float32, Float32, Float32]],
    forward: Bool,
) raises -> Float32:
    var bytes = _pack_points(points)
    var n = len(points)

    var raw_dev = ctx.enqueue_create_buffer[DType.int8](len(bytes))
    var out_dev = ctx.enqueue_create_buffer[DType.float32](1)

    _fill_i8(ctx, raw_dev, bytes)
    out_dev.enqueue_fill(Float32(1.0))
    ctx.synchronize()

    var num_blocks = (n + WG_SIZE - 1) // WG_SIZE
    var row_step = n * PC_STRIDE

    ctx.enqueue_function[
        critical_zone_pointcloud_kernel, critical_zone_pointcloud_kernel,
    ](
        raw_dev.unsafe_ptr(), len(bytes), n,
        PC_STRIDE, row_step, n,
        Int32(1),  # is_contiguous = true
        PC_X_OFFSET, PC_Y_OFFSET, PC_Z_OFFSET,
        MIN_Z, MAX_Z,
        Float32(1.0), Float32(0.0), Float32(0.0),
        Float32(0.0), Float32(1.0), Float32(0.0),
        COS_SQ_CRIT_ANGLE,
        ROBOT_RADIUS, CRIT_DIST, INV_DIST_RANGE,
        SAFE_THRESHOLD_SQ,
        Int32(1) if forward else Int32(0),
        out_dev.unsafe_ptr(),
        grid_dim=num_blocks, block_dim=WG_SIZE,
    )
    ctx.synchronize()

    var result = _read_f32(ctx, out_dev, 1)
    return result[0]


def test_pointcloud_stop_front(ctx: DeviceContext) raises:
    """Single point at (0.7, 0, 0.5). Distance = 0.7, surface = 0.19,
    which is ≤ crit_dist=0.3 → stop. Expected factor = 0.0."""
    var points = List[Tuple[Float32, Float32, Float32]]()
    points.append((Float32(0.7), Float32(0.0), Float32(0.5)))

    var factor = _launch_pointcloud(ctx, points, forward=True)
    assert_almost_equal(
        factor, Float32(0.0), atol=1e-5,
        msg="pointcloud_stop_front: expected 0.0, got " + String(factor),
    )
    print("  PASS: test_pointcloud_stop_front (factor=", factor, ")")


def test_pointcloud_slowdown_zone(ctx: DeviceContext) raises:
    """Single point at (0.95, 0, 0.5). Distance = 0.95, surface = 0.44.
    Factor = (0.44 - 0.3) / (0.6 - 0.3) = 0.467."""
    var points = List[Tuple[Float32, Float32, Float32]]()
    points.append((Float32(0.95), Float32(0.0), Float32(0.5)))

    var factor = _launch_pointcloud(ctx, points, forward=True)
    assert_true(
        factor > Float32(0.4) and factor < Float32(0.6),
        "pointcloud_slowdown_zone: expected in (0.4, 0.6), got " + String(factor),
    )
    print("  PASS: test_pointcloud_slowdown_zone (factor=", factor, ")")


def test_pointcloud_mixed_stop_wins(ctx: DeviceContext) raises:
    """Slowdown-zone points + height-filtered outliers + one stop-zone
    point should still yield factor = 0.0 (stop dominates min)."""
    var points = List[Tuple[Float32, Float32, Float32]]()
    # Slowdown candidates
    points.append((Float32(0.95), Float32(0.0), Float32(0.5)))
    points.append((Float32(1.0), Float32(0.5), Float32(0.5)))
    # Z-filtered (above ceiling) — must be ignored
    points.append((Float32(0.5), Float32(0.0), Float32(3.0)))
    points.append((Float32(0.5), Float32(0.0), Float32(-1.0)))
    # Stop-zone point — must dominate
    points.append((Float32(0.75), Float32(0.0), Float32(0.5)))

    var factor = _launch_pointcloud(ctx, points, forward=True)
    assert_almost_equal(
        factor, Float32(0.0), atol=1e-5,
        msg="pointcloud_mixed_stop_wins: expected 0.0, got " + String(factor),
    )
    print("  PASS: test_pointcloud_mixed_stop_wins (factor=", factor, ")")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main() raises:
    var ctx = DeviceContext()
    print("Running critical zone tests on:", ctx.name())

    test_laserscan_stop_front(ctx)
    test_laserscan_slowdown_front(ctx)
    test_pointcloud_stop_front(ctx)
    test_pointcloud_slowdown_zone(ctx)
    test_pointcloud_mixed_stop_wins(ctx)

    print("All tests passed.")
